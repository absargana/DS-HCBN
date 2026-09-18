from __future__ import annotations
import argparse, copy, json, math, os, random, time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, balanced_accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from dshcbn.models import CONCEPT_SPECS, build_model

# sklearn has no specificity_score; kept explicit below.

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def parse_values(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return []
    try:
        x = json.loads(str(v))
        return [int(a) for a in x] if isinstance(x,list) else []
    except Exception:
        return []

def ordinal_soft_target(vals, ncls=5):
    vals = [int(v) for v in vals if 1 <= int(v) <= ncls]
    if not vals: return torch.zeros(ncls-1), torch.tensor(0.0)
    a = np.asarray(vals, dtype=np.float32)
    t = np.asarray([(a > k).mean() for k in range(1,ncls)], dtype=np.float32)
    return torch.from_numpy(t), torch.tensor(1.0)

def categorical_soft_target(vals, ncls):
    vals = [int(v) for v in vals if 1 <= int(v) <= ncls]
    if not vals: return torch.zeros(ncls), torch.tensor(0.0)
    c = np.bincount(np.asarray(vals)-1, minlength=ncls).astype(np.float32)
    return torch.from_numpy(c/c.sum()), torch.tensor(1.0)

class NoduleDataset(Dataset):
    def __init__(self, df, augment=False, augment_prob=0.60, noise_prob=0.25):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_prob = noise_prob

    def __len__(self): return len(self.df)

    def _read(self, p):
        return torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)[None])

    def _aug_pair(self, a, b):
        # Same geometry for local/context; intensity perturbation remains very mild.
        if random.random() < self.augment_prob:
            for dim in (1,2,3):
                if random.random() < 0.5:
                    a = torch.flip(a, dims=(dim,))
                    b = torch.flip(b, dims=(dim,))
            if random.random() < 0.50:
                k = random.randint(0,3)
                plane = random.choice(((2,3),(1,3),(1,2)))
                a = torch.rot90(a, k, dims=plane)
                b = torch.rot90(b, k, dims=plane)
        if random.random() < self.noise_prob:
            shift = random.uniform(-0.015, 0.015)
            sigma = random.uniform(0.0, 0.010)
            a = (a + shift + torch.randn_like(a)*sigma).clamp(0,1)
            b = (b + shift + torch.randn_like(b)*sigma).clamp(0,1)
        return a,b

    def __getitem__(self, i):
        r = self.df.iloc[i]
        xL = self._read(r["local_patch_path"])
        xC = self._read(r["context_patch_path_96"])
        if self.augment:
            xL,xC = self._aug_pair(xL,xC)

        concepts, masks = {}, {}
        for name,(kind,ncls) in CONCEPT_SPECS.items():
            vals = parse_values(r.get(f"{name}_values_json", "[]"))
            if kind == "ordinal":
                t,m = ordinal_soft_target(vals,ncls)
            else:
                t,m = categorical_soft_target(vals,ncls)
            concepts[name] = t
            masks[name] = m

        return {
            "local": xL, "context": xC,
            "y": torch.tensor(float(r["label"]), dtype=torch.float32),
            "patient_id": str(r["patient_id"]),
            "patch_id": str(r.get("patch_id", i)),
            "concept_targets": concepts,
            "concept_masks": masks,
        }

def weighted_sampler(df):
    y = df["label"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2).astype(np.float64)
    inv = np.where(counts>0, 1.0/counts, 0.0)
    return WeightedRandomSampler(torch.as_tensor(inv[y],dtype=torch.double), len(y), replacement=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.60, gamma=1.5):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, y):
        y = y.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = y*p + (1-y)*(1-p)
        at = y*self.alpha + (1-y)*(1-self.alpha)
        return (at * (1-pt).pow(self.gamma) * bce).mean()

def concept_loss(out, batch):
    total = torch.tensor(0.0, device=out["mal_logit"].device)
    denom = 0.0
    for name,(kind,ncls) in CONCEPT_SPECS.items():
        z = out["concept_logits"][name]
        t = batch["concept_targets"][name].to(z.device)
        m = batch["concept_masks"][name].to(z.device)
        if kind == "ordinal":
            per = F.binary_cross_entropy_with_logits(z, t, reduction="none").mean(dim=1)
        else:
            per = -(t * F.log_softmax(z,dim=1)).sum(dim=1)
        total = total + (per*m).sum()
        denom += float(m.sum().detach().cpu())
    return total / max(denom,1.0)

def metrics(y,p,thr=0.5):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    yh=(p>=thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,yh,labels=[0,1]).ravel()
    return {
        "accuracy":float(accuracy_score(y,yh)),
        "precision":float(precision_score(y,yh,zero_division=0)),
        "sensitivity":float(recall_score(y,yh,zero_division=0)),
        "specificity":float(tn/max(tn+fp,1)),
        "f1":float(f1_score(y,yh,zero_division=0)),
        "f2":float(fbeta_score(y,yh,beta=2,zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,yh)),
        "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),
        "pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))>1 else float("nan"),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"threshold":float(thr)
    }

def best_f2(y,p):
    best=None
    for t in np.linspace(0.05,0.95,181):
        m=metrics(y,p,float(t))
        key=(m["f2"],m["sensitivity"],m["specificity"])
        if best is None or key>best[0]: best=(key,m)
    return best[1]

def high_sens(y,p,target=0.90):
    best=None
    for t in np.linspace(0.01,0.99,197):
        m=metrics(y,p,float(t))
        if m["sensitivity"]+1e-12 >= target:
            key=(m["specificity"],m["precision"],m["f2"],t)
            if best is None or key>best[0]: best=(key,m)
    return best[1] if best else metrics(y,p,0.0)

@torch.no_grad()
def predict(model, dl, device, amp=True):
    model.eval(); rows=[]
    for b in dl:
        xL=b["local"].to(device,non_blocking=True)
        xC=b["context"].to(device,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
            o=model(xL,xC)
        p=torch.sigmoid(o["mal_logit"].float()).cpu().numpy()
        z=o["mal_logit"].float().cpu().numpy()
        for i in range(len(p)):
            rows.append({
                "patient_id":b["patient_id"][i],
                "patch_id":b["patch_id"][i],
                "y":int(b["y"][i].item()),
                "logit":float(z[i]),"prob":float(p[i]),
                "context_cnn_gate":float(o["context_cnn_gate"][i].float().cpu()),
                "attention_gate":float(o["attention_gate"][i].float().cpu()),
                "concept_scale":float(o["concept_scale"][i].float().cpu()),
            })
    return pd.DataFrame(rows)


def pairwise_ranking_loss(logits, y, margin=0.50):
    """
    Pairwise logistic margin loss on the primary binary logits.
    It is active only when the current batch contains at least one malignant
    and one benign sample. With the existing weighted sampler and batch size 2,
    this occurs frequently without increasing GPU memory.
    """
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    diff = pos[:, None] - neg[None, :]
    return F.softplus(margin - diff).mean()

def total_loss(model, batch, device, mal_loss, lambda_concept, lambda_rank, rank_margin, amp):
    xL=batch["local"].to(device,non_blocking=True)
    xC=batch["context"].to(device,non_blocking=True)
    y=batch["y"].to(device,non_blocking=True)
    with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
        out=model(xL,xC)
        lm=mal_loss(out["mal_logit"],y)
        lc=concept_loss(out,batch)
        lrk=pairwise_ranking_loss(out["mal_logit"],y,margin=rank_margin)
        loss=lm + lambda_concept*lc + lambda_rank*lrk
    return loss,lm.detach(),lc.detach(),lrk.detach()

def make_scheduler(opt, epochs, warmup=3):
    def f(ep):
        if ep < warmup: return float(ep+1)/float(max(warmup,1))
        x=(ep-warmup)/float(max(epochs-warmup,1))
        return 0.5*(1.0+math.cos(math.pi*min(max(x,0.0),1.0)))
    return LambdaLR(opt,f)

def diagnosis(train_m,val_m,best_epoch):
    ag=train_m["roc_auc"]-val_m["roc_auc"]
    pg=train_m["pr_auc"]-val_m["pr_auc"]
    lines=[]
    if ag >= 0.07:
        lines.append("STRONG_OVERFIT: train-val ROC-AUC gap >= 7 pp.")
    elif ag >= 0.04:
        lines.append("MODERATE_OVERFIT: train-val ROC-AUC gap >= 4 pp.")
    else:
        lines.append("CONTROLLED_GAP: train-val ROC-AUC gap < 4 pp.")
    if pg >= 0.07: lines.append("PR_OVERFIT: train-val PR-AUC gap >= 7 pp.")
    if val_m["roc_auc"] < 0.90 and train_m["roc_auc"] < 0.94:
        lines.append("UNDERFIT_SIGNAL: both train and validation discrimination are limited.")
    if val_m["sensitivity"] < 0.80:
        lines.append("RECALL_SIGNAL: operating threshold/positive representation needs attention.")
    if best_epoch <= 8 and ag >= 0.04:
        lines.append("EARLY_OVERFIT: best generalization occurs early; further capacity is unlikely to help.")
    if not lines: lines.append("NO_MAJOR_FAILURE_RULE_TRIGGERED.")
    return lines

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_csv",required=True)
    ap.add_argument("--encoder", choices=["residual_cnn", "unet"], default="residual_cnn")
    ap.add_argument("--split_manifest",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=45)
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--num_workers",type=int,default=0)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--weight_decay",type=float,default=1e-3)
    ap.add_argument("--dropout",type=float,default=0.25)
    ap.add_argument("--encoder_drop",type=float,default=0.05)
    ap.add_argument("--encoder_norm",choices=["batch","group","instance"],default="batch")
    ap.add_argument("--encoder_final_channels",type=int,choices=[192,256],default=256)
    ap.add_argument("--lambda_concept",type=float,default=0.35)
    ap.add_argument("--lambda_rank",type=float,default=0.10)
    ap.add_argument("--rank_margin",type=float,default=0.50)
    ap.add_argument("--focal_alpha",type=float,default=0.60)
    ap.add_argument("--focal_gamma",type=float,default=1.5)
    ap.add_argument("--ema_decay",type=float,default=0.995)
    ap.add_argument("--early_stop",type=int,default=8)
    ap.add_argument("--target_sensitivity",type=float,default=0.90)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=(device.type=="cuda")
    print(f"[Device] {device} | {torch.cuda.get_device_name(0) if amp else 'CPU'}")
    print("[Policy] TRAIN+VAL ONLY. The manifest test split is never instantiated or evaluated.")

    df=pd.read_csv(args.labels_csv).copy()
    if "is_augmented" in df.columns:
        df=df[df.is_augmented.astype(int)==0].copy()
    df=df[df.label.astype(float).isin([0.0,1.0])].copy()
    man=pd.read_csv(args.split_manifest)
    man["patient_id"]=man.patient_id.astype(str).str.strip()
    df["patient_id"]=df.patient_id.astype(str).str.strip()
    mp=dict(zip(man.patient_id,man.split))
    df["split"]=df.patient_id.map(mp)
    if df["split"].isna().any():
        raise ValueError("Some binary patients are absent from split manifest")
    tr=df[df.split=="train"].copy()
    va=df[df.split=="val"].copy()
    te=df[df.split=="test"].copy()
    assert set(tr.patient_id).isdisjoint(set(va.patient_id))
    assert set(tr.patient_id).isdisjoint(set(te.patient_id))
    assert set(va.patient_id).isdisjoint(set(te.patient_id))

    summary={
        "train":{"patients":tr.patient_id.nunique(),"nodules":len(tr),"benign":int((tr.label==0).sum()),"malignant":int((tr.label==1).sum())},
        "val":{"patients":va.patient_id.nunique(),"nodules":len(va),"benign":int((va.label==0).sum()),"malignant":int((va.label==1).sum())},
        "test_held_out_not_evaluated":{"patients":te.patient_id.nunique(),"nodules":len(te),"benign":int((te.label==0).sum()),"malignant":int((te.label==1).sum())},
    }
    print(json.dumps(summary,indent=2))

    ds_tr=NoduleDataset(tr,augment=True)
    ds_tr_clean=NoduleDataset(tr,augment=False)
    ds_va=NoduleDataset(va,augment=False)
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,sampler=weighted_sampler(tr),num_workers=args.num_workers,
                     pin_memory=amp,drop_last=False)
    dl_tr_clean=DataLoader(ds_tr_clean,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=amp)
    dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=amp)

    model=build_model(args.encoder, dropout=args.dropout, encoder_drop=args.encoder_drop,
                      encoder_norm=args.encoder_norm,
                      encoder_final_channels=args.encoder_final_channels).to(device)
    ema=AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True).to(device)
    opt=AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sch=make_scheduler(opt,args.epochs,warmup=3)
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    mal_loss=FocalLoss(args.focal_alpha,args.focal_gamma)

    npar=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] trainable parameters: {npar:,}")
    history=[]; best_score=-1e9; best_epoch=0; bad=0

    for ep in range(1,args.epochs+1):
        model.train(); sumloss=0.; sumrank=0.; rank_batches=0; n=0
        for b in dl_tr:
            opt.zero_grad(set_to_none=True)
            loss,lm,lc,lrk=total_loss(
                model,b,device,mal_loss,args.lambda_concept,
                args.lambda_rank,args.rank_margin,amp
            )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            ema.update_parameters(model)
            bs=len(b["y"]); sumloss += float(loss.detach().cpu())*bs; n+=bs
            if float(lrk.detach().cpu()) > 0:
                sumrank += float(lrk.detach().cpu())
                rank_batches += 1
        sch.step()

        trp=predict(ema,dl_tr_clean,device,amp)
        vap=predict(ema,dl_va,device,amp)
        tm=metrics(trp.y,trp.prob,0.5)
        vm05=metrics(vap.y,vap.prob,0.5)
        vf2=best_f2(vap.y,vap.prob)
        vhs=high_sens(vap.y,vap.prob,args.target_sensitivity)
        auc_gap=max(tm["roc_auc"]-vm05["roc_auc"],0.0)
        pr_gap=max(tm["pr_auc"]-vm05["pr_auc"],0.0)
        score=0.55*vm05["roc_auc"]+0.45*vm05["pr_auc"]-0.20*auc_gap-0.10*pr_gap
        row={
            "epoch":ep,"train_loss":sumloss/max(n,1),
            "mean_active_rank_loss":sumrank/max(rank_batches,1),
            "active_rank_batches":int(rank_batches),
            "lr":opt.param_groups[0]["lr"],
            "train_auc":tm["roc_auc"],"train_pr_auc":tm["pr_auc"],"train_f1_05":tm["f1"],
            "val_auc":vm05["roc_auc"],"val_pr_auc":vm05["pr_auc"],"val_f1_05":vm05["f1"],
            "auc_gap":tm["roc_auc"]-vm05["roc_auc"],"pr_gap":tm["pr_auc"]-vm05["pr_auc"],
            "generalization_score":score,
            "val_hs_sensitivity":vhs["sensitivity"],"val_hs_specificity":vhs["specificity"],"val_hs_threshold":vhs["threshold"],
            "val_f2":vf2["f2"],"val_f2_threshold":vf2["threshold"]
        }
        history.append(row)
        print(
            f"Ep{ep:03d} loss={row['train_loss']:.4f} | "
            f"Train AUC/PR={tm['roc_auc']:.3f}/{tm['pr_auc']:.3f} | "
            f"Val AUC/PR={vm05['roc_auc']:.3f}/{vm05['pr_auc']:.3f} | "
            f"Gap AUC/PR={row['auc_gap']:.3f}/{row['pr_gap']:.3f} | "
            f"HS90 S/Sp={vhs['sensitivity']:.3f}/{vhs['specificity']:.3f} thr={vhs['threshold']:.3f} | "
            f"GScore={score:.4f}"
        )

        if score > best_score + 1e-4:
            best_score=score; best_epoch=ep; bad=0
            torch.save({
                "model_state":ema.module.state_dict(),
                "epoch":ep,"generalization_score":score,
                "args":vars(args),"split_summary":summary
            },out/"best_model.pt")
            trp.to_csv(out/"best_train_predictions.csv",index=False)
            vap.to_csv(out/"best_validation_predictions.csv",index=False)
        else:
            bad += 1
        pd.DataFrame(history).to_csv(out/"epoch_history.csv",index=False)
        if bad >= args.early_stop:
            print(f"[EarlyStop] no generalization-score improvement for {args.early_stop} epochs.")
            break

    ck=torch.load(out/"best_model.pt",map_location=device,weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    trp=predict(model,dl_tr_clean,device,amp)
    vap=predict(model,dl_va,device,amp)
    tm05=metrics(trp.y,trp.prob,0.5); vm05=metrics(vap.y,vap.prob,0.5)
    vf2=best_f2(vap.y,vap.prob); vhs=high_sens(vap.y,vap.prob,args.target_sensitivity)
    diag=diagnosis(tm05,vm05,best_epoch)

    # Threshold sweep is validation-only and diagnostic.
    sweep=[]
    for t in np.linspace(0.10,0.90,33):
        sweep.append(metrics(vap.y,vap.prob,float(t)))
    pd.DataFrame(sweep).to_csv(out/"validation_threshold_sweep.csv",index=False)

    results={
        "experiment":"v42_pairwise_ranking_seed42",
        "seed":args.seed,
        "test_evaluated":False,
        "split_summary":summary,
        "trainable_parameters":npar,
        "best_epoch":best_epoch,
        "best_generalization_score":best_score,
        "best_train_metrics_at_0.5":tm05,
        "best_validation_metrics_at_0.5":vm05,
        "validation_f2_operating_point":vf2,
        "validation_high_sensitivity_operating_point":vhs,
        "train_val_gap":{
            "roc_auc":tm05["roc_auc"]-vm05["roc_auc"],
            "pr_auc":tm05["pr_auc"]-vm05["pr_auc"],
            "f1_at_0.5":tm05["f1"]-vm05["f1"]
        },
        "learned_gates":{
            "context_cnn_gate":float(torch.sigmoid(model.context_cnn_gate_logit).detach().cpu()),
            "attention_gate":float(torch.sigmoid(model.attn_gate_logit).detach().cpu()),
            "concept_scale":float(torch.sigmoid(model.concept_scale_logit).detach().cpu()),
        },
        "automatic_diagnosis":diag,
        "design":{
            "framework":"dual-scale CNN + context Transformer + local-anchored cross-attention + concept supervision",
            "normalization":"GroupNorm",
            "shared_cnn_weights":True,
            "transformer_depth":2,
            "transformer_dim":128,
            "dropout":args.dropout,
            "weighted_sampler":True,
            "augmentation":"mild paired all-class spatial/intensity, train only",
            "ema_decay":args.ema_decay,
            "checkpoint_rule":"0.55*val_AUC + 0.45*val_PR_AUC - 0.20*positive_AUC_gap - 0.10*positive_PR_gap",
            "malignancy_loss":{"type":"focal","alpha_positive":args.focal_alpha,"gamma":args.focal_gamma},
            "lambda_concept":args.lambda_concept,
            "lambda_pairwise_rank":args.lambda_rank,
            "pairwise_rank_margin":args.rank_margin,
            "pairwise_rank_note":"primary-logit positive-vs-negative separation loss; no target-cohort information",
        }
    }
    with open(out/"results.json","w") as f: json.dump(results,f,indent=2)
    with open(out/"DIAGNOSIS.txt","w") as f:
        f.write("\n".join(diag)+"\n")
    print("\n[RESULT]")
    print(json.dumps(results,indent=2))
    print("\n[STOP] Do not run the test. Upload V42_REVIEW.zip for comparison against V4.0.")

if __name__=="__main__":
    main()
