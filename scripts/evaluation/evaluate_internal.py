from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from dshcbn.models import build_model

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""): h.update(x)
    return h.hexdigest()

class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def read(self,p):
        a=sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
        return torch.from_numpy(a[None])
    def __getitem__(self,i):
        r=self.df.iloc[i]
        return {
            "local":self.read(r["local_patch_path"]),
            "context":self.read(r["context_patch_path_96"]),
            "y":torch.tensor(float(r["label"]),dtype=torch.float32),
            "patient_id":str(r["patient_id"]),
            "patch_id":str(r.get("patch_id",i))
        }

def metrics(y,p,t):
    y=np.asarray(y,int); p=np.asarray(p,float); yh=(p>=t).astype(int)
    tn=int(((y==0)&(yh==0)).sum()); fp=int(((y==0)&(yh==1)).sum())
    fn=int(((y==1)&(yh==0)).sum()); tp=int(((y==1)&(yh==1)).sum())
    prec=tp/max(tp+fp,1); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    f1=2*prec*sens/max(prec+sens,1e-12)
    f2=5*prec*sens/max(4*prec+sens,1e-12)
    return {
        "threshold":float(t),"accuracy":float((tp+tn)/len(y)),
        "precision":float(prec),"sensitivity":float(sens),"specificity":float(spec),
        "f1":float(f1),"f2":float(f2),"balanced_accuracy":float((sens+spec)/2),
        "roc_auc":float(roc_auc_score(y,p)),
        "pr_auc":float(average_precision_score(y,p)),
        "brier":float(np.mean((p-y)**2)),
        "TN":tn,"FP":fp,"FN":fn,"TP":tp
    }

def ece(y,p,bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float)
    edges=np.linspace(0,1,bins+1); s=0.; rows=[]
    ids=np.minimum(np.digitize(p,edges[1:-1]),bins-1)
    for b in range(bins):
        m=ids==b
        if not m.any():
            rows.append({"bin":b,"n":0,"mean_prob":None,"observed_rate":None}); continue
        mp=float(p[m].mean()); op=float(y[m].mean()); n=int(m.sum())
        s += n/len(y)*abs(mp-op)
        rows.append({"bin":b,"n":n,"mean_prob":mp,"observed_rate":op})
    return float(s),pd.DataFrame(rows)

def bootstrap(df,t,n_boot=10000,seed=20260828):
    rng=np.random.default_rng(seed)
    groups=[(g.y.to_numpy(np.int8),g.prob.to_numpy(float))
            for _,g in df.groupby("patient_id",sort=True)]
    vals={k:[] for k in ["accuracy","precision","sensitivity","specificity","f1","f2",
                         "balanced_accuracy","roc_auc","pr_auc","brier"]}
    t0=time.time()
    for b in range(n_boot):
        idx=rng.integers(0,len(groups),size=len(groups))
        y=np.concatenate([groups[j][0] for j in idx]); p=np.concatenate([groups[j][1] for j in idx])
        if len(np.unique(y))<2: continue
        m=metrics(y,p,t)
        for k in vals: vals[k].append(m[k])
        if (b+1)%500==0 or b==0:
            rate=(b+1)/max(time.time()-t0,1e-9)
            eta=(n_boot-b-1)/max(rate,1e-9)
            print(f"[Bootstrap] {b+1}/{n_boot} ETA~{eta:.1f}s",flush=True)
    point=metrics(df.y,df.prob,t)
    return {k:{
        "estimate":float(point[k]),
        "ci95_low":float(np.quantile(vals[k],0.025)),
        "ci95_high":float(np.quantile(vals[k],0.975)),
        "bootstrap_n":len(vals[k])
    } for k in vals}

@torch.no_grad()
def infer(model,dl,device):
    rows=[]; amp=device.type=="cuda"; total=len(dl); t0=time.time()
    model.eval()
    for bi,b in enumerate(dl,1):
        xL=b["local"].to(device,non_blocking=True)
        xC=b["context"].to(device,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
            o=model(xL,xC)
        z=o["mal_logit"].float().cpu().numpy()
        p=torch.sigmoid(o["mal_logit"].float()).cpu().numpy()
        for i in range(len(p)):
            rows.append({"patient_id":b["patient_id"][i],"patch_id":b["patch_id"][i],
                         "y":int(b["y"][i].item()),"logit":float(z[i]),"prob":float(p[i])})
        if bi==1 or bi%10==0 or bi==total:
            print(f"[Inference] {bi}/{total} batches | {time.time()-t0:.1f}s",flush=True)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_csv",required=True)
    ap.add_argument("--encoder", choices=["residual_cnn", "unet"], default="residual_cnn")
    ap.add_argument("--split_manifest",required=True)
    ap.add_argument("--frozen_spec",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--num_workers",type=int,default=0)
    ap.add_argument("--bootstrap_n",type=int,default=10000)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    result_path=out/"V42_NEW_SPLIT_TEST_RESULTS.json"
    if result_path.exists():
        raise RuntimeError("Test results already exist. Refusing to re-run the frozen test automatically.")

    spec=json.load(open(args.frozen_spec))
    if sha256(args.split_manifest)!=spec["split_manifest_sha256"]:
        raise RuntimeError("Split manifest changed after model freeze.")
    ckpt=spec["checkpoint_path"]
    if sha256(ckpt)!=spec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after freeze.")

    labels=pd.read_csv(args.labels_csv)
    if "is_augmented" in labels.columns:
        labels=labels[labels.is_augmented.astype(int)==0].copy()
    labels=labels[labels.label.astype(float).isin([0.0,1.0])].copy()
    labels["patient_id"]=labels.patient_id.astype(str).str.strip()
    m=pd.read_csv(args.split_manifest); m["patient_id"]=m.patient_id.astype(str).str.strip()
    labels["split"]=labels.patient_id.map(dict(zip(m.patient_id,m.split)))
    test=labels[labels.split=="test"].copy()
    print("[Frozen test cohort]",{
        "patients":int(test.patient_id.nunique()),"nodules":int(len(test)),
        "benign":int((test.label==0).sum()),"malignant":int((test.label==1).sum())
    },flush=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(ckpt,map_location=device,weights_only=False)
    ck_args=ck.get("args",{})
    model=build_model(args.encoder, dropout=float(ck_args.get("dropout",0.25)),
                      encoder_drop=float(ck_args.get("encoder_drop",0.05)),
                      encoder_norm=ck_args.get("encoder_norm","batch"),
                      encoder_final_channels=int(ck_args.get("encoder_final_channels",256))).to(device)
    state=ck["model_state"] if "model_state" in ck else ck
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch missing={missing}, unexpected={unexpected}")

    dl=DataLoader(DS(test),batch_size=args.batch_size,shuffle=False,
                  num_workers=args.num_workers,pin_memory=device.type=="cuda")
    pred=infer(model,dl,device)

    t_primary=float(spec["primary_threshold"]["value"])
    t_secondary=float(spec["secondary_threshold"]["value"])
    pmet=metrics(pred.y,pred.prob,t_primary)
    smet=metrics(pred.y,pred.prob,t_secondary)
    pmet["ece"],cal=ece(pred.y,pred.prob)
    smet["ece"]=pmet["ece"]
    print("[Primary frozen threshold]",json.dumps(pmet,indent=2),flush=True)
    print("[Secondary 0.50]",json.dumps(smet,indent=2),flush=True)

    ci=bootstrap(pred,t_primary,args.bootstrap_n)

    for tag,t in [("primary",t_primary),("secondary",t_secondary)]:
        d=pred.copy(); d["pred"]=(d.prob>=t).astype(int)
        d["error_type"]=np.where((d.y==1)&(d.pred==0),"FN",
                         np.where((d.y==0)&(d.pred==1),"FP","correct"))
        d[d.error_type=="FN"].sort_values("prob").to_csv(out/f"false_negatives_{tag}.csv",index=False)
        d[d.error_type=="FP"].sort_values("prob",ascending=False).to_csv(out/f"false_positives_{tag}.csv",index=False)

    result={
        "experiment":"DS-HCBN V4.2 new stratified-random patient split frozen test",
        "seed":42,
        "split_manifest_sha256":spec["split_manifest_sha256"],
        "checkpoint_sha256":spec["checkpoint_sha256"],
        "best_epoch":spec["best_epoch"],
        "target_cohort_tuning":False,
        "threshold_search_on_test":False,
        "primary_operating_point":pmet,
        "secondary_operating_point_0_50":smet,
        "patient_bootstrap_95ci_primary":ci
    }
    pred.to_csv(out/"V42_NEW_SPLIT_TEST_PREDICTIONS.csv",index=False)
    cal.to_csv(out/"V42_NEW_SPLIT_CALIBRATION_BINS.csv",index=False)
    with open(result_path,"w") as f: json.dump(result,f,indent=2)
    print("[PASS] Frozen test evaluation completed exactly with the pre-frozen checkpoint/threshold.",flush=True)

if __name__=="__main__":
    main()
