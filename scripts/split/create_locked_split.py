from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def parse_ratings(v):
    if v is None or (isinstance(v,float) and np.isnan(v)):
        return []
    try:
        x=json.loads(str(v))
        return [float(a) for a in x] if isinstance(x,list) else []
    except Exception:
        return []

def make_patient_table(binary):
    rows=[]
    for pid,g in binary.groupby("patient_id",sort=True):
        mal=g[g.label.astype(float)==1.0]
        ben=g[g.label.astype(float)==0.0]
        n_single=0
        for _,r in mal.iterrows():
            vals=parse_ratings(r.get("malignancy_values_json","[]"))
            nr=len(vals)
            if nr==0 and "rad_count" in r.index and pd.notna(r["rad_count"]):
                nr=int(r["rad_count"])
            if nr==1:
                n_single += 1
        rows.append({
            "patient_id":str(pid),
            "n_nodules":int(len(g)),
            "n_benign":int(len(ben)),
            "n_malignant":int(len(mal)),
            "has_malignant":int(len(mal)>0),
            "n_single_rating_malignant":int(n_single),
            "has_single_rating_malignant":int(n_single>0),
            "n_multi_rating_malignant":int(len(mal)-n_single),
            "n_borderline_malignant_mean4":int(
                np.isclose(mal["malignancy_mean"].astype(float),4.0).sum()
            ) if len(mal) and "malignancy_mean" in mal.columns else 0,
            "mean_malignancy_if_malignant":(
                float(mal["malignancy_mean"].astype(float).mean())
                if len(mal) and "malignancy_mean" in mal.columns else None
            ),
        })
    p=pd.DataFrame(rows)

    # Predeclared strata use only pre-model dataset properties.
    # No prediction, confidence, false-negative status, or test result enters the split.
    def stratum(r):
        if int(r.n_malignant)==0:
            return "B1" if int(r.n_benign)<=1 else "B2p"
        burden="M1" if int(r.n_malignant)==1 else "M2p"
        support="S" if int(r.has_single_rating_malignant)==1 else "M"
        return f"{burden}_{support}"
    p["stratum"]=p.apply(stratum,axis=1)
    return p

def split_summary(binary,manifest):
    x=binary.merge(manifest[["patient_id","split","stratum"]],on="patient_id",how="left")
    rows=[]
    for s in ["train","val","test"]:
        d=x[x.split==s]
        mal=d[d.label.astype(float)==1.0]
        single=0
        for _,r in mal.iterrows():
            vals=parse_ratings(r.get("malignancy_values_json","[]"))
            nr=len(vals)
            if nr==0 and "rad_count" in r.index and pd.notna(r["rad_count"]):
                nr=int(r["rad_count"])
            single += int(nr==1)
        rows.append({
            "split":s,
            "patients":int(d.patient_id.nunique()),
            "nodules":int(len(d)),
            "benign":int((d.label.astype(float)==0.0).sum()),
            "malignant":int((d.label.astype(float)==1.0).sum()),
            "single_rating_malignant":int(single),
            "multi_rating_malignant":int(len(mal)-single),
            "single_rating_malignant_pct":float(100*single/max(len(mal),1)),
            "borderline_malignant_mean4":int(
                np.isclose(mal["malignancy_mean"].astype(float),4.0).sum()
            ) if len(mal) and "malignancy_mean" in mal.columns else 0,
            "borderline_malignant_mean4_pct":float(
                100*np.isclose(mal["malignancy_mean"].astype(float),4.0).mean()
            ) if len(mal) and "malignancy_mean" in mal.columns else None,
            "malignancy_mean":float(mal["malignancy_mean"].astype(float).mean())
                if len(mal) and "malignancy_mean" in mal.columns else None,
            "malignant_subtlety_mean":float(mal["subtlety_mean"].astype(float).mean())
                if len(mal) and "subtlety_mean" in mal.columns else None,
            "malignant_spiculation_mean":float(mal["spiculation_mean"].astype(float).mean())
                if len(mal) and "spiculation_mean" in mal.columns else None,
            "malignant_lobulation_mean":float(mal["lobulation_mean"].astype(float).mean())
                if len(mal) and "lobulation_mean" in mal.columns else None,
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_csv",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--train_patients",type=int,default=412)
    ap.add_argument("--val_patients",type=int,default=88)
    ap.add_argument("--test_patients",type=int,default=89)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"LOCKED_SPLIT_MANIFEST.csv"
    lock_path=out/"SPLIT_LOCK.json"

    labels=pd.read_csv(args.labels_csv)
    if "is_augmented" in labels.columns:
        labels=labels[labels.is_augmented.astype(int)==0].copy()
    labels=labels[labels.label.astype(float).isin([0.0,1.0])].copy()
    labels["patient_id"]=labels.patient_id.astype(str).str.strip()

    if len(labels)!=1106 or labels.patient_id.nunique()!=589:
        raise RuntimeError(
            f"Expected corrected binary dataset 1106 nodules/589 patients, got "
            f"{len(labels)} nodules/{labels.patient_id.nunique()} patients."
        )

    patient=make_patient_table(labels)
    expected_total=args.train_patients+args.val_patients+args.test_patients
    if len(patient)!=expected_total:
        raise RuntimeError(f"Requested split totals {expected_total}, but found {len(patient)} patients.")

    # If already locked, verify reproducibility and do not create a different split.
    if manifest_path.exists() and lock_path.exists():
        print("[LOCK] Existing split found. It will not be regenerated.",flush=True)
        m=pd.read_csv(manifest_path)
        if sha256(manifest_path)!=json.load(open(lock_path))["manifest_sha256"]:
            raise RuntimeError("Existing manifest hash does not match SPLIT_LOCK.json.")
        print(m.split.value_counts().to_dict(),flush=True)
        print("[PASS] Existing frozen split verified.",flush=True)
        return

    # Stage 1: exact 89-patient test, stratified random.
    trainval,test=train_test_split(
        patient,
        test_size=args.test_patients,
        random_state=args.seed,
        shuffle=True,
        stratify=patient["stratum"]
    )
    # Stage 2: exact 88-patient validation from the remaining 500, same fixed seed.
    train,val=train_test_split(
        trainval,
        test_size=args.val_patients,
        random_state=args.seed,
        shuffle=True,
        stratify=trainval["stratum"]
    )

    assert len(train)==args.train_patients
    assert len(val)==args.val_patients
    assert len(test)==args.test_patients

    m=pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test")
    ],ignore_index=True)
    m=m.sort_values("patient_id").reset_index(drop=True)

    # Hard leakage checks.
    assert m.patient_id.nunique()==len(m)==589
    assert set(train.patient_id).isdisjoint(set(val.patient_id))
    assert set(train.patient_id).isdisjoint(set(test.patient_id))
    assert set(val.patient_id).isdisjoint(set(test.patient_id))

    m.to_csv(manifest_path,index=False)
    patient.to_csv(out/"PATIENT_STRATA_ALL.csv",index=False)
    summary=split_summary(labels,m)
    summary.to_csv(out/"SPLIT_BALANCE_SUMMARY.csv",index=False)

    ctab=pd.crosstab(m["stratum"],m["split"])
    ctab.to_csv(out/"STRATUM_COUNTS.csv")

    lock={
        "seed":args.seed,
        "method":"two-stage patient-level stratified random split",
        "strata_definition":{
            "B1":"benign-only patient with <=1 benign nodule",
            "B2p":"benign-only patient with >=2 benign nodules",
            "M1_M":"exactly 1 malignant nodule; no single-rating malignant nodule",
            "M1_S":"exactly 1 malignant nodule; has a single-rating malignant nodule",
            "M2p_M":">=2 malignant nodules; no single-rating malignant nodule",
            "M2p_S":">=2 malignant nodules; has a single-rating malignant nodule"
        },
        "sizes":{"train":args.train_patients,"val":args.val_patients,"test":args.test_patients},
        "dataset":{"binary_nodules":int(len(labels)),"binary_patients":int(labels.patient_id.nunique())},
        "manifest_sha256":sha256(manifest_path),
        "uses_model_outputs":False,
        "uses_previous_error_status":False,
        "uses_test_predictions":False
    }
    with open(lock_path,"w") as f: json.dump(lock,f,indent=2)

    print("\n[STRATUM COUNTS]",flush=True)
    print(ctab.to_string(),flush=True)
    print("\n[SPLIT BALANCE]",flush=True)
    print(summary.to_string(index=False),flush=True)
    print(f"\n[LOCKED] {manifest_path}",flush=True)
    print(f"[SHA256] {lock['manifest_sha256']}",flush=True)
    print("[PASS] Patient-level stratified-random split created and frozen.",flush=True)

if __name__=="__main__":
    main()
