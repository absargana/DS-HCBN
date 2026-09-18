import argparse, hashlib, json, os
from pathlib import Path
import pandas as pd
import torch

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""): h.update(x)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--labels_csv",required=True)
ap.add_argument("--split_manifest",required=True)
ap.add_argument("--split_lock",required=True)
args=ap.parse_args()

for p in [args.labels_csv,args.split_manifest,args.split_lock]:
    if not os.path.isfile(p): raise FileNotFoundError(p)

lock=json.load(open(args.split_lock))
if sha256(args.split_manifest)!=lock["manifest_sha256"]:
    raise RuntimeError("Split manifest hash mismatch: frozen split was modified.")

df=pd.read_csv(args.labels_csv)
if "is_augmented" in df.columns:
    df=df[df.is_augmented.astype(int)==0].copy()
df=df[df.label.astype(float).isin([0.0,1.0])].copy()
df["patient_id"]=df.patient_id.astype(str).str.strip()
m=pd.read_csv(args.split_manifest)
m["patient_id"]=m.patient_id.astype(str).str.strip()

if m.patient_id.nunique()!=589: raise RuntimeError("Manifest must contain 589 unique patients.")
if set(df.patient_id.unique())!=set(m.patient_id.unique()):
    raise RuntimeError("Manifest patient set differs from corrected binary dataset.")

mp=dict(zip(m.patient_id,m.split))
df["split"]=df.patient_id.map(mp)
report={}
for s in ["train","val","test"]:
    d=df[df.split==s]
    report[s]={
        "patients":int(d.patient_id.nunique()),"nodules":int(len(d)),
        "benign":int((d.label.astype(float)==0).sum()),
        "malignant":int((d.label.astype(float)==1).sum())
    }

required=["local_patch_path","context_patch_path_96"]
for c in required:
    if c not in df.columns: raise RuntimeError(f"Missing column {c}")
for c in required:
    for p in df[c].dropna().astype(str).head(20):
        if not os.path.isfile(p): raise FileNotFoundError(f"Patch missing: {p}")

print(json.dumps({
    "split":report,
    "manifest_sha256":lock["manifest_sha256"],
    "cuda_available":torch.cuda.is_available(),
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch":torch.__version__
},indent=2))
if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available.")
print("[PASS] Frozen split and patch paths verified.")
