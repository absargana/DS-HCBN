from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""): h.update(x)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--checkpoint",required=True)
ap.add_argument("--results_json",required=True)
ap.add_argument("--split_manifest",required=True)
ap.add_argument("--split_lock",required=True)
ap.add_argument("--out_dir",required=True)
args=ap.parse_args()

for p in [args.checkpoint,args.results_json,args.split_manifest,args.split_lock]:
    if not os.path.isfile(p): raise FileNotFoundError(p)

out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
spec_path=out/"FROZEN_EVAL_SPEC.json"
if spec_path.exists():
    raise RuntimeError("FROZEN_EVAL_SPEC.json already exists. Refusing to overwrite frozen evaluation specification.")

r=json.load(open(args.results_json))
lock=json.load(open(args.split_lock))

# Predeclared clinical priority: highest specificity on validation while sensitivity >= 0.90.
hs=r["validation_high_sensitivity_operating_point"]
f2=r["validation_f2_operating_point"]

spec={
    "model":"DS-HCBN V4.2 pairwise-ranking",
    "seed":42,
    "checkpoint_path":os.path.abspath(args.checkpoint),
    "checkpoint_sha256":sha256(args.checkpoint),
    "split_manifest_path":os.path.abspath(args.split_manifest),
    "split_manifest_sha256":sha256(args.split_manifest),
    "split_lock_sha256":sha256(args.split_lock),
    "best_epoch":r.get("best_epoch"),
    "primary_threshold":{
        "value":float(hs["threshold"]),
        "rule":"validation-only: highest specificity subject to sensitivity >= 0.90",
        "validation_metrics":hs
    },
    "secondary_threshold":{
        "value":0.50,
        "rule":"predefined conventional threshold"
    },
    "f2_threshold_for_reference_only":{
        "value":float(f2["threshold"]),
        "validation_metrics":f2
    },
    "test_tuning_allowed":False,
    "temperature_scaling":False,
    "test_evaluated_before_freeze":False
}
if spec["split_manifest_sha256"]!=lock["manifest_sha256"]:
    raise RuntimeError("Split hash differs from SPLIT_LOCK.json.")

with open(spec_path,"w") as f: json.dump(spec,f,indent=2)
print(json.dumps(spec,indent=2))
print(f"[FROZEN] {spec_path}")
print("[IMPORTANT] No architecture, threshold, loss, or checkpoint changes after this point.")
