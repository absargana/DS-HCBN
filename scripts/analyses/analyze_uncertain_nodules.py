from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Patches(Dataset):
    def __init__(self, frame): self.frame = frame.reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        read = lambda p: torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)[None])
        return read(r.local_patch_path), read(r.context_patch_path_96), str(r.patient_id), str(r.patch_id), float(r.malignancy_mean), int(r.rad_count)


def summarize(d):
    rho, p = spearmanr(d.malignancy_mean, d.prob)
    return {"n_nodules": int(len(d)), "n_patients": int(d.patient_id.nunique()),
            "probability_median": float(d.prob.median()), "probability_q1": float(d.prob.quantile(.25)),
            "probability_q3": float(d.prob.quantile(.75)), "spearman_rho": float(rho), "spearman_p": float(p)}


def patient_bootstrap(d, n, seed):
    rng = np.random.default_rng(seed); ids = d.patient_id.unique(); vals = []
    for _ in range(n):
        chosen = rng.choice(ids, len(ids), replace=True)
        parts = [d[d.patient_id == pid].assign(_boot=i) for i, pid in enumerate(chosen)]
        x = pd.concat(parts, ignore_index=True)
        r = spearmanr(x.malignancy_mean, x.prob).statistic
        if np.isfinite(r): vals.append(r)
    return [float(x) for x in np.quantile(vals, [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True); ap.add_argument("--frozen_spec", required=True); ap.add_argument("--checkpoint")
    ap.add_argument("--encoder", choices=["residual_cnn", "unet"], default="residual_cnn"); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=2); ap.add_argument("--bootstrap_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(a.frozen_spec, encoding="utf-8")); ckpt = Path(a.checkpoint or spec["checkpoint_path"])
    if sha256(ckpt) != spec["checkpoint_sha256"]: raise RuntimeError("Frozen checkpoint hash mismatch")
    df = pd.read_csv(a.labels_csv); df = df[(df.label.astype(float) == .5) & (df.is_augmented.astype(int) == 0)].copy()
    for c in ["local_patch_path", "context_patch_path_96"]:
        missing = df.loc[~df[c].map(lambda p: Path(str(p)).is_file()), c]
        if len(missing): raise FileNotFoundError(f"{len(missing)} missing paths in {c}; first={missing.iloc[0]}")
    from dshcbn.models import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    model = build_model(a.encoder, dropout=float(ck_args.get("dropout", .25)),
                        encoder_drop=float(ck_args.get("encoder_drop", .05)),
                        encoder_norm=ck_args.get("encoder_norm", "batch"),
                        encoder_final_channels=int(ck_args.get("encoder_final_channels", 256))).to(device)
    model.load_state_dict(ck.get("model_state", ck), strict=True); model.eval()
    rows = []
    dl = DataLoader(Patches(df), batch_size=a.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    with torch.no_grad():
        for xl, xc, pid, patch, rating, nr in tqdm(dl, desc="CNN uncertain-nodule inference", unit="batch", dynamic_ncols=True):
            xl, xc = xl.to(device), xc.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(xl, xc)["mal_logit"]).float().cpu().numpy()
            rows += [{"patient_id": pid[i], "patch_id": patch[i], "malignancy_mean": float(rating[i]), "rad_count": int(nr[i]), "prob": float(prob[i])} for i in range(len(prob))]
    pred = pd.DataFrame(rows); primary = float(spec["primary_threshold"]["value"]); secondary = float(spec["secondary_threshold"]["value"])
    pred["above_primary_threshold"] = pred.prob >= primary; pred["above_0_5"] = pred.prob >= secondary
    strata = pred.groupby("malignancy_mean").prob.agg(["count", "median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).reset_index()
    strata.columns = ["malignancy_mean", "n", "median_probability", "q1_probability", "q3_probability"]
    result = {"analysis": "Frozen primary-model behavior on uncertain LIDC nodules", "checkpoint_sha256": spec["checkpoint_sha256"],
              "no_retraining": True, "no_binary_accuracy_claimed": True, "device": str(device), "all_uncertain": summarize(pred),
              "spearman_patient_bootstrap_95ci": patient_bootstrap(pred, a.bootstrap_n, a.seed),
              "multi_reader_only": summarize(pred[pred.rad_count >= 2]),
              "proportion_above_frozen_primary": float(pred.above_primary_threshold.mean()),
              "proportion_above_0_5": float(pred.above_0_5.mean()), "frozen_primary_threshold": primary}
    pred.to_csv(out / "uncertain_nodule_predictions.csv", index=False); strata.to_csv(out / "uncertain_rating_strata.csv", index=False)
    json.dump(result, open(out / "uncertain_nodule_summary.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
