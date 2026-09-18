#!/usr/bin/env python3
"""Frozen DS-HCBN V4.2 inference/evaluation on the official LNDb test cases.

No training, fine-tuning, calibration fitting, or threshold search is performed.
The official LNDb testNodules.csv has no public ground-truth labels, so inference
always runs and metrics are produced only when an optional labels CSV is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_cube(a: np.ndarray, center_xyz: tuple[int, int, int], size: int) -> np.ndarray:
    cx, cy, cz = center_xyz
    half = size // 2
    x0, y0, z0 = cx - half, cy - half, cz - half
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    zd, yd, xd = a.shape
    ix0, ix1 = max(0, x0), min(xd, x1)
    iy0, iy1 = max(0, y0), min(yd, y1)
    iz0, iz1 = max(0, z0), min(zd, z1)
    out = np.zeros((size, size, size), dtype=np.float32)
    ox0, oy0, oz0 = ix0 - x0, iy0 - y0, iz0 - z0
    out[oz0:oz0 + iz1 - iz0, oy0:oy0 + iy1 - iy0, ox0:ox0 + ix1 - ix0] = (
        a[iz0:iz1, iy0:iy1, ix0:ix1]
    )
    return out


def load_normalized_isotropic(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    old_spacing, old_size = img.GetSpacing(), img.GetSize()
    spacing = (1.0, 1.0, 1.0)
    new_size = [int(round(old_size[i] * old_spacing[i] / spacing[i])) for i in range(3)]
    rs = sitk.Resample(
        img, new_size, sitk.Transform(), sitk.sitkLinear,
        img.GetOrigin(), spacing, img.GetDirection(), -1000.0, sitk.sitkFloat32,
    )
    return (sitk.Clamp(rs, lowerBound=-1000.0, upperBound=400.0) + 1000.0) / 1400.0


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yh = (p >= threshold).astype(int)
    tn = int(((y == 0) & (yh == 0)).sum())
    fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum())
    tp = int(((y == 1) & (yh == 1)).sum())
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    f2 = 5 * precision * sensitivity / max(4 * precision + sensitivity, 1e-12)
    return {
        "threshold": float(threshold), "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision), "sensitivity": float(sensitivity),
        "specificity": float(specificity), "f1": float(f1), "f2": float(f2),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    total = 0.0
    rows = []
    for b in range(bins):
        mask = ids == b
        if not mask.any():
            rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "n": 0,
                         "mean_prob": None, "observed_rate": None})
            continue
        mean_prob = float(p[mask].mean())
        observed = float(y[mask].mean())
        count = int(mask.sum())
        total += count / len(y) * abs(mean_prob - observed)
        rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "n": count,
                     "mean_prob": mean_prob, "observed_rate": observed})
    return float(total), pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame, threshold: float, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = [(g.label.to_numpy(dtype=np.int8), g.prob.to_numpy(dtype=float))
              for _, g in frame.groupby("LNDbID", sort=True)]
    keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "f2",
            "balanced_accuracy", "roc_auc", "pr_auc", "brier"]
    values = {k: [] for k in keys}
    started = time.time()
    for iteration in range(n):
        chosen = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([groups[j][0] for j in chosen])
        p = np.concatenate([groups[j][1] for j in chosen])
        if np.unique(y).size < 2:
            continue
        m = metrics(y, p, threshold)
        for k in keys:
            values[k].append(m[k])
        if iteration == 0 or (iteration + 1) % 1000 == 0:
            print(f"[Bootstrap] {iteration + 1}/{n}; elapsed={time.time()-started:.1f}s", flush=True)
    point = metrics(frame.label.to_numpy(), frame.prob.to_numpy(), threshold)
    return {k: {"estimate": point[k], "ci95_low": float(np.quantile(values[k], .025)),
                "ci95_high": float(np.quantile(values[k], .975)), "bootstrap_n": len(values[k])}
            for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_dir", type=Path, required=True)
    ap.add_argument("--candidates_csv", type=Path, required=True)
    ap.add_argument("--encoder", choices=["residual_cnn", "unet"], default="residual_cnn")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--frozen_spec", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path, help="Optional CSV with LNDbID, FindingID, label")
    ap.add_argument("--dataset_name", default="LNDb official 58-CT test split")
    ap.add_argument("--output_prefix", default="LNDb_test")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.frozen_spec.read_text(encoding="utf-8"))
    actual_hash = sha256(args.checkpoint)
    if actual_hash != spec["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: expected {spec['checkpoint_sha256']}, got {actual_hash}")

    from dshcbn.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    model = build_model(args.encoder, dropout=float(ck_args.get("dropout", .25)),
                        encoder_drop=float(ck_args.get("encoder_drop", .05)),
                        encoder_norm=ck_args.get("encoder_norm", "batch"),
                        encoder_final_channels=int(ck_args.get("encoder_final_channels", 256))).to(device)
    state = ck["model_state"] if "model_state" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[Device] {device}" + (f" - {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    candidates = pd.read_csv(args.candidates_csv)
    required = {"LNDbID", "FindingID", "x", "y", "z"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidates CSV missing {sorted(required - set(candidates.columns))}")
    candidates = candidates.sort_values(["LNDbID", "FindingID"]).reset_index(drop=True)
    rows: list[dict] = []
    started = time.time()
    amp = device.type == "cuda"

    with torch.no_grad():
        for scan_i, (scan_id, group) in enumerate(candidates.groupby("LNDbID", sort=True), 1):
            ct_path = args.ct_dir / f"LNDb-{int(scan_id):04d}.mhd"
            if not ct_path.exists():
                raise FileNotFoundError(ct_path)
            image = load_normalized_isotropic(ct_path)
            array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
            items = []
            for r in group.itertuples(index=False):
                idx = image.TransformPhysicalPointToContinuousIndex((float(r.x), float(r.y), float(r.z)))
                center = tuple(int(math.floor(v + .5)) for v in idx)
                items.append((r, center))
            for start in range(0, len(items), args.batch_size):
                batch = items[start:start + args.batch_size]
                local = torch.from_numpy(np.stack([extract_cube(array, c, 64) for _, c in batch])[:, None]).to(device)
                context = torch.from_numpy(np.stack([extract_cube(array, c, 96) for _, c in batch])[:, None]).to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(local, context)
                logits = out["mal_logit"].float().cpu().numpy()
                probs = torch.sigmoid(out["mal_logit"].float()).cpu().numpy()
                for (r, center), logit, prob in zip(batch, logits, probs):
                    rows.append({"LNDbID": int(r.LNDbID), "FindingID": int(r.FindingID),
                                 "x": float(r.x), "y": float(r.y), "z": float(r.z),
                                 "resampled_i": center[0], "resampled_j": center[1], "resampled_k": center[2],
                                 "logit": float(logit), "prob": float(prob),
                                 "pred_primary_0_23": int(prob >= float(spec["primary_threshold"]["value"])),
                                 "pred_secondary_0_50": int(prob >= float(spec["secondary_threshold"]["value"]))})
            print(f"[Inference] scan {scan_i}/{candidates.LNDbID.nunique()} LNDb-{int(scan_id):04d}; "
                  f"candidates={len(group)} elapsed={time.time()-started:.1f}s", flush=True)

    pred = pd.DataFrame(rows)
    pred_path = args.out_dir / f"{args.output_prefix}_predictions.csv"
    pred.to_csv(pred_path, index=False)
    result = {
        "dataset": args.dataset_name, "checkpoint_sha256": actual_hash,
        "retraining": False, "fine_tuning": False, "threshold_search_on_external_test": False,
        "n_scans": int(pred.LNDbID.nunique()), "n_candidates": int(len(pred)),
        "primary_threshold": float(spec["primary_threshold"]["value"]),
        "secondary_threshold": float(spec["secondary_threshold"]["value"]),
        "ground_truth_available": bool(args.labels_csv),
    }
    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv)
        needed = {"LNDbID", "FindingID", "label"}
        if not needed.issubset(labels.columns):
            raise ValueError(f"Labels CSV missing {sorted(needed - set(labels.columns))}")
        scored = pred.merge(labels[["LNDbID", "FindingID", "label"]], on=["LNDbID", "FindingID"], how="inner", validate="one_to_one")
        if len(scored) != len(pred):
            raise ValueError(f"Labels cover {len(scored)}/{len(pred)} predictions; refusing partial scoring")
        if not scored.label.astype(float).isin([0, 1]).all() or scored.label.nunique() < 2:
            raise ValueError("Labels must be binary 0/1 and contain both classes")
        primary = float(spec["primary_threshold"]["value"])
        secondary = float(spec["secondary_threshold"]["value"])
        result["primary_metrics"] = metrics(scored.label.to_numpy(), scored.prob.to_numpy(), primary)
        result["secondary_metrics"] = metrics(scored.label.to_numpy(), scored.prob.to_numpy(), secondary)
        result["patient_bootstrap_95ci_primary"] = bootstrap(scored, primary, args.bootstrap_n, args.seed)
        scored.to_csv(args.out_dir / f"{args.output_prefix}_predictions_scored.csv", index=False)
    else:
        result["metrics_status"] = (
            "Not computed: the public official LNDb test files withhold ground-truth annotations. "
            "Supply --labels_csv only if authoritative challenge labels are available."
        )
    (args.out_dir / f"{args.output_prefix}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
