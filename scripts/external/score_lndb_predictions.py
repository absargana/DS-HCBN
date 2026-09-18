#!/usr/bin/env python3
"""Score completed LNDb inference without repeating CT preprocessing/inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_lndb import bootstrap, calibration, metrics, sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--frozen_spec", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_scored_csv", type=Path, required=True)
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    spec = json.loads(args.frozen_spec.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != spec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    pred = pd.read_csv(args.predictions_csv)
    labels = pd.read_csv(args.labels_csv)
    scored = pred.merge(labels, on=["LNDbID", "FindingID"], how="inner", validate="one_to_one")
    if len(scored) != len(labels):
        raise RuntimeError(f"Eligible-label coverage mismatch: pred={len(pred)}, labels={len(labels)}, matched={len(scored)}")
    primary = float(spec["primary_threshold"]["value"])
    secondary = float(spec["secondary_threshold"]["value"])
    primary_metrics = metrics(scored.label.to_numpy(), scored.prob.to_numpy(), primary)
    secondary_metrics = metrics(scored.label.to_numpy(), scored.prob.to_numpy(), secondary)
    ece, calibration_bins = calibration(scored.label.to_numpy(), scored.prob.to_numpy())
    primary_metrics["ece"] = ece
    secondary_metrics["ece"] = ece
    result = {
        "dataset": "LNDb public annotated cohort",
        "checkpoint_sha256": checkpoint_hash,
        "retraining": False, "fine_tuning": False,
        "threshold_search_on_external_test": False,
        "n_input_predictions": int(len(pred)),
        "n_predictions_excluded_by_eligibility": int(len(pred) - len(scored)),
        "n_scans": int(scored.LNDbID.nunique()), "n_nodules": int(len(scored)),
        "n_benign": int((scored.label == 0).sum()),
        "n_malignant": int((scored.label == 1).sum()),
        "primary_threshold": primary, "secondary_threshold": secondary,
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "patient_bootstrap_95ci_primary": bootstrap(scored, primary, args.bootstrap_n, args.seed),
    }
    args.out_scored_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_scored_csv, index=False)
    scored.loc[(scored.label == 0) & (scored.prob >= primary)].to_csv(
        args.out_scored_csv.with_name("LNDb_public_annotated_false_positives.csv"), index=False
    )
    scored.loc[(scored.label == 1) & (scored.prob < primary)].to_csv(
        args.out_scored_csv.with_name("LNDb_public_annotated_false_negatives.csv"), index=False
    )
    calibration_bins.to_csv(args.out_json.with_name("LNDb_public_annotated_calibration_bins.csv"), index=False)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
