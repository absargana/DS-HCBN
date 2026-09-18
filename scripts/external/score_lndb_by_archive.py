#!/usr/bin/env python3
"""Stratify the completed frozen LNDb evaluation by source data*.rar archive."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from evaluate_lndb import bootstrap, calibration, metrics


def archive_scan_ids(seven_zip: Path, archive: Path) -> set[int]:
    completed = subprocess.run(
        [str(seven_zip), "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        int(match.group(1))
        for match in re.finditer(r"^Path = .*LNDb-(\d+)\.mhd\s*$", completed.stdout, re.MULTILINE)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", type=Path, required=True)
    ap.add_argument("--archive_dir", type=Path, required=True)
    ap.add_argument("--seven_zip", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--primary_threshold", type=float, default=0.23)
    ap.add_argument("--secondary_threshold", type=float, default=0.50)
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    frame = pd.read_csv(args.scored_csv)
    memberships: dict[int, str] = {}
    archive_counts: dict[str, int] = {}
    for index in range(6):
        name = f"data{index}"
        ids = archive_scan_ids(args.seven_zip, args.archive_dir / f"{name}.rar")
        archive_counts[name] = len(ids)
        for scan_id in ids:
            if scan_id in memberships:
                raise RuntimeError(f"LNDb-{scan_id:04d} occurs in multiple archives")
            memberships[scan_id] = name

    frame["source_archive"] = frame["LNDbID"].map(memberships)
    if frame["source_archive"].isna().any():
        missing = sorted(frame.loc[frame.source_archive.isna(), "LNDbID"].unique())
        raise RuntimeError(f"Evaluated scans missing from archive listings: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    results = {
        "dataset": "LNDb public annotated cohort stratified by distributed CT archive",
        "retraining": False,
        "fine_tuning": False,
        "threshold_search_on_external_test": False,
        "primary_threshold": args.primary_threshold,
        "secondary_threshold": args.secondary_threshold,
        "bootstrap_unit": "patient/CT",
        "subsets": {},
    }
    for index in range(6):
        name = f"data{index}"
        subset = frame.loc[frame.source_archive == name].copy()
        primary = metrics(subset.label.to_numpy(), subset.prob.to_numpy(), args.primary_threshold)
        secondary = metrics(subset.label.to_numpy(), subset.prob.to_numpy(), args.secondary_threshold)
        ece, bins = calibration(subset.label.to_numpy(), subset.prob.to_numpy())
        primary["ece"] = ece
        secondary["ece"] = ece
        item = {
            "archive": f"{name}.rar",
            "n_cts_in_archive": archive_counts[name],
            "n_evaluated_scans": int(subset.LNDbID.nunique()),
            "n_nodules": int(len(subset)),
            "n_benign": int((subset.label == 0).sum()),
            "n_malignant": int((subset.label == 1).sum()),
            "primary_metrics": primary,
            "secondary_metrics": secondary,
            "patient_bootstrap_95ci_primary": bootstrap(
                subset, args.primary_threshold, args.bootstrap_n, args.seed + index
            ),
        }
        results["subsets"][name] = item
        summaries.append({
            "subset": name,
            "archive": item["archive"],
            "n_cts_in_archive": item["n_cts_in_archive"],
            "n_evaluated_scans": item["n_evaluated_scans"],
            "n_nodules": item["n_nodules"],
            "n_benign": item["n_benign"],
            "n_malignant": item["n_malignant"],
            **{f"primary_{key}": value for key, value in primary.items()},
            **{f"secondary_{key}": value for key, value in secondary.items()},
        })
        subset.to_csv(args.out_dir / f"LNDb_{name}_predictions_scored.csv", index=False)
        bins.to_csv(args.out_dir / f"LNDb_{name}_calibration_bins.csv", index=False)
        (args.out_dir / f"LNDb_{name}_results.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )

    pd.DataFrame(summaries).to_csv(args.out_dir / "LNDb_subsets_metrics_summary.csv", index=False)
    (args.out_dir / "LNDb_subsets_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: {
        "n_scans": value["n_evaluated_scans"],
        "n_nodules": value["n_nodules"],
        "n_benign": value["n_benign"],
        "n_malignant": value["n_malignant"],
        "primary_metrics": value["primary_metrics"],
    } for key, value in results["subsets"].items()}, indent=2))


if __name__ == "__main__":
    main()
