#!/usr/bin/env python3
"""Create the frozen binary LNDb public-annotation evaluation cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def any_reader_called_nodule(value: object) -> bool:
    if pd.isna(value):
        return False
    return any(part.strip() == "1" for part in str(value).split(","))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_nods_csv", type=Path, required=True)
    ap.add_argument("--test_cts_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.all_nods_csv)
    test_ids = set(pd.read_csv(args.test_cts_csv)["LNDbID"].astype(int))
    source["is_nodule"] = source["Nodule"].map(any_reader_called_nodule)
    rated = source[source.is_nodule & source.Malignancy.notna()].copy()
    if set(rated.LNDbID.astype(int)) & test_ids:
        raise RuntimeError("Public annotated cohort overlaps the hidden LNDb test IDs")

    # LNDb uses the 1--5 malignancy scale. Values outside this range (notably 0)
    # are unavailable/invalid annotations and must be excluded before labeling.
    valid = rated[rated.Malignancy.between(1.0, 5.0, inclusive="both")].copy()
    determinate = valid[(valid.Malignancy <= 2.0) | (valid.Malignancy >= 4.0)].copy()
    determinate["label"] = (determinate.Malignancy >= 4.0).astype(int)
    determinate["LNDbID"] = determinate.LNDbID.astype(int)
    determinate["FindingID"] = determinate.FindingID.astype(int)
    determinate = determinate.sort_values(["LNDbID", "FindingID"]).reset_index(drop=True)
    if determinate.duplicated(["LNDbID", "FindingID"]).any():
        raise RuntimeError("Duplicate merged nodule keys found")

    candidates = determinate[["LNDbID", "FindingID", "x", "y", "z"]]
    labels = determinate[["LNDbID", "FindingID", "label", "Malignancy", "RadID", "RadFinding"]]
    candidates.to_csv(args.out_dir / "LNDb_public_annotated_candidates.csv", index=False)
    labels.to_csv(args.out_dir / "LNDb_public_annotated_labels.csv", index=False)

    audit = {
        "source": str(args.all_nods_csv.resolve()),
        "validity_rule": "retain only malignancy values in the inclusive range 1-5",
        "label_rule": "benign if valid merged mean malignancy <=2; malignant if >=4; uncertain (2,4) excluded",
        "n_source_rows": int(len(source)),
        "n_rated_nodules": int(len(rated)),
        "n_invalid_malignancy_excluded": int(len(rated) - len(valid)),
        "n_malignancy_zero_excluded": int((rated.Malignancy == 0).sum()),
        "n_uncertain_excluded": int(((valid.Malignancy > 2) & (valid.Malignancy < 4)).sum()),
        "n_indeterminate_excluded": int(((valid.Malignancy > 2) & (valid.Malignancy < 4)).sum()),
        "n_determinate_nodules": int(len(determinate)),
        "n_scans": int(determinate.LNDbID.nunique()),
        "n_benign": int((determinate.label == 0).sum()),
        "n_malignant": int((determinate.label == 1).sum()),
        "hidden_test_overlap": 0,
    }
    (args.out_dir / "LNDb_public_annotated_cohort_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
