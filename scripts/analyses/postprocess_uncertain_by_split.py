from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summarize(d, bootstrap_n=5000, seed=42):
    rho, p = spearmanr(d.malignancy_mean, d.prob)
    ids = d.patient_id.unique(); rng = np.random.default_rng(seed); boot = []
    groups = {pid: d.loc[d.patient_id == pid, ["malignancy_mean", "prob"]].to_numpy() for pid in ids}
    for _ in range(bootstrap_n):
        chosen = rng.choice(ids, len(ids), replace=True)
        x = np.concatenate([groups[pid] for pid in chosen], axis=0)
        r = spearmanr(x[:, 0], x[:, 1]).statistic
        if np.isfinite(r): boot.append(r)
    return {"n_nodules": int(len(d)), "n_patients": int(d.patient_id.nunique()), "spearman_rho": float(rho),
            "spearman_p": float(p), "patient_bootstrap_95ci": [float(v) for v in np.quantile(boot, [.025, .975])],
            "probability_median": float(d.prob.median()), "probability_q1": float(d.prob.quantile(.25)),
            "probability_q3": float(d.prob.quantile(.75)), "proportion_above_0_23": float((d.prob >= .23).mean()),
            "proportion_above_0_5": float((d.prob >= .5).mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--predictions", required=True); ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bootstrap_n", type=int, default=5000); a = ap.parse_args()
    d = pd.read_csv(a.predictions); m = pd.read_csv(a.split_manifest, usecols=["patient_id", "split"]).drop_duplicates()
    d = d.merge(m, on="patient_id", how="left", validate="many_to_one")
    d["split"] = d["split"].fillna("unassigned_no_confident_nodule")
    result = {"interpretation_guardrail": "Full cohort is descriptive; test-patient subset is the held-out generalization analysis.",
              "by_split": {s: summarize(d[d.split == s].copy(), a.bootstrap_n, 42)
                           for s in ["train", "val", "test", "unassigned_no_confident_nodule"]}}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); json.dump(result, open(out, "w"), indent=2)
    d.to_csv(out.with_name("uncertain_nodule_predictions_with_split.csv"), index=False); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
