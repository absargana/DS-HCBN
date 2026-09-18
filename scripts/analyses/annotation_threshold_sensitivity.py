from __future__ import annotations

import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def label(values):
    m = float(np.mean(values))
    return 0.0 if m <= 2 else (1.0 if m >= 4 else .5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessing_module", required=True); ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--xml_root", required=True); ap.add_argument("--dicom_root", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[5, 7.5, 10, 12.5, 15]); a = ap.parse_args()
    module_path = Path(a.preprocessing_module).resolve()
    spec = importlib.util.spec_from_file_location("dshcbn_preprocessing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preprocessing module: {module_path}")
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    xml_index = p.build_xml_index(a.xml_root); records = []; assignments = {}
    meta_files = sorted(Path(a.meta_dir).glob("*_meta.json"))
    for mp in tqdm(meta_files, desc="Threshold sensitivity"):
        meta = json.load(open(mp, encoding="utf-8")); pid = str(meta["patient_id"]); sel = meta["selected_series"]
        uid, series_dir = str(sel["series_uid"]), str(sel["series_dir"]); xp = xml_index.get(uid)
        if not Path(series_dir).is_dir():
            candidate = Path(a.dicom_root) / pid / Path(series_dir).parent.name / Path(series_dir).name
            if candidate.is_dir(): series_dir = str(candidate)
        if not xp: continue
        try:
            img = p.read_series_sitk(series_dir, uid); sop = p.build_sop_to_k(series_dir, uid)
            ann = p.build_reader_centroids(p.parse_reader_nodules(xp), sop)
            pts = [tuple(map(float, img.TransformContinuousIndexToPhysicalPoint(x["centroid_ijk_original"]))) for x in ann]
        except Exception as e:
            records.append({"patient_id": pid, "threshold_mm": None, "status": f"ERROR: {e}"}); continue
        keys = [f"{pid}|{x['reader_id']}|{x['nodule_id']}" for x in ann]
        for t in a.thresholds:
            clusters = p.cluster_constrained_complete_linkage(ann, pts, t); amap = {}
            rcounts = []; eligible = []; label_counts = {0.0: 0, 0.5: 0, 1.0: 0}
            for ci, ids in enumerate(clusters):
                vals = [int(ann[i]["malignancy"]) for i in ids if ann[i].get("malignancy") is not None and 1 <= int(ann[i]["malignancy"]) <= 5]
                lab = label(vals) if vals else np.nan; rcounts.append(len(ids))
                if vals:
                    eligible.append(ids); label_counts[lab] += 1
                for i in ids: amap[keys[i]] = {"cluster": f"{pid}|{ci}", "label": lab}
            assignments[(pid, t)] = amap
            records.append({"patient_id": pid, "threshold_mm": t, "status": "OK", "annotations": len(ann), "all_clusters": len(clusters),
                            "eligible_clusters": len(eligible), "benign_clusters": label_counts[0.0], "uncertain_clusters": label_counts[0.5],
                            "malignant_clusters": label_counts[1.0], "single_reader_clusters": int(sum(len(x) == 1 for x in eligible)),
                            "multi_reader_clusters": int(sum(len(x) >= 2 for x in eligible))})
    detail = pd.DataFrame(records); ref = 10.0; summary = []
    for t in a.thresholds:
        d = detail[(detail.threshold_mm == t) & (detail.status == "OK")]
        changed = comparable = all_labeled = any_changed = confident_to_uncertain = uncertain_to_confident = opposite_confident = 0
        same_cluster_pairs = ref_cluster_pairs = 0
        for pid in d.patient_id:
            x, y = assignments.get((pid, t), {}), assignments.get((pid, ref), {}); common = sorted(set(x) & set(y))
            for k in common:
                lx, ly = x[k]["label"], y[k]["label"]
                if lx in (0., .5, 1.) and ly in (0., .5, 1.):
                    all_labeled += 1
                    if lx != ly:
                        any_changed += 1
                        confident_to_uncertain += int(ly in (0., 1.) and lx == .5)
                        uncertain_to_confident += int(ly == .5 and lx in (0., 1.))
                        opposite_confident += int(lx in (0., 1.) and ly in (0., 1.))
                if x[k]["label"] in (0.,1.) and y[k]["label"] in (0.,1.):
                    comparable += 1; changed += int(x[k]["label"] != y[k]["label"])
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    sx = x[common[i]]["cluster"] == x[common[j]]["cluster"]; sy = y[common[i]]["cluster"] == y[common[j]]["cluster"]
                    same_cluster_pairs += int(sx and sy); ref_cluster_pairs += int(sy)
        summary.append({"threshold_mm": t, "patients_processed": int(d.patient_id.nunique()),
                        "patients_with_eligible_clusters": int(d.loc[d.eligible_clusters.astype(int) > 0, "patient_id"].nunique()),
                        "eligible_clusters": int(d.eligible_clusters.sum()), "benign_clusters": int(d.benign_clusters.sum()),
                        "uncertain_clusters": int(d.uncertain_clusters.sum()), "malignant_clusters": int(d.malignant_clusters.sum()),
                        "multi_reader_clusters": int(d.multi_reader_clusters.sum()), "comparable_confident_annotation_assignments": comparable,
                        "changed_confident_label_assignments_vs_10mm": changed,
                        "changed_fraction": float(changed/comparable) if comparable else None,
                        "all_labeled_annotation_assignments": all_labeled,
                        "any_label_changed_vs_10mm": any_changed,
                        "any_label_changed_fraction": float(any_changed/all_labeled) if all_labeled else None,
                        "reference_confident_to_alternative_uncertain": confident_to_uncertain,
                        "reference_uncertain_to_alternative_confident": uncertain_to_confident,
                        "opposite_confident_changes": opposite_confident,
                        "reference_cocluster_pair_retention": float(same_cluster_pairs/ref_cluster_pairs) if ref_cluster_pairs else None})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); detail.to_csv(out / "threshold_patient_details.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "threshold_sensitivity_summary.csv", index=False)
    json.dump({"method": "constrained complete linkage; one annotation per reader", "reference_threshold_mm": ref,
               "thresholds_mm": a.thresholds, "summary": summary}, open(out / "threshold_sensitivity_summary.json", "w"), indent=2)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__": main()
