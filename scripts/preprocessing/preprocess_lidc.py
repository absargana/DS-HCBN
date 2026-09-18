#!/usr/bin/env python3
"""
Reproducible preprocessing for DS-HCBN / LIDC-IDRI.

Pipeline properties
-------------------
1. Preprocessing NEVER creates augmented samples. Augmentation is performed on-the-fly
   in the training Dataset *after* the patient-wise split, and only for training samples.
2. Reader-level concept ratings are preserved in the CSV (JSON arrays); non-integer
   reader means are therefore not rounded for supervision.
3. LIDC concepts use their native structures:
      - ordinal 1..5: subtlety, sphericity, margin, lobulation, spiculation, texture
      - nominal 1..4: internalStructure
      - nominal 1..6: calcification
   Invalid codes are excluded and counted in an audit file.
4. Nodule grouping supports a constrained complete-linkage centroid rule that never
   places two annotations from the same reader into one cluster and requires every
   member pair to be within --cluster_eps_mm.
5. Multiple context sizes can be generated in one run for controlled context-size
   experiments (e.g. 64,80,96,112 mm at 1-mm isotropic spacing).
6. An audit report records label counts, reader counts, concept validity, and clustering
   distances to make dataset construction transparent in the manuscript/rebuttal.

The code intentionally leaves CT denoising disabled. It performs HU clipping,
linear resampling, and normalization. Noise robustness augmentation belongs in the
training pipeline only, so validation/test CTs remain unmodified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


LOCALIZER_KEYWORDS = ("localizer", "scout", "topogram", "surview")
NS = {"nih": "http://www.nih.gov"}

# Native LIDC/IDRI concept structures. Only internalStructure and calcification are nominal.
CONCEPT_SPECS: Dict[str, Dict[str, object]] = {
    "subtlety": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
    "internalStructure": {"kind": "categorical", "codes": [1, 2, 3, 4]},
    "calcification": {"kind": "categorical", "codes": [1, 2, 3, 4, 5, 6]},
    "sphericity": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
    "margin": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
    "lobulation": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
    "spiculation": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
    "texture": {"kind": "ordinal", "codes": [1, 2, 3, 4, 5]},
}
CONCEPT_TAGS = list(CONCEPT_SPECS.keys())

# Optional environment defaults; command-line paths take precedence.
PROJECT_ROOT = os.environ.get("DSHCBN_DATA_ROOT", "")
DICOM_ROOT = os.environ.get("DSHCBN_LIDC_DICOM_ROOT", "")
XML_ROOT = os.environ.get("DSHCBN_LIDC_XML_ROOT", "")
PREPROC_ROOT = os.environ.get("DSHCBN_PREPROCESSED_ROOT", "")
TCIA_METADATA_CSV = os.environ.get("DSHCBN_TCIA_METADATA_CSV", "")
TCIA_NODULE_COUNTS_XLSX = os.environ.get("DSHCBN_TCIA_NODULE_COUNTS_XLSX", "")
TCIA_DIAGNOSIS_XLS = os.environ.get("DSHCBN_TCIA_DIAGNOSIS_XLS", "")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_dicom_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def find_series_leaf_dirs(patient_dir: str) -> List[str]:
    candidates: List[str] = []
    for root, _, files in os.walk(patient_dir):
        if any(is_dicom_file(os.path.join(root, fn)) for fn in files):
            candidates.append(root)
    return list(dict.fromkeys(candidates))


def discover_patient_dirs(dicom_root: str) -> List[str]:
    """Discover LIDC patient folders robustly even if an extra collection folder exists.

    Once a directory named ``LIDC-IDRI-*`` is found it is recorded as a patient root
    and removed from ``os.walk`` traversal, preventing an expensive recursive scan
    through every study/series/slice beneath that patient.
    """
    immediate = [
        os.path.join(dicom_root, d)
        for d in os.listdir(dicom_root)
        if os.path.isdir(os.path.join(dicom_root, d))
    ]
    named = [d for d in immediate if os.path.basename(d).upper().startswith("LIDC-IDRI-")]
    if named:
        return sorted(named)

    recursive: List[str] = []
    for root, dirs, _ in os.walk(dicom_root):
        patient_names = [d for d in dirs if d.upper().startswith("LIDC-IDRI-")]
        recursive.extend(os.path.join(root, d) for d in patient_names)
        if patient_names:
            patient_set = set(patient_names)
            dirs[:] = [d for d in dirs if d not in patient_set]
    if recursive:
        return sorted(list(dict.fromkeys(recursive)))

    # Last-resort compatibility with nonstandard renamed patient folders.
    return sorted(immediate)


def get_series_info_from_one_file(dcm_path: str) -> Dict[str, str]:
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    image_type = getattr(ds, "ImageType", [])
    if isinstance(image_type, (list, tuple)):
        image_type = " ".join(str(x) for x in image_type)
    return {
        "modality": str(getattr(ds, "Modality", "")).upper(),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")).strip(),
        "series_desc": str(getattr(ds, "SeriesDescription", "")).strip(),
        "image_type": str(image_type),
    }


def looks_like_localizer(series_desc: str, image_type: str) -> bool:
    text = f"{series_desc or ''} {image_type or ''}".lower()
    return any(k in text for k in LOCALIZER_KEYWORDS)


@dataclass
class SeriesCandidate:
    series_dir: str
    num_slices: int
    modality: str
    series_uid: str
    series_desc: str
    image_type: str


def collect_series_candidates(patient_dir: str) -> List[SeriesCandidate]:
    reader = sitk.ImageSeriesReader()
    out: List[SeriesCandidate] = []
    for leaf in find_series_leaf_dirs(patient_dir):
        try:
            series_ids = reader.GetGDCMSeriesIDs(leaf) or []
            for sid in series_ids:
                files = reader.GetGDCMSeriesFileNames(leaf, sid)
                if not files:
                    continue
                info = get_series_info_from_one_file(files[0])
                out.append(
                    SeriesCandidate(
                        series_dir=leaf,
                        num_slices=len(files),
                        modality=info["modality"],
                        series_uid=info["series_uid"] or sid,
                        series_desc=info["series_desc"],
                        image_type=info["image_type"],
                    )
                )
        except Exception:
            continue
    return out


def choose_best_ct_series(
    cands: Sequence[SeriesCandidate], min_slices: int = 50, exclude_localizers: bool = True
) -> Optional[SeriesCandidate]:
    ct = [c for c in cands if c.modality == "CT" and c.num_slices >= min_slices]
    if exclude_localizers:
        nonloc = [c for c in ct if not looks_like_localizer(c.series_desc, c.image_type)]
        if nonloc:
            ct = nonloc
    if not ct:
        return None
    return sorted(ct, key=lambda x: x.num_slices, reverse=True)[0]


def read_series_sitk(series_dir: str, series_uid: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir, series_uid)
    if not files:
        raise RuntimeError(f"No DICOM files for series {series_uid} in {series_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def resample_to_spacing(img: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = [
        int(round(old_size[i] * old_spacing[i] / target_spacing[i])) for i in range(3)
    ]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(target_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetInterpolator(sitk.sitkLinear)
    return r.Execute(img)


def clamp_and_normalize(
    img_hu: sitk.Image, hu_clip: Tuple[int, int] = (-1000, 400)
) -> sitk.Image:
    lo, hi = hu_clip
    clipped = sitk.Clamp(img_hu, lowerBound=float(lo), upperBound=float(hi))
    f = sitk.Cast(clipped, sitk.sitkFloat32)
    return (f - float(lo)) / float(hi - lo)


def get_series_uid_from_xml(xml_path: str) -> Optional[str]:
    try:
        root = ET.parse(xml_path).getroot()
        suid = root.findtext(
            "nih:ResponseHeader/nih:SeriesInstanceUid", default="", namespaces=NS
        ).strip()
        return suid or None
    except Exception:
        return None


def build_xml_index(xml_root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for root, _, files in os.walk(xml_root):
        for fn in files:
            if fn.lower().endswith(".xml"):
                p = os.path.join(root, fn)
                suid = get_series_uid_from_xml(p)
                if suid and suid not in idx:
                    idx[suid] = p
    return idx


def build_sop_to_k(series_dir: str, series_uid: str) -> Dict[str, int]:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir, series_uid)
    if not files:
        raise RuntimeError(f"No DICOM files for series {series_uid}")
    out: Dict[str, int] = {}
    for k, fp in enumerate(files):
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        sop = str(getattr(ds, "SOPInstanceUID", "")).strip()
        if sop:
            out[sop] = k
    return out


def _safe_int(text: object) -> Optional[int]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def parse_reader_nodules(xml_path: str) -> List[dict]:
    root = ET.parse(xml_path).getroot()
    out: List[dict] = []
    for si, sess in enumerate(root.findall("nih:readingSession", NS)):
        reader_id = sess.findtext(
            "nih:servicingRadiologistID", default=f"R{si+1}", namespaces=NS
        ).strip()
        # Radiologist ID is not stable across cases. It is only used within this XML to
        # prevent two annotations from the same reading session entering one cluster.
        if not reader_id:
            reader_id = f"session_{si+1}"
        reader_key = f"{si}:{reader_id}"
        for n in sess.findall("nih:unblindedReadNodule", NS):
            nid = n.findtext("nih:noduleID", default="", namespaces=NS).strip()
            chars = n.find("nih:characteristics", NS)
            concepts = {k: None for k in CONCEPT_TAGS}
            malignancy = None
            if chars is not None:
                for tag in CONCEPT_TAGS:
                    concepts[tag] = _safe_int(
                        chars.findtext(f"nih:{tag}", default="", namespaces=NS)
                    )
                malignancy = _safe_int(
                    chars.findtext("nih:malignancy", default="", namespaces=NS)
                )
            rois: List[dict] = []
            for roi in n.findall("nih:roi", NS):
                sop = roi.findtext("nih:imageSOP_UID", default="", namespaces=NS).strip()
                pts: List[Tuple[int, int]] = []
                for e in roi.findall("nih:edgeMap", NS):
                    x = _safe_int(e.findtext("nih:xCoord", default="", namespaces=NS))
                    y = _safe_int(e.findtext("nih:yCoord", default="", namespaces=NS))
                    if x is not None and y is not None:
                        pts.append((x, y))
                rois.append({"sop_uid": sop, "points": pts})
            out.append(
                {
                    "reader_id": reader_key,
                    "nodule_id": nid,
                    "malignancy": malignancy,
                    "concepts": concepts,
                    "rois": rois,
                }
            )
    return out


def centroid_xy(points: Sequence[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    a = np.asarray(points, dtype=np.float32)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def build_reader_centroids(reader_nodules: Sequence[dict], sop_to_k: Dict[str, int]) -> List[dict]:
    out: List[dict] = []
    for rn in reader_nodules:
        pts: List[Tuple[float, float, float]] = []
        for roi in rn["rois"]:
            sop = roi["sop_uid"]
            if not sop or sop not in sop_to_k:
                continue
            cxy = centroid_xy(roi["points"])
            if cxy is None:
                continue
            x, y = cxy
            pts.append((x, y, float(sop_to_k[sop])))
        if not pts:
            continue
        a = np.asarray(pts, dtype=np.float32)
        cx, cy, ck = a.mean(axis=0)
        out.append(
            {
                "reader_id": rn["reader_id"],
                "nodule_id": rn["nodule_id"],
                "malignancy": rn["malignancy"],
                "concepts": rn["concepts"],
                "centroid_ijk_original": (float(cx), float(cy), float(ck)),
            }
        )
    return out


def euclidean_mm(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def cluster_constrained_complete_linkage(
    annotations: Sequence[dict], pts_mm: Sequence[Tuple[float, float, float]], eps_mm: float
) -> List[List[int]]:
    """Greedy complete-linkage clustering with a one-annotation-per-reader constraint.

    A candidate annotation may join a cluster only if:
      1) its reader is not already represented in the cluster, and
      2) it is within eps_mm of *every* current cluster member.

    This is deliberately conservative and avoids the transitive-chain failure mode of
    unrestricted single-linkage/union-find clustering.
    """
    order = list(range(len(pts_mm)))
    # Stable order by z/y/x makes runs deterministic.
    order.sort(key=lambda i: (pts_mm[i][2], pts_mm[i][1], pts_mm[i][0], annotations[i]["reader_id"]))
    clusters: List[List[int]] = []
    for i in order:
        reader = annotations[i]["reader_id"]
        candidates: List[Tuple[float, int]] = []
        for ci, cluster in enumerate(clusters):
            readers = {annotations[j]["reader_id"] for j in cluster}
            if reader in readers:
                continue
            distances = [euclidean_mm(pts_mm[i], pts_mm[j]) for j in cluster]
            if distances and max(distances) <= eps_mm:
                candidates.append((float(np.mean(distances)), ci))
        if not candidates:
            clusters.append([i])
        else:
            _, best_ci = min(candidates, key=lambda t: (t[0], t[1]))
            clusters[best_ci].append(i)
    return clusters


def cluster_single_linkage_original(
    pts_mm: Sequence[Tuple[float, float, float]], eps_mm: float
) -> List[List[int]]:
    """Reproduction option matching the original union-find centroid clustering."""
    n = len(pts_mm)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_mm(pts_mm[i], pts_mm[j]) <= eps_mm:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def extract_cube(
    arr_zyx: np.ndarray,
    center_xyz: Tuple[int, int, int],
    size: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    cx, cy, cz = center_xyz
    half = size // 2
    x0, y0, z0 = cx - half, cy - half, cz - half
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    zdim, ydim, xdim = arr_zyx.shape
    ix0, ix1 = max(0, x0), min(xdim, x1)
    iy0, iy1 = max(0, y0), min(ydim, y1)
    iz0, iz1 = max(0, z0), min(zdim, z1)
    out = np.full((size, size, size), pad_value, dtype=arr_zyx.dtype)
    ox0, oy0, oz0 = ix0 - x0, iy0 - y0, iz0 - z0
    out[
        oz0 : oz0 + (iz1 - iz0),
        oy0 : oy0 + (iy1 - iy0),
        ox0 : ox0 + (ix1 - ix0),
    ] = arr_zyx[iz0:iz1, iy0:iy1, ix0:ix1]
    return out, (x0, y0, z0)


def write_patch_nifti(
    patch_zyx: np.ndarray,
    img_rs: sitk.Image,
    x0: int,
    y0: int,
    z0: int,
    out_path: str,
) -> None:
    patch = sitk.GetImageFromArray(patch_zyx)
    patch.SetSpacing(img_rs.GetSpacing())
    patch.SetDirection(img_rs.GetDirection())
    patch.SetOrigin(
        img_rs.TransformContinuousIndexToPhysicalPoint((float(x0), float(y0), float(z0)))
    )
    sitk.WriteImage(patch, out_path)


def malignancy_mean_to_label(
    mal_mean: float, benign_max: float = 2.0, malignant_min: float = 4.0
) -> float:
    if mal_mean <= benign_max:
        return 0.0
    if mal_mean >= malignant_min:
        return 1.0
    return 0.5


def valid_concept_values(tag: str, values: Iterable[Optional[int]]) -> Tuple[List[int], List[int]]:
    allowed = set(int(x) for x in CONCEPT_SPECS[tag]["codes"])
    valid: List[int] = []
    invalid: List[int] = []
    for v in values:
        if v is None:
            continue
        iv = int(v)
        if iv in allowed:
            valid.append(iv)
        else:
            invalid.append(iv)
    return valid, invalid


def majority_vote(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    m = counts.max()
    tied = vals[counts == m]
    # deterministic tie break: smallest native code
    return int(tied.min())


def preprocess_subset(
    dicom_root: str,
    out_root: str,
    max_patients: int = 0,
    seed: int = 42,
    min_slices: int = 50,
    exclude_localizers: bool = True,
    spacing: float = 1.0,
    hu_low: int = -1000,
    hu_high: int = 400,
    xml_root: Optional[str] = None,
    require_xml_match: bool = True,
) -> None:
    safe_makedirs(out_root)
    out_norm = os.path.join(out_root, "nifti_norm")
    out_meta = os.path.join(out_root, "meta")
    safe_makedirs(out_norm)
    safe_makedirs(out_meta)

    if not os.path.isdir(dicom_root):
        raise FileNotFoundError(f"DICOM root does not exist: {dicom_root}")

    xml_index: Dict[str, str] = {}
    if xml_root:
        if not os.path.isdir(xml_root):
            raise FileNotFoundError(f"XML root does not exist: {xml_root}")
        xml_index = build_xml_index(xml_root)
        if require_xml_match and not xml_index:
            raise RuntimeError(f"No LIDC XML annotations were indexed under: {xml_root}")
        print(f"Indexed {len(xml_index)} annotated SeriesInstanceUID values from XML.")

    patients = discover_patient_dirs(dicom_root)
    if not patients:
        raise RuntimeError(f"No patient directories found under DICOM root: {dicom_root}")
    print(f"Discovered {len(patients)} patient directories under DICOM root.")
    rng = random.Random(seed)
    rng.shuffle(patients)
    if max_patients > 0:
        patients = patients[:max_patients]

    log_path = os.path.join(out_meta, "preprocess_log.csv")
    fields = [
        "patient_id",
        "status",
        "message",
        "selected_series_dir",
        "selected_series_uid",
        "series_selection_source",
        "xml_annotation_path",
        "num_slices",
        "orig_size",
        "orig_spacing",
        "resampled_size",
        "resampled_spacing",
        "out_norm_nifti",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pdir in tqdm(patients, desc="DICOM -> normalized NIfTI"):
            pid = os.path.basename(os.path.normpath(pdir))
            row = {k: "" for k in fields}
            row["patient_id"] = pid
            try:
                candidates = collect_series_candidates(pdir)
                eligible = [
                    c for c in candidates
                    if c.modality == "CT"
                    and c.num_slices >= min_slices
                    and (not exclude_localizers or not looks_like_localizer(c.series_desc, c.image_type))
                ]

                # Prefer the CT series explicitly referenced by the LIDC XML. This avoids
                # silently choosing an unannotated CT when a patient directory contains
                # multiple CT series.
                annotated = [c for c in eligible if c.series_uid in xml_index] if xml_index else []
                if annotated:
                    chosen = sorted(annotated, key=lambda x: x.num_slices, reverse=True)[0]
                    row["series_selection_source"] = "xml_uid_match"
                    row["xml_annotation_path"] = xml_index.get(chosen.series_uid, "")
                elif require_xml_match and xml_index:
                    row["status"] = "FAIL"
                    row["message"] = "No eligible CT series matched any XML SeriesInstanceUID"
                    row["series_selection_source"] = "no_xml_match"
                    writer.writerow(row)
                    continue
                else:
                    chosen = choose_best_ct_series(candidates, min_slices, exclude_localizers)
                    row["series_selection_source"] = "largest_eligible_ct_fallback"

                if chosen is None:
                    row["status"] = "FAIL"
                    row["message"] = "No eligible CT series"
                    writer.writerow(row)
                    continue
                img = read_series_sitk(chosen.series_dir, chosen.series_uid)
                row.update(
                    {
                        "selected_series_dir": chosen.series_dir,
                        "selected_series_uid": chosen.series_uid,
                        "num_slices": chosen.num_slices,
                        "orig_size": str(img.GetSize()),
                        "orig_spacing": str(img.GetSpacing()),
                    }
                )
                img_rs = resample_to_spacing(img, (spacing, spacing, spacing))
                img_norm = clamp_and_normalize(img_rs, (hu_low, hu_high))
                out_nii = os.path.join(out_norm, f"{pid}_ct_norm_sp{spacing:.1f}.nii.gz")
                sitk.WriteImage(img_norm, out_nii)
                row.update(
                    {
                        "resampled_size": str(img_rs.GetSize()),
                        "resampled_spacing": str(img_rs.GetSpacing()),
                        "out_norm_nifti": out_nii,
                        "status": "OK",
                        "message": "Processed",
                    }
                )
                with open(os.path.join(out_meta, f"{pid}_meta.json"), "w", encoding="utf-8") as jf:
                    json.dump(
                        {
                            "patient_id": pid,
                            "selected_series": asdict(chosen),
                            "out_norm_nifti": out_nii,
                            "target_spacing": [spacing, spacing, spacing],
                            "hu_clip": [hu_low, hu_high],
                            "normalization": "linear_to_[0,1]",
                            "denoising": "none",
                            "series_selection_source": row.get("series_selection_source", ""),
                            "xml_annotation_path": row.get("xml_annotation_path", ""),
                        },
                        jf,
                        indent=2,
                    )
                writer.writerow(row)
            except Exception as e:
                row["status"] = "FAIL"
                row["message"] = f"{type(e).__name__}: {e}"
                writer.writerow(row)
    print(f"Saved preprocessing log: {log_path}")


def build_patches_and_labels(
    preproc_root: str,
    xml_root: str,
    local_size: int = 64,
    context_sizes: Sequence[int] = (64, 80, 96, 112),
    primary_context_size: int = 96,
    cluster_eps_mm: float = 10.0,
    cluster_mode: str = "constrained_complete_linkage",
    benign_max: float = 2.0,
    malignant_min: float = 4.0,
) -> None:
    meta_dir = os.path.join(preproc_root, "meta")
    labels_dir = os.path.join(preproc_root, "labels")
    safe_makedirs(labels_dir)

    context_sizes = sorted(set(int(s) for s in context_sizes))
    if primary_context_size not in context_sizes:
        context_sizes.append(int(primary_context_size))
        context_sizes.sort()
    if local_size % 8 != 0 or any(s % 8 != 0 for s in context_sizes):
        raise ValueError("local/context sizes should be divisible by 8 for the current Transformer patch embedding")

    out_local_dir = os.path.join(preproc_root, f"patches_local{local_size}")
    safe_makedirs(out_local_dir)
    ctx_dirs = {s: os.path.join(preproc_root, f"patches_ctx{s}") for s in context_sizes}
    for d in ctx_dirs.values():
        safe_makedirs(d)

    xml_index = build_xml_index(xml_root)
    if not xml_index:
        raise RuntimeError(f"No XML files indexed under {xml_root}")

    rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []
    cluster_audit: List[Dict[str, object]] = []
    meta_files = sorted(
        os.path.join(meta_dir, fn)
        for fn in os.listdir(meta_dir)
        if fn.endswith("_meta.json")
    )

    for mp in tqdm(meta_files, desc="XML -> original nodule patches"):
        try:
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pid = str(meta["patient_id"])
            series_uid = str(meta["selected_series"]["series_uid"])
            series_dir = str(meta["selected_series"]["series_dir"])
            nifti_path = str(meta["out_norm_nifti"])
            xml_path = xml_index.get(series_uid)
            if not xml_path:
                continue

            img_orig = read_series_sitk(series_dir, series_uid)
            sop_to_k = build_sop_to_k(series_dir, series_uid)
            rc = build_reader_centroids(parse_reader_nodules(xml_path), sop_to_k)
            if not rc:
                continue
            phys_pts: List[Tuple[float, float, float]] = []
            for rci in rc:
                p = img_orig.TransformContinuousIndexToPhysicalPoint(rci["centroid_ijk_original"])
                phys_pts.append((float(p[0]), float(p[1]), float(p[2])))

            if cluster_mode == "constrained_complete_linkage":
                clusters = cluster_constrained_complete_linkage(rc, phys_pts, cluster_eps_mm)
            elif cluster_mode == "original_single_linkage":
                clusters = cluster_single_linkage_original(phys_pts, cluster_eps_mm)
            else:
                raise ValueError(f"Unknown cluster_mode={cluster_mode}")

            img_rs = sitk.ReadImage(nifti_path)
            arr_rs = sitk.GetArrayFromImage(img_rs).astype(np.float32)

            for ci, idxs in enumerate(clusters):
                members = [rc[i] for i in idxs]
                member_phys = [phys_pts[i] for i in idxs]
                malignancy_values = [
                    int(m["malignancy"])
                    for m in members
                    if m.get("malignancy") is not None and 1 <= int(m["malignancy"]) <= 5
                ]
                if not malignancy_values:
                    continue
                mal_mean = float(np.mean(malignancy_values))
                label = malignancy_mean_to_label(mal_mean, benign_max, malignant_min)
                centroid_phys = np.asarray(member_phys, dtype=np.float64).mean(axis=0)
                ix, iy, iz = img_rs.TransformPhysicalPointToIndex(tuple(centroid_phys.tolist()))
                center = (int(ix), int(iy), int(iz))
                patch_id = f"{pid}_C{ci:03d}"

                local_patch, (lx0, ly0, lz0) = extract_cube(arr_rs, center, local_size, 0.0)
                local_path = os.path.join(out_local_dir, f"{patch_id}_local{local_size}.nii.gz")
                write_patch_nifti(local_patch, img_rs, lx0, ly0, lz0, local_path)

                ctx_paths: Dict[int, str] = {}
                ctx_bounds: Dict[int, Tuple[int, int, int]] = {}
                for s in context_sizes:
                    patch, (x0, y0, z0) = extract_cube(arr_rs, center, s, 0.0)
                    path = os.path.join(ctx_dirs[s], f"{patch_id}_ctx{s}.nii.gz")
                    write_patch_nifti(patch, img_rs, x0, y0, z0, path)
                    ctx_paths[s] = path
                    ctx_bounds[s] = (x0, y0, z0)

                pair_dists = [
                    euclidean_mm(member_phys[a], member_phys[b])
                    for a in range(len(member_phys))
                    for b in range(a + 1, len(member_phys))
                ]
                readers = [m["reader_id"] for m in members]
                duplicate_reader = len(readers) != len(set(readers))
                cluster_audit.append(
                    {
                        "patient_id": pid,
                        "patch_id": patch_id,
                        "n_annotations": len(members),
                        "n_readers": len(set(readers)),
                        "duplicate_reader_in_cluster": int(duplicate_reader),
                        "max_pairwise_centroid_mm": max(pair_dists) if pair_dists else 0.0,
                        "mean_pairwise_centroid_mm": float(np.mean(pair_dists)) if pair_dists else 0.0,
                        "cluster_mode": cluster_mode,
                        "cluster_eps_mm": float(cluster_eps_mm),
                    }
                )

                row: Dict[str, object] = {
                    "patient_id": pid,
                    "series_uid": series_uid,
                    "xml_path": xml_path,
                    "cluster_id": ci,
                    "patch_id": patch_id,
                    "rad_count": len(set(readers)),
                    "malignancy_values_json": json.dumps(malignancy_values),
                    "malignancy_mean": mal_mean,
                    "malignancy_median": float(np.median(malignancy_values)),
                    "malignancy_std": float(np.std(malignancy_values)),
                    "label": float(label),
                    "local_patch_path": local_path,
                    "context_patch_path": ctx_paths[int(primary_context_size)],
                    "ct_path": nifti_path,
                    "ct_x": int(ix),
                    "ct_y": int(iy),
                    "ct_z": int(iz),
                    "local_size": int(local_size),
                    "primary_context_size": int(primary_context_size),
                    "is_augmented": 0,
                    "cluster_mode": cluster_mode,
                    "cluster_eps_mm": float(cluster_eps_mm),
                    "cluster_max_pairwise_centroid_mm": max(pair_dists) if pair_dists else 0.0,
                }
                for s in context_sizes:
                    row[f"context_patch_path_{s}"] = ctx_paths[s]
                    x0, y0, z0 = ctx_bounds[s]
                    row[f"ctx{s}_x0"] = x0
                    row[f"ctx{s}_y0"] = y0
                    row[f"ctx{s}_z0"] = z0

                for tag in CONCEPT_TAGS:
                    raw = [(m.get("concepts") or {}).get(tag) for m in members]
                    valid, invalid = valid_concept_values(tag, raw)
                    if invalid:
                        invalid_rows.append(
                            {
                                "patient_id": pid,
                                "patch_id": patch_id,
                                "concept": tag,
                                "invalid_values_json": json.dumps(invalid),
                                "valid_codes_json": json.dumps(CONCEPT_SPECS[tag]["codes"]),
                            }
                        )
                    row[f"{tag}_values_json"] = json.dumps(valid)
                    row[f"{tag}_n_raters"] = len(valid)
                    row[f"{tag}_mean"] = float(np.mean(valid)) if valid else np.nan
                    row[f"{tag}_median"] = float(np.median(valid)) if valid else np.nan
                    row[f"{tag}_std"] = float(np.std(valid)) if valid else np.nan
                    row[f"{tag}_majority"] = majority_vote(valid) if valid else np.nan
                    row[f"{tag}_kind"] = CONCEPT_SPECS[tag]["kind"]
                rows.append(row)
        except Exception as e:
            print(f"[WARN] {mp}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(labels_dir, "nodules.csv")
    df.to_csv(out_csv, index=False)
    pd.DataFrame(invalid_rows).to_csv(
        os.path.join(labels_dir, "invalid_concept_codes.csv"), index=False
    )
    pd.DataFrame(cluster_audit).to_csv(
        os.path.join(labels_dir, "cluster_audit.csv"), index=False
    )

    # Compact dataset audit for manuscript/rebuttal.
    audit: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_patients": int(df.patient_id.nunique()) if len(df) else 0,
        "label_counts": df["label"].value_counts(dropna=False).to_dict() if len(df) else {},
        "context_sizes": context_sizes,
        "local_size": int(local_size),
        "primary_context_size": int(primary_context_size),
        "cluster_mode": cluster_mode,
        "cluster_eps_mm": float(cluster_eps_mm),
        "preprocessing_time_augmentation": False,
        "concept_specs": CONCEPT_SPECS,
        "invalid_concept_codes_n": int(len(invalid_rows)),
    }
    if len(df):
        audit["concept_rater_counts"] = {
            tag: {
                "samples_with_rating": int((df[f"{tag}_n_raters"] > 0).sum()),
                "mean_raters": float(df[f"{tag}_n_raters"].mean()),
            }
            for tag in CONCEPT_TAGS
        }
    with open(os.path.join(labels_dir, "dataset_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"Saved revised nodule CSV: {out_csv}")
    print(f"Original samples only: {len(df)}")
    if len(df):
        print(df["label"].value_counts(dropna=False).sort_index())



def validate_project_paths() -> bool:
    """Print a startup audit of all fixed local paths.

    DICOM_ROOT and XML_ROOT are required for LIDC preprocessing. The three TCIA
    metadata files are supplementary and therefore reported separately.
    """
    required = {
        "DICOM_ROOT": DICOM_ROOT,
        "XML_ROOT": XML_ROOT,
    }
    optional = {
        "TCIA_METADATA_CSV": TCIA_METADATA_CSV,
        "TCIA_NODULE_COUNTS_XLSX": TCIA_NODULE_COUNTS_XLSX,
        "TCIA_DIAGNOSIS_XLS": TCIA_DIAGNOSIS_XLS,
    }
    print("\nFixed project paths")
    print("-" * 78)
    ok = True
    for name, path in required.items():
        exists = os.path.isdir(path)
        ok = ok and exists
        print(f"[{'OK' if exists else 'MISSING'}] REQUIRED {name}: {path}")
    for name, path in optional.items():
        exists = os.path.isfile(path)
        print(f"[{'OK' if exists else 'MISSING'}] OPTIONAL {name}: {path}")
    print(f"[OUTPUT] PREPROC_ROOT: {PREPROC_ROOT}")

    # Lightweight structural checks. They provide immediate evidence that the two
    # required trees are not merely present but look like LIDC-IDRI inputs.
    if os.path.isdir(DICOM_ROOT):
        try:
            patient_dirs = discover_patient_dirs(DICOM_ROOT)
            print(f"[INFO] Discovered patient directories: {len(patient_dirs)}")
            if not patient_dirs:
                ok = False
        except Exception as e:
            ok = False
            print(f"[ERROR] Could not inspect DICOM_ROOT: {type(e).__name__}: {e}")
    if os.path.isdir(XML_ROOT):
        try:
            xml_index = build_xml_index(XML_ROOT)
            print(f"[INFO] Indexed XML SeriesInstanceUIDs: {len(xml_index)}")
            if not xml_index:
                ok = False
        except Exception as e:
            ok = False
            print(f"[ERROR] Could not inspect XML_ROOT: {type(e).__name__}: {e}")

    print("-" * 78)
    return ok


def run_all_fixed_paths(
    max_patients: int = 0,
    spacing: float = 1.0,
    hu_low: int = -1000,
    hu_high: int = 400,
    local_size: int = 64,
    context_sizes: Sequence[int] = (64, 80, 96, 112),
    primary_context_size: int = 96,
    cluster_eps_mm: float = 10.0,
    cluster_mode: str = "constrained_complete_linkage",
) -> None:
    """Run both LIDC preprocessing stages using the fixed Windows paths."""
    if not validate_project_paths():
        raise FileNotFoundError(
            "One or more required fixed paths are missing. Correct DICOM_ROOT/XML_ROOT "
            "at the top of this script before running preprocessing."
        )
    preprocess_subset(
        DICOM_ROOT,
        PREPROC_ROOT,
        max_patients=max_patients,
        spacing=spacing,
        hu_low=hu_low,
        hu_high=hu_high,
        xml_root=XML_ROOT,
        require_xml_match=True,
    )
    build_patches_and_labels(
        PREPROC_ROOT,
        XML_ROOT,
        local_size=local_size,
        context_sizes=context_sizes,
        primary_context_size=primary_context_size,
        cluster_eps_mm=cluster_eps_mm,
        cluster_mode=cluster_mode,
    )


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="LIDC-IDRI preprocessing and dual-scale patch construction"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("check_paths", help="Verify the fixed local dataset/metadata paths")

    p1 = sub.add_parser("preprocess_ct")
    p1.add_argument("--dicom_root", default=DICOM_ROOT, required=not bool(DICOM_ROOT))
    p1.add_argument("--out_root", default=PREPROC_ROOT, required=not bool(PREPROC_ROOT))
    p1.add_argument("--xml_root", default=XML_ROOT, required=not bool(XML_ROOT))
    p1.add_argument("--max_patients", type=int, default=0, help="0 = all patients")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--min_slices", type=int, default=50)
    p1.add_argument("--include_localizers", action="store_true")
    p1.add_argument("--allow_unmatched_series", action="store_true")
    p1.add_argument("--spacing", type=float, default=1.0)
    p1.add_argument("--hu_low", type=int, default=-1000)
    p1.add_argument("--hu_high", type=int, default=400)

    p2 = sub.add_parser("build_patches")
    p2.add_argument("--preproc_root", default=PREPROC_ROOT, required=not bool(PREPROC_ROOT))
    p2.add_argument("--xml_root", default=XML_ROOT, required=not bool(XML_ROOT))
    p2.add_argument("--local_size", type=int, default=64)
    p2.add_argument("--context_sizes", default="64,80,96,112")
    p2.add_argument("--primary_context_size", type=int, default=96)
    p2.add_argument("--cluster_eps_mm", type=float, default=10.0)
    p2.add_argument(
        "--cluster_mode",
        choices=["constrained_complete_linkage", "original_single_linkage"],
        default="constrained_complete_linkage",
    )
    p2.add_argument("--benign_max", type=float, default=2.0)
    p2.add_argument("--malignant_min", type=float, default=4.0)

    p3 = sub.add_parser(
        "run_all", help="Run CT preprocessing and patch construction"
    )
    p3.add_argument("--dicom_root", default=DICOM_ROOT, required=not bool(DICOM_ROOT))
    p3.add_argument("--xml_root", default=XML_ROOT, required=not bool(XML_ROOT))
    p3.add_argument("--out_root", default=PREPROC_ROOT, required=not bool(PREPROC_ROOT))
    p3.add_argument("--max_patients", type=int, default=0, help="0 = all patients")
    p3.add_argument("--spacing", type=float, default=1.0)
    p3.add_argument("--hu_low", type=int, default=-1000)
    p3.add_argument("--hu_high", type=int, default=400)
    p3.add_argument("--local_size", type=int, default=64)
    p3.add_argument("--context_sizes", default="64,80,96,112")
    p3.add_argument("--primary_context_size", type=int, default=96)
    p3.add_argument("--cluster_eps_mm", type=float, default=10.0)
    p3.add_argument(
        "--cluster_mode",
        choices=["constrained_complete_linkage", "original_single_linkage"],
        default="constrained_complete_linkage",
    )

    a = p.parse_args()
    if a.cmd == "check_paths":
        if not validate_project_paths():
            raise SystemExit(2)
    elif a.cmd == "preprocess_ct":
        preprocess_subset(
            a.dicom_root,
            a.out_root,
            max_patients=a.max_patients,
            seed=a.seed,
            min_slices=a.min_slices,
            exclude_localizers=not a.include_localizers,
            spacing=a.spacing,
            hu_low=a.hu_low,
            hu_high=a.hu_high,
            xml_root=a.xml_root,
            require_xml_match=not a.allow_unmatched_series,
        )
    elif a.cmd == "build_patches":
        build_patches_and_labels(
            a.preproc_root,
            a.xml_root,
            local_size=a.local_size,
            context_sizes=parse_int_list(a.context_sizes),
            primary_context_size=a.primary_context_size,
            cluster_eps_mm=a.cluster_eps_mm,
            cluster_mode=a.cluster_mode,
            benign_max=a.benign_max,
            malignant_min=a.malignant_min,
        )
    else:
        preprocess_subset(
            a.dicom_root, a.out_root, max_patients=a.max_patients, seed=42,
            min_slices=50, exclude_localizers=True, spacing=a.spacing,
            hu_low=a.hu_low, hu_high=a.hu_high, xml_root=a.xml_root,
            require_xml_match=True,
        )
        build_patches_and_labels(
            a.out_root, a.xml_root,
            local_size=a.local_size,
            context_sizes=parse_int_list(a.context_sizes),
            primary_context_size=a.primary_context_size,
            cluster_eps_mm=a.cluster_eps_mm,
            cluster_mode=a.cluster_mode,
        )


if __name__ == "__main__":
    main()
