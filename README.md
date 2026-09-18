# DS-HCBN: Dual-Scale Hybrid Concept Bottleneck Network

Official reproducibility code for dual-scale 3D lung-nodule malignancy classification with interpretable radiological concepts. The **shared residual 3D CNN encoder is the primary manuscript model**. A shared 3D U-Net encoder is included as a controlled ablation/comparison using the same split, losses, fusion mechanism, and evaluation protocol.

## Repository contents

```text
configs/                 Reproducible model and path examples
docs/                    Data-layout and reproducibility notes
reproducibility/         Locked split, split hash, and compact verified results
scripts/
  preprocessing/         LIDC-IDRI CT, annotation, clustering, and patch pipeline
  split/                 Patient-wise stratified split and leakage preflight
  training/              Shared trainer for residual-CNN and U-Net encoders
  evaluation/            Checkpoint/threshold freezing and internal test evaluation
  external/              LNDb preparation, inference, and scoring
  analyses/              Uncertain-nodule, clustering-sensitivity, and complexity analyses
src/dshcbn/models/        Exact residual-CNN and U-Net DS-HCBN implementations
tests/                    Fast installation/model smoke test
```

Datasets, CT volumes, generated patch arrays, checkpoints, prediction-level files, caches, manuscripts, and temporary logs are intentionally excluded.

## Installation

Python 3.10 or newer is recommended. Install PyTorch for the CUDA version available on your machine, then install this repository:

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
python tests/smoke_test.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. The scripts automatically use CUDA when available.

## 1. Preprocess LIDC-IDRI

```bash
python scripts/preprocessing/preprocess_lidc.py run_all \
  --dicom_root data/LIDC-IDRI \
  --xml_root data/tcia-lidc-xml \
  --out_root data/preprocessed \
  --context_sizes 64,80,96,112 \
  --primary_context_size 96 \
  --cluster_eps_mm 10
```

The preprocessing pipeline performs series selection, HU clipping, isotropic resampling, XML parsing, constrained complete-linkage annotation grouping, concept aggregation, and dual-scale patch generation. It does not create augmented samples; augmentation occurs only in the training dataset after the patient-wise split.

## 2. Create and verify the locked split

```bash
python scripts/split/create_locked_split.py \
  --labels_csv data/preprocessed/labels/nodules.csv \
  --out_dir outputs/split --seed 42 \
  --train_patients 412 --val_patients 88 --test_patients 89

python scripts/split/preflight_split.py \
  --labels_csv data/preprocessed/labels/nodules.csv \
  --split_manifest outputs/split/LOCKED_SPLIT_MANIFEST.csv \
  --split_lock outputs/split/SPLIT_LOCK.json
```

Do not recreate or edit the split between experiments. The manuscript split is archived under `reproducibility/`.

## 3. Train the primary residual-CNN model

```bash
python scripts/training/train.py \
  --encoder residual_cnn \
  --labels_csv data/preprocessed/labels/nodules.csv \
  --split_manifest outputs/split/LOCKED_SPLIT_MANIFEST.csv \
  --out_dir outputs/residual_cnn --seed 42 --epochs 45 \
  --batch_size 2 --lr 0.0001 --weight_decay 0.001 \
  --dropout 0.25 --lambda_concept 0.35 --lambda_rank 0.10 \
  --rank_margin 0.50 --focal_alpha 0.60 --focal_gamma 1.5 \
  --ema_decay 0.995 --early_stop 8 --target_sensitivity 0.90
```

## 4. Freeze and evaluate the internal test set

Freeze the chosen checkpoint and validation-derived threshold before test evaluation:

```bash
python scripts/evaluation/freeze_checkpoint.py \
  --checkpoint outputs/residual_cnn/best_model.pt \
  --results_json outputs/residual_cnn/results.json \
  --split_manifest outputs/split/LOCKED_SPLIT_MANIFEST.csv \
  --split_lock outputs/split/SPLIT_LOCK.json \
  --out_dir outputs/residual_cnn/frozen

python scripts/evaluation/evaluate_internal.py \
  --encoder residual_cnn \
  --labels_csv data/preprocessed/labels/nodules.csv \
  --split_manifest outputs/split/LOCKED_SPLIT_MANIFEST.csv \
  --frozen_spec outputs/residual_cnn/frozen/FROZEN_EVAL_SPEC.json \
  --out_dir outputs/residual_cnn/test
```

## 5. U-Net encoder comparison

Use the same locked split. Candidate settings are recorded in `configs/unet_validation_tuning.json`. A representative controlled candidate is:

```bash
python scripts/training/train.py \
  --encoder unet --encoder_norm group --encoder_final_channels 192 \
  --encoder_drop 0.10 --lr 0.00005 --weight_decay 0.002 \
  --labels_csv data/preprocessed/labels/nodules.csv \
  --split_manifest outputs/split/LOCKED_SPLIT_MANIFEST.csv \
  --out_dir outputs/unet/gn192_lr5e5_drop10 \
  --seed 42 --epochs 12 --batch_size 2 --early_stop 4
```

Compare candidates on validation data only. Freeze the selected candidate with the same command used above, then call `evaluate_internal.py --encoder unet`. Do not use test performance for hyperparameter selection.

## 6. External LNDb evaluation

```bash
python scripts/external/prepare_lndb_annotated.py \
  --all_nods_csv data/LNDb/trainNodules.csv \
  --test_cts_csv data/LNDb/testCTs.csv \
  --out_dir outputs/lndb/prepared

python scripts/external/evaluate_lndb.py \
  --encoder residual_cnn --ct_dir data/LNDb/data0 \
  --candidates_csv outputs/lndb/prepared/LNDb_public_annotated_candidates.csv \
  --labels_csv outputs/lndb/prepared/LNDb_public_annotated_labels.csv \
  --checkpoint outputs/residual_cnn/best_model.pt \
  --frozen_spec outputs/residual_cnn/frozen/FROZEN_EVAL_SPEC.json \
  --out_dir outputs/lndb/evaluation
```

The scoring and archive-stratified utilities in `scripts/external/` support prediction-only rescoring and LNDb archive-level reporting.

## 7. Additional robustness analyses

- `scripts/analyses/analyze_uncertain_nodules.py`: frozen-model behavior for mean malignancy strictly between 2 and 4.
- `scripts/analyses/postprocess_uncertain_by_split.py`: split-wise uncertain-nodule summaries.
- `scripts/analyses/annotation_threshold_sensitivity.py`: clustering-distance sensitivity analysis.
- `scripts/analyses/benchmark_complexity.py`: parameter count, latency, memory, and profiler-supported FLOPs for both encoders.

Run any script with `--help` for its exact arguments.

## Reproducibility and integrity

The split is patient-wise, fixed before training, and protected by SHA-256. Checkpoint selection and operating-threshold selection use validation data only; the test set is evaluated after freezing. Compact audit results are provided under `reproducibility/results/`. See `docs/REPRODUCIBILITY.md` for the full protocol.

## Citation

Please cite the associated DS-HCBN manuscript. 

## License

A license has deliberately not been assigned without rights-holder approval. Complete `LICENSE_SELECTION_REQUIRED.md` before making the repository public.
