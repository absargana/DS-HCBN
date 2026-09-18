# Reproducibility protocol

1. Preprocess the CT scans without offline augmentation.
2. Construct dual-scale local (64 cubed) and context (96 cubed) patches and preserve reader-level concept ratings.
3. Create the patient-wise stratified split once with seed 42, then verify its hash against `reproducibility/SPLIT_LOCK.json`.
4. Train the residual-CNN primary model. Training-only augmentation is applied after splitting.
5. Select the checkpoint and operating threshold using validation data only.
6. Freeze the checkpoint SHA-256 and threshold specification before evaluating the test set.
7. Run U-Net tuning on the same locked split, select only from validation results, freeze, and then evaluate once for the encoder comparison.
8. Run LNDb external evaluation without retraining.

The compact JSON files under `reproducibility/results/` are retained audit outputs. Checkpoints and prediction-level files are excluded because of size; the frozen specification records the primary checkpoint digest.
