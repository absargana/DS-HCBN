# Data layout

Datasets are not redistributed. Download LIDC-IDRI from TCIA and its XML annotations from the official LIDC-IDRI annotation release. LNDb is required only for external evaluation.

After preprocessing, the labels CSV contains one row per grouped nodule and absolute paths to the local and context patch arrays. Expected core fields include `patient_id`, `label`, local/context patch paths, and the eight radiological concept targets. Reader-level ratings are retained as JSON arrays where applicable.

Recommended local layout (all ignored by Git):

```text
data/
  LIDC-IDRI/
  tcia-lidc-xml/
  preprocessed/
  LNDb/
outputs/
checkpoints/
```

Use `configs/paths.example.json` as a private local template. Never commit protected data, credentials, or machine-specific absolute paths.
