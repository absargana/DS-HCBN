param(
  [Parameter(Mandatory=$true)][string]$LabelsCsv,
  [Parameter(Mandatory=$true)][string]$OutputRoot,
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Repo "src"
$Split = Join-Path $OutputRoot "split"
$Train = Join-Path $OutputRoot "residual_cnn"
$Frozen = Join-Path $Train "frozen"

& $Python (Join-Path $Repo "scripts\split\create_locked_split.py") --labels_csv $LabelsCsv --out_dir $Split --seed 42 --train_patients 412 --val_patients 88 --test_patients 89
if ($LASTEXITCODE -ne 0) { throw "Split creation failed." }
& $Python (Join-Path $Repo "scripts\split\preflight_split.py") --labels_csv $LabelsCsv --split_manifest (Join-Path $Split "LOCKED_SPLIT_MANIFEST.csv") --split_lock (Join-Path $Split "SPLIT_LOCK.json")
if ($LASTEXITCODE -ne 0) { throw "Split preflight failed." }
& $Python (Join-Path $Repo "scripts\training\train.py") --encoder residual_cnn --labels_csv $LabelsCsv --split_manifest (Join-Path $Split "LOCKED_SPLIT_MANIFEST.csv") --out_dir $Train --seed 42 --epochs 45 --batch_size 2 --lr 0.0001 --weight_decay 0.001 --dropout 0.25 --lambda_concept 0.35 --lambda_rank 0.10 --rank_margin 0.50 --focal_alpha 0.60 --focal_gamma 1.5 --ema_decay 0.995 --early_stop 8 --target_sensitivity 0.90
if ($LASTEXITCODE -ne 0) { throw "Primary training failed." }
& $Python (Join-Path $Repo "scripts\evaluation\freeze_checkpoint.py") --checkpoint (Join-Path $Train "best_model.pt") --results_json (Join-Path $Train "results.json") --split_manifest (Join-Path $Split "LOCKED_SPLIT_MANIFEST.csv") --split_lock (Join-Path $Split "SPLIT_LOCK.json") --out_dir $Frozen
if ($LASTEXITCODE -ne 0) { throw "Checkpoint freeze failed." }
Write-Host "[PASS] Primary model trained and frozen. Run test evaluation only when ready." -ForegroundColor Green
