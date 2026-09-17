param(
  [Parameter(Mandatory=$true)][string]$LabelsCsv,
  [Parameter(Mandatory=$true)][string]$SplitManifest,
  [Parameter(Mandatory=$true)][string]$OutputRoot,
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Repo "src"
$Candidates = @(
  @{Name="gn192_lr5e5_drop10"; Lr="0.00005"},
  @{Name="gn192_lr1e4_drop10"; Lr="0.00010"}
)
foreach ($Candidate in $Candidates) {
  $Out = Join-Path $OutputRoot $Candidate.Name
  if (Test-Path (Join-Path $Out "results.json")) {
    Write-Host "[SKIP] $($Candidate.Name) already completed."
    continue
  }
  & $Python (Join-Path $Repo "scripts\training\train.py") --encoder unet --encoder_norm group --encoder_final_channels 192 --encoder_drop 0.10 --lr $Candidate.Lr --weight_decay 0.002 --labels_csv $LabelsCsv --split_manifest $SplitManifest --out_dir $Out --seed 42 --epochs 12 --batch_size 2 --early_stop 4
  if ($LASTEXITCODE -ne 0) { throw "U-Net candidate failed: $($Candidate.Name)" }
}
Write-Host "[PASS] Validation-only U-Net candidates completed. Select by validation results before freezing." -ForegroundColor Green
