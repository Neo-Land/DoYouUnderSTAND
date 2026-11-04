# LEGACY: needs mediapipe 0.10.14 (mp.solutions). The repo .venv runs 0.10.35 for dyu/, so this
# script will not run there until asl/ is ported to the Tasks API.
# One-shot ASL pipeline: download dataset -> extract hand landmarks -> train model.
# Safe to re-run: skips the download if the dataset is already there.
# Writes a log to logs/train_asl.log
# Python/MediaPipe log to stderr, so check exit codes instead of using ErrorActionPreference=Stop
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force logs | Out-Null
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { & .\setup.ps1 }

$log = "logs\train_asl.log"
"=== $(Get-Date) ===" | Out-File $log

function Step($name, [string[]]$pyArgs) {
    Write-Host ">> $name"
    & $py @pyArgs *>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: $name (see $log)"; exit 1 }
}

Step "download" @("asl\download_asl.py")
Step "extract landmarks" @("asl\import_asl.py", "--fresh", "--per-class", "1500")
Step "train" @("asl\train.py", "--mode", "asl")
Write-Host "ASL pipeline finished. Model: models\asl.joblib  Log: logs\train_asl.log"
