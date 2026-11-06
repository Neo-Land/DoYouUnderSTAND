# Run the whole ASL sign pipeline for a vocabulary size, unattended, logging to logs/signs_vocab<N>.log.
#
#   powershell -ExecutionPolicy Bypass -File scripts/signs/run_vocab.ps1 -Vocab 300
#
# Start it detached (keeps running if the terminal or Claude is closed):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy Bypass -File scripts/signs/run_vocab.ps1 -Vocab 300'
#
# Steps: unpack the signs' videos from the ASL Citizen zip -> landmarks for any videos not done yet
# (resumable) -> logistic-regression baseline -> GRU (3 seeds) -> live-pipeline replay. Outputs for 300
# are named *_300 (reports/signs/gru_300.json, models/sign_gru_300.pt), so the app's 100-sign model and
# the published 100-sign numbers are untouched. Python/MediaPipe write progress to stderr, so each step's
# exit code is checked instead of relying on $ErrorActionPreference.
param([int]$Vocab = 300)

Set-Location (Join-Path $PSScriptRoot "..\..")
New-Item -ItemType Directory -Force logs | Out-Null
$py = ".\.venv\Scripts\python.exe"
$log = "logs\signs_vocab$Vocab.log"
"=== $(Get-Date -Format s)  vocab $Vocab  started ===" | Out-File $log -Encoding utf8

function Step($name, [string[]]$pyArgs) {
    "`n=== $(Get-Date -Format s)  $name ===" | Out-File $log -Append -Encoding utf8
    & $py @pyArgs *>&1 | ForEach-Object { "$_" } |
        Where-Object { $_ -notmatch '^(W0000|I0000|E0000|INFO:)|clearcut|Source Location|^\s+@' } |
        Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        "=== $(Get-Date -Format s)  FAILED at: $name (exit $LASTEXITCODE) ===" | Out-File $log -Append -Encoding utf8
        exit 1
    }
}

Step "unpack videos"   @("scripts\signs\extract_asl_citizen.py", "--vocab", "$Vocab")
Step "landmarks"       @("scripts\signs\extract_landmarks.py", "--workers", "8")
Step "train logreg"    @("scripts\signs\train.py", "--model", "logreg", "--vocab", "$Vocab")
Step "train gru"       @("scripts\signs\train.py", "--model", "gru", "--vocab", "$Vocab")
Step "live replay"     @("scripts\signs\eval_live.py", "--vocab", "$Vocab")
"`n=== $(Get-Date -Format s)  DONE ===" | Out-File $log -Append -Encoding utf8
