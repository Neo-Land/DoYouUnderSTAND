# Windows setup: creates .venv with Python 3.11, installs the dependencies and the dyu package.
# If PowerShell blocks this script, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    # Prefer the py launcher; fall back to a python on PATH (e.g. the Microsoft Store install).
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv }
    else { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"
& $py --version
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip install -e . --no-deps   # makes `import dyu` work from scripts/ and app.py

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
