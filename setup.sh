#!/usr/bin/env bash
# Mac/Linux setup: creates .venv with Python 3.11 and installs dependencies.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.11}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Done. Activate with:  source .venv/bin/activate"
