#!/usr/bin/env bash
# Mac/Linux setup: creates .venv with Python 3.11, installs the dependencies and the dyu package.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.11}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps   # makes `import dyu` work from scripts/ and app.py
echo "Done. Activate with:  source .venv/bin/activate"
