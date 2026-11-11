#!/usr/bin/env bash
# Create a local .venv with everything TokToxStability's TokaMaker/TORAX coupling
# needs, so PythonCall can be pointed at it (see README.md) with no local OFT build
# and no GPEC checkout required.
#
# Usage:   ./setup_venv.sh
#
# Prefers `uv` (fast) if available, otherwise falls back to the stdlib venv.
# Safe to re-run; it recreates .venv from scratch.
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION="3.12"
VENV_DIR=".venv"

echo ">> Creating $VENV_DIR (Python $PY_VERSION)"
if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY_VERSION" "$VENV_DIR"
    PY="$VENV_DIR/bin/python"
    uv pip install --python "$PY" pip
else
    python"$PY_VERSION" -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
    PY="$VENV_DIR/bin/python"
    "$PY" -m pip install --upgrade pip
fi

echo ">> Installing OpenFUSIONToolkit, TORAX, and helpers (from requirements.txt)"
"$PY" -m pip install -r requirements.txt

echo ""
echo ">> Done. Installed:"
"$PY" -m pip list | grep -iE "^(OpenFUSIONToolkit|torax|omfit_classes|uncertainties|h5py) " || true
echo ""
echo "Point PythonCall at this venv before 'using TokToxStability' (e.g. in your"
echo "notebook's first cell) -- this is a Julia notebook, so Jupyter never launches"
echo "this venv as its own kernel; PythonCall just loads its libpython in-process:"
echo '    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"'
echo "    ENV[\"JULIA_PYTHONCALL_EXE\"] = \"$(pwd)/$VENV_DIR/bin/python\""
