#!/usr/bin/env bash
# Set up a virtual environment and run the end-to-end Assignment 3 workflow.
#
#   ./run_workflow.sh                  # all 9 models, full MBPP test split
#   ./run_workflow.sh --quick          # smoke test (20 problems, 2 training epochs)
#   ./run_workflow.sh --only scratch   # just the from-scratch transformers (runs on a Mac)
#   ./run_workflow.sh --setup-only     # create/update the venv, don't run anything
#
# Any other arguments are passed straight to Assignment03_KshaunishSoni.py.
# Use a specific interpreter with:  PYTHON=python3.11 ./run_workflow.sh
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [[ ! -d "$VENV" ]]; then
    echo ">> Creating virtual environment in $VENV with $("$PYTHON" --version)"
    "$PYTHON" -m venv "$VENV"
fi

echo ">> Installing requirements"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements.txt

"$VENV/bin/python" - <<'PY'
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f">> torch {torch.__version__}, device: {device}")
if device != "cuda":
    print(">> No CUDA GPU: the pretrained LLMs will be skipped (they need a GPU); "
          "the from-scratch transformers will run. Run --only pretrained on a GPU machine.")
PY

# CodeLlama models are gated: accept Meta's license on Hugging Face, then log in once.
if [[ -z "${HF_TOKEN:-}" ]] && ! "$VENV/bin/python" -c \
        "from huggingface_hub import get_token; import sys; sys.exit(get_token() is None)"; then
    echo ">> Not logged in to Hugging Face. CodeLlama downloads will fail until you run:"
    echo "     $VENV/bin/hf auth login     (or export HF_TOKEN=...)"
fi

if [[ "${1:-}" == "--setup-only" ]]; then
    echo ">> Setup complete. Activate with: source $VENV/bin/activate"
    exit 0
fi

echo ">> Running: python Assignment03_KshaunishSoni.py $*"
"$VENV/bin/python" Assignment03_KshaunishSoni.py "$@"
