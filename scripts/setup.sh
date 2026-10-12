#!/usr/bin/env bash
# Create a virtual environment and install the core harness.
# GPU extras are deliberately separate: get the mock path green first.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "creating .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install numpy pyyaml matplotlib tqdm datasets rank-bm25

echo
echo "core harness installed."
echo "next:  ./scripts/smoke_test.sh"
echo
echo "when you have a GPU, add the serving stack:"
echo "  pip install torch transformers accelerate vllm openai sentence-transformers"
