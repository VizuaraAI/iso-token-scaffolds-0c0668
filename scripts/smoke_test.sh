#!/usr/bin/env bash
# End-to-end wiring check on the mock backend. No GPU, no network, seconds.
# If this is green, every moving part of the harness is connected.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

echo "== 1/4  import check"
python -c "import src.cost, src.models, src.data, src.scaffolds, src.retrieval, src.evaluate; print('   imports ok')"

echo "== 2/4  sweep grid (dry run)"
python -m src.run_sweep --config configs/default.yaml --dry-run | head -5

echo "== 3/4  full mock sweep"
python -m src.run_sweep --config configs/default.yaml

echo "== 4/4  analysis"
python -m src.analysis --results results/sweep.jsonl --cost-model raw

echo
echo "smoke test passed."
echo "Compare the raw and price accountings and watch the ranking move:"
echo "  python -m src.analysis --results results/sweep.jsonl --cost-model price"
