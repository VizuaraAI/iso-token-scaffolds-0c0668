#!/usr/bin/env bash
# The real sweep on the Qwen2.5 ladder, then every analysis view.
#
# Read configs/qwen_ladder.yaml and shrink the limits before the first run.
# A full grid is thousands of generations per cell.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

CONFIG="${1:-configs/qwen_ladder.yaml}"
RESULTS="${2:-results/qwen_ladder.jsonl}"

echo "== sweep: $CONFIG"
python -m src.run_sweep --config "$CONFIG" --out "$RESULTS"

for MODEL in raw price flop; do
  echo
  echo "== analysis under the '$MODEL' cost accounting"
  python -m src.analysis \
    --results "$RESULTS" \
    --cost-model "$MODEL" \
    --figure "results/frontier_${MODEL}.png"
done

echo
echo "== cost proxy validation"
python -m src.calibrate --results "$RESULTS"

echo
echo "done. If the three accountings rank configurations differently,"
echo "that disagreement is the paper, not a bug."
