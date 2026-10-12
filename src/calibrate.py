"""Fit each token cost proxy against measured GPU-seconds.

This is the paper's "proxy validation" ablation and it is the step that turns
an opinion into a recommendation. If `price` explains GPU-seconds with
R^2 = 0.97 and `raw` only manages 0.71, you have earned the right to tell the
community which accounting to standardise on.

    python -m src.calibrate --results results/sweep.jsonl

Requires that your sweep ran against real hardware, so the latency column is
meaningful. The mock backend produces a degenerate fit, which is expected.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

COST_FIELDS = ["cost_raw", "cost_price", "cost_flop"]


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares y = a*x + b. Returns (a, b, r_squared)."""
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/sweep.jsonl")
    parser.add_argument(
        "--target", default="mean_latency_s",
        help="measured cost column to predict (mean_latency_s or gpu_seconds)",
    )
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) < 3:
        raise SystemExit("need at least 3 sweep rows to fit anything")

    y = np.array([r[args.target] for r in rows], dtype=float)
    if float(np.std(y)) == 0.0:
        print(
            f"warning: {args.target} is constant across rows. "
            "You are almost certainly looking at mock-backend output."
        )

    print(f"\npredicting {args.target} from each cost accounting "
          f"({len(rows)} sweep cells)\n")
    print(f"{'accounting':<14} {'slope':>12} {'intercept':>12} {'R^2':>8}")
    print("-" * 48)

    results = {}
    for field in COST_FIELDS:
        x = np.array([r[field] for r in rows], dtype=float)
        a, b, r2 = fit(x, y)
        results[field] = r2
        print(f"{field:<14} {a:>12.3e} {b:>12.4f} {r2:>8.4f}")

    best = max(results, key=lambda k: results[k])
    print(
        f"\nbest proxy: {best}  (R^2 = {results[best]:.4f})\n"
        "Report this table in the ablations section. A cost accounting that "
        "does not track measured hardware cost is not worth standardising on."
    )


if __name__ == "__main__":
    main()
