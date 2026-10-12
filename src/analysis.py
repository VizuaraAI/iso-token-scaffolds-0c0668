"""Evaluation entry point: turn the sweep into the paper's Table 1 and Figure 2.

    python -m src.analysis --results results/sweep.jsonl
    python -m src.analysis --results results/sweep.jsonl --cost-model price

Outputs a LaTeX table body whose rows line up with Table 1 of the draft, plus
the accuracy-versus-cost figure. Paste the table body straight into main.tex
and the XX.X placeholders become your numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .cost import accuracy_at_budget, crossover_cost

COST_FIELDS = {"raw": "cost_raw", "price": "cost_price", "flop": "cost_flop"}


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def curves(rows: list[dict], cost_field: str) -> dict[tuple, list[tuple]]:
    """Group rows into accuracy-versus-cost curves keyed by (dataset, model, scaffold)."""
    out: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["model"], r["scaffold"])
        out[key].append((r[cost_field], r["accuracy"]))
    return {k: sorted(v) for k, v in out.items()}


def reference_point(
    rows: list[dict], dataset: str, cost_field: str,
    ref_model: str, ref_scaffold: str = "greedy_cot",
) -> tuple[float, float] | None:
    """The large model's single-shot (cost, accuracy). Everything is read against this."""
    for r in rows:
        if (r["dataset"] == dataset and r["model"] == ref_model
                and r["scaffold"] == ref_scaffold):
            return r[cost_field], r["accuracy"]
    return None


def build_table(rows: list[dict], cost_model: str, ref_model: str) -> list[dict]:
    cost_field = COST_FIELDS[cost_model]
    all_curves = curves(rows, cost_field)
    table = []

    for dataset in sorted({r["dataset"] for r in rows}):
        ref = reference_point(rows, dataset, cost_field, ref_model)
        if ref is None:
            print(f"  ! no reference row for {dataset}/{ref_model}, skipping")
            continue
        ref_cost, ref_acc = ref

        for (ds, model, scaffold), curve in sorted(all_curves.items()):
            if ds != dataset:
                continue
            best = max(curve, key=lambda p: p[1])
            cstar = crossover_cost(curve, ref_acc)
            iso = accuracy_at_budget(curve, ref_cost)
            table.append({
                "dataset": ds,
                "model": model,
                "scaffold": scaffold,
                "best_accuracy": best[1],
                "best_cost": best[0],
                # Accuracy when forced down to the reference model's budget.
                "iso_cost_accuracy": iso,
                "margin_param_matched": round(best[1] - ref_acc, 2),
                "margin_iso_cost": (
                    round(iso - ref_acc, 2) if iso is not None else None
                ),
                "crossover_ratio": (
                    round(cstar / ref_cost, 2) if cstar else None
                ),
            })
    return table


def to_latex(table: list[dict], rows_src: list[dict]) -> str:
    """Emit the body of Table 1. Column order matches the draft."""
    tok = {
        (r["dataset"], r["model"], r["scaffold"], r["budget"]):
        (r["mean_tokens_in"], r["mean_tokens_out"]) for r in rows_src
    }
    lines = []
    for e in table:
        ins = [v[0] for k, v in tok.items() if k[:3] ==
               (e["dataset"], e["model"], e["scaffold"])]
        outs = [v[1] for k, v in tok.items() if k[:3] ==
                (e["dataset"], e["model"], e["scaffold"])]
        tin = f"{max(ins):.0f}" if ins else "--"
        tout = f"{max(outs):.0f}" if outs else "--"
        iso = e["iso_cost_accuracy"]
        iso_s = f"{iso:.1f}" if iso is not None else "--"
        cs = e["crossover_ratio"]
        cs_s = f"{cs:.1f}" if cs is not None else "$>$ sweep"
        name = e["scaffold"].replace("_", " ")
        lines.append(
            f"    {e['model']} & {name} & {tin} & {tout} & "
            f"{e['best_accuracy']:.1f} & {iso_s} & {cs_s} \\\\"
        )
    return "\n".join(lines)


def plot_frontier(rows: list[dict], cost_model: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost_field = COST_FIELDS[cost_model]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (ds, model, scaffold), curve in sorted(curves(rows, cost_field).items()):
        xs = [c for c, _ in curve]
        ys = [a for _, a in curve]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                label=f"{model} + {scaffold.replace('_', ' ')}")
    ax.set_xscale("log")
    ax.set_xlabel(f"total token cost per query ({cost_model}, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy vs total token cost")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/sweep.jsonl")
    parser.add_argument(
        "--cost-model", default="raw", choices=sorted(COST_FIELDS)
    )
    parser.add_argument(
        "--ref-model", default="large",
        help="name of the large single-shot reference model in the config",
    )
    parser.add_argument("--figure", default="results/frontier.png")
    args = parser.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"no rows in {args.results}, run src.run_sweep first")

    table = build_table(rows, args.cost_model, args.ref_model)

    header = (
        f"{'model':<10} {'scaffold':<20} {'best':>7} {'iso-cost':>9} "
        f"{'d param':>8} {'d iso':>7} {'C*/Cref':>8}"
    )
    print(f"\ncost model: {args.cost_model}   reference: {args.ref_model}")
    print(header)
    print("-" * len(header))
    for e in table:
        iso = e["iso_cost_accuracy"]
        cs = e["crossover_ratio"]
        d_iso = e["margin_iso_cost"]
        iso_s = f"{iso:.1f}" if iso is not None else "--"
        d_iso_s = f"{d_iso:+.1f}" if d_iso is not None else "--"
        cs_s = f"{cs:.2f}" if cs is not None else "> sweep"
        print(
            f"{e['model']:<10} {e['scaffold']:<20} "
            f"{e['best_accuracy']:>7.1f} {iso_s:>9} "
            f"{e['margin_param_matched']:>+8.1f} {d_iso_s:>7} {cs_s:>8}"
        )

    print("\n% --- paste into Table 1 of main.tex ---")
    print(to_latex(table, rows))

    plot_frontier(rows, args.cost_model, args.figure)


if __name__ == "__main__":
    main()
