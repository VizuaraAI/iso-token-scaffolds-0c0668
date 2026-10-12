"""Main experiment entry point: sweep budget knobs and record cost.

This produces one JSONL row per (dataset, model, scaffold, budget) cell. Each
row carries the accuracy AND the measured token spend under all three cost
accountings. That file is the only input the analysis needs.

    python -m src.run_sweep --config configs/default.yaml
    python -m src.run_sweep --config configs/default.yaml --dry-run

Start with the mock backend. Confirm the sweep shape and the analysis output
are what you expect before you touch a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import yaml

from .cost import TokenAccountant
from .data import Example, load_dataset_by_name
from .evaluate import accuracy, is_correct
from .models import build_backend
from .retrieval import build_retriever
from .scaffolds import SCAFFOLD_FAMILY, get_scaffold


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def exemplar_pool(cfg: dict) -> list[Example]:
    """Training-split examples for the retriever index. Never test data."""
    pool_cfg = cfg.get("retrieval", {}).get("pool", {})
    if not pool_cfg:
        return []
    return load_dataset_by_name(
        pool_cfg["dataset"],
        limit=pool_cfg.get("limit", 500),
        **({"split": pool_cfg["split"]} if "split" in pool_cfg else {}),
    )


def run_cell(
    examples: list[Example],
    backend,
    scaffold_name: str,
    budget: int,
    retriever,
) -> dict:
    """Run one (model, scaffold, budget) cell over the whole dataset."""
    scaffold = get_scaffold(scaffold_name)
    acct = TokenAccountant()
    predictions: dict[str, str] = {}
    correct: dict[str, bool] = {}

    start = time.perf_counter()
    for example in examples:
        if hasattr(retriever, "set_query_gold"):
            retriever.set_query_gold(example.uid)
        with acct.query(example.uid) as qcost:
            try:
                prediction = scaffold(
                    example, backend, qcost, budget=budget, retriever=retriever
                )
            except Exception as exc:  # a failed query still costs tokens
                print(f"  ! {example.uid} failed: {exc}")
                prediction = ""
        predictions[example.uid] = prediction
        correct[example.uid] = is_correct(prediction, example)
    wall_s = time.perf_counter() - start

    tok_in, tok_out = acct.mean_tokens()
    return {
        "scaffold": scaffold_name,
        "family": SCAFFOLD_FAMILY.get(scaffold_name, "unknown"),
        "budget": budget,
        "n_examples": len(examples),
        "accuracy": round(accuracy(predictions, examples), 3),
        "mean_calls": round(acct.mean_calls(), 3),
        "mean_tokens_in": round(tok_in, 2),
        "mean_tokens_out": round(tok_out, 2),
        "cost_raw": round(acct.mean_cost("raw"), 2),
        "cost_price": round(acct.mean_cost("price"), 2),
        "cost_flop": round(acct.mean_cost("flop"), 2),
        "mean_latency_s": round(acct.mean_latency(), 4),
        "wall_s": round(wall_s, 2),
        "per_query_correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="override output path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sweep grid and exit without any model calls",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = args.out or cfg["output"]["results"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    grid = [
        (d["name"], m["name"], s["name"], b)
        for d in cfg["datasets"]
        for m in cfg["models"]
        for s in cfg["scaffolds"]
        for b in s["budgets"]
    ]
    print(f"sweep grid: {len(grid)} cells")
    if args.dry_run:
        for cell in grid:
            print("  ", cell)
        return

    pool = exemplar_pool(cfg)
    rows = []

    for dcfg in cfg["datasets"]:
        examples = load_dataset_by_name(
            dcfg["name"],
            limit=dcfg.get("limit"),
            **({"split": dcfg["split"]} if "split" in dcfg else {}),
        )
        print(f"\n== {dcfg['name']}: {len(examples)} examples")

        retriever = build_retriever(
            cfg.get("retrieval", {}).get("kind", "null"),
            pool,
            gold_pool=pool,
        )

        for mcfg in cfg["models"]:
            backend = build_backend(mcfg["backend"])
            for scfg in cfg["scaffolds"]:
                for budget in scfg["budgets"]:
                    label = (
                        f"{dcfg['name']} | {mcfg['name']} | "
                        f"{scfg['name']} | b={budget}"
                    )
                    print(f"  running {label}")
                    row = run_cell(
                        examples, backend, scfg["name"], budget, retriever
                    )
                    row.update(
                        {
                            "dataset": dcfg["name"],
                            "model": mcfg["name"],
                            "model_id": mcfg["backend"].get("model", mcfg["name"]),
                        }
                    )
                    print(
                        f"    acc={row['accuracy']:.1f}  "
                        f"raw={row['cost_raw']:.0f}  "
                        f"price={row['cost_price']:.0f}  "
                        f"calls={row['mean_calls']:.1f}"
                    )
                    rows.append(row)

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} rows to {out_path}")
    print("now run: python -m src.analysis --results " + out_path)


if __name__ == "__main__":
    main()
