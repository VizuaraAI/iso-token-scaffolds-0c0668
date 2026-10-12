"""Token accounting and the three cost models from the paper.

This is the heart of the project. Everything else exists to feed it.

A scaffold makes many language model calls. Each call reports how many prompt
tokens it read and how many tokens it generated. The accountant collects those
per-call records and turns them into three different prices for the same run:

    raw    = sum(n_in + n_out)
    price  = sum(w * n_in + n_out)        w defaults to 0.25
    flop   = sum(n_in + n_out) * params   (tokens times active parameters)

The whole argument of the paper is that these three numbers rank
configurations differently, so a comparison that reports only accuracy is
underspecified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterable


# Active parameter counts, in billions, for the models on the ladder.
# Used by the `flop` cost model so that an 8B token is not priced the same as
# a 32B token. Add your own checkpoints here.
MODEL_PARAMS_B: dict[str, float] = {
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen2.5-32B-Instruct": 32.8,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "mock-small": 7.6,
    "mock-large": 32.8,
}


@dataclass
class CallRecord:
    """One language model call, metered."""

    model: str
    role: str  # "generate", "critique", "verify", "rerank"
    n_in: int
    n_out: int
    latency_s: float = 0.0


@dataclass
class QueryCost:
    """All the calls spent on a single benchmark query."""

    query_id: str
    calls: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.calls.append(record)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def tokens_in(self) -> int:
        return sum(c.n_in for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.n_out for c in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def cost(self, model: str = "raw", input_weight: float = 0.25) -> float:
        """Total token cost under one of the three accountings.

        `model` is one of "raw", "price", "flop". Note that `flop` uses each
        call's own model, which matters for scaffolds that mix sizes.
        """
        if model == "raw":
            return float(self.tokens_in + self.tokens_out)
        if model == "price":
            return float(
                sum(input_weight * c.n_in + c.n_out for c in self.calls)
            )
        if model == "flop":
            return float(
                sum(
                    (c.n_in + c.n_out) * MODEL_PARAMS_B.get(c.model, 1.0)
                    for c in self.calls
                )
            )
        raise ValueError(
            f"unknown cost model {model!r}, expected raw, price or flop"
        )

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "n_calls": self.n_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_s": round(self.latency_s, 4),
            "cost_raw": self.cost("raw"),
            "cost_price": self.cost("price"),
            "cost_flop": self.cost("flop"),
            "calls": [asdict(c) for c in self.calls],
        }


class TokenAccountant:
    """Collects QueryCost objects across a whole evaluation run.

    Usage inside a scaffold::

        acct = TokenAccountant()
        with acct.query("gsm8k-0042") as q:
            response = backend.generate(prompt, cost=q, role="generate")
    """

    def __init__(self) -> None:
        self.queries: list[QueryCost] = []
        self._current: QueryCost | None = None

    def query(self, query_id: str) -> "_QueryContext":
        return _QueryContext(self, query_id)

    def _open(self, query_id: str) -> QueryCost:
        self._current = QueryCost(query_id=query_id)
        return self._current

    def _close(self) -> None:
        if self._current is not None:
            self.queries.append(self._current)
            self._current = None

    # ---- aggregate views -------------------------------------------------

    def mean_cost(self, model: str = "raw", input_weight: float = 0.25) -> float:
        if not self.queries:
            return 0.0
        total = sum(q.cost(model, input_weight) for q in self.queries)
        return total / len(self.queries)

    def mean_tokens(self) -> tuple[float, float]:
        if not self.queries:
            return 0.0, 0.0
        n = len(self.queries)
        return (
            sum(q.tokens_in for q in self.queries) / n,
            sum(q.tokens_out for q in self.queries) / n,
        )

    def mean_calls(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.n_calls for q in self.queries) / len(self.queries)

    def mean_latency(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.latency_s for q in self.queries) / len(self.queries)

    def dump_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for q in self.queries:
                fh.write(json.dumps(q.to_dict()) + "\n")


class _QueryContext:
    def __init__(self, accountant: TokenAccountant, query_id: str) -> None:
        self.accountant = accountant
        self.query_id = query_id
        self.record: QueryCost | None = None

    def __enter__(self) -> QueryCost:
        self.record = self.accountant._open(self.query_id)
        return self.record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.accountant._close()
        return False


def crossover_cost(
    curve: Iterable[tuple[float, float]], target_accuracy: float
) -> float | None:
    """Smallest cost at which an accuracy-versus-cost curve reaches a target.

    `curve` is an iterable of (cost, accuracy) points. Returns None if the
    curve never reaches the target inside the swept budget, which is a real
    and reportable outcome (the paper writes it as "> sweep").
    """
    points = sorted(curve)
    for i, (cost, acc) in enumerate(points):
        if acc >= target_accuracy:
            if i == 0:
                return cost
            prev_cost, prev_acc = points[i - 1]
            if acc == prev_acc:
                return cost
            # Linear interpolation in cost between the bracketing points.
            frac = (target_accuracy - prev_acc) / (acc - prev_acc)
            return prev_cost + frac * (cost - prev_cost)
    return None


def accuracy_at_budget(
    curve: Iterable[tuple[float, float]], budget: float
) -> float | None:
    """Accuracy of a configuration interpolated at a given cost budget."""
    points = sorted(curve)
    if not points or budget < points[0][0]:
        return None
    if budget >= points[-1][0]:
        return points[-1][1]
    for (c0, a0), (c1, a1) in zip(points, points[1:]):
        if c0 <= budget <= c1:
            if c1 == c0:
                return a1
            frac = (budget - c0) / (c1 - c0)
            return a0 + frac * (a1 - a0)
    return None
