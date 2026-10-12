"""Exemplar retrieval for the input-token scaffold.

Two retrievers plus an oracle:

  BM25Retriever    lexical, zero GPU, a strong and honest baseline
  DenseRetriever   E5-family sentence embeddings
  OracleRetriever  retrieves by gold-answer similarity, giving the ceiling

The oracle matters. It separates "retrieval helps" from "this retriever
helps", and without it a weak result is uninterpretable.

HARD RULE: build the index from TRAINING splits only. If test questions or
their solutions enter the index, your 28-point gain is leakage and a reviewer
will find it.
"""

from __future__ import annotations

import re

from .data import Example


def _format_exemplar(example: Example) -> str:
    rationale = example.meta.get("rationale")
    if rationale:
        body = re.sub(r"####\s*", "The answer is ", rationale).strip()
    else:
        body = f"The answer is {example.answer}."
    return f"Problem: {example.question}\nSolution: {body}\n"


class BM25Retriever:
    """Lexical retrieval over the training split. Needs `rank-bm25`."""

    def __init__(self, pool: list[Example]) -> None:
        from rank_bm25 import BM25Okapi

        self.pool = pool
        self.texts = [_format_exemplar(e) for e in pool]
        self.index = BM25Okapi([self._tok(e.question) for e in pool])

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        if k <= 0:
            return []
        scores = self.index.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # Most relevant last: models attend more strongly to recent context.
        return [self.texts[i] for i in reversed(order)]


class DenseRetriever:
    """Dense retrieval with a sentence-transformers encoder."""

    def __init__(
        self,
        pool: list[Example],
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.pool = pool
        self.texts = [_format_exemplar(e) for e in pool]
        self.encoder = SentenceTransformer(model_name, device=device)
        # E5 models expect the "passage:" / "query:" prefixes.
        self.embeddings = self.encoder.encode(
            [f"passage: {e.question}" for e in pool],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        if k <= 0:
            return []
        q = self.encoder.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )
        scores = (self.embeddings @ q[0])
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.texts[i] for i in reversed(order)]


class OracleRetriever:
    """Retrieves exemplars whose gold answer is closest to the gold answer.

    This is a ceiling, not a method. Report it as a dashed line on the
    accuracy-versus-cost plot so readers can see how much headroom a better
    retriever would buy at identical prompt-token cost.
    """

    def __init__(self, pool: list[Example], gold_lookup: dict[str, str]) -> None:
        self.pool = pool
        self.texts = [_format_exemplar(e) for e in pool]
        self.gold_lookup = gold_lookup
        self._current_gold: str | None = None

    def set_query_gold(self, uid: str) -> None:
        self._current_gold = self.gold_lookup.get(uid)

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        if k <= 0:
            return []
        target = self._current_gold or ""

        def distance(example: Example) -> float:
            try:
                return abs(float(example.answer) - float(target))
            except (TypeError, ValueError):
                return 0.0 if example.answer == target else 1e9

        order = sorted(range(len(self.pool)), key=lambda i: distance(self.pool[i]))[:k]
        return [self.texts[i] for i in reversed(order)]


class NullRetriever:
    """Returns nothing. Lets you price a scaffold's overhead at k = 0."""

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        return []


def build_retriever(kind: str, pool: list[Example], **kwargs):
    if kind == "bm25":
        return BM25Retriever(pool)
    if kind == "dense":
        return DenseRetriever(pool, **kwargs)
    if kind == "oracle":
        lookup = {e.uid: e.answer for e in kwargs.pop("gold_pool", pool)}
        return OracleRetriever(pool, gold_lookup=lookup)
    if kind in ("null", "none"):
        return NullRetriever()
    raise ValueError(f"unknown retriever {kind!r}")
