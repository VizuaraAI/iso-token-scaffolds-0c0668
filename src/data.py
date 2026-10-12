"""Dataset loading. Nothing large is committed; everything downloads.

Each loader returns a list of `Example`. Keep the interface this narrow: the
scaffolds should not know which benchmark they are running on.

`gsm1k` is the contamination control from Zhang et al. 2024. If your
scaffolded small model's advantage shrinks more on GSM1k than the baseline's
does, part of your effect is memorisation, and you need to know that before
you write the abstract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Example:
    uid: str
    question: str
    answer: str  # normalised gold answer string
    context: list[str] = field(default_factory=list)  # for multi-hop sets
    meta: dict = field(default_factory=dict)


def _norm_number(text: str) -> str:
    """Pull the final numeric answer out of a string and normalise it."""
    text = text.replace(",", "").replace("$", "").strip()
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return text.strip().lower()
    value = matches[-1]
    # Drop a trailing ".0" so "42" and "42.0" compare equal.
    if value.endswith(".0"):
        value = value[:-2]
    return value


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        gold = row["answer"].split("####")[-1]
        out.append(
            Example(
                uid=f"gsm8k-{split}-{i}",
                question=row["question"],
                answer=_norm_number(gold),
                meta={"rationale": row["answer"], "kind": "math"},
            )
        )
    return out


def load_gsm1k(limit: int | None = None) -> list[Example]:
    """Scale AI's GSM1k contamination control."""
    from datasets import load_dataset

    ds = load_dataset("scaleapi/gsm1k", split="test")
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(
            Example(
                uid=f"gsm1k-{i}",
                question=row["question"],
                answer=_norm_number(str(row["answer"])),
                meta={"kind": "math"},
            )
        )
    return out


def load_math500(limit: int | None = None) -> list[Example]:
    """The 500-problem MATH subset released with the PRM800K work."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(
            Example(
                uid=f"math500-{i}",
                question=row["problem"],
                answer=str(row["answer"]).strip(),
                meta={"level": row.get("level"), "subject": row.get("subject"),
                      "kind": "math"},
            )
        )
    return out


def load_hotpotqa(split: str = "validation", limit: int | None = None) -> list[Example]:
    """Multi-hop question answering, where retrieval should genuinely win."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        paragraphs = [
            " ".join(sents) for sents in row["context"]["sentences"]
        ]
        out.append(
            Example(
                uid=f"hotpotqa-{split}-{i}",
                question=row["question"],
                answer=str(row["answer"]).strip(),
                context=paragraphs,
                meta={"level": row.get("level"), "kind": "qa"},
            )
        )
    return out


def load_mmlu_pro(limit: int | None = None) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        options = row["options"]
        rendered = "\n".join(
            f"({chr(65 + j)}) {opt}" for j, opt in enumerate(options)
        )
        out.append(
            Example(
                uid=f"mmlupro-{i}",
                question=f"{row['question']}\n{rendered}",
                answer=str(row["answer"]).strip(),
                meta={"category": row.get("category"), "kind": "mcq"},
            )
        )
    return out


def load_synthetic(limit: int = 40) -> list[Example]:
    """Offline arithmetic set so `make smoke` needs no network at all.

    Only for wiring tests. Never report a number measured on this.
    """
    out = []
    for i in range(limit):
        a, b = 6 + i, 7 + (i % 5)
        out.append(
            Example(
                uid=f"synthetic-{i}",
                question=f"A box holds {a} rows of {b} apples. How many apples?",
                answer=str(a * b),
                meta={"kind": "math"},
            )
        )
    return out


LOADERS = {
    "gsm8k": load_gsm8k,
    "gsm1k": load_gsm1k,
    "math500": load_math500,
    "hotpotqa": load_hotpotqa,
    "mmlu_pro": load_mmlu_pro,
    "synthetic": load_synthetic,
}


def load_dataset_by_name(name: str, limit: int | None = None, **kwargs) -> list[Example]:
    if name not in LOADERS:
        raise ValueError(
            f"unknown dataset {name!r}, available: {sorted(LOADERS)}"
        )
    return LOADERS[name](limit=limit, **kwargs)
