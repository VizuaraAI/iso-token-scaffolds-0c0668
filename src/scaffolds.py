"""The scaffold library.

Every scaffold has the same signature and the same contract:

    scaffold(example, backend, cost, budget, retriever=None) -> str

It returns a predicted answer string, and it bills every language model call
it makes to `cost`. `budget` is the single knob that the sweep varies.

Two families matter for the paper's main claim:

  OUTPUT-TOKEN scaffolds buy accuracy with generated tokens
  (self_consistency, best_of_n, self_refine). Generated tokens are
  sequential and memory-bandwidth-bound, so they are the expensive kind.

  INPUT-TOKEN scaffolds buy accuracy with prompt tokens
  (retrieved_scaffold). Prompt tokens prefill in parallel and can be cached
  across queries, so they are the cheap kind.

Hypothesis H3 is that the second family survives iso-cost matching and the
first does not. Keep the two groups clearly separated in your results.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from .cost import QueryCost
from .data import Example
from .models import Backend

ANSWER_RE = re.compile(r"answer is\s*:?\s*\$?(-?[\d,]*\.?\d+|[A-Z])", re.IGNORECASE)

COT_INSTRUCTION = (
    "Solve the problem. Think step by step, then finish with a single line "
    'of exactly the form "The answer is X."'
)


def extract_answer(text: str) -> str:
    """Pull the final answer out of a chain of thought."""
    match = ANSWER_RE.search(text)
    if match:
        value = match.group(1).replace(",", "").strip()
        if value.endswith(".0"):
            value = value[:-2]
        return value
    # Fall back to the last number in the transcript.
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def build_prompt(example: Example, exemplars: list[str] | None = None) -> str:
    parts = [COT_INSTRUCTION, ""]
    if exemplars:
        parts.append("Here are solved problems of the same kind.")
        parts.append("")
        parts.extend(exemplars)
        parts.append("")
    if example.context:
        parts.append("Context:")
        parts.extend(example.context)
        parts.append("")
    parts.append(f"Problem: {example.question}")
    parts.append("Solution:")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Output-token scaffolds
# --------------------------------------------------------------------------


def greedy_cot(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 1,
    retriever=None,
) -> str:
    """Single greedy chain of thought. The reference operating point."""
    prompt = build_prompt(example)
    gen = backend.generate_one(prompt, cost=cost, temperature=0.0)
    return extract_answer(gen.text)


def self_consistency(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 8,
    retriever=None,
) -> str:
    """Sample `budget` chains, return the majority answer.

    `budget` is k. Cost grows linearly in k on the output side, which is why
    this is the scaffold most likely to lose under iso-cost matching.
    """
    prompt = build_prompt(example)
    gens = backend.generate(
        prompt, cost=cost, role="generate", n=budget, temperature=0.7
    )
    answers = [extract_answer(g.text) for g in gens]
    answers = [a for a in answers if a]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def best_of_n(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 8,
    retriever=None,
    verifier: Backend | None = None,
) -> str:
    """Sample `budget` candidates, score each with a verifier, take the best.

    Note that the verifier's own tokens are billed to the same QueryCost.
    Best-of-N papers routinely omit this, which is one of the accounting
    errors the project is designed to surface.
    """
    prompt = build_prompt(example)
    gens = backend.generate(
        prompt, cost=cost, role="generate", n=budget, temperature=0.8
    )
    judge = verifier or backend

    best_answer, best_score = "", float("-inf")
    for gen in gens:
        score_prompt = (
            "Rate the correctness of this solution from 0 to 9. "
            "Reply with a single digit only.\n\n"
            f"Problem: {example.question}\n\nSolution: {gen.text}\n\nRating:"
        )
        judgement = judge.generate_one(
            score_prompt, cost=cost, role="verify", temperature=0.0
        )
        digits = re.findall(r"\d", judgement.text)
        score = int(digits[0]) if digits else 0
        if score > best_score:
            best_score, best_answer = score, extract_answer(gen.text)
    return best_answer


def self_refine(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 2,
    retriever=None,
) -> str:
    """Generate, then critique and revise `budget` times.

    Cost grows super-linearly in rounds because every round re-reads the
    previous answer, and it also costs serial latency, so report latency
    alongside tokens for this one.
    """
    prompt = build_prompt(example)
    current = backend.generate_one(prompt, cost=cost, temperature=0.3).text

    for _ in range(budget):
        critique_prompt = (
            "Critique the solution below. Name any arithmetic or reasoning "
            "error concretely. If it is correct, reply exactly CORRECT.\n\n"
            f"Problem: {example.question}\n\nSolution: {current}\n\nCritique:"
        )
        critique = backend.generate_one(
            critique_prompt, cost=cost, role="critique", temperature=0.0
        ).text
        if "CORRECT" in critique.upper():
            break
        revise_prompt = (
            f"{COT_INSTRUCTION}\n\n"
            f"Problem: {example.question}\n\n"
            f"Previous attempt: {current}\n\n"
            f"Critique: {critique}\n\n"
            "Revised solution:"
        )
        current = backend.generate_one(
            revise_prompt, cost=cost, role="generate", temperature=0.3
        ).text

    return extract_answer(current)


# --------------------------------------------------------------------------
# Input-token scaffolds
# --------------------------------------------------------------------------


def retrieved_scaffold(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 4,
    retriever=None,
) -> str:
    """Prepend `budget` retrieved worked exemplars, then one greedy pass.

    This is the scaffold the paper expects to survive iso-cost matching: it
    makes exactly one call and spends its budget on prompt tokens. Make sure
    your retriever index is built from training splits only.
    """
    if retriever is None:
        raise ValueError("retrieved_scaffold needs a retriever")
    exemplars = retriever.retrieve(example.question, k=budget)
    prompt = build_prompt(example, exemplars=exemplars)
    gen = backend.generate_one(prompt, cost=cost, temperature=0.0)
    return extract_answer(gen.text)


def retrieved_plus_sc(
    example: Example,
    backend: Backend,
    cost: QueryCost,
    budget: int = 4,
    retriever=None,
    k: int = 8,
) -> str:
    """The combination row: retrieved exemplars plus majority voting.

    Interesting because it mixes a cheap input-token budget with an expensive
    output-token budget, so its position on the cost axis depends strongly on
    which accounting you use.
    """
    if retriever is None:
        raise ValueError("retrieved_plus_sc needs a retriever")
    exemplars = retriever.retrieve(example.question, k=budget)
    prompt = build_prompt(example, exemplars=exemplars)
    gens = backend.generate(
        prompt, cost=cost, role="generate", n=k, temperature=0.7
    )
    answers = [extract_answer(g.text) for g in gens]
    answers = [a for a in answers if a]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


SCAFFOLDS: dict[str, Callable] = {
    "greedy_cot": greedy_cot,
    "self_consistency": self_consistency,
    "best_of_n": best_of_n,
    "self_refine": self_refine,
    "retrieved_scaffold": retrieved_scaffold,
    "retrieved_plus_sc": retrieved_plus_sc,
}

# Which cost side each scaffold spends on. Used by the analysis to group rows
# the way Table 1 of the paper groups them.
SCAFFOLD_FAMILY = {
    "greedy_cot": "reference",
    "self_consistency": "output_token",
    "best_of_n": "output_token",
    "self_refine": "output_token",
    "retrieved_scaffold": "input_token",
    "retrieved_plus_sc": "mixed",
}


def get_scaffold(name: str) -> Callable:
    if name not in SCAFFOLDS:
        raise ValueError(
            f"unknown scaffold {name!r}, available: {sorted(SCAFFOLDS)}"
        )
    return SCAFFOLDS[name]
