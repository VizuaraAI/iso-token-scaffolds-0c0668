"""Model backends that report their own token usage.

Three backends:

  mock    deterministic fake model, no GPU, no network. Use it to smoke test
          the harness and the analysis before you burn any compute.
  vllm    in-process vLLM. Fastest option when you own the GPU.
  openai  any OpenAI-compatible HTTP endpoint, including `vllm serve`.

Every backend returns a `Generation` carrying the text AND the exact token
counts reported by the engine. Do not re-tokenize the transcript to estimate
cost: that silently under-counts retries, discarded votes and truncated
candidates, which is precisely the error the paper is about.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass

from .cost import CallRecord, QueryCost


@dataclass
class Generation:
    text: str
    n_in: int
    n_out: int
    latency_s: float


class Backend:
    """Common interface. Subclasses implement `_generate`."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate(self, prompt: str, n: int, temperature: float) -> list[Generation]:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        cost: QueryCost | None = None,
        role: str = "generate",
        n: int = 1,
        temperature: float | None = None,
    ) -> list[Generation]:
        """Generate `n` completions and bill them to `cost`."""
        temp = self.temperature if temperature is None else temperature
        gens = self._generate(prompt, n=n, temperature=temp)
        if cost is not None:
            for g in gens:
                cost.add(
                    CallRecord(
                        model=self.model,
                        role=role,
                        n_in=g.n_in,
                        n_out=g.n_out,
                        latency_s=g.latency_s / max(len(gens), 1),
                    )
                )
        return gens

    def generate_one(self, prompt: str, **kwargs) -> Generation:
        return self.generate(prompt, n=1, **kwargs)[0]


class MockBackend(Backend):
    """A deterministic stand-in so the whole pipeline runs anywhere.

    It solves the synthetic arithmetic set correctly with probability
    `effective`, and otherwise returns a near miss. `effective` rises with the
    model's nominal `skill` and with prompt length, so a bigger model looks
    better and a retrieval-padded prompt looks better, both in a stable,
    seeded way.

    That is enough structure to exercise every claim the harness makes:
    self-consistency improves when the per-sample rate is above one half, the
    retrieved scaffold buys accuracy with input tokens, and the three cost
    accountings rank the results differently. It is a harness test, not a
    model, so never report a number measured with it.
    """

    #: Accuracy gained per prompt token beyond the bare question. Stands in
    #: for "more retrieved context helps".
    CONTEXT_GAIN = 0.0006

    def __init__(self, model: str = "mock-small", skill: float = 0.52, **kwargs):
        super().__init__(model, **kwargs)
        self.skill = skill

    @staticmethod
    def _solve(prompt: str) -> int | None:
        """Work out the gold answer for the synthetic set, if this is one.

        Reads only the text after the LAST "Problem:" marker, so retrieved
        exemplars earlier in the prompt cannot leak their own numbers in.
        """
        tail = prompt.rsplit("Problem:", 1)[-1]
        numbers = [int(x) for x in re.findall(r"\b\d+\b", tail)]
        if len(numbers) >= 2:
            return numbers[0] * numbers[1]
        return None

    def _generate(self, prompt: str, n: int, temperature: float) -> list[Generation]:
        n_in = max(len(prompt) // 4, 1)
        gold = self._solve(prompt)
        # Padding beyond the bare question stands in for retrieved context.
        effective = min(self.skill + self.CONTEXT_GAIN * max(n_in - 60, 0), 0.95)

        out = []
        for i in range(n):
            seed = f"{self.model}|{prompt}|{i}|{temperature}"
            draw = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if gold is None:
                answer = 42 if draw < effective else 17
            elif draw < effective:
                answer = gold
            else:
                # A plausible near miss, varied so wrong votes do not agree.
                answer = gold + (1 + i % 4) * (1 if i % 2 else -1)
            text = (
                "Let me work through this step by step.\n"
                "Step 1: restate the given quantities.\n"
                "Step 2: combine them.\n"
                f"The answer is {answer}."
            )
            out.append(
                Generation(
                    text=text,
                    n_in=n_in,
                    n_out=max(len(text) // 4, 1),
                    latency_s=0.001,
                )
            )
        return out


class VLLMBackend(Backend):
    """In-process vLLM. Requires a GPU and `pip install vllm`."""

    def __init__(self, model: str, gpu_memory_utilization: float = 0.90, **kwargs):
        super().__init__(model, **kwargs)
        from vllm import LLM  # imported lazily so the mock path needs no GPU

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
        )

    def _generate(self, prompt: str, n: int, temperature: float) -> list[Generation]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        start = time.perf_counter()
        outputs = self.llm.generate([prompt], params)
        elapsed = time.perf_counter() - start

        request = outputs[0]
        n_in = len(request.prompt_token_ids)
        return [
            Generation(
                text=c.text,
                n_in=n_in,
                n_out=len(c.token_ids),
                latency_s=elapsed / n,
            )
            for c in request.outputs
        ]


class OpenAIBackend(Backend):
    """Any OpenAI-compatible endpoint, including `vllm serve`.

    Set OPENAI_BASE_URL and OPENAI_API_KEY. Usage counts come from the
    response's `usage` field, which is the engine's own accounting.
    """

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    def _generate(self, prompt: str, n: int, temperature: float) -> list[Generation]:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
            n=n,
        )
        elapsed = time.perf_counter() - start

        usage = resp.usage
        # The server reports prompt tokens once and completion tokens in total,
        # so split the completion side evenly across the n choices.
        n_choices = max(len(resp.choices), 1)
        per_choice_out = (usage.completion_tokens if usage else 0) // n_choices
        return [
            Generation(
                text=choice.message.content or "",
                n_in=(usage.prompt_tokens if usage else 0),
                n_out=per_choice_out,
                latency_s=elapsed / n_choices,
            )
            for choice in resp.choices
        ]


def build_backend(spec: dict) -> Backend:
    """Build a backend from a config dict, for example::

        {"kind": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct", "max_tokens": 1024}
    """
    spec = dict(spec)
    kind = spec.pop("kind", "mock")
    if kind == "mock":
        return MockBackend(**spec)
    if kind == "vllm":
        return VLLMBackend(**spec)
    if kind == "openai":
        return OpenAIBackend(**spec)
    raise ValueError(f"unknown backend kind {kind!r}")
