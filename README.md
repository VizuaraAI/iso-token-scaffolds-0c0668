# iso-token-scaffolds

**Scaffolding at matched total token cost.** Starter repository for Ashish
Ranjan, Vizuara SLM Bootcamp.

## The one-sentence version

Everyone reports that a scaffolded 8B model beats a 33B model, and that
retrieved scaffolds add 28 to 29 points, but every one of those comparisons
is matched on **parameters** and unmatched on **token spend**, which is the
budget that actually governs deployment. This repo re-runs the comparison on
the cost axis that matters.

## Why this is a real paper and not a blog post

A scaffold is not free. Self-consistency at k=32 issues 32 generations. A
retrieval scaffold with eight exemplars adds thousands of prompt tokens to
every single query. None of that appears in the headline number. So the
honest question is not "does the scaffolded 8B beat the 33B" but:

1. How much of the margin survives when total token spend is held fixed?
2. What is the crossover cost `C*`, the budget multiple at which a scaffolded
   small model first matches the big one?
3. What happens when you give the big model the **same scaffold at the same
   budget**, which is the control nobody runs?

Our prediction, and the thing that makes this publishable: scaffolds that buy
accuracy with **output tokens** (voting, refinement) largely collapse under
matched cost, while scaffolds that buy it with **input tokens** (retrieval)
largely survive, because prefill is parallel and cacheable while decode is
sequential and memory-bandwidth-bound. If that holds, the practical advice
reverses for half the scaffolds people currently recommend.

## Setup

```bash
git clone <your-fork> iso-token-scaffolds
cd iso-token-scaffolds
./scripts/setup.sh
source .venv/bin/activate
```

That installs the core harness only. It runs on a laptop with no GPU and no
network, because the default config uses a mock backend and an offline
synthetic dataset.

When you have a GPU:

```bash
pip install torch transformers accelerate vllm openai sentence-transformers
```

## The first three things to run

**1. Prove the harness is wired up (about 10 seconds, no GPU).**

```bash
./scripts/smoke_test.sh
```

Watch the last table. You should see a row where `self_consistency` has a
large positive `d param` margin and a **negative** `d iso` margin. That single
sign flip is the entire thesis of the paper, reproduced on a toy problem.

**2. Watch the cost accounting change the answer.**

```bash
python -m src.analysis --results results/sweep.jsonl --cost-model raw
python -m src.analysis --results results/sweep.jsonl --cost-model price
python -m src.analysis --results results/sweep.jsonl --cost-model flop
```

Three accountings of the exact same run. Where they disagree about which
configuration wins, you have found something worth writing down.

**3. Run the real ladder (needs a GPU).**

Open `configs/qwen_ladder.yaml`, cut `limit` to 100, cut the budget lists to
`[1, 4, 16]`, then:

```bash
./scripts/run_sweep.sh configs/qwen_ladder.yaml results/pilot.jsonl
```

Check the curve shapes before you spend real compute. Then scale up.

## Layout

```
src/cost.py          the token accountant and the three cost models  <- read this first
src/models.py        backends (mock, vLLM, OpenAI-compatible) that report usage
src/data.py          dataset loaders, nothing large is committed
src/retrieval.py     BM25, dense and oracle exemplar retrieval
src/scaffolds.py     the scaffold library, each with one budget knob
src/evaluate.py      answer scoring and difficulty stratification
src/run_sweep.py     main experiment entry point
src/analysis.py      iso-cost curves, crossover costs, the LaTeX table
src/calibrate.py     fit the cost proxies against measured hardware cost
configs/             default (mock) and qwen_ladder (real)
scripts/             setup, smoke test, full sweep
experiments/         the log template, whose rows are the paper's Table 1
```

`src/analysis.py` prints a LaTeX block you paste directly into Table 1 of the
paper draft. That is the point of the whole repo: a pipeline from a sweep to
a table in your paper, with nothing retyped by hand.

## Three rules that will save your paper

1. **Never estimate token counts.** Read them from the engine's usage
   records. Re-tokenizing the final transcript silently drops retries,
   discarded votes and the verifier's own calls, which is the exact error
   this paper exists to correct.
2. **Build the retriever index from training splits only.** If test questions
   or their solutions get into the index, your 28-point gain is leakage and a
   reviewer will find it in ten minutes.
3. **Always give the big model the same scaffold at the same budget.** A
   comparison between "8B with every trick" and "32B with none" is not a
   result, it is a confound.

## Where this goes

Primary target is an **ICLR 2027 workshop** (suggested paper deadline
1 February 2027) on efficiency or reasoning. The paper draft in
`../paper/main.tex` already has the structure; your job over the next eight
weeks is to replace every `XX.X` with a measured number.

See `experiments/log_template.md` and keep it current from day one.
