# CLAUDE.md

Instructions for the coding agent working on this repository with Ashish.

## Project goal

Measure how much of the published "small model plus scaffold beats a large
model" advantage survives when the comparison is matched on **total token
cost** instead of on **parameter count**. Produce a workshop paper by
February 2027.

The claim being tested, in the author's words: a scaffolded 8B model beats a
33B model, and retrieved scaffolds add 28 to 29 points, but every published
comparison matches on parameters, never on total token spend, which is the
budget that actually governs deployment.

Four hypotheses drive every experiment:

- **H1 replication.** Under the conventional parameter-matched protocol, the
  scaffolded small model beats the 32B single-shot baseline.
- **H2 shrinkage.** Under matched raw token cost, that margin shrinks a lot
  and goes negative for sampling-based scaffolds.
- **H3 asymmetry.** Under price-weighted cost, retrieval scaffolds recover
  most of their margin and sampling scaffolds do not, because retrieval
  spends input tokens (parallel prefill, cacheable) and voting spends output
  tokens (sequential decode).
- **H4 task dependence.** The crossover cost is small on knowledge-intensive
  tasks and may exceed any realistic budget on competition mathematics.

If a change you are asked to make does not serve one of these four, say so
before writing the code.

## Layout

```
src/cost.py          TokenAccountant, QueryCost, the three cost models, C* helpers
src/models.py        Backend ABC + MockBackend / VLLMBackend / OpenAIBackend
src/data.py          Example dataclass and dataset loaders
src/retrieval.py     BM25Retriever / DenseRetriever / OracleRetriever / NullRetriever
src/scaffolds.py     scaffold functions, all with the same signature
src/evaluate.py      normalise, is_correct, accuracy, difficulty_buckets
src/run_sweep.py     entry point: config -> results JSONL
src/analysis.py      entry point: results JSONL -> table + figure
src/calibrate.py     entry point: cost proxies vs measured hardware cost
configs/             default.yaml (mock, laptop) and qwen_ladder.yaml (real)
experiments/         log_template.md, rows map to Table 1 of the paper
```

## Conventions

- **Every scaffold has the signature**
  `scaffold(example, backend, cost, budget, retriever=None) -> str`
  and bills **every** call it makes to `cost`. A scaffold that makes an
  unbilled call is a bug, even if the output is correct.
- **One budget knob per scaffold.** The sweep varies exactly one integer. If
  a scaffold needs two knobs, fix one in the config and document it.
- Token counts come from the engine's `usage`, never from re-tokenizing the
  transcript. Do not "optimise" this.
- `small`, `mid` and `large` are the model names the analysis keys on. Keep
  them stable in configs.
- Answer extraction lives only in `scaffolds.extract_answer` and
  `evaluate.normalise`. Changing either moves every number in the paper, so
  flag it loudly in the PR description and re-run everything.
- Python 3.10+, type hints on public functions, `from __future__ import
  annotations` at the top of each module.
- No new dependencies without asking. The core harness must keep running
  with no GPU and no network.

## Working agreement

- Default to the mock backend when developing. `./scripts/smoke_test.sh` must
  stay green and must stay under about 30 seconds.
- Before a GPU run, print the grid with `--dry-run` and estimate the
  generation count. A cell at k=32 over 500 examples is 16,000 generations.
  Tell Ashish the estimate before launching.
- After any experiment, append to `experiments/log_template.md`. Include the
  runs that showed nothing; those become the ablations section.
- When results land, regenerate the LaTeX table with `src.analysis` and paste
  it into `../paper/main.tex`. Never retype numbers by hand.

## Things that will invalidate the paper, in priority order

1. **Retrieval leakage.** The exemplar index must come from training splits
   only. Check this every time the retrieval config changes.
2. **Unbilled calls.** Verifier calls, critique calls, retries and discarded
   candidates all cost tokens. If `mean_calls` does not match what the
   scaffold should be issuing, stop and find out why.
3. **A missing control.** Every claim of the form "the 7B plus scaffold wins"
   needs the 32B running the same scaffold at the same budget in the same
   table.
4. **Contamination.** GSM8K is widely memorised. The GSM1k re-run is not
   optional.
5. **Changing answer extraction mid-project** without re-running everything.

## Next steps (keep this list current)

- [ ] Pilot the Qwen2.5 ladder on GSM8K at `limit: 100` and confirm the
      accuracy-versus-cost curves have the expected shape.
- [ ] Add GPU-seconds measurement to the vLLM path so `src.calibrate` has a
      real target instead of wall-clock latency.
- [ ] Implement the prefix-caching on/off comparison for retrieval scaffolds;
      this is the main threat to H3 being an artifact of the serving stack.
- [ ] Add the process reward model to `best_of_n` and bill the verifier.
- [ ] Wire `difficulty_buckets` into `src.analysis` for the stratified
      non-monotonicity plot.
- [ ] Sweep the input weight `w` from 0.05 to 1.0 and find the sign-change
      point per scaffold.
