# Experiment log

One row per sweep cell. The columns are deliberately the columns of Table 1
in the paper draft, so filling this in fills in the paper. Copy the block
below for each run and never delete an old one, including the runs that
failed: "we tried X and it did nothing" is a result you will want in the
ablations section, and you will not remember it in week 7.

---

## Run template

**Run ID:** `RUN-001`
**Date:** YYYY-MM-DD
**Config:** `configs/qwen_ladder.yaml`
**Commit:** `git rev-parse --short HEAD`
**Hardware:** 1x A100 80GB, bf16, vLLM 0.7.x, prefix caching ON
**Question this run answers:** one sentence, written BEFORE you launch it.

### Results

| Model | Scaffold | Budget | tok in | tok out | calls | Acc (%) | Cost raw | Cost price | Cost flop | Acc @ iso-cost | C*/C_32B | GPU-s |
|-------|----------|--------|--------|---------|-------|---------|----------|------------|-----------|----------------|----------|-------|
| 32B | greedy CoT | 1 | | | 1 | | | | | ref | n/a | |
| 32B | self-consistency | 8 | | | 8 | | | | | | n/a | |
| 14B | greedy CoT | 1 | | | 1 | | | | | | | |
| 7B | greedy CoT | 1 | | | 1 | | | | | | | |
| 7B | self-consistency | 32 | | | 32 | | | | | | | |
| 7B | best-of-N + PRM | 32 | | | 64 | | | | | | | |
| 7B | self-refine | 3 | | | 7 | | | | | | | |
| 7B | retrieved scaffold | 4 | | | 1 | | | | | | | |
| 7B | retrieved scaffold | 8 | | | 1 | | | | | | | |
| 7B | retrieved + SC | 4, k=8 | | | 8 | | | | | | | |

### Margin summary

| Cost accounting | Best output-token scaffold | Best input-token scaffold |
|-----------------|---------------------------|---------------------------|
| parameter-matched | | |
| raw-token matched | | |
| price-weighted matched | | |

### What happened

- Expected:
- Observed:
- Surprise (the most valuable line in this file):
- Does this change a hypothesis (H1 replication / H2 shrinkage / H3 asymmetry / H4 task dependence)?

### Sanity checks before you believe any of it

- [ ] Token counts came from the engine's usage records, not from re-tokenizing the transcript.
- [ ] Retriever index was built from TRAINING splits only.
- [ ] The large model got the same scaffold at the same budget somewhere in this run.
- [ ] Answer extraction is unchanged from the previous run (if not, say so loudly).
- [ ] Seeds and prompts are identical across the configurations being compared.
- [ ] Prefix caching state is recorded, because it moves retrieval costs a lot.

### Next action

One concrete next run, not a wish list.

---

## Index of runs

| Run | Date | Question | Headline | Made it into the paper? |
|-----|------|----------|----------|-------------------------|
| RUN-001 | | | | |
