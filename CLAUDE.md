# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# HRM-Text × Open-Jev

## What this project is

LoRA fine-tune **sapientinc/HRM-Text-1B** into a typed decision model on **Open-Jev
`release-v2-redistributable`**, then compare it against published decision models.

> **Research question:** Can a LoRA-fine-tuned 1B *base* HRM-Text model match published
> decision models (e.g. Open-Jev-2B, an instruction-tuned Qwen3.5-2B) on Open-Jev
> test/OOD and on public decision-model leaderboards?

**Scope of claims.** We do **not** train matched controls, so we cannot claim "HRM's
architecture beats a standard one" — recipes (LoRA rank, steps, head, base vs instruct)
differ across models. Claims are limited to "HRM-Text-1B + our recipe scores X vs model Y
on identical rows / the same leaderboard harness."

## Working style

- Be concise. No walls of text in reports or commit messages.
- Label every claim **VERIFIED** (seen in code, data, or a run) or **UNVERIFIED** (from a
  card, paper, or inference). Never upgrade a card claim to verified without checking it.
- Report bad results as plainly as good ones. "HRM-Text doesn't match Open-Jev-2B" is a
  valid finding.

## Plan

1. **Phase 0** — verify data, model, LoRA feasibility, fp16 stability, timing (below).
2. **Train** HRM-Text LoRA on Kaggle. Seed 1 first; seeds 2–3 only after seed 1 works.
3. **Evaluate** on our own pipeline: HRM-Text + re-evaluated published baselines on
   identical rows and metrics.
4. **Submit** the best checkpoint to public decision leaderboards (see Leaderboards).

## Prior work (UNVERIFIED — from cards/papers)

| Model | Setup | Reported result |
|---|---|---|
| Open-Jev-2B | Qwen3.5-2B, LoRA r8/α16 + scalar decision head, 1 pass over 80,816 rows of frozen `release-v2`, temperature 1.519 | Test hard acc 94.71%, OOD 86.02%; ECE 0.0116 / 0.1056. Hard acc **excludes soft-target rows**. |
| SatellaJev-100M | Trained from scratch, 101.5M params, custom 7k tokenizer, `release-v2-redistributable` | Test acc 82.03%, OOD 72.09%; ECE 0.0287 / 0.0997 |
| SanSi (Ouro-1.4B looped, frozen + LoRA r64) | Not Open-Jev data | Loop 1/3/8 acc 58.4/70.4/72.0 vs non-looped SmolLM2-1.7B 58.4. Looping only helps with looped pre-training. ECE best at loop 3; averaging per-loop probs cut ECE 0.093→0.044. |
| Jeff (Harvard) | LoRA r32 + letter-readout head | Base (non-instruct) backbone 70.3 vs instruct 84.4 with same recipe; head-only fit gave no gain; ranks 16–128 similar. |

No published HRM-Text decision model or HRM-Text LoRA fine-tune exists (as of 2026-11-19).

## Data

- **Version: `release-v2-redistributable`** (chosen so published numbers are comparable).
  Do **not** mix in v1.1 — its test/OOD composition differs.
- Expected split sizes (from SatellaJev card, VERIFY): train 79,116 · calibration 4,672 ·
  validation 3,723 · test 10,356 · OOD 15,701. Open-Jev-2B evaluated on 10,532 / 15,920 —
  the gap is the omitted `wikispeedia-v1` rows; note it in every comparison table.
- Record kinds: `choice`, `noul` (`[P(no), P(yes)]`), `score`. Targets are distributions,
  some soft. `metadata.target_basis` gives the target source.
- **Training:** English rows only (HRM-Text is English-only).
- **Evaluation:** report **both** the English subset **and** the full split. The full split
  is what the published cards measured; HRM-Text will be weak on non-English rows — report
  it anyway.
- Split usage: `train` → training · `validation` → early stopping/hparams · `calibration` →
  temperature fitting only · `test`, `ood` → final eval, once per final checkpoint.
- **OOD is the headline metric.** Never let a `group_id` straddle train and eval.
- Licence: check the `release-v2-redistributable` card before redistributing anything.

## Phase 0 — verify before writing training code

Record answers in `notes/phase0.md`.

1. Load 50 rows per split and per `kind`; print full records; confirm schema and split sizes.
2. Count rows by `kind`, `source`, language, `target_basis`, per split.
3. Option-count distribution and prompt token lengths with the HRM-Text tokenizer, vs the
   4,096 context limit. Count rows that would need truncation (leaderboards forbid it).
4. Confirm one-hot vs soft targets per `target_basis`.
5. Load HRM-Text-1B; one forward pass with the PrefixLM format
   (`<|im_start|><|object_ref_start|>…<|im_end|>`, `token_type_ids=1` over the prompt).
6. **LoRA feasibility (blocking).** Check peft attaches to the shared looped weights and
   gradients reach every loop iteration. The base model runs explorer L-cycles (0, 1) under
   `torch.no_grad()` in training; `L_bp_cycles: [0, 3]` in config is likely related — read
   `modeling_hrm_text.py`. Decide whether to patch it out (a community chat fine-tune did) and
   log the choice in `notes/decisions.md`. The official repo's SFT is full-parameter only, so
   the LoRA loop is our own code.
7. Config (VERIFIED from card): hidden 1536, 16 layers, vocab 65,536, context 4,096,
   `H_cycles` 2, `L_cycles` 3. On a Kaggle T4: peak memory + wall time for 50 LoRA steps at
   real prompt lengths; extrapolate run time.
8. **fp16 stability** (T4 has no bf16; HRM-Text was trained in bf16): 200 LoRA steps, check
   for NaN/inf, activation overflow across loop passes, non-decreasing loss. If unstable, try
   fp32 for recurrent state/norms before abandoning T4s.
9. Load the published Open-Jev-2B adapter (pinned Qwen3.5-2B revision + Open-Jev loader) and
   reproduce its card numbers on a 1k-row sample before trusting our eval harness.

## Method

- **Score options in one forward pass**: all options enumerated in the prompt, read the
  option-letter logits at the last prompt position, softmax over valid options. Never one
  pass per option (multiplies cost by option count). No free-text generation.
- Loss: soft cross-entropy against `target` (+ optionally 0.1 × Brier, as SatellaJev did).
- **LoRA** rank 16 on attention + MLP projections, trained jointly with the readout head.
  ~16k–80k rows is likely enough (SanSi 16k, Open-Jev-2B 80k); one epoch max to start.
- Calibration: temperature scaling on the `calibration` split. Also try averaging readout
  probabilities across the 2 H-cycles (SanSi's per-loop averaging trick) if the readout can
  be taken per cycle.
- HRM-Text is a **base** checkpoint; Jeff found base backbones far behind instruct ones with
  the same recipe. Expect a handicap; note it in every results table.

### Stretch: memory-optimized full fine-tune (only after LoRA results exist)

An ablation of HRM-Text LoRA vs HRM-Text full FT, same data and steps. Run only if the
Phase 0 memory probe shows it fits. In order of impact:
1. 8-bit AdamW (bitsandbytes) — optimizer states ~8 GB → ~2 GB.
2. Optimizer step fused into backward (`register_post_accumulate_grad_hook`) — no full
   gradient buffer.
3. Gradient checkpointing per HRM cycle; batch 1 + gradient accumulation.
4. If still too tight: freeze embeddings or train a subset (e.g. L-module + embeddings).

Checkpointing and accumulation do **not** reduce weight/gradient/optimizer memory, and no
trick reduces compute. fp16-only full FT of deep recurrence is risky — short stability run
first.

## Baselines (re-evaluated, not retrained)

- **Open-Jev-2B**: download the adapter + head, run through **our** eval code on the exact
  same rows and metrics as HRM-Text. Report our numbers next to the card numbers.
- **SatellaJev-100M**: same, if weights are downloadable; otherwise card numbers only,
  clearly labelled.
- No Qwen3 / Qwen3.5 / Ouro controls are trained (dropped to save quota). Revisit a single
  matched control only if HRM-Text looks competitive.

## Metrics

- Hard accuracy (argmax vs argmax target, **excluding soft-target rows**, matching the
  Open-Jev-2B card) and accuracy on all rows; NLL; Brier; ECE before/after temperature.
- Per split (`test`, `ood`) × subset (English, full) × `kind`.
- HRM-Text: seeds 1–3, mean ± std for final claims. Zero-shot HRM-Text as a floor.

## Leaderboards (after our own eval)

Submit only a checkpoint that already passed our test/OOD eval.

| Leaderboard | Submission | Notes |
|---|---|---|
| Jev Decision Index (HF Space, multimodalart) | Pull request with dataset link + hardware specs; re-scored in review | 132,422 frozen requests, 37 benchmarks. No truncation, no dropping options, one fixed prompt rendering. |
| JevBench — Benchmark Heaven | Process not yet found | 300 open + 1,200 sealed decisions; open-weights board run on their hardware. >10 s per decision = wrong + zero calibration. |
| LeanMCP JevBench | Open submissions | Varies by dataset |

Not Terminal-Bench (agentic shell tasks — wrong task type).

**Packaging requirements:** loads with a few lines; bundles HRM-Text custom code, PrefixLM
formatting, LoRA adapter, readout head, temperature; documents context limit (4,096) and
max option count; measured per-decision latency on a stated GPU.

## Where things run (decided)

- **All training and full evaluation on Kaggle** (2×T4, fp16). Local machine is not used
  for GPU-heavy work (overheating concern).
- **Local:** code, CPU-only analysis of saved predictions (`runs/<id>/preds.parquet`), plots.
- Full eval costs hours per checkpoint (HRM-Text ≈ 8 module passes per token). Run it in the
  same Kaggle session right after training; save per-row probabilities.
- During development, evaluate on a **fixed stratified sample** (e.g. 2k test + 2k OOD,
  ids saved once to `data/dev_eval_ids.json`).
- Rough estimate (UNVERIFIED, replace with Phase 0 timing): ~12–16 h per full English epoch
  on one T4, ~7–8 h on 2×T4; 16k–80k-row runs ~3–14 h.

## Hardware & environment

- Kaggle: ~30 GPU-h/week (2×T4 or 1×P100) + ~20 TPU-h/week, 12 h GPU sessions, quota counts
  wall-clock time. **No bf16 on T4/P100.** HRM-Text on TPU is untested — don't plan on it.
- Run training as a headless commit ("Save & Run All"); checkpoint every N steps; save
  checkpoints and predictions as Kaggle output; every run resumable across sessions.
- Local: CachyOS, HP ZBook, RTX A2000 8 GB, 16 GB RAM. Python env: `uv`. Shell: fish.

**Planned commands. None of these exist yet** (no `src/`, `configs/` or `pyproject.toml`).
Create them in Phase 0, then remove this note.

```bash
uv sync                                   # install
uv run python -m src.data.inspect         # Phase 0 data checks
uv run python -m src.train --config configs/<name>.yaml
uv run python -m src.eval --run runs/<run_id> --split test,ood
uv run python -m src.eval --baseline open-jev-2b --split test,ood
uv run pytest -q
```

## Repo layout (proposed, none of it exists yet)

```
configs/      one YAML per (seed, lr) run
src/data/     loading, filtering, prompt templates, option-letter collate
src/models/   HRM-Text loader + PrefixLM handling, baseline loaders
src/train.py  LoRA training loop
src/eval.py   metrics, calibration, per-subset tables, baseline re-eval
notebooks/    Kaggle notebooks (thin wrappers around src/)
notes/        phase0.md, results.md, decisions.md
runs/         outputs (gitignored)
```

## Gotchas

- LoRA on shared looped weights is untested; if adapters only see one loop iteration,
  results are invalid.
- Open-Jev-2B's "hard accuracy" excludes soft-target rows — match the definition or say so.
- Card numbers and our re-evaluated numbers may differ; report both, trust ours.
- Never compare against v1.1 or other-version scores.
- Never commit HF tokens, Kaggle credentials, or `.env`.
