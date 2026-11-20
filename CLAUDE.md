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

Summary table of Open-Jev-2B, SatellaJev-100M, SanSi and Jeff results lives in
[`notes/prior_work.md`](notes/prior_work.md). No published HRM-Text decision model or
HRM-Text LoRA fine-tune exists (as of 2026-11-19).

## Data

- **Version: `release-v2-redistributable`** (chosen so published numbers are comparable).
  Do **not** mix in v1.1 — its test/OOD composition differs.
- Split sizes (VERIFIED against the files, enforced by the loader): train 79,116 · calibration
  4,672 · validation 3,723 · test 10,356 · OOD 15,701. Open-Jev-2B evaluated on 10,532 / 15,920 —
  the gap is the omitted `wikispeedia-v1` rows; note it in every comparison table.
- Record kinds: `choice`, `noul` (`[P(no), P(yes)]`), `score`. Targets are distributions,
  some soft. `metadata.target_basis` gives the target source.
- **Training:** English rows only (HRM-Text is English-only). "English" = no CJK text, which is
  exactly metadata `language != "en+zh"`; non-English rows are ~1% of every split.
- **Evaluation:** report **both** the English subset **and** the full split. The full split
  is what the published cards measured; HRM-Text will be weak on non-English rows — report
  it anyway.
- Split usage: `train` → training · `validation` → early stopping/hparams · `calibration` →
  temperature fitting only · `test`, `ood` → final eval, once per final checkpoint.
- **OOD is the headline metric.** Never let a `group_id` straddle train and eval.
- Licence: check the `release-v2-redistributable` card before redistributing anything.

## Phase 0 — verify before writing training code

Record answers in `notes/phase0.md`. **Status:** items 1–8 done (VERIFIED, incl. T4 timing/memory and fp16 stability over 1,000 real steps); 9 pending
(Kaggle; runbook in `notes/kaggle.md`).

1. Load 50 rows per split and per `kind`; print full records; confirm schema and split sizes.
2. Count rows by `kind`, `source`, language, `target_basis`, per split.
3. Option-count distribution and prompt token lengths with the HRM-Text tokenizer, vs the
   4,096 context limit. Count rows that would need truncation (leaderboards forbid it).
4. Confirm one-hot vs soft targets per `target_basis`.
5. Load HRM-Text-1B; one forward pass with the PrefixLM format
   (`<|im_start|><|object_ref_start|>…<|im_end|>`, `token_type_ids=1` over the prompt).
6. **LoRA feasibility — DONE, VERIFIED** (`tests/test_lora_feasibility.py`). peft attaches to both the
   L and H stacks and every adapter gets gradient. With the default `L_bp_cycles=[0, 3]`, all three
   L passes of H-cycle 0 run under `torch.no_grad()` (as in pretraining); `[3, 3]` backpropagates
   through everything. Default kept; `[3, 3]` is a config switch. Details: `notes/phase0.md`.
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

## Where things run & environment (decided)

- **All training and full evaluation on Kaggle** (2×T4, fp16; **no bf16 on T4/P100**). Local
  machine is not used for GPU-heavy work (overheating concern).
- Kaggle quota: ~30 GPU-h/week (2×T4 or 1×P100) + ~20 TPU-h/week, 12 h GPU sessions, counted
  as wall-clock time. HRM-Text on TPU is untested — don't plan on it.
- Run training as a headless commit ("Save & Run All"); checkpoint every N steps; save
  checkpoints and predictions as Kaggle output; every run resumable across sessions.
- Full eval costs hours per checkpoint (HRM-Text ≈ 8 module passes per token). Run it in the
  same Kaggle session right after training; save per-row probabilities.
- During development, evaluate on a **fixed stratified sample** (e.g. 2k test + 2k OOD,
  ids saved once to `data/dev_eval_ids.json`; `src.eval --limit N` is the quick option).
- Measured on one T4 (50-step probe, fp16, batch 4×4): ~465 tok/s, 3.4 GB peak. Extrapolated (UNVERIFIED): ~3 h for a 16k-row run,
  ~15 h for a full English epoch; eval measured at ~0.32 s/row (~2.7 h for calibration+test+OOD on one T4, ~1.4 h sharded over 2, per seed). Two T4s run two seeds
  in parallel, or one run data-parallel via `torchrun --nproc_per_node=2` (~1.9x, NCCL on Kaggle UNVERIFIED). Details: `notes/phase0.md`.
- **Local:** code, CPU-only analysis of saved predictions (`runs/<id>/preds/<ckpt>/<split>.parquet`), plots.
  CachyOS, HP ZBook, RTX A2000 8 GB, 16 GB RAM. Python env: `uv`. Shell: fish.

## Commands

```bash
uv sync                                                   # install (CPU torch locally; on Kaggle use its torch + pip install transformers>=5.9 peft)
uv run pytest -q                                          # all tests (CPU, ~20 s; first run downloads the tokenizer + data)
uv run pytest tests/test_train.py::test_lr_schedule -q    # single test
uv run python -m src.data.inspect --samples 1             # Phase 0 data checks (markdown to stdout)
python -m src.probe_forward --device cpu --dtype fp32 --rows 8  # Phase 0 item 5 (done; ~2.5 min on CPU; use cuda + fp16 to also check fp16)
uv run python -m src.train --config configs/seed1.yaml [--set max_steps=50 --set run_id=x]   # resumable: rerun to continue
torchrun --nproc_per_node=2 -m src.train --config configs/seed1_long.yaml   # same recipe data-parallel on 2 GPUs (accumulation % 2 == 0)
uv run python -m src.eval --run runs/seed1 --ckpt best --split calibration,test,ood [--shard 0/2 --device cuda:1] [--limit 2000]
uv run python -m src.eval --run runs/seed1 --ckpt best --hub-repo <user>/<repo> --restore-only   # fetch adapter from the Hub in a fresh session, then run the shards
uv run python -m src.report --run runs/seed1 --ckpt best  # CPU: temperature fit + tables -> preds/<ckpt>/results.{json,md}
```

Not built yet: baseline re-evaluation (`--baseline open-jev-2b`), multi-seed aggregation, packaging.

## Architecture

Data flow: `src/data/records.py` (pinned parquet -> `Record`) -> `src/data/prompt.py` (PrefixLM prompt
`<|im_start|><|object_ref_start|>Context… Options: A. … Answer:<|im_end|>`; `collate` right-pads, sets
`token_type_ids=1` over the prompt, optionally shuffles `choice` options with the target permuted
alongside) -> `src/models/hrm_decision.py` (`HrmDecisionModel`: HRM backbone + peft LoRA + 16-way letter
readout at the last prompt token, masked to each row's option count) -> `decision_loss` (soft CE + 0.1 Brier).

- `src/train.py` is resumable by construction: batch order is a pure function of (seed, epoch), option
  shuffling is seeded per micro-batch, checkpoints (`runs/<id>/ckpt`) hold adapter + readout + optimizer +
  scaler + step. `best/` (lowest val NLL) and `final/` hold adapter + `readout.pt` + `model.json` only;
  base weights are re-downloaded at the pinned revision. Hyperparameters live only in `configs/*.yaml`
  (`src/config.py:TrainConfig`); each run writes its resolved config + revisions to `run.json`.
  Optional W&B mirror via `wandb_project` (`src/tracking.py`); `metrics.jsonl` stays the source of truth.
  Optional upload of adapter + resumable `ckpt/` to a private HF model repo via `hub_repo` (`src/hub.py`; best-effort, never kills a run);
  `train` restores `ckpt/` from it when the local run dir is gone, and refuses to resume if training settings (`RESUME_KEYS`) changed.
- Eval is split in two so GPU time is never spent on analysis: `src.eval` (GPU) saves per-row logits in
  canonical option order on the *full* split; `src.report` (CPU) fits one temperature on the full
  calibration split and computes metrics per split x {english, full} x kind. `src/metrics.py` follows the
  Open-Jev-2B card definitions (hard accuracy excludes soft-target rows).
- HRM-Text (`transformers>=5.9`, native `hrm_text`): two stacks `L_module`/`H_module`, 16 layers each, applied
  3×L then H, twice (8 stack passes per token) — compute per token is ~8× a plain 1B model. Pin both
  `MODEL_REVISION` and `DATASET_REVISION`; bump deliberately and note it in `notes/decisions.md`.
- Choices and rationale (readout, shuffling, calibration, GPU use): `notes/decisions.md`.

## Repo layout

```
configs/      one YAML per run (probe_timing, probe_fp16, seed1, seed1_long)
src/data/     records.py (loader), prompt.py (render/collate), inspect.py (Phase 0)
src/models/   hrm_decision.py
src/          config, train, predict, eval, report, metrics, probe_forward
tests/        CPU tests on tiny random-weight HRM models + real tokenizer/data
notes/        phase0.md, decisions.md, prior_work.md   (results.md to come)
runs/ data/   outputs and downloaded data (gitignored)
```

## Gotchas

- LoRA feasibility is verified on a tiny model only; confirm gradient flow/finite loss on the real model
  in the first Kaggle probe. Under the default `L_bp_cycles=[0, 3]` the H-cycle-0 L passes get no gradient (by design).
- Third-party code in the cloned Open-Jev repo is read, not executed, until reviewed.
- Open-Jev-2B's "hard accuracy" excludes soft-target rows — match the definition or say so.
- Card numbers and our re-evaluated numbers may differ; report both, trust ours.
- Never compare against v1.1 or other-version scores.
- Never commit HF tokens, Kaggle credentials, or `.env`.
