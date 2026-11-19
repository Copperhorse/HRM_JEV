# Phase 0 results

Date: 2026-11-19. Dataset `ZefanCai/Open-Jev` @ `c67699e1…`, config `release-v2-redistributable`.
Model `sapientinc/HRM-Text-1B` @ `22097cbc…`. transformers 5.23.0, peft 0.19+.
Reproduce items 1–4: `uv run python -m src.data.inspect --samples 1`.

## 1. Schema and split sizes: VERIFIED
Train 79,116 · calibration 4,672 · validation 3,723 · test 10,356 · OOD 15,701: exact match to the card
(the loader raises on any mismatch). Columns: `id, group_id, split, source, kind, question, options, target,
state_json, metadata_json, record_json, original_line_number`. Model inputs are only `state`, `question`,
`kind`, `options`; `metadata_json` carries teacher labels and is never rendered.

## 2. Composition: VERIFIED
- Kinds (train): noul 48,907 · choice 18,286 · score 11,923.
- 12 sources; `painting-geometry-v1` (23.5k), `snake-v1` (12.5k) and the four `workflow-controls-v1/*` dominate train.
  OOD has no `tic_tac_toe`, `tile_platformer` or `trex_runner`.
- `language` metadata exists only for `customer-control-v1` rows. `language == "en+zh"` is **exactly** the set of rows with
  CJK text, so the English filter is a CJK-regex check. Non-English: train 600 (0.76%), calibration 78, validation 60,
  test 108 (1.04%), OOD 78 (0.50%). So "English subset" vs "full split" differ by about 1% of rows; both are still reported.
- Group leakage: 0 `group_id` shared between train and any other split.

## 3. Lengths: VERIFIED
With our prompt format (letters + instruction line): mean ~330 tokens, p99 ~740, **max 827** across all splits vs the
4,096 limit. **0 rows need truncation.** Options per row ≤ 16 (letters A–P suffice); 49,661/79,116 train rows are binary.

## 4. Targets: VERIFIED
- All `noul` rows are one-hot `[P(no), P(yes)]` with options `['no','yes']`.
- Soft targets (train): `tic_tac_toe` choice 50% (minimax ties), `snake` choice 31%, `customer-control` choice/score 9%
  (uniform latent worlds). Overall 3.6% of train rows, 3.5% test, 2.0% OOD.
- **Answer-position bias exists** (train choice rows): `tile_platformer` 89% at index 4, `tic_tac_toe` 57% at index 0,
  `trex_runner` 71% at index 1. Open-Jev-2B scores each candidate independently so it cannot use position. We
  therefore shuffle `choice` option order in training (never `score`: ordinal; never `noul`). Eval uses canonical order.

## 5. Real-model forward pass with PrefixLM format: VERIFIED (CPU, fp32, 8 test rows; `python -m src.probe_forward`)
Weights `model.safetensors` (2,365,606,568 bytes, revision `22097cbc…`) load as `HrmTextForCausalLM`: 1.183B parameters
including embedding and `lm_head` (the two stacks are ~0.98B). Results:
- Logits finite (letter logits range 10.5 to 20.6).
- Padding invariance: max |diff| 3.8e-6 between a row batched with padding and the same row alone.
- PrefixLM mask is honoured: max |prefix − causal| logit difference 3.9 (so `token_type_ids` matters, as the card says).
- CPU speed: 143 s for 8 rows at padded length 632 (fp32). Not informative for GPU timing (item 7).
- Zero-shot accuracy on these 8 rows (0.50 vs ~0.40 chance) is **not evidence of anything** (n=8); the readout is untrained.
Tokenizer facts VERIFIED: `<|im_start|>`=6, `<|im_end|>`=7, `<|object_ref_start|>`=8, letters A–P are single tokens (ids 63–78),
and `tokenizer(text, add_special_tokens=False)` is correct (no BOS is added by default).
Hub download note: the 2.4 GB file ran at ~0.2–1 MB/s; the xet client stalled, plain HTTP (`HF_HUB_DISABLE_XET=1`) finished.

## 6. LoRA feasibility: VERIFIED (tiny random-weight model, `tests/test_lora_feasibility.py`)
Read from `transformers/models/hrm_text/modeling_hrm_text.py` (5.23.0):
- Two separate stacks `L_module`, `H_module`, 16 layers each (~0.98B params in the stacks, 1.183B with embedding and `lm_head`); the loop applies L three times then H, twice
  (8 stack passes/token, 128 layer applications/token).
- `L_bp_cycles=[0, 3]`: in H-cycle 0 **all three L passes run under `torch.no_grad()`**; in H-cycle 1 all three have
  gradients; the H stack always has gradients. (This corrects the earlier note about "explorer L-cycles (0, 1)".) The
  `no_grad` is unconditional, i.e. also at inference; it only affects autograd.
- peft attaches to both stacks (`q/k/v/o_proj` + attention `gate_proj`, MLP `gate/up/down_proj`).
- Every LoRA parameter receives a finite, non-zero gradient under `[0, 3]` and `[3, 3]`. Gradients differ between
  the two, so truncation is not inert.
- Consequence: with the default, LoRA gradients ignore how the adapters change the H-cycle-0 L passes. That matches
  pretraining. Decision in `decisions.md`: default `[0,3]`, ablate `[3,3]` if memory allows.

## 7. Timing / memory on Kaggle T4: PENDING (run `configs/probe_timing.yaml`, needs GPU)
## 8. fp16 stability: PENDING (run `configs/probe_fp16.yaml`, needs GPU)
## 9. Reproduce Open-Jev-2B card numbers on 1k rows: PENDING (needs GPU + Qwen3.5-2B + Open-Jev loader)
The Open-Jev repo's code is untrusted third-party code; review before running. It scores each candidate in its own
forward pass with a scalar Yes/No head, so our baseline comparison is on identical rows and metrics but not an identical prompt.

## Real-weights end-to-end smoke test: VERIFIED (CPU, fp32, 2026-11-20)
`src.train --set device=cpu --set dtype=fp32 --set max_steps=2 --set batch_size=1 --set accumulation=1`, then `src.eval --limit 4`
and `src.report` on the saved checkpoint. Only proves the code path works on real weights; the numbers are meaningless (n≤4, 2 steps).
- peft found its targets on the real model: **16,515,072 LoRA parameters** (rank 16; matches the hand count
  32 layers × (5×16×3072 + 3×16×5632) exactly) + 24,592 readout parameters.
- Loss and gradient norms finite on both steps (gradients clipped at 1.0; raw norms 52 and 20 at step 1–2 with LR still in warmup).
- Checkpoints written (`ckpt/`, `best/`, `final/`), reloaded via `load_trained` into a fresh real model, predictions saved, temperature fitted, report built.
- CPU speed ~12–14 tok/s training, ~8 s/row eval; not a GPU number. Items 7 (T4 timing/memory), 8 (fp16 stability), 9 (baseline) remain PENDING.
