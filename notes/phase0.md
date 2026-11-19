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

## 5. Real-model forward pass with PrefixLM format: PENDING (GPU; `python -m src.probe_forward`, see `notes/kaggle.md`)
Not done locally: the 2.4 GB `model.safetensors` download from the Hub ran at ~200 KB/s and stalled, and a 1B-parameter
CPU forward at 8 stack passes/token is impractical anyway. The same checks (finite logits, padding invariance, prefix mask
honoured) already pass on tiny random-weight models (`tests/test_decision_model.py`); the probe repeats them on real weights.
Tokenizer facts needed for the format are VERIFIED: `<|im_start|>`=6, `<|im_end|>`=7, `<|object_ref_start|>`=8, letters A–P are
single tokens (ids 63–78), and `tokenizer(text, add_special_tokens=False)` is correct (no BOS is added by default).

## 6. LoRA feasibility: VERIFIED (tiny random-weight model, `tests/test_lora_feasibility.py`)
Read from `transformers/models/hrm_text/modeling_hrm_text.py` (5.23.0):
- Two separate stacks `L_module`, `H_module`, 16 layers each (~983M params total); the loop applies L three times then H, twice
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
