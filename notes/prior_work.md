# Prior work (UNVERIFIED — from cards/papers)

| Model | Setup | Reported result |
|---|---|---|
| Open-Jev-2B | Qwen3.5-2B, LoRA r8/α16 + scalar decision head, 1 pass over 80,816 rows of frozen `release-v2`, temperature 1.519 | Test hard acc 94.71%, OOD 86.02%; ECE 0.0116 / 0.1056. Hard acc **excludes soft-target rows**. |
| SatellaJev-100M | Trained from scratch, 101.5M params, custom 7k tokenizer, `release-v2-redistributable` | Test acc 82.03%, OOD 72.09%; ECE 0.0287 / 0.0997 |
| SanSi (Ouro-1.4B looped, frozen + LoRA r64) | Not Open-Jev data | Loop 1/3/8 acc 58.4/70.4/72.0 vs non-looped SmolLM2-1.7B 58.4. Looping only helps with looped pre-training. ECE best at loop 3; averaging per-loop probs cut ECE 0.093→0.044. |
| Jeff (Harvard) | LoRA r32 + letter-readout head | Base (non-instruct) backbone 70.3 vs instruct 84.4 with same recipe; head-only fit gave no gain; ranks 16–128 similar. |
| QJev-1.5B (AnkitNayak-dev; HF card read 2026-11-21) | **Instruct** Qwen2.5-1.5B-Instruct, QLoRA r16 on q/k/v/o, MLP head (LayerNorm, 1536->512 GELU ->10) on the final hidden state, single pass, up to 10 options. Stage 1: head only, 50,000 steps; stage 2: 1 pass over 655,806 rows of `SargeDev/jev-distill-corpus-v3` (**not Open-Jev `release-v2`**); stage 3: 1,250 steps of cross-format consistency KL. RTX 5060 laptop | BenchmarkHeaven `jevbench-mini` (240 cases, 50 scored metamorphic relations, 1,468 trials): **62.53% at tol=0.05 (official)**, 71.50% at tol=0.10 (the card calls it "exploratory relaxed"). Pre-consistency 58.67%. ~31 ms, 3.4 GB fp16. No Open-Jev test/OOD, no ECE, no accuracy. Weights downloadable (`decision_head.safetensors`, `lora_adapter/`); Apache-2.0 on the card, base under the Qwen licence. |

No published HRM-Text decision model or HRM-Text LoRA fine-tune exists (as of 2026-11-19).

QJev-1.5B caveats (card numbers UNVERIFIED, not re-run by us):
- Its benchmark is BenchmarkHeaven's metamorphic *consistency* suite (invariance, bounds, partition relations scored within a tolerance), not accuracy against labelled answers. Our 231-item public JevBench accuracy (63.6 / 61.0 for seed1/seed2) is a different measurement; 62.53 vs 61-63.6 is not a like-for-like gap.
- The Reddit post set 71.50% beside the 9B (72.0%) and 27B (76.5%) scores as "matching 9B". Per the card, 71.50% is the tol=0.10 *relaxed* score while the 9B/27B rows are tol=0.05 strict, so that comparison is wrong. The like-for-like figure is 62.53 vs 72.0 / 76.5.
- Different training data (distillation corpus, 655k rows) and an instruct base; nothing on Open-Jev `release-v2` test/OOD, so no row-level comparison with HRM-Text is possible without re-running its adapter on our rows.
- Recipe is the nearest relative to ours (head + LoRA on attention, single forward pass). Its extra ingredient is the permutation/format-consistency stage, which we do not have.
