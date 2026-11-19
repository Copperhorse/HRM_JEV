# Prior work (UNVERIFIED — from cards/papers)

| Model | Setup | Reported result |
|---|---|---|
| Open-Jev-2B | Qwen3.5-2B, LoRA r8/α16 + scalar decision head, 1 pass over 80,816 rows of frozen `release-v2`, temperature 1.519 | Test hard acc 94.71%, OOD 86.02%; ECE 0.0116 / 0.1056. Hard acc **excludes soft-target rows**. |
| SatellaJev-100M | Trained from scratch, 101.5M params, custom 7k tokenizer, `release-v2-redistributable` | Test acc 82.03%, OOD 72.09%; ECE 0.0287 / 0.0997 |
| SanSi (Ouro-1.4B looped, frozen + LoRA r64) | Not Open-Jev data | Loop 1/3/8 acc 58.4/70.4/72.0 vs non-looped SmolLM2-1.7B 58.4. Looping only helps with looped pre-training. ECE best at loop 3; averaging per-loop probs cut ECE 0.093→0.044. |
| Jeff (Harvard) | LoRA r32 + letter-readout head | Base (non-instruct) backbone 70.3 vs instruct 84.4 with same recipe; head-only fit gave no gain; ranks 16–128 similar. |

No published HRM-Text decision model or HRM-Text LoRA fine-tune exists (as of 2026-11-19).
