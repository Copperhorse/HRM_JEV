# Results

## Training runs, seed1 + seed2 (Kaggle T4 ×2, 2026-11-20): VERIFIED from the W&B logs (user-supplied screenshot)
Config `configs/seed1.yaml`: 16k English train rows (seeded sample), 1 pass = 1,000 steps × 16 rows, LoRA r16, lr 2e-4, fp16, `L_bp_cycles=[0,3]`,
batch 4 × accumulation 4. One run per GPU in parallel (seed1 on cuda:0, seed2 on cuda:1; seed2 used `--set seed=2`). Total wall-clock about 3.7 h.

Validation at step 1000 (fixed random sample of 1,000 English validation rows; temperature NOT applied; NOT test/OOD):

| run | hard acc | expected acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| seed1 | 0.8912 | 0.8635 | 0.3115 | 0.1474 | 0.0154 |
| seed2 | 0.8532 | 0.8288 | 0.3664 | 0.1794 | 0.0376 |

- fp16: 3 skipped steps per run (all in the first ~10), loss scale 8192 afterwards, no non-finite values. Throughput 435–452 tok/s.
- Seeds differ by ~4 points of hard accuracy (NLL 0.31 vs 0.37) on the same 1,000 rows: the seed-to-seed spread is not small; use mean ± std over seeds for claims.
- Validation at step 50 (probes) was NLL ≈ 0.71, so training clearly learns. The step-1000 numbers above are the final-step validation, not necessarily `best/`.
- **Not comparable to Open-Jev-2B yet**: that needs test/OOD evaluation on identical rows (see `notes/kaggle.md`). Validation is in-distribution; OOD is the headline metric.
- Logging note: seed2's `mem_gb` read 0.0 in this run (metric queried the wrong CUDA device; fixed in `0d5387a`, training unaffected).
