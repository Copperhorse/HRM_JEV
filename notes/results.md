# Results

## Training runs, seed1 + seed2 (Kaggle T4 ×2, 2026-11-20): VERIFIED from the full per-step logs (user-supplied `output.log` files)
Config `configs/seed1.yaml`: 16k English train rows (seeded sample; **the seed also changes which 16k rows**), 1 pass = 1,000 steps × 16 rows, LoRA r16, lr 2e-4
(cosine to 10%), fp16, `L_bp_cycles=[0,3]`, batch 4 × accumulation 4. One run per GPU in parallel (seed1 on cuda:0, seed2 on cuda:1). Wall-clock ≈ 3.7 h
(≈ 3.3 h training + 4 validation passes of ≈ 320 s + checkpoint uploads).

Validation = fixed random sample of 1,000 English validation rows (same rows for both seeds), temperature NOT applied, **not test/OOD**:

| step | seed1 hard acc | seed1 NLL | seed1 Brier | seed1 ECE | seed2 hard acc | seed2 NLL | seed2 Brier | seed2 ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.7962 | 0.4780 | 0.2593 | 0.0419 | 0.7772 | 0.5375 | 0.2968 | 0.0691 |
| 500 | 0.8374 | 0.4171 | 0.2072 | 0.0369 | 0.8279 | 0.4474 | 0.2297 | 0.0646 |
| 750 | 0.8564 | 0.3570 | 0.1765 | 0.0371 | 0.8458 | 0.4091 | 0.1916 | 0.0481 |
| 1000 | **0.8912** | **0.3115** | **0.1474** | **0.0154** | **0.8532** | **0.3664** | **0.1794** | **0.0376** |

- Validation NLL decreased at every checkpoint for both seeds and was still falling at the end (no plateau, no overfitting signal) → **`best/` == `final/` (step 1000) for both seeds**.
  Hypothesis (UNVERIFIED): more rows/steps would help; decide after test/OOD evaluation.
- fp16: skipped steps stayed at **3 from step 20 to step 1000** on both seeds (seed2: 2 at step 10), loss scale 8192, every grad norm finite (max ≈ 16.6). No NaN/inf.
- seed1 is better than seed2 at all four checkpoints (NLL gap 0.03–0.06; hard acc gap 2–4 points at the end). Seed variance includes the different 16k-row subset.
  Use mean ± std over seeds (plan: 3) for claims.
- Throughput 410–459 tok/s; the 108–131 tok/s values right after steps 250/500/750 are the validation pass inside the timing window, not slowdowns.
- Validation pass: 1,000 rows in 318–321 s ≈ **0.32 s/row on one T4** (batch 16). Full calibration+test+OOD (30,729 rows) ≈ 2.7 h on one T4, ≈ 1.4 h sharded over two, per seed.
- **Not comparable to Open-Jev-2B yet**: that needs test/OOD evaluation on identical rows (see `notes/kaggle.md`). Validation is in-distribution; OOD is the headline metric.
- Logging note: seed2's `mem_gb` read 0.0 (metric queried the wrong CUDA device; fixed in `0d5387a`, training unaffected).
