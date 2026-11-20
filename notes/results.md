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

## Test / OOD evaluation of `best/` (= step 1000), seed1 + seed2 (Kaggle T4 ×2, 2026-11-20): VERIFIED from `report_seed{1,2}.log` (user-supplied)
`src.eval` on the full splits in canonical option order, `src.report`: one temperature fitted on the full calibration split (4,672 rows): **seed1 T = 1.0656, seed2 T = 1.1170**.
NLL / Brier / ECE below use that temperature; hard accuracy excludes soft-target rows. Subset "full" (all rows) unless noted; the English subset differs by ≤ 0.1 point.

| split | n (hard n) | metric | seed1 | seed2 | mean of 2 |
|---|---|---|---:|---:|---:|
| test | 10,356 (9,995) | hard acc | 0.9210 | 0.9098 | 0.9154 |
| | | expected acc | 0.9016 | 0.8910 | 0.8963 |
| | | NLL | 0.2177 | 0.2366 | 0.2272 |
| | | Brier | 0.1066 | 0.1201 | 0.1134 |
| | | ECE | 0.0095 | 0.0150 | 0.0123 |
| ood | 15,701 (15,389) | hard acc | 0.8520 | 0.8414 | 0.8467 |
| | | expected acc | 0.8445 | 0.8337 | 0.8391 |
| | | NLL | 0.7183 | 0.7182 | 0.7183 |
| | | Brier | 0.2531 | 0.2582 | 0.2557 |
| | | ECE | 0.0786 | 0.0769 | 0.0778 |

Hard accuracy by kind (full subset), seed1 / seed2: test choice 0.838 / 0.811, noul 0.937 / 0.929, score 0.966 / 0.962; **OOD choice 0.747 / 0.702**, noul 0.880 / 0.877, score 0.866 / 0.863.
OOD choice NLL is 1.46 / 1.38 (the weakest part). OOD ECE ≈ 0.077–0.079 even after temperature (the calibration split is in-distribution, so T ≈ 1 and does not fix OOD overconfidence).

- **Seed spread on the full splits is ~1 point of hard accuracy** (test 0.921 vs 0.910; OOD 0.852 vs 0.841), much smaller than the 3.8 points seen on the 1,000-row validation sample. Mean ± std over 2 seeds is not a reliable
  estimate (test 91.5 ± 0.8, OOD 84.7 ± 0.8 points, sample std, n = 2); the plan still needs ≥ 3 seeds for a final claim.
- **Comparison with the Open-Jev-2B card (UNVERIFIED, card numbers on a different row set: 10,532 test / 15,920 OOD including the omitted `wikispeedia-v1` rows; not our re-evaluation; item 9 pending):**
  card test hard acc 94.71%, NLL 0.1954, Brier 0.0761, ECE 0.0116; card OOD hard acc 86.02%, NLL 0.8030, Brier 0.2465, ECE 0.1056.
  Our 2-seed means: test hard acc −3.2 points, NLL worse by 0.03, Brier worse by 0.04; OOD hard acc −1.5 points, **NLL better by 0.085**, Brier worse by 0.009, ECE better (0.078 vs 0.106).
  Context: 16k training rows, 1 pass, base (non-instruct) 1B model vs 80k rows, instruction-tuned 2B. No matched controls: no architecture claims.
- Per-row predictions (`preds/best/*.parquet`) are only in the Kaggle version output (the Hub copy holds `results.*`); download them for local analysis by source/kind.

## Two-GPU data-parallel smoke test (Kaggle 2×T4, 30 steps, 2026-11-20): VERIFIED
NCCL init/all-reduce/barriers work (`NCCL_P2P_DISABLE=1`). Throughput per 5-step window 780 / 727 / 541 (contains a validation pass) / 734 / 754 tok/s, i.e. ~740 clean, ~1.6x one T4 (~470).
Loss and fp16 scaler normal (3 skipped steps in the first 25; val NLL 0.817 → 0.750 on 64 rows, noise). The speed gate (> 750) failed by 2% and aborted the notebook as designed;
the threshold was my own arbitrary line, now > 650. Cause of the gap (VERIFIED by simulation on the real batches): unequal padded length between the two ranks' micro-batches; fixed by length-balanced assignment.

## Public JevBench items (231), seed1 / seed2 at 1,000 steps: VERIFIED (local RTX A2000, fp16, 2026-11-20; ~5 min per seed)
`python -m src.jevbench_eval` (JevBench's definitions re-implemented, items read as data; JevBench repo commit 609a664). T from our Open-Jev calibration split. 1 hard item (4,412 tokens) exceeds the context: counted wrong, not truncated.
Selection hygiene: both seeds were chosen/trained before this was run; the public split was not used for any decision.

| system | acc % | easy | standard | hard acc % | hard ECE | Intelligence |
|---|---:|---:|---:|---:|---:|---:|
| HRM-Text LoRA seed1 (T=1.0656) | 63.6 | 100.0 | 73.6 | 41.4 | 0.342 | 48.2 |
| HRM-Text LoRA seed2 (T=1.1170) | 61.0 | 97.9 | 69.4 | 39.6 | 0.297 | 44.1 |
| Kev-0.8B (published, UNVERIFIED) | 63.6 | | | 36.9 | 0.172 | 49.3 |
| Qwen3-1.7B dense control (published, UNVERIFIED) | 63.6 | | | 34.2 | 0.253 | 50.0 |
| LEV-1.4B depth 4 (published, UNVERIFIED) | 68.0 | | | 44.1 | 0.231 | 55.7 |
| Kev-4B / LEV-2.6B d4 / Jev 1.13.0 (published, UNVERIFIED) | 75.8 / 76.2 / 86.6 | | | 54.1 / 56.8 / 73.0 | 0.112 / 0.192 / 0.086 | 67.2 / 67.3 / 82.2 |

- Accuracy 95% CI (Wilson, n=231): seed1 57.3-69.6%. So level with Kev-0.8B / Qwen3-1.7B / LEV-1.4B within noise; clearly below Kev-4B, LEV-2.6B, Jev.
- By kind (seed1/seed2): choice 0.647/0.619, **noul 0.595/0.568 (near chance for binary)**, score 0.722/0.722. Easy tier ~solved, hard tier ~40%.
- **Calibration is poor here**: hard ECE 0.30-0.34 vs 0.11-0.25 for published rows (our temperature fitted on in-distribution Open-Jev data barely helps: 0.353 -> 0.342).
- Seeds agree on 133 items, seed1-only 14, seed2-only 8. Different training data from the published rows (Kev data / closed), so this is a placement, not a matched comparison.
- Open-Jev test 92% / OOD 85% vs JevBench 61-64%: the gap is domain shift (JevBench items are long policy / rule-application documents), not a bug in the harness (VERIFIED easy tier 98-100%).

### Why noul is near chance on JevBench (investigated 2026-11-20; seed1/seed2 at 1,000 steps)
Noul accuracy on the public items (n=74: 12 easy, 24 standard, 38 hard): easy 1.00/1.00, standard 0.62/0.62, hard 0.45/0.39 (seed1/seed2). OOD Open-Jev noul is 0.877, so this is specific to these items.
- **VERIFIED, the hard tier is adversarial by construction:** every hard item with a `surface_answer` (106 of 111) has surface answer != expected (0 of 106). For a binary question accuracy is therefore exactly 1 - P(picks the surface answer); we pick it 60% on hard noul (35 items with a surface answer) and 70% of all our hard errors (44 of 63) are the surface answer. Below-chance is the designed outcome for a model that follows surface cues. Hard noul n=38: 15/38 correct is not distinguishable from chance (P=0.31).
- **VERIFIED, yes bias with high confidence:** hard noul predicted "yes" 26/38 (expected yes 17/38), mean P(yes) 0.68; when the truth is "no" accuracy is only 0.29/0.24; mean confidence 0.86-0.89 -> this is also where the poor hard ECE comes from. The bias is NOT our training prior (train noul P(yes) = 0.269): it comes from the content of these documents.
- **VERIFIED, not a prompt-format artefact:** our training noul questions never contain the "Yes means:/No means:" lines (0 of 48,907) but JevBench's do. Re-scoring the 74 noul items without those lines: seed1 standard 0.625 -> 0.708, hard 0.447 -> 0.368; seed2 standard 0.625 -> 0.708, hard 0.395 -> 0.421. Changes of 1-3 items in inconsistent directions: no effect.
- **UNVERIFIED hypothesis:** 43% of train noul rows are structured-state generators (painting-geometry 11.8k, snake 9.4k) plus workflow-controls; only 1.8k are `reasoning-control-v1`. The model learned fact-to-label patterns that do not transfer to applying policy text. Kev-0.8B also scores only 36.9% on hard overall, so a small base model may simply lack the capacity; not tested.
- Not done on purpose: no tuning against the public items (JevBench requires disclosing any training/selection on the public split). Open question for the next recipe: up-weighting `reasoning-control-v1` / customer-service noul, judged on our validation split and OOD, not on JevBench.
