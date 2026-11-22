# Decisions

## 2026-11-19
- **Readout:** letters A–P at the last prompt token (`<|im_end|>`), via a 16×1536 readout initialised from the
  pretrained `lm_head` rows for the bare letter tokens and trained jointly with LoRA. `lm_head` is dropped. Noul is
  presented as options `A. no / B. yes` (uniform interface; the Yes/No-token variant is an untested alternative).
- **Condition prefix:** `direct` (`<|object_ref_start|>`), per the model card's guidance for classification-style tasks.
- **PrefixLM:** `token_type_ids = 1` over the entire prompt, 0 on padding; right padding; verified padding-invariant.
- **LoRA:** rank 16, alpha 32, on attention q/k/v/o + attention gate + MLP gate/up/down of both stacks.
- **`L_bp_cycles`:** keep the model default `[0, 3]` (matches pretraining). `[3, 3]` is a config switch
  (`l_bp_cycles: [3, 3]`) for an ablation if the T4 memory probe allows.
- **Option shuffling:** train-time only, `choice` rows only (answer-position shortcuts exist; see phase0.md §4).
- **Loss:** soft cross-entropy + 0.1 × Brier (as SatellaJev did).
- **English subset:** defined by absence of CJK text (equals metadata `language != "en+zh"`).
- **Calibration:** one temperature fitted on the full calibration split, applied to every split and subset.
- **Two GPUs:** run one seed per T4 (`device: cuda:1`) rather than DDP; shard eval with `--shard i/n`.
- **Third-party code:** the cloned Open-Jev repo is read, not executed, until reviewed.

## 2026-11-20
- **Experiment tracking:** optional Weights & Biases mirror (`wandb_project` config key; off by default), chosen over MLflow because it
  works from a Kaggle notebook with just an API key (MLflow would need a reachable server or a downloaded file store).
  `runs/<id>/metrics.jsonl`, `run.json` and the saved predictions remain the source of truth; W&B receives config + numeric metrics only.
  Not yet mirrored: eval/report tables (CPU-side; add if wanted).
- **Adapter persistence:** optional upload of adapter + readout + metrics to a *private* Hugging Face model repo (`hub_repo`), after every checkpoint and at the end,
  so an expiring Kaggle session cannot lose a run. Failure only warns. No optimizer state, predictions or base weights are uploaded.
- **Resumability across lost sessions:** the Hub upload now includes `ckpt/` (optimizer state, step), and `train` restores it when the local run dir is missing
  (tested: wipe + resume reproduces the uninterrupted weights). Resume refuses if training settings changed (`RESUME_KEYS` in `src/train.py`).
  Cost: ~200 MB uploaded per `save_every`, paused training for the upload; private repo storage grows with each checkpoint version.
- **Both GPUs:** one run per GPU in parallel (seed1 on cuda:0, seed2 on cuda:1) instead of DDP: same wall-clock and quota as one run, gives an early
  variance estimate, no distributed code. DDP would roughly halve one run's time but only gives one result.

## ShARC mix (2026-11-21): `configs/seed1_sharc.yaml`, `src/data/sharc.py`
Why: the JevBench gap is on free-text rule/policy application (Open-Jev train is 95% JSON-state generators), see `notes/jevbench_vs_openjev.md`. `SargeDev/jev-distill-corpus-v3` was rejected (contains 14,336 of our validation/calibration/test rows; labels distilled from a closed API).
- Source: ShARC official zip (authors' site), SHA-256 pinned, read in place with `zipfile`. CC-BY-SA-3.0 (HF card `UCLNLP/sharc`); rule texts from public gov.uk / IRS pages. Share-alike consequences for a published adapter are UNVERIFIED (not legal advice): check before releasing weights.
- VERIFIED properties: train/dev/test share 0 `tree_id`s (1 rule text crosses train/test); train = 21,850 utterances after exact-duplicate removal (yes 6,763 / no 7,028 / inquire 6,803 / irrelevant 1,256) but **only 619 distinct rule texts**; choice-view prompts p50 217 / p99 378 / max 481 tokens (short: this does NOT address JevBench's long-document hard items, median 656 tokens, 10% > 3,000).
- Two views: `choice` (yes / no / irrelevant / ask a follow-up; all rows) and `noul` ([no, yes]; Yes/No rows only), disjoint utterances in a mix. State = rule text + scenario + answered follow-ups; no JevBench wording ("Yes means:" lines) is copied.
- Mix = identical 16k Open-Jev rows as seed1 + 6,000 choice + 2,000 noul ShARC rows (24k rows, 1,500 steps, 1.5x seed1 compute). Not compute-matched to seed1: a positive result on Open-Jev/ShARC says "this mix helps", not "ShARC beats the same compute spent on Open-Jev".
- Selection of `best/` unchanged (Open-Jev validation NLL); `val_sharc` (500 ShARC dev rows) is logged for information. ShARC `test` (8,276) is reserved for one final evaluation (no `src.eval` support yet).
- Judge on: Open-Jev test/OOD vs seed1 (regression check), ShARC dev/test (did it learn the new data), and only then JevBench (report-only, never selection).

## Public release of the seed1_long adapter (2026-11-22)
Published `CopperHorse/hrm-text-1b-open-jev-lora` (public, Apache-2.0, tagged `not-recommended`; model card opens with a NOT RECOMMENDED banner and lists the limits). Contents: `seed1_long/best` (adapter, readout, model.json, temperature.json), `run.json`, `results.md`; layout matches `src.hub.restore_run`, so `--hub-repo CopperHorse/hrm-text-1b-open-jev-lora` works for `src.eval` / `src.jevbench_eval`. Chosen over `seed1_long_sharc` because it has full test/OOD results and no ShARC (CC-BY-SA-3.0) content, so no share-alike question. The private `hrm-jev-adapters` repo (with optimizer states and all runs) stays private.
