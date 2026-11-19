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
