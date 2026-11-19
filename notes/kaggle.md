# Kaggle runbook (2×T4, fp16)

Kaggle notebook cells (Internet on, GPU T4 ×2). The repo is the source of truth; notebooks stay thin.

```bash
# 1. code + deps (Kaggle ships torch; do not reinstall it). Clone into /kaggle/working, then `cd HRM_JEV` once (not twice)
git clone https://github.com/Copperhorse/HRM_JEV.git /kaggle/working/HRM_JEV && cd /kaggle/working/HRM_JEV
pip install -q -r requirements.txt                    # not torch: Kaggle's CUDA build is already installed
pip uninstall -y torchao                              # Kaggle's torchao 0.10 breaks peft ("incompatible version of torchao", needs >=0.16); we don't use it
python -m pytest -q                                   # CPU tests (first run downloads tokenizer + data)

# 2. Phase 0 GPU items, in order
python -m src.probe_forward --device cuda:0 --dtype fp16                      # item 5 already verified on CPU fp32; this checks fp16 forward is finite
python -m src.train --config configs/probe_timing.yaml --set device=cuda:0    # item 7: tok_per_s, mem_gb in runs/probe_timing/metrics.jsonl
python -m src.train --config configs/probe_fp16.yaml                          # item 8: skipped_steps, loss_scale collapse, non-finite loss
# memory/speed with full backprop through all loops:
python -m src.train --config configs/probe_timing.yaml --set l_bp_cycles=[3,3] --set run_id=probe_bp33

# 3. Real run: one independent run per GPU, in parallel (same wall-clock = same quota as one run). A single run never uses both GPUs (no DDP).
#    %%bash cell; `wait` makes it block, so it also works in "Save & Run All". Needs the secrets cell (WANDB_*, HF_TOKEN) run earlier in the same session.
%%bash
cd /kaggle/working/HRM_JEV
python -m src.train --config configs/seed1.yaml --set device=cuda:0 --set wandb_project=HRM_JEV_Finetune --set hub_repo=<hf-user>/hrm-jev-adapters > seed1.log 2>&1 &
python -m src.train --config configs/seed1.yaml --set device=cuda:1 --set seed=2 --set run_id=seed2 --set wandb_project=HRM_JEV_Finetune --set hub_repo=<hf-user>/hrm-jev-adapters > seed2.log 2>&1 &
wait
#    Follow progress from another cell: !tail -n 3 /kaggle/working/HRM_JEV/seed1.log   (and the W&B charts)

# 4. Eval right after training, same session; shard across both GPUs
python -m src.eval --run runs/seed1 --ckpt best --split calibration,test,ood --shard 0/2 --device cuda:0 &
python -m src.eval --run runs/seed1 --ckpt best --split calibration,test,ood --shard 1/2 --device cuda:1 &
wait; python -m src.report --run runs/seed1 --ckpt best
```

Notes
- Training is resumable: restore `runs/<id>/` from the previous session's output, rerun the same command.
- Save `runs/` as notebook output; copy `preds/` and `results.md` back locally for CPU analysis.
- Hyperparameters live in `configs/`; override with `--set key=value` and a distinct `run_id`.
- Local Hub downloads of the 2.4 GB weights were ~200 KB/s on 2026-11-19; Kaggle should not have this problem.
- Experiment tracking (optional, W&B project `HRM_JEV_Finetune`): `pip install wandb` (commented out in `requirements.txt`). In the
  notebook's Add-ons > Secrets, add `WANDB_API_KEY` and `WANDB_ENTITY` and tick them for this notebook (leave other secrets unticked).
  Secrets are not env vars by themselves; run this cell first (never print the values, notebook output is saved):
  ```python
  import os
  from kaggle_secrets import UserSecretsClient
  s = UserSecretsClient()
  os.environ["WANDB_API_KEY"] = s.get_secret("WANDB_API_KEY")   # the label exactly as named in the Secrets panel
  os.environ["WANDB_ENTITY"] = s.get_secret("WANDB_ENTITY")     # entity is kept out of this public repo
  ```
  Later `!python -m src.train ... --set wandb_project=HRM_JEV_Finetune` cells inherit these variables.
  Mirrors `metrics.jsonl` only (config + numeric train/val metrics; no rows, prompts or predictions). Resumed sessions continue the
  same W&B run. A bad key (e.g. truncated: W&B keys are 40+ chars) only prints a `W&B disabled for this run` warning and training continues; check `len(key)` without printing it. Without a key an interactive notebook shows a wandb login prompt (stop the cell, run the secrets cell, rerun; or answer `3` to disable W&B for that run), and a headless "Save & Run All" run cannot answer it. Env vars do not survive a kernel restart: re-run the secrets cell in every new session. No network? `WANDB_MODE=offline`,
  then `wandb sync runs/<id>/wandb`.
- Saving the adapter to the Hugging Face Hub (optional, private repo): create a **write** token at hf.co/settings/tokens, store it as a Kaggle
  Secret named `HF_TOKEN` (ticked for the notebook) and export it in the same secrets cell:
  ```python
  os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")   # must have write access; a read-only token gives a 403 warning, training continues
  ```
  then add `--set hub_repo=<hf-user>/hrm-jev-adapters` to `src.train`. After every checkpoint (`save_every`) and at the end it uploads
  `best/`, `final/`, `run.json`, `metrics.jsonl` and `MODEL_NOTES.md` to `<run_id>/` in that repo (private; unchanged files are skipped).
  Optimizer checkpoints, predictions and base weights are never uploaded. Manual upload: `python -m src.hub --run runs/seed1 --repo <hf-user>/hrm-jev-adapters`.
  Download later: `hf download <hf-user>/hrm-jev-adapters --include "seed1/*" --local-dir adapters` (needs HF_TOKEN for a private repo).

## If the session dies (timeout, crash, quota, wiped disk)
Checkpoints (adapter + readout + optimizer + scaler + step) are written every `save_every` steps (100 for seed1 ≈ 20 min of work) and, when
`hub_repo` is set, uploaded to the Hub (`<run_id>/ckpt`). At most one `save_every` interval is lost. To continue: start a new session, re-run the
setup cells (clone, `pip install -r requirements.txt`, `pip uninstall -y torchao`, secrets cell) and the **identical** training command.
It resumes from the local `runs/<id>/ckpt` if the disk survived ("Files only" persistence), otherwise restores it from the Hub; if neither
exists it starts fresh. Resuming refuses (with the changed keys listed) if training settings differ from the checkpoint's (seed, lr, batch,
rows, lora_*, epochs, max_steps, ...). Eval/logging/hub/dtype/device settings may change. Use `--no-resume` or a new `run_id` for a new recipe.

## Unattended "Save & Run All" notebook (seed1 + seed2, then eval)
Save & Run All re-runs the whole notebook from a fresh, empty `/kaggle/working`: keep **only** these cells (delete probe/debug cells), stop any open
draft session first (double quota), and confirm Settings > Accelerator = GPU T4 x2.
1. `import torch; assert torch.cuda.device_count() == 2` · `!cd /kaggle/working && git clone <repo url>` · `%cd /kaggle/working/HRM_JEV` · `!pip install -q -r requirements.txt` · `!pip uninstall -y torchao`
2. secrets cell (WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN → `os.environ`)
3. `%%bash` train: seed1 on cuda:0 and seed2 on cuda:1 in the background with `--set wandb_project=HRM_JEV_Finetune --set hub_repo=<hf-user>/hrm-jev-adapters`, then `wait` (~3 h)
4. `%%bash` for each seed: `src.eval --ckpt best --split calibration,test,ood` sharded `0/2` on cuda:0 and `1/2` on cuda:1, `wait`, `src.report`, `src.hub` (~1.5 h per seed)
Total ≈ 6 h of the 12 h session limit. A crash in step 4 does not lose the adapters (Hub + notebook output); rerun eval from an interactive session.
Logs (`seed*.log`, `eval_*.log`, `report_*.log`) land in the saved version's Output together with `runs/`.
