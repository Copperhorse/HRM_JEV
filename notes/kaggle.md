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

# 3. Real run: one independent run per GPU, in parallel (same wall-clock = same quota as one run). For ONE run on both GPUs see "One run on both GPUs" below.
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

## One run on both GPUs (data-parallel; `configs/seed1_long.yaml` = 60k rows, 3,750 steps)
`torchrun --nproc_per_node=2 -m src.train ...` starts one process per T4. Each rank takes half of every step's micro-batches (batch 4 x accumulation 4 -> 2 per rank),
gradients are summed before the optimizer step, so the recipe is the same as on one GPU (VERIFIED on CPU/gloo: 2-process weights equal 1-process weights, and a
crash + resume equals an uninterrupted run; **NCCL on the two T4s is UNVERIFIED**). Expected ~1.9x faster: ~7 s/step, 3,750 steps ≈ 7.3 h, plus ~1 h of validation (every 250 steps) and checkpoint uploads ≈ 8.3 h (UNVERIFIED extrapolation from 13 s/step on one T4).
Only rank 0 logs, checkpoints and uploads to the Hub; validation is split across the ranks. `accumulation` must be divisible by 2. Warm the caches first, then **smoke test 20 steps** before the long run:
```bash
%%bash
cd /kaggle/working/HRM_JEV
python -c "from huggingface_hub import snapshot_download as d; from src.models.hrm_decision import MODEL_ID, MODEL_REVISION as R; d(MODEL_ID, revision=R)"
python -c "from src.data.records import load_split; load_split('train', 'data/raw')"
torchrun --nproc_per_node=2 -m src.train --config configs/seed1_long.yaml --set run_id=smoke --set max_steps=20 --set eval_every=10 --set save_every=10 --set eval_rows=64
# check: tok_per_s ≈ 2x the single-GPU ~470 in runs/smoke/metrics.jsonl, no NCCL error, skipped_steps low. Then the real run (resumable: rerun the same command):
torchrun --nproc_per_node=2 -m src.train --config configs/seed1_long.yaml --set wandb_project=HRM_JEV_Finetune --set hub_repo=<hf-user>/hrm-jev-adapters > seed1_long.log 2>&1
```
If NCCL hangs or errors, try `export NCCL_P2P_DISABLE=1` (T4 pairs without peer-to-peer) before `torchrun`; gradients are only ~66 MB per step, so it costs almost nothing.
Eval of the result: same shard commands as step 4 with `--run runs/seed1_long`.

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

## Evaluate a trained adapter in a fresh session (restore from the Hub)
Use after a training-only commit run, or whenever `runs/<id>/best` is not on disk. Same setup cells as above (GPU check, clone, install, `pip uninstall -y torchao`,
secrets cell), then. Restore **once** per run, then fan out the shards (parallel restores into one folder would race):
```bash
%%bash
cd /kaggle/working/HRM_JEV
for run in seed1 seed2; do
  python -m src.eval --run runs/$run --ckpt best --hub-repo <hf-user>/hrm-jev-adapters --restore-only
  python -m src.eval --run runs/$run --ckpt best --split calibration,test,ood --shard 0/2 --device cuda:0 > eval_${run}_0.log 2>&1 &
  python -m src.eval --run runs/$run --ckpt best --split calibration,test,ood --shard 1/2 --device cuda:1 > eval_${run}_1.log 2>&1 &
  wait
  python -m src.report --run runs/$run --ckpt best > report_$run.log 2>&1
  python -m src.hub --run runs/$run --repo <hf-user>/hrm-jev-adapters >> report_$run.log 2>&1   # uploads results.* next to the adapter
done
```
`--restore-only` fetches only `best/` (or `final/` with `--ckpt final`) and `run.json`, not the optimizer checkpoint. A quick dev check: add `--limit 200`.
Warm the dataset/model cache first (the two `!python -c ...` pre-download cells above) so parallel shards do not race on downloads.
