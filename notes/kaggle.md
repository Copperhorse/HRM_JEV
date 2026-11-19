# Kaggle runbook (2×T4, fp16)

Kaggle notebook cells (Internet on, GPU T4 ×2). The repo is the source of truth; notebooks stay thin.

```bash
# 1. code + deps (Kaggle ships torch; do not reinstall it)
git clone <this repo> /kaggle/working/HRM_JEV && cd /kaggle/working/HRM_JEV
pip install -q "transformers>=5.9.0" peft pyyaml pyarrow pandas tqdm pytest
python -m pytest -q                                   # CPU tests (first run downloads tokenizer + data)

# 2. Phase 0 GPU items, in order
python -m src.probe_forward --device cuda:0 --dtype fp16                      # item 5 already verified on CPU fp32; this checks fp16 forward is finite
python -m src.train --config configs/probe_timing.yaml --set device=cuda:0    # item 7: tok_per_s, mem_gb in runs/probe_timing/metrics.jsonl
python -m src.train --config configs/probe_fp16.yaml                          # item 8: skipped_steps, loss_scale collapse, non-finite loss
# memory/speed with full backprop through all loops:
python -m src.train --config configs/probe_timing.yaml --set l_bp_cycles=[3,3] --set run_id=probe_bp33

# 3. Real run (one seed per GPU; start seed 2 only after seed 1 looks sane)
python -m src.train --config configs/seed1.yaml --set device=cuda:0 &
python -m src.train --config configs/seed1.yaml --set device=cuda:1 --set seed=2 --set run_id=seed2 &

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
- Experiment tracking (optional): `pip install wandb`, add `WANDB_API_KEY` as a Kaggle Secret (export it in the cell), then add
  `--set wandb_project=hrm-jev` to `src.train`. Mirrors `metrics.jsonl` only (config + numeric train/val metrics; no rows, prompts
  or predictions). Resumed sessions continue the same W&B run. No network? `WANDB_MODE=offline`, then `wandb sync runs/<id>/wandb`.
