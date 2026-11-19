"""Optional upload of a run's adapter to the Hugging Face Hub (private repo by default).

    python -m src.hub --run runs/seed1 --repo <hf-user>/hrm-jev-adapters      # manual
    python -m src.train --config ... --set hub_repo=<hf-user>/hrm-jev-adapters  # automatic, after each checkpoint + at the end

Uploads `best/` and `final/` (LoRA adapter + readout + model.json), `run.json`, `metrics.jsonl`, eval `results.*` and a short
MODEL_NOTES.md, under `<run_id>/` in the repo. NOT uploaded: optimizer checkpoints (`ckpt/`), predictions, datasets, base weights.
Auth: `HF_TOKEN` env var with WRITE access (Kaggle: add as a Secret and export it in the notebook).
Unchanged files are not re-uploaded, so calling this repeatedly during training is cheap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERNS = ["best/**", "final/**", "run.json", "metrics.jsonl", "MODEL_NOTES.md", "preds/*/results.*"]


def write_notes(run_dir: Path) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = run["config"]
    (run_dir / "MODEL_NOTES.md").write_text(
        f"# {cfg['run_id']}\n\n"
        "LoRA adapter + 16-way letter readout for a typed decision model (Open-Jev `release-v2-redistributable`), trained on top of "
        f"`{run['model'][0]}` @ `{run['model'][1]}`.\n\n"
        "- Not a standalone model: load the pinned base weights, then the adapter and `readout.pt` with the code in the training repo "
        f"(commit `{run.get('git_commit')}`).\n"
        f"- Data: `{run['dataset'][0]}` @ `{run['dataset'][1]}`; {run['train_rows']} training rows, seed {cfg['seed']}, "
        f"LoRA rank {cfg['lora_rank']}, {run['total_steps']} steps, dtype {cfg['dtype']}.\n"
        "- `best/` = lowest validation NLL, `final/` = last step. Evaluation results (if any) are in `preds/*/results.md`.\n"
        "- Licenses of the base model and dataset apply; this card makes no performance claim.\n")


def sync_run(run_dir: str | Path, repo: str, *, private: bool = True, token: str | None = None) -> None:
    from huggingface_hub import HfApi

    run_dir = Path(run_dir)
    write_notes(run_dir)
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(run_dir), repo_id=repo, repo_type="model", path_in_repo=run_dir.name,
                      allow_patterns=PATTERNS, commit_message=f"Update {run_dir.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True, help="<hf-user>/<repo-name>")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    args = ap.parse_args()
    sync_run(args.run, args.repo, private=not args.public)
    print(f"uploaded {args.run} -> https://huggingface.co/{args.repo}/tree/main/{Path(args.run).name}")


if __name__ == "__main__":
    main()
