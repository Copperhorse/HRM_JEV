"""Optional upload of a run's adapter to the Hugging Face Hub (private repo by default).

    python -m src.hub --run runs/seed1 --repo <hf-user>/hrm-jev-adapters      # manual
    python -m src.train --config ... --set hub_repo=<hf-user>/hrm-jev-adapters  # automatic, after each checkpoint + at the end

Uploads `ckpt/` (latest resumable checkpoint incl. optimizer state), `best/` and `final/` (LoRA adapter + readout + model.json), `run.json`,
`metrics.jsonl`, eval `results.*` and a short MODEL_NOTES.md, under `<run_id>/` in the repo. NOT uploaded: predictions, datasets, base weights.
If the local run dir is gone (e.g. a wiped Kaggle session), `restore_run` pulls `ckpt/` back so training resumes exactly where it stopped.
Auth: `HF_TOKEN` env var with WRITE access (Kaggle: add as a Secret and export it in the notebook).
Unchanged files are not re-uploaded, so calling this repeatedly during training is cheap.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

PATTERNS = ["ckpt/**", "best/**", "final/**", "run.json", "metrics.jsonl", "MODEL_NOTES.md", "preds/*/results.*"]


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


TRAIN_ITEMS = ("ckpt", "best", "metrics.jsonl", "run.json")
_MARKER = {"ckpt": "state.json", "best": "model.json", "final": "model.json"}  # file proving a restored folder is complete


def restore_run(run_dir: str | Path, repo: str, *, items=TRAIN_ITEMS, token: str | None = None) -> bool:
    """Pull `<run_id>/<item>` for each item (default: ckpt, best, metrics.jsonl, run.json) from the Hub into `run_dir`.

    Training resumes with the default items; evaluation needs only e.g. ("best", "run.json"). Returns False if the repo does
    not hold a complete first folder item. Do not call concurrently for the same run_dir (restore once, then fan out).
    """
    from huggingface_hub import snapshot_download

    run_dir = Path(run_dir)
    name = run_dir.name
    patterns = [f"{name}/{i}/**" if i in _MARKER else f"{name}/{i}" for i in items]
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo, repo_type="model", token=token, local_dir=tmp, allow_patterns=patterns)
        src = Path(tmp) / name
        first = next(i for i in items if i in _MARKER)
        if not (src / first / _MARKER[first]).exists():
            return False
        run_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            if (src / item).exists():
                dst = run_dir / item
                shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink(missing_ok=True)
                shutil.move(str(src / item), str(dst))
    return True


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
