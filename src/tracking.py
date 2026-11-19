"""Optional Weights & Biases tracking. Off unless `wandb_project` is set in the config.

`runs/<id>/metrics.jsonl` stays the source of truth; this only mirrors it. Only the resolved config and the
numeric training/validation metrics are sent: no dataset rows, prompts or predictions. Code saving is off.
Auth: `WANDB_API_KEY` env var (Kaggle: add it as a Secret and export it in the notebook; without it wandb prompts interactively). For no network during a run use `WANDB_MODE=offline`
and `wandb sync runs/<id>/wandb` afterwards. The wandb run id is stored in `runs/<id>/wandb_id.txt`, so a resumed
training session continues the same W&B run.
"""
from __future__ import annotations

import uuid
from pathlib import Path


def _flatten(row: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) or v is None:
            if v is not None:
                out[key] = v
    return out


class Tracker:
    def __init__(self, cfg, run_dir: Path):
        self.run = None
        if not cfg.wandb_project:
            return
        try:
            import wandb

            id_file = Path(run_dir) / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
            id_file.write_text(run_id)
            self.run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_id, id=run_id, resume="allow",
                config=cfg.to_dict(), dir=str(run_dir), settings=wandb.Settings(save_code=False))
        except Exception as e:  # noqa: BLE001
            # Tracking is only a mirror of metrics.jsonl: a bad key or an outage must not kill a training run.
            print(f"[train] WARNING: W&B disabled for this run ({type(e).__name__}: {e}); training continues", flush=True)
            self.run = None

    def log(self, row: dict) -> None:
        if self.run is not None and "step" in row:
            try:
                self.run.log({k: v for k, v in _flatten(row).items() if k != "step"}, step=int(row["step"]))
            except Exception as e:  # noqa: BLE001
                print(f"[train] WARNING: W&B logging failed, disabling it ({type(e).__name__}: {e})", flush=True)
                self.run = None

    def finish(self, summary: dict | None = None) -> None:
        if self.run is not None:
            try:
                for k, v in (summary or {}).items():
                    self.run.summary[k] = v
                self.run.finish()
            except Exception as e:  # noqa: BLE001
                print(f"[train] WARNING: W&B finish failed ({type(e).__name__}: {e})", flush=True)
