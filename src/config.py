from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    run_id: str
    seed: int = 1
    data_dir: str = "data/raw"
    out_root: str = "runs"
    # model
    dtype: str = "fp16"  # fp16 on Kaggle T4/P100 (no bf16); bf16 on Ampere+; fp32 for CPU tests
    device: str = "cuda:0"
    lora_rank: int = 16
    lora_alpha: int | None = None  # default 2 * rank
    lora_dropout: float = 0.0
    l_bp_cycles: list[int] | None = None  # None = model default [0, 3]; [3, 3] = backprop through all loops
    grad_checkpointing: bool = True
    # data
    english_only: bool = True
    max_rows: int | None = None  # seeded random subset of train; None = all
    # extra training rows from ShARC (src/data/sharc.py; CC-BY-SA rule-application data). 0 = Open-Jev only (default).
    sharc_choice_rows: int = 0  # utterances as the 4-way choice view (yes / no / irrelevant / ask a follow-up)
    sharc_noul_rows: int = 0  # Yes/No utterances as the binary noul view (distinct from the choice rows when possible)
    sharc_val_rows: int = 500  # fixed seeded sample of ShARC dev (choice view), logged as `val_sharc`; only used when a ShARC mix is on
    shuffle_choice_options: bool = True
    # optimisation
    epochs: float = 1.0
    batch_size: int = 8
    accumulation: int = 4
    lr: float = 2e-4
    readout_lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0
    brier_weight: float = 0.1
    max_steps: int | None = None  # hard cap (probes / smoke tests)
    # bookkeeping
    log_every: int = 10
    eval_every: int = 500
    eval_rows: int = 1000  # fixed seeded sample of validation
    eval_batch_size: int = 16
    save_every: int = 250
    # optional Weights & Biases mirror of metrics.jsonl (off when wandb_project is null); see src/tracking.py
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # optional upload of adapter + metrics to a Hugging Face model repo, e.g. "<user>/hrm-jev-adapters" (see src/hub.py)
    hub_repo: str | None = None
    hub_private: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.run_id
