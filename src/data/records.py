"""Open-Jev `release-v2-redistributable` records.

Only `state`, `question`, `kind` and `options` may reach the model. `target` is the label;
`language` / `target_basis` / `source` / `group_id` are analysis-only and never rendered.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_ID = "ZefanCai/Open-Jev"
DATASET_CONFIG = "release-v2-redistributable"
# Pinned so numbers stay comparable; bump deliberately (and note it in notes/decisions.md).
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
SPLITS = ("train", "calibration", "validation", "test", "ood")
EXPECTED_SIZES = {"train": 79_116, "calibration": 4_672, "validation": 3_723, "test": 10_356, "ood": 15_701}

# Rows whose metadata language is "en+zh" are exactly the rows with CJK text (verified in Phase 0).
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


@dataclass(frozen=True)
class Record:
    id: str
    group_id: str
    source: str
    kind: str  # choice | noul | score
    question: str
    options: tuple[str, ...]
    target: tuple[float, ...]
    state: Any  # str, or decoded JSON (dict/list)
    language: str | None  # analysis only
    target_basis: str | None  # analysis only

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def hard(self) -> bool:
        """One-hot target. Soft-target rows are excluded from 'hard accuracy' (Open-Jev-2B card)."""
        return max(self.target) == 1.0

    @property
    def is_english(self) -> bool:
        return not _CJK.search(render_state(self.state) + self.question + " ".join(self.options))


def render_state(state: Any) -> str:
    """Same rendering as the Open-Jev reference (`jev.api._render`)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _row_to_record(row) -> Record:
    meta = json.loads(row.metadata_json)
    return Record(
        id=row.id, group_id=row.group_id, source=row.source, kind=row.kind, question=row.question,
        options=tuple(row.options), target=tuple(float(x) for x in row.target),
        state=json.loads(row.state_json), language=meta.get("language"), target_basis=meta.get("target_basis"),
    )


def split_file(split: str, data_dir: str | Path = "data/raw") -> Path:
    """Local path of a split's parquet, downloading the pinned revision on first use."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        DATASET_ID, f"data/{DATASET_CONFIG}/{split}-00000-of-00001.parquet", repo_type="dataset",
        revision=DATASET_REVISION, local_dir=str(data_dir),
    ))


def load_split(split: str, data_dir: str | Path = "data/raw", *, english_only: bool = False,
               limit: int | None = None) -> list[Record]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    df = pd.read_parquet(split_file(split, data_dir))
    if split in EXPECTED_SIZES and len(df) != EXPECTED_SIZES[split]:
        raise ValueError(f"{split}: {len(df)} rows, expected {EXPECTED_SIZES[split]} (wrong dataset revision?)")
    records = [_row_to_record(r) for r in df.itertuples(index=False)]
    for r in records:
        t = np.asarray(r.target)
        if len(t) != r.n_options or abs(t.sum() - 1.0) > 1e-6 or (t < 0).any():
            raise ValueError(f"bad target for {r.id}")
    if english_only:
        records = [r for r in records if r.is_english]
    return records[:limit] if limit else records
