"""ShARC (Saeidi et al. 2018, https://sharc-data.github.io) converted to our typed-decision `Record`s.

ShARC = a rule text (gov.uk / IRS pages), a question, an optional user scenario and the follow-up Q&A so far; the label is
Yes / No / Irrelevant, or "Inquire" (a condition that decides the answer has not been established, so ask a follow-up).
Human-annotated, licence CC-BY-SA-3.0 (HF card `UCLNLP/sharc`), text from public government pages.

Two views, one `Record` each (never both for the same utterance in a training mix, see `sample_sharc`):
  * `choice`: 4 options  yes / no / irrelevant / ask a follow-up   (all rows)
  * `noul`:   [no, yes]                                            (Yes/No rows only)

The zip is downloaded once from the authors' site and read in place with `zipfile` (nothing is extracted or executed);
its SHA-256 is pinned. Splits are disjoint by `tree_id` (VERIFIED: 0 shared tree ids across train/dev/test), so
`group_id = tree_id` keeps the Open-Jev rule (a group never straddles train and eval).
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from .records import Record

SHARC_URL = "https://sharc-data.github.io/data/sharc1-official.zip"
SHARC_SHA256 = "72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e"  # verified 2026-11-21
SHARC_SPLITS = {"train": "train", "dev": "dev", "test": "test"}  # we use train for training, dev as validation, test once
EXPECTED_SIZES = {"train": 21_890, "dev": 2_270, "test": 8_276}  # utterances before exact-duplicate removal

CHOICE_OPTIONS = (
    "yes",
    "no",
    "irrelevant (the rule text does not apply to this question)",
    "ask a follow-up (a condition that decides the answer has not been established yet)",
)
_LABEL_INDEX = {"yes": 0, "no": 1, "irrelevant": 2}  # anything else is an inquiry (the answer text is the follow-up question)
_CHOICE_ASK = "Using only the rule text and what the user has said so far, which response is correct?"
_NOUL_ASK = "Answer from the rule text and what the user has said so far."


def zip_path(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / "sharc" / "sharc1-official.zip"


def fetch(data_dir: str | Path = "data") -> Path:
    """Download the official zip (if missing) and verify its SHA-256. Call from ONE process, then barrier."""
    p = zip_path(data_dir)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(SHARC_URL, timeout=120) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != SHARC_SHA256:
        raise ValueError(f"{p}: sha256 {digest} != pinned {SHARC_SHA256}; delete the file and retry")
    return p


def render_state(item: dict) -> str:
    """Rule text, then the user's scenario and the follow-ups answered so far (each section only when present)."""
    parts = [f"Rule text:\n{item['snippet'].strip()}"]
    if item["scenario"].strip():
        parts.append(f"User scenario: {item['scenario'].strip()}")
    hist = [h for h in item["history"] if isinstance(h, dict)]
    if hist:
        parts.append("Follow-up questions already answered:\n" + "\n".join(
            f"Q: {h['follow_up_question'].strip()}\nA: {h['follow_up_answer'].strip()}" for h in hist))
    return "\n\n".join(parts)


def label_index(answer: str) -> int:
    return _LABEL_INDEX.get(answer.strip().lower(), 3)


def to_record(item: dict, view: str) -> Record:
    """One ShARC utterance as a `Record`. `noul` is defined only for Yes/No answers."""
    idx = label_index(item["answer"])
    base = dict(id=f"sharc:{item['utterance_id']}:{view}", group_id=item["tree_id"], source="sharc",
                state=render_state(item), language="en", target_basis="human_annotated_sharc")
    if view == "choice":
        t = [0.0] * 4
        t[idx] = 1.0
        return Record(kind="choice", question=f"{item['question'].strip()}\n{_CHOICE_ASK}", options=CHOICE_OPTIONS,
                      target=tuple(t), **base)
    if view == "noul":
        if idx > 1:
            raise ValueError("noul view needs a Yes/No answer")
        return Record(kind="noul", question=f"{item['question'].strip()}\n{_NOUL_ASK}", options=("no", "yes"),
                      target=(0.0, 1.0) if idx == 0 else (1.0, 0.0), **base)
    raise ValueError(f"unknown view {view!r}")


def load_items(split: str, data_dir: str | Path = "data") -> list[dict]:
    """Raw utterances of `train` / `dev` / `test` with exact duplicates (same rule, question, scenario, history) removed."""
    if split not in SHARC_SPLITS:
        raise ValueError(f"unknown ShARC split {split!r}; expected one of {sorted(SHARC_SPLITS)}")
    with zipfile.ZipFile(fetch(data_dir)) as z:
        raw = json.loads(z.read(f"sharc1-official/json/sharc_{split}.json"))
    if len(raw) != EXPECTED_SIZES[split]:
        raise ValueError(f"sharc {split}: {len(raw)} utterances, expected {EXPECTED_SIZES[split]}")
    seen, out = set(), []
    for it in raw:
        key = (it["snippet"], it["question"], it["scenario"], json.dumps(it["history"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def load_sharc(split: str, data_dir: str | Path = "data", *, view: str = "choice") -> list[Record]:
    items = load_items(split, data_dir)
    if view == "noul":
        items = [it for it in items if label_index(it["answer"]) <= 1]
    return [to_record(it, view) for it in items]


def sample_sharc(n_choice: int, n_noul: int, seed: int, data_dir: str | Path = "data") -> list[Record]:
    """Seeded training sample: `n_choice` utterances as the 4-way view plus `n_noul` Yes/No utterances as the binary view.

    The two sets are disjoint whenever enough Yes/No utterances remain, so no utterance is trained on twice.
    """
    items = load_items("train", data_dir)
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    choice_idx = order[:n_choice]
    taken = set(choice_idx)
    yes_no = [i for i in order if label_index(items[i]["answer"]) <= 1]
    noul_idx = [i for i in yes_no if i not in taken][:n_noul]
    if len(noul_idx) < n_noul:  # not enough unused Yes/No rows: top up from the rest
        noul_idx += [i for i in yes_no if i in taken][: n_noul - len(noul_idx)]
    if n_choice > len(items):
        raise ValueError(f"asked for {n_choice} choice rows, only {len(items)} ShARC train utterances")
    return [to_record(items[i], "choice") for i in choice_idx] + [to_record(items[i], "noul") for i in noul_idx]
