# HRM-Text × Open-Jev

LoRA fine-tune of [`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B) into a
typed decision model on Open-Jev `release-v2-redistributable`, then compare it against
published decision models.

## Research question

Can a LoRA-fine-tuned 1B *base* HRM-Text model match published decision models (e.g.
Open-Jev-2B, an instruction-tuned Qwen3.5-2B) on Open-Jev test/OOD and on public
decision-model leaderboards?

**Scope of claims:** no matched controls are trained, so this project cannot claim that HRM's
architecture beats a standard one. Results are limited to "HRM-Text-1B + this recipe scores X
vs model Y on identical rows / the same leaderboard harness."

## Status

Phase 0 data/LoRA checks done; training, eval and report code written and tested on CPU (GPU probes pending on Kaggle). The plan lives in [`CLAUDE.md`](CLAUDE.md).

## Plan

1. **Phase 0**: verify data, model, LoRA feasibility, fp16 stability and timing.
2. **Train** HRM-Text LoRA on Kaggle (seed 1 first).
3. **Evaluate** on our own pipeline against re-evaluated published baselines.
4. **Submit** the best checkpoint to public decision leaderboards.

## Where things run

- Training and full evaluation: Kaggle (2×T4, fp16).
- Local: code, CPU-only analysis of saved predictions, plots.

## License

Code: MIT (see [LICENSE](LICENSE)). The Open-Jev dataset, HRM-Text-1B weights and Open-Jev-2B checkpoint are third-party and keep
their own licenses; none of them are redistributed here (they are downloaded from the Hugging Face Hub at pinned revisions).

## Setup

- Local (CPU torch, via [uv](https://docs.astral.sh/uv/)): `uv sync`, then `uv run pytest -q`.
- Plain `pip` (e.g. Kaggle, which already provides a CUDA torch): `pip install -r requirements.txt`.
