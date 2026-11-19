"""HRM-Text backbone + LoRA + letter-readout head -> option logits in one forward pass.

The readout is a [MAX_OPTIONS, hidden] matrix (+ bias) initialised from the pretrained `lm_head`
rows of the letter tokens A-P, trained jointly with LoRA. `lm_head` itself is dropped (saves
~200 MB; the 65k-way vocabulary projection is never used).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ..data.prompt import MAX_OPTIONS

MODEL_ID = "sapientinc/HRM-Text-1B"
MODEL_REVISION = "22097cbcecdd1301afe30a19a3ee61b96a9863e5"
# Attention (q/k/v/o + sigmoid output gate) and MLP (gate/up/down) of both the L and H stacks.
# Note `gate_proj` is both the attention output gate and the MLP gate; both are adapted.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_MASK_VALUE = -1e9
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_hrm(dtype: str = "fp32", device: str = "cpu", model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    from transformers import HrmTextForCausalLM

    return HrmTextForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=DTYPES[dtype], attn_implementation="sdpa",
    ).to(device)


class HrmDecisionModel(nn.Module):
    def __init__(self, causal_lm, letter_ids: list[int], *, lora_rank: int = 16, lora_alpha: int | None = None,
                 lora_dropout: float = 0.0, grad_checkpointing: bool = False,
                 l_bp_cycles: list[int] | None = None):
        super().__init__()
        if len(letter_ids) != MAX_OPTIONS:
            raise ValueError(f"need {MAX_OPTIONS} letter token ids")
        self.settings = dict(lora_rank=lora_rank, lora_alpha=lora_alpha or 2 * lora_rank,
                             lora_dropout=lora_dropout, l_bp_cycles=l_bp_cycles)
        lm_head = causal_lm.lm_head.weight
        hidden = lm_head.shape[1]
        self.readout = nn.Linear(hidden, MAX_OPTIONS, bias=True)
        with torch.no_grad():
            self.readout.weight.copy_(lm_head[letter_ids].float())
            self.readout.bias.zero_()
        self.readout.to(lm_head.device)
        backbone = causal_lm.model
        del causal_lm.lm_head
        backbone.requires_grad_(False)
        if l_bp_cycles is not None:
            # Default [0, 3] (from config) = H-cycle 0's L passes run under no_grad, as in pretraining.
            # [3, 3] backpropagates through every loop pass (more memory, untruncated gradient).
            backbone.L_bp_cycles_padded = list(l_bp_cycles)
        if grad_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_rank:
            from peft import LoraConfig, get_peft_model

            backbone = get_peft_model(backbone, LoraConfig(
                r=lora_rank, lora_alpha=self.settings["lora_alpha"], lora_dropout=lora_dropout,
                target_modules=list(LORA_TARGETS), bias="none"))
        self.backbone = backbone

    def forward(self, input_ids, attention_mask, token_type_ids, n_options) -> torch.Tensor:
        """Logits [B, MAX_OPTIONS] (float32); positions >= n_options are masked to a large negative."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            use_cache=False)
        last = attention_mask.sum(-1) - 1  # right padding: last prompt token (<|im_end|>)
        hidden = out.last_hidden_state[torch.arange(len(last), device=last.device), last]
        logits = self.readout(hidden.to(self.readout.weight.dtype)).float()
        valid = torch.arange(MAX_OPTIONS, device=logits.device)[None] < n_options[:, None]
        return logits.masked_fill(~valid, _MASK_VALUE)

    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        lora = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        return {"lora": lora, "readout": list(self.readout.parameters())}

    # -- persistence: adapter + readout + settings (base weights are re-downloaded at the pinned revision)
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.settings["lora_rank"]:
            self.backbone.save_pretrained(path / "adapter")
        torch.save(self.readout.state_dict(), path / "readout.pt")
        (path / "model.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION,
                                                     **self.settings}, indent=2) + "\n")

    def load_trained(self, path: str | Path) -> "HrmDecisionModel":
        """Load adapter + readout saved by `save` into a freshly built model with the same settings."""
        path = Path(path)
        if self.settings["lora_rank"]:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            result = set_peft_model_state_dict(self.backbone, load_file(path / "adapter" / "adapter_model.safetensors"))
            if getattr(result, "unexpected_keys", None):
                raise RuntimeError(f"adapter keys not loaded: {result.unexpected_keys[:3]}")
        self.readout.load_state_dict(torch.load(path / "readout.pt", map_location=self.readout.weight.device,
                                                weights_only=True))
        return self


def decision_loss(logits: torch.Tensor, target: torch.Tensor, brier_weight: float = 0.1) -> torch.Tensor:
    """Soft cross-entropy against `target` (+ brier_weight * sum-of-squares distance), mean over rows.

    Masked positions carry logit -1e9 and target 0, so they contribute nothing.
    """
    probs = logits.log_softmax(-1).exp()
    ce = -(target * logits.log_softmax(-1)).sum(-1)
    brier = ((probs - target) ** 2).sum(-1)
    return (ce + brier_weight * brier).mean()
