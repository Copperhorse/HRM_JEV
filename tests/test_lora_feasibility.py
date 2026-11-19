"""Phase 0 item 6: does peft LoRA attach to HRM-Text's looped stacks, and where do gradients land?

Uses a tiny random-weight HrmText (CPU, seconds). The structure (two stacks L/H applied
H_cycles * (L_cycles + 1) times, `L_bp_cycles` no-grad trick) is identical to the real model.
"""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import HrmTextConfig, HrmTextForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_config(l_bp_cycles):
    return HrmTextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, H_cycles=2, L_cycles=3,
        L_bp_cycles=l_bp_cycles, max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, tie_word_embeddings=False,
    )


def lora_grads(l_bp_cycles, seed=0):
    torch.manual_seed(seed)
    base = HrmTextForCausalLM(tiny_config(l_bp_cycles))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, bias="none"))
    # LoRA B starts at zero, which zeroes A's grads; randomize B so every adapter can show a grad.
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    ids = torch.randint(3, 128, (2, 10))
    out = model(input_ids=ids, token_type_ids=torch.ones_like(ids))
    out.logits[:, -1].logsumexp(-1).sum().backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}, model


def stack_of(name):
    return "L_module" if ".L_module." in name else "H_module" if ".H_module." in name else "other"


def test_lora_attaches_to_both_stacks():
    _, model = lora_grads([0, 3])
    names = [n for n, p in model.named_parameters() if "lora_A" in n]
    assert any(stack_of(n) == "L_module" for n in names)
    assert any(stack_of(n) == "H_module" for n in names)
    # 7 targets per layer; the attention gate and MLP gate are both named gate_proj -> 8 modules/layer.
    assert len(names) == 2 * 8 * 2  # 2 stacks * 8 linear modules * 2 layers


@pytest.mark.parametrize("bp", [[0, 3], [3, 3]])
def test_every_lora_adapter_gets_gradient(bp):
    grads, model = lora_grads(bp)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n in grads for n in trainable), "some LoRA params received no gradient"
    assert all(torch.isfinite(g).all() for g in grads.values())
    for stack in ("L_module", "H_module"):
        assert any(g.abs().sum() > 0 for n, g in grads.items() if stack_of(n) == stack)


def test_default_truncation_changes_gradient():
    """[0,3] (pretraining default) skips grads through H-cycle-0's L passes; [3,3] does not.
    If these were equal, the no_grad trick would be inert and no patching decision would matter."""
    g_default, _ = lora_grads([0, 3])
    g_full, _ = lora_grads([3, 3])
    assert g_default.keys() == g_full.keys()
    diffs = [(g_default[n] - g_full[n]).abs().max().item() for n in g_default]
    assert max(diffs) > 1e-6
