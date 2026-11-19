"""Tiny random-weight HRM (real tokenizer ids) -> checks wrapper, collation, loss, persistence."""
import random

import pytest
import torch
from transformers import AutoTokenizer, HrmTextConfig, HrmTextForCausalLM

from src.data.prompt import MAX_OPTIONS, collate, letter_token_ids, render_prompt
from src.data.records import Record
from src.models.hrm_decision import HrmDecisionModel, decision_loss


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")


def make_lm(seed=0):
    torch.manual_seed(seed)
    cfg = HrmTextConfig(
        vocab_size=65536, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[0, 3],
        max_position_embeddings=1024, pad_token_id=5, bos_token_id=6, eos_token_id=11, tie_word_embeddings=False)
    return HrmTextForCausalLM(cfg).eval()


def rec(i, kind="choice", n=3, state="s"):
    opts = ("no", "yes") if kind == "noul" else tuple(f"option {j} for {i}" for j in range(n))
    tgt = [0.0] * len(opts)
    tgt[i % len(opts)] = 1.0
    return Record(f"r{i}", f"g{i}", "src", kind, f"Question {i}? " + "word " * (i * 7), opts, tuple(tgt),
                  state, None, None)


def build(tok, **kw):
    return HrmDecisionModel(make_lm(), letter_token_ids(tok), lora_rank=4, **kw)


def test_prompt_format(tok):
    p = render_prompt(rec(0))
    assert p.startswith("<|im_start|><|object_ref_start|>Context:\ns") and p.endswith("Answer:<|im_end|>")
    ids = tok(p, add_special_tokens=False)["input_ids"]
    assert ids[0] == 6 and ids[1] == 8 and ids[-1] == 7


def test_padding_invariance(tok):
    """A row's logits must not depend on what it is batched with (right padding + prefix mask)."""
    m = build(tok).eval()
    # random LoRA B so the adapter is not a no-op
    for n, p in m.backbone.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    rows = [rec(1, n=3), rec(5, "noul"), rec(9, n=6)]
    with torch.no_grad():
        b = collate(rows, tok)
        batched = m(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options)
        for i, r in enumerate(rows):
            s = collate([r], tok)
            single = m(s.input_ids, s.attention_mask, s.token_type_ids, s.n_options)
            valid = r.n_options
            assert torch.allclose(batched[i, :valid], single[0, :valid], atol=1e-4), r.id
    assert b.input_ids.shape[1] > collate([rows[0]], tok).input_ids.shape[1]  # padding really happened


def test_masking_and_loss(tok):
    m = build(tok).eval()
    b = collate([rec(1, n=3), rec(2, "noul")], tok)
    logits = m(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options)
    assert (logits[0, 3:] < -1e8).all() and (logits[1, 2:] < -1e8).all()
    probs = logits.softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2)) and (probs[0, 3:] == 0).all()
    uniform = torch.zeros(1, MAX_OPTIONS).masked_fill(torch.arange(MAX_OPTIONS)[None] >= 4, -1e9)
    tgt = torch.tensor([[1.0, 0, 0, 0] + [0.0] * 12])
    assert decision_loss(uniform, tgt, 0.0).item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-5)


def test_gradients_only_on_lora_and_readout(tok):
    m = build(tok).train()
    for n, p in m.backbone.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    b = collate([rec(1), rec(2, n=4)], tok)
    decision_loss(m(b.input_ids, b.attention_mask, b.token_type_ids, b.n_options), b.target).backward()
    params = m.trainable_parameters()
    assert params["lora"] and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params["lora"])
    assert any(p.grad.abs().sum() > 0 for p in params["readout"])
    assert all(p.grad is None for n, p in m.backbone.named_parameters() if not p.requires_grad)


def test_readout_initialised_from_lm_head(tok):
    lm = make_lm()
    ids = letter_token_ids(tok)
    expect = lm.lm_head.weight[ids].clone()
    m = HrmDecisionModel(lm, ids, lora_rank=4)
    assert torch.equal(m.readout.weight.data, expect) and not hasattr(lm, "lm_head")


def test_save_load_roundtrip(tok, tmp_path):
    a = build(tok).eval()
    for n, p in a.backbone.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    a.readout.bias.data.normal_()
    a.save(tmp_path)
    b = build(tok).eval().load_trained(tmp_path)
    batch = collate([rec(3), rec(4, "noul")], tok)
    args = (batch.input_ids, batch.attention_mask, batch.token_type_ids, batch.n_options)
    with torch.no_grad():
        assert torch.allclose(a(*args), b(*args), atol=1e-6)
    fresh = build(tok).eval()
    with torch.no_grad():
        assert not torch.allclose(a(*args), fresh(*args), atol=1e-3)


def test_choice_shuffle_permutes_target_consistently(tok):
    r = rec(1, n=5)
    b = collate([r], tok, rng=random.Random(0))
    order = b.order[0]
    assert sorted(order) == list(range(5)) and order != list(range(5))
    assert b.target[0, order.index(1)] == 1.0  # original answer index 1 follows its option
    for kind_rec in (rec(2, "noul"), Record("s", "g", "x", "score", "q", ("a", "b", "c"), (1.0, 0, 0), "s", None, None)):
        assert collate([kind_rec], tok, rng=random.Random(0)).order[0] == list(range(kind_rec.n_options))


def test_refuses_to_truncate(tok):
    with pytest.raises(ValueError, match="refusing to truncate"):
        collate([rec(1, state="word " * 200)], tok, max_length=50)
