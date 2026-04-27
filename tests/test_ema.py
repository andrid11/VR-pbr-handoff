"""Tests for src/ema.py shadow-weight helper."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def _make_model():
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))


def test_ema_initial_shadow_equals_model():
    from src.ema import EMA
    torch.manual_seed(0)
    m = _make_model()
    ema = EMA(m, decay=0.99)
    for k, v in m.state_dict().items():
        assert torch.equal(ema.shadow[k], v), f"Mismatch at {k}"


def test_ema_update_moves_toward_model():
    from src.ema import EMA
    torch.manual_seed(0)
    m = _make_model()
    ema = EMA(m, decay=0.5)  # aggressive so we can see the move

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)

    before = ema.shadow["0.weight"].clone()
    ema.update(m)
    after = ema.shadow["0.weight"]

    # Shadow should be halfway between the old shadow and the new weight
    expected = 0.5 * before + 0.5 * m.state_dict()["0.weight"]
    assert torch.allclose(after, expected, atol=1e-6)


def test_ema_preserves_int_buffers():
    """BatchNorm's num_batches_tracked is a long tensor — must not be decayed."""
    from src.ema import EMA
    m = _make_model()
    ema = EMA(m, decay=0.9)

    # Bump the counter by running one batch through BN
    m.train()
    m(torch.randn(2, 4))
    ema.update(m)

    nbt = ema.shadow["1.num_batches_tracked"]
    assert nbt.dtype == torch.long
    assert nbt.item() == m.state_dict()["1.num_batches_tracked"].item()


def test_ema_state_dict_roundtrip():
    from src.ema import EMA
    m = _make_model()
    ema = EMA(m, decay=0.99)
    sd = ema.state_dict()
    assert set(sd.keys()) == set(m.state_dict().keys())
