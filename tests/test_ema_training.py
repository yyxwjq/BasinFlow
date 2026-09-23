"""EMA weights must be tracked, saved and separable from the raw weights.

ReactOT trains with ``ema=True, ema_decay=0.999`` and evaluates the EMA copy,
so a reproduction that silently reports raw weights would be measuring a
different model. These tests pin the tracking rule (a lagged average), the
checkpoint contract and the validation of the decay.
"""
from __future__ import annotations

import numpy as np
import pytest

from basinflow.data import Event, EventCatalog, Structure
from basinflow.seeds import ZeroInit


def _toy_dataset() -> EventCatalog:
    reactant = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
        movable_mask=[False, True, False],
    )
    product = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )
    return EventCatalog(events={"e": Event("e", reactant, product)})


def test_ema_decay_must_be_a_fraction() -> None:
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    model = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)
    with pytest.raises(ValueError, match="ema_decay"):
        train_product_flow(
            model, _toy_dataset(), [ZeroInit()], epochs=1, lr=1e-3, ema_decay=1.0
        )


def test_checkpoint_carries_a_lagged_ema_copy(tmp_path) -> None:
    import torch

    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    torch.manual_seed(5)
    model = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    train_product_flow(
        model,
        _toy_dataset(),
        [ZeroInit()],
        epochs=3,
        lr=0.05,
        checkpoint_path=tmp_path / "ckpt.pt",
        ema_decay=0.9,
    )
    saved = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert saved["ema_decay"] == pytest.approx(0.9)
    raw, ema = saved["model_state_dict"], saved["ema_model_state_dict"]
    assert set(raw) == set(ema)
    lagged = 0
    for name in raw:
        raw_delta = float((raw[name].float() - before[name].float()).abs().sum())
        if raw_delta <= 1e-6:
            continue
        ema_delta = float((ema[name].float() - before[name].float()).abs().sum())
        assert ema_delta < raw_delta
        lagged += 1
    assert lagged > 0, "training did not move any parameter, so the test proves nothing"


def test_without_ema_the_checkpoint_says_so(tmp_path) -> None:
    import torch

    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    model = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)
    train_product_flow(
        model, _toy_dataset(), [ZeroInit()], epochs=1, lr=1e-3,
        checkpoint_path=tmp_path / "plain.pt",
    )
    saved = torch.load(tmp_path / "plain.pt", map_location="cpu", weights_only=False)
    assert saved["ema_model_state_dict"] is None
    assert saved["ema_decay"] is None
