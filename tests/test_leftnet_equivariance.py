"""The ported LEFTNet must keep the reference's O(3) behaviour and contracts.

Two levels are covered:

* the backbone (``basinflow.models.leftnet.leftnet.LEFTNet``) -- rotation
  equivariance, translation invariance, reflection handling and finiteness;
* the ``R + P -> TS`` adapter (``LeftNetBridge``) -- the ``{"scaled": ...}``
  contract, the TS-block-only readout, per-sample mean removal, and the
  ``leftnet_sb`` factory registration.

A tiny configuration is used (2 layers, 32 channels) so the suite stays fast; the
equations, not the capacity, are what these tests pin.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from basinflow.models.factory import build_model, model_spec
from basinflow.models.leftnet import LEFTNet
from basinflow.models.leftnet.adapter import LeftNetBridge

ATOMS = 6


def _ring_edge_index(n: int = ATOMS) -> torch.Tensor:
    source, target = [], []
    for i in range(n):
        source += [i, (i + 1) % n]
        target += [(i + 1) % n, i]
    return torch.tensor([source, target])


def _features(n: int = ATOMS, width: int = 8) -> torch.Tensor:
    h = torch.zeros(n, width)
    h[:, 0] = 1.0
    return h


def _backbone(**overrides) -> LEFTNet:
    parameters = dict(
        cutoff=10.0,
        num_layers=2,
        hidden_channels=32,
        num_radial=16,
        in_hidden_channels=8,
        reflect_equiv=True,
        legacy=True,
        update=True,
        single_layer_output=True,
        object_aware=True,
    )
    parameters.update(overrides)
    torch.manual_seed(0)
    model = LEFTNet(**parameters).eval()
    return model


def _random_rotation(seed: int = 0, *, proper: bool = True) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator)
    q, _ = torch.linalg.qr(matrix)
    if proper:
        q = q * torch.sign(torch.det(q))
    return q


def _displacement(model: LEFTNet, pos: torch.Tensor) -> torch.Tensor:
    h = _features()
    edge_index = _ring_edge_index()
    mask = torch.ones(edge_index.size(1), 1)
    _, pos_out, _ = model(h, pos, edge_index, subgraph_mask=mask)
    return pos_out - pos


def test_backbone_returns_the_reference_shapes() -> None:
    model = _backbone()
    h, pos_out, edge_attr = model(_features(), torch.randn(ATOMS, 3), _ring_edge_index())
    assert h.shape == (ATOMS, 8)  # in_hidden_channels
    assert pos_out.shape == (ATOMS, 3)
    assert edge_attr is None  # the reference always returns None here
    assert torch.isfinite(pos_out).all()


def test_backbone_is_rotation_equivariant() -> None:
    model = _backbone()
    pos = torch.randn(ATOMS, 3)
    rotation = _random_rotation(1)
    reference = _displacement(model, pos)
    rotated = _displacement(model, pos @ rotation.T)
    assert torch.allclose(rotated, reference @ rotation.T, atol=1e-4)


def test_backbone_is_translation_invariant() -> None:
    model = _backbone()
    pos = torch.randn(ATOMS, 3)
    reference = _displacement(model, pos)
    shifted = _displacement(model, pos + torch.tensor([3.0, -1.5, 2.0]))
    assert torch.allclose(shifted, reference, atol=1e-4)


def test_reflect_equivariant_flag_changes_the_frame_construction() -> None:
    """The ``reflect_equiv`` switch must actually reach the node frame.

    The reference's parity handling is the ``abs`` applied to the second
    scalrization row; if the flag were ignored the two arms would be identical,
    which is exactly the ablation gap the ReactOT config relies on. The
    direction of the effect is not asserted here -- it is the subject of a
    training run, not of a unit test.
    """
    pos = torch.randn(ATOMS, 3)
    outputs = {flag: _displacement(_backbone(reflect_equiv=flag), pos) for flag in (True, False)}
    assert torch.isfinite(outputs[True]).all()
    assert torch.isfinite(outputs[False]).all()
    assert not torch.allclose(outputs[True], outputs[False])


def test_backbone_gradients_reach_the_coordinates() -> None:
    model = _backbone()
    pos = torch.randn(ATOMS, 3, requires_grad=True)
    loss = _displacement(model, pos).pow(2).mean()
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().sum()) > 0.0


def _bridge_batch(*, samples: int = 2, atoms: int = 4, seed: int = 0) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)
    z = torch.tensor([6, 1, 8, 7, 6, 1, 1, 8])[: samples * atoms].to(torch.long)
    reactant = torch.randn(samples * atoms, 3, generator=generator)
    state = reactant + 0.1 * torch.randn(samples * atoms, 3, generator=generator)
    endpoint = reactant + 0.2 * torch.randn(samples * atoms, 3, generator=generator)
    index = torch.repeat_interleave(torch.arange(samples), atoms)
    return SimpleNamespace(
        z=z,
        reactant_pos=reactant,
        pos=state,
        endpoint_pos=endpoint,
        batch=index,
        movable_mask=torch.ones(samples * atoms, dtype=torch.bool),
        flow_time=torch.full((samples, 1), 0.5),
        cell=torch.eye(3).expand(samples, 3, 3),
        pbc=torch.zeros(samples, 3, dtype=torch.bool),
        num_graphs=samples,
    )


def _bridge(**overrides) -> LeftNetBridge:
    parameters = dict(
        hidden_channels=32,
        num_layers=2,
        num_radial=16,
        cutoff=10.0,
        num_elements=5,
    )
    parameters.update(overrides)
    torch.manual_seed(0)
    return LeftNetBridge(**parameters).eval()


def test_adapter_emits_only_the_scaled_residual_for_the_ts_block() -> None:
    model = _bridge()
    batch = _bridge_batch()
    out = model(batch)
    assert set(out) == {"scaled"}
    assert out["scaled"].shape == batch.pos.shape
    assert torch.isfinite(out["scaled"]).all()
    # No "velocity" key: this task has no flow field, and the loss selector keys
    # off the presence of "scaled".
    assert "velocity" not in out


def test_adapter_removes_the_per_sample_mean() -> None:
    model = _bridge()
    batch = _bridge_batch()
    scaled = model(batch)["scaled"]
    for sample in range(batch.num_graphs):
        block = scaled[batch.batch == sample]
        assert torch.allclose(block.mean(dim=0), torch.zeros(3), atol=1e-5)


def test_adapter_is_rotation_equivariant() -> None:
    model = _bridge()
    batch = _bridge_batch()
    rotation = _random_rotation(3)
    reference = model(batch)["scaled"]
    rotated = _bridge_batch()
    rotated.reactant_pos = rotated.reactant_pos @ rotation.T
    rotated.pos = rotated.pos @ rotation.T
    rotated.endpoint_pos = rotated.endpoint_pos @ rotation.T
    assert torch.allclose(model(rotated)["scaled"], reference @ rotation.T, atol=1e-4)


def test_adapter_rejects_unsupported_elements() -> None:
    model = _bridge()
    batch = _bridge_batch()
    batch.z = batch.z.clone()
    batch.z[0] = 16  # sulfur is outside the reference vocabulary
    with pytest.raises(ValueError, match="unsupported"):
        model(batch)


def test_adapter_requires_the_product_endpoint() -> None:
    model = _bridge()
    batch = _bridge_batch()
    batch.endpoint_pos = None
    with pytest.raises(ValueError, match="endpoint_pos"):
        model(batch)


def test_factory_registers_the_leftnet_bridge_backend() -> None:
    spec_backend, parameters = model_spec({"backend": "leftnet_sb"})
    assert spec_backend == "leftnet_sb"
    assert parameters["hidden_channels"] == 196
    assert parameters["num_layers"] == 6
    assert parameters["num_radial"] == 96
    assert parameters["cutoff"] == 10.0
    assert parameters["num_elements"] == 5
    assert parameters["object_aware"] is True

    model = build_model("leftnet_sb", {"hidden_channels": 32, "num_layers": 2, "num_radial": 16})
    assert isinstance(model, LeftNetBridge)
    assert model.cutoff == 10.0
    assert model.builds_own_graph is True
    assert model.bridge is True


def test_factory_reads_reactot_style_flags_from_strings() -> None:
    _, parameters = model_spec(
        {
            "backend": "leftnet_sb",
            "object_aware": "true",
            "legacy": "False",
            "single_layer_output": "1",
            "reflect_equiv": "on",
        }
    )
    assert parameters["object_aware"] is True
    assert parameters["legacy"] is False
    assert parameters["single_layer_output"] is True
    assert parameters["reflect_equiv"] is True
