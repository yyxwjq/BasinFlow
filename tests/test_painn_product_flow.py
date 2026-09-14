import pytest

torch = pytest.importorskip("torch")


def _batch(periodic=False):
    reference = torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.1, 0.4], [0.2, 1.3, 0.6], [1.1, 1.2, 0.8]], dtype=torch.float64)
    return {
        "z": torch.tensor([1, 8, 6, 1]),
        "pos": reference + torch.tensor([[0.1, 0.0, 0.2], [0.0, 0.1, 0.0], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]]),
        "reactant_pos": reference,
        "movable_mask": torch.tensor([True, True, True, False]),
        "active_prior": torch.zeros(4, dtype=torch.float64),
        "cell": torch.tensor([[[3.1, 0.0, 0.0], [0.5, 3.3, 0.0], [0.2, 0.4, 3.5]]], dtype=torch.float64),
        "pbc": torch.full((1, 3), periodic),
        "batch": torch.zeros(4, dtype=torch.long),
        "flow_time": torch.tensor([[0.4]], dtype=torch.float64),
    }


def _model(**kwargs):
    from basinflow.models.painn import PaiNN

    torch.manual_seed(4)
    return PaiNN(num_features=16, num_layers=2, num_radial_basis=8, r_max=2.2, **kwargs).double()


@pytest.mark.parametrize("stability_mode", ["none", "scaled", "bounded"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("reflection", [False, True])
def test_painn_vector_equivariance(stability_mode, periodic, reflection):
    # `bounded` rescales the vector stream by a per-node scalar, so equivariance
    # must hold for every mode, not only the default one.
    model, batch = _model(stability_mode=stability_mode), _batch(periodic)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)).Q
    if reflection:
        rotation[:, 0] *= -1
    transformed = {key: value.clone() for key, value in batch.items()}
    for key in ("pos", "reactant_pos"):
        transformed[key] = batch[key] @ rotation.T + 2.3
    transformed["cell"] = batch["cell"] @ rotation.T
    original = model(batch)["velocity"]
    actual = model(transformed)["velocity"]
    torch.testing.assert_close(actual, original @ rotation.T, atol=1e-9, rtol=1e-8)
    assert original.shape == (4, 3)
    assert original[3].count_nonzero() == 0


@pytest.mark.parametrize("stability_mode", ["scaled", "bounded"])
def test_painn_bounded_mode_suppresses_vector_stream_growth(stability_mode):
    """The bounded mode must stop the quadratic vector-dot amplification.

    ``UpdateBlock`` forms an inner product of two vector streams, which is
    quadratic in their magnitude; a constant divisor cannot bound it. Driving
    the input far from the training manifold is what produced the Pt
    non-finite gradient, so the two modes are compared on the same stressed
    batch: the bounded stream must stay finite and far below the scaled one.
    """
    from basinflow.models import flow_loss

    model, batch = _model(stability_mode=stability_mode), _batch(False)
    stressed = {key: value.clone() for key, value in batch.items()}
    stressed["pos"] = batch["pos"] * 40.0
    stressed["reactant_pos"] = batch["reactant_pos"] * 40.0
    output = model(stressed)
    loss, _ = flow_loss(output, {**stressed, "target_velocity": torch.zeros_like(output["velocity"])})
    peak = output["velocity"].abs().max()
    assert torch.isfinite(output["velocity"]).all(), "bounded mode produced non-finite velocity"
    assert torch.isfinite(loss)
    if stability_mode == "bounded":
        assert peak < 1e3, f"bounded velocity peak {peak} is not controlled"


def test_time_is_read_and_seed_noise_is_not_a_persistent_condition():
    model, batch = _model(), _batch()
    original = model(batch)["velocity"]
    torch.testing.assert_close(original, model(batch, t=0.4)["velocity"])
    assert not torch.allclose(original, model(batch, t=0.9)["velocity"])
    batch["seed_displacement"] = torch.randn(4, 3)
    batch["seed_direction"] = torch.randn(4, 3)
    torch.testing.assert_close(original, model(batch)["velocity"])


def test_permutation_batching_and_periodic_image_invariance():
    model, batch = _model(), _batch(True)
    original = model(batch)["velocity"]
    permutation = torch.tensor([2, 0, 3, 1])
    atom_keys = ("z", "pos", "reactant_pos", "movable_mask", "active_prior", "batch")
    permuted = {key: value[permutation] if key in atom_keys else value for key, value in batch.items()}
    torch.testing.assert_close(model(permuted)["velocity"], original[permutation])
    shifted = {key: value.clone() for key, value in batch.items()}
    for key in ("pos", "reactant_pos"):
        shifted[key][1] += batch["cell"][0, 0] * 2 - batch["cell"][0, 1]
    torch.testing.assert_close(model(shifted)["velocity"], original, atol=1e-9, rtol=1e-8)
    other = _batch(False)
    combined = {key: torch.cat([batch[key], other[key]]) for key in batch}
    combined["batch"][4:] = 1
    actual = model(combined)["velocity"]
    torch.testing.assert_close(actual[:4], original)
    torch.testing.assert_close(actual[4:], model(other)["velocity"])


def test_backward_checkpoint_and_empty_edges(tmp_path):
    model, batch = _model(), _batch()
    batch["pos"].requires_grad_(True)
    velocity = model(batch)["velocity"]
    velocity.square().sum().backward()
    assert torch.isfinite(batch["pos"].grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(batch)["velocity"], velocity)
    batch["pos"] = batch["pos"].detach() * 100
    batch["reactant_pos"] = batch["reactant_pos"] * 100
    assert torch.isfinite(model(batch)["velocity"]).all()


@pytest.mark.parametrize("basis", ["bessel", "gaussian"])
@pytest.mark.parametrize("envelope", ["polynomial", "cosine"])
def test_radial_choices_and_cutoff_smoothness(basis, envelope):
    from basinflow.models.painn.layers import RadialBasis

    radial = RadialBasis(8, 3.0, basis, envelope).double()
    distances = torch.tensor([0.0, 1.0, 3.0, 3.1], dtype=torch.float64, requires_grad=True)
    features, cutoff = radial(distances)
    assert torch.isfinite(features).all()
    assert cutoff[0] == 1
    assert cutoff[2:].count_nonzero() == 0
    derivative = torch.autograd.grad(cutoff.sum(), distances)[0]
    assert derivative[2:].abs().max() < 1e-12
    assert torch.isfinite(_model(radial_basis=basis, envelope=envelope)(_batch())["velocity"]).all()


def test_liflow_tensor_api_matches_event_adapter():
    model, batch = _model(), _batch()
    from basinflow.models.painn.modules import prepare_flow_input

    prepared = prepare_flow_input(batch, model.r_max)
    direct = model.backbone(prepared)
    torch.testing.assert_close(model(batch)["velocity"], direct["velocity"] * batch["movable_mask"][:, None])
    assert {"positions_1", "positions_2", "elements", "edge_index", "shifts", "time"} <= prepared.keys()


def test_dense_intermediate_flow_geometry_has_finite_forward_and_backward():
    from basinflow.models.painn import PaiNN

    torch.manual_seed(42)
    model = PaiNN()
    count = 64
    reference = torch.randn(count, 3) * 0.2
    batch = {
        "z": torch.full((count,), 78),
        "pos": reference + torch.randn_like(reference) * 0.2,
        "reactant_pos": reference,
        "movable_mask": torch.ones(count, dtype=torch.bool),
        "active_prior": torch.ones(count),
        "cell": torch.zeros(1, 3, 3),
        "pbc": torch.zeros(1, 3, dtype=torch.bool),
        "flow_time": torch.tensor([0.5]),
        "batch": torch.zeros(count, dtype=torch.long),
    }
    velocity = model(batch)["velocity"]
    assert torch.isfinite(velocity).all()
    velocity.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_long_displacement_in_periodic_environment_has_finite_training_gradients():
    from ase.build import bulk
    from basinflow.models.painn import PaiNN

    torch.manual_seed(42)
    atoms = bulk("Pt", "fcc", a=3.92, cubic=True).repeat((2, 2, 2))
    reference = torch.as_tensor(atoms.positions, dtype=torch.float32)
    model = PaiNN(r_max=4.5)
    current = reference.clone()
    current[0] += torch.tensor([5.2, 0.0, 0.0])
    batch = {
        "z": torch.full((len(atoms),), 78),
        "pos": current,
        "reactant_pos": reference,
        "movable_mask": torch.ones(len(atoms), dtype=torch.bool),
        "active_prior": torch.ones(len(atoms)),
        "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float32)[None],
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "flow_time": torch.tensor([0.86]),
        "batch": torch.zeros(len(atoms), dtype=torch.long),
    }
    velocity = model(batch)["velocity"]
    velocity.square().mean().backward()
    assert torch.isfinite(velocity).all()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    assert torch.isfinite(norm)
