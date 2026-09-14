"""The R+P -> TS task: dataset contract and the endpoint-message ablation."""
import numpy as np
import pytest

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.loader import DataLoader

from basinflow.data.pyg import TransitionStateDataset
from basinflow.models import FlowLossWeights, flow_loss
from basinflow.models.factory import build_model, model_spec

REACTANT = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 3.0]]
PRODUCT = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1.5, 0.0]]
STATE = [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [0.0, 0.7, 0.0]]


def _catalog(*, with_state: bool = True):
    structures = {
        "r": StructureRecord("r", ["H", "H", "H"], REACTANT, cell=np.eye(3) * 15.0, pbc=False),
        "p": StructureRecord("p", ["H", "H", "H"], PRODUCT, cell=np.eye(3) * 15.0, pbc=False),
    }
    if with_state:
        structures["ts"] = StructureRecord("ts", ["H", "H", "H"], STATE, cell=np.eye(3) * 15.0, pbc=False)
    events = {"e": EventRecord("e", "r", "p", "b", "ts" if with_state else None)}
    return EventCatalog(structures, events, {"b": BasinRecord("b", "r", ["e"])})


def test_transition_state_dataset_conditions_on_both_endpoints():
    dataset = TransitionStateDataset(_catalog(), flow_time=0.0)

    sample = dataset[0]

    assert torch.allclose(sample.endpoint_pos, torch.tensor(PRODUCT, dtype=torch.float32))
    assert torch.allclose(sample.reactant_pos, torch.tensor(REACTANT, dtype=torch.float32))
    assert torch.allclose(sample.target_pos, torch.tensor(STATE, dtype=torch.float32))
    # The source is the midpoint bridge, which is ReactOT's prior.
    assert torch.allclose(sample.pos, torch.tensor(0.5 * (np.array(REACTANT) + np.array(PRODUCT)), dtype=torch.float32))
    assert torch.allclose(sample.source_pos, sample.pos)
    assert torch.allclose(sample.target_velocity, sample.target_pos - sample.source_pos)
    assert bool(sample.movable_mask.all())


def test_transition_state_dataset_requires_a_transition_state():
    catalog = _catalog(with_state=False)
    with pytest.raises(ValueError, match="no transition state"):
        TransitionStateDataset(catalog, flow_time=0.0)


def test_transition_state_dataset_rejects_a_negative_source_scale():
    with pytest.raises(ValueError, match="source_scale"):
        TransitionStateDataset(_catalog(), flow_time=0.0, source_scale=-1.0)


def _batch():
    dataset = TransitionStateDataset(_catalog(), flow_time=0.0)
    return next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))


def _model(**overrides):
    config = {
        "backend": "painn", "num_features": 8, "num_layers": 1, "num_radial_basis": 6,
        "r_max": 12.0, "stability_mode": "bounded", "endpoint_condition": "true",
        **overrides,
    }
    backend, parameters = model_spec(config)
    return build_model(backend, parameters)


def test_endpoint_channel_is_opt_in_and_ablates_cleanly():
    batch = _batch()
    invariant = _model(endpoint_equivariant="false")
    equivariant = _model(endpoint_equivariant="true")
    plain = _model(endpoint_condition="false", endpoint_equivariant="false")

    assert plain.endpoint_condition is False
    assert plain.backbone.messages[0].endpoint_filter is None
    assert invariant.backbone.messages[0].endpoint_filter is not None
    assert invariant.backbone.messages[0].endpoint_vector_filter is None
    assert equivariant.backbone.messages[0].endpoint_vector_filter is not None

    weights = FlowLossWeights(velocity=1.0, active=0.0, direction=0.0, active_pos_weight=1.0, norm="l1")
    outputs = {}
    for name, model in (("plain", plain), ("invariant", invariant), ("equivariant", equivariant)):
        out = model(batch)
        assert torch.isfinite(out["velocity"]).all()
        loss, _ = flow_loss(out, batch, weights)
        assert torch.isfinite(loss)
        outputs[name] = out["velocity"]

    # The two endpoint arms differ, so the ablation is not a no-op.
    assert not torch.allclose(outputs["invariant"], outputs["equivariant"])


def test_endpoint_condition_must_be_enabled_for_the_equivariant_flag():
    with pytest.raises(ValueError, match="requires endpoint_condition"):
        _model(endpoint_condition="false", endpoint_equivariant="true")


def test_missing_endpoint_tensor_is_reported():
    model = _model(endpoint_equivariant="true")
    batch = _batch()
    del batch.endpoint_pos
    with pytest.raises(ValueError, match="endpoint_condition is enabled"):
        model(batch)


def test_molgen_style_time_sampling_is_available_and_reproducible():
    from basinflow.data.pyg import TransitionStateDataset

    fixed = TransitionStateDataset(_catalog(), flow_time=0.25)
    assert fixed[0].flow_time.tolist() == [0.25]
    assert fixed._flow_time(0) == 0.25

    beta = TransitionStateDataset(_catalog(), seed=7, flow_time=None, time_distribution="beta",
                                  beta_alpha=0.8)
    times = [beta._flow_time(index) for index in range(200)]
    assert all(0.0 <= value <= 1.0 for value in times)
    assert len(set(times)) > 1
    assert times == [TransitionStateDataset(_catalog(), seed=7, time_distribution="beta",
                                            beta_alpha=0.8)._flow_time(index) for index in range(200)]

    with pytest.raises(ValueError, match="time_distribution"):
        TransitionStateDataset(_catalog(), time_distribution="cosine")
    with pytest.raises(ValueError, match="beta_alpha"):
        TransitionStateDataset(_catalog(), time_distribution="beta", beta_alpha=0.0)


# --- Flow-matching degeneracy contract -------------------------------------
#
# A pinned time plus a deterministic source is direct regression, not flow
# matching: the field is supervised at one point and one sampler step is the
# whole model. These tests pin the two properties that make the TS task a flow.

def test_pinned_time_cannot_be_combined_with_a_sampled_distribution():
    from basinflow.data.pyg import TransitionStateDataset

    with pytest.raises(ValueError, match="cannot be combined"):
        TransitionStateDataset(_catalog(), flow_time=0.0, time_distribution="beta")
    with pytest.raises(ValueError, match="cannot be combined"):
        TransitionStateDataset(_catalog(), flow_time=0.0, time_distribution="uniform")
    # The deterministic bridge stays legal.
    assert TransitionStateDataset(_catalog(), flow_time=0.0).time_distribution == "fixed"


def test_sampled_source_and_time_make_the_supervision_a_field():
    from basinflow.data.pyg import TransitionStateDataset

    dataset = TransitionStateDataset(_catalog(), flow_time=None, source_scale=1.0,
                                     time_distribution="beta", seed=3)
    midpoint = 0.5 * (np.asarray(REACTANT) + np.asarray(PRODUCT))

    first = dataset[0]
    # The source is no longer the midpoint: this is what stops x_t from
    # carrying the answer algebraically at t > 0.
    assert not np.allclose(first.source_pos.numpy(), midpoint)
    assert np.allclose(first.pos.numpy(),
                       (1.0 - float(first.flow_time)) * first.source_pos.numpy()
                       + float(first.flow_time) * np.asarray(STATE), atol=1e-5)
    # The regression target follows the source, not the midpoint.
    assert torch.allclose(first.target_velocity, first.target_pos - first.source_pos)

    # A new epoch must resample both the source and the time, otherwise the
    # network can memorise one displacement per event.
    dataset.set_epoch(1)
    second = dataset[0]
    assert not torch.allclose(first.source_pos, second.source_pos)
    assert not torch.allclose(first.pos, second.pos)
    assert not torch.allclose(first.target_velocity, second.target_velocity)

    # Same seed and epoch reproduce exactly; a different seed does not.
    same_epoch = TransitionStateDataset(_catalog(), flow_time=None, source_scale=1.0,
                                        time_distribution="beta", seed=3)
    same_epoch.set_epoch(1)
    assert torch.allclose(second.source_pos, same_epoch[0].source_pos)
    assert float(second.flow_time) == float(same_epoch[0].flow_time)
    other_seed = TransitionStateDataset(_catalog(), flow_time=None, source_scale=1.0,
                                        time_distribution="beta", seed=4)
    other_seed.set_epoch(1)
    assert not torch.allclose(second.source_pos, other_seed[0].source_pos)
    # Returning to epoch 0 restores the epoch-0 field exactly.
    dataset.set_epoch(0)
    assert torch.allclose(dataset[0].source_pos, first.source_pos)
    assert float(dataset[0].flow_time) == float(first.flow_time)


def test_field_is_non_trivial_across_the_path():
    """The trained quantity must vary with t, or one step is the whole model."""
    from basinflow.data.pyg import TransitionStateDataset

    dataset = TransitionStateDataset(_catalog(), flow_time=None, source_scale=0.0,
                                     time_distribution="uniform", seed=11)
    velocities, positions = [], []
    for epoch in range(60):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        velocities.append(sample.target_velocity.numpy().copy())
        positions.append(sample.pos.numpy().copy())
        assert 0.0 <= float(sample.flow_time) <= 1.0

    spread = np.concatenate([v.reshape(-1) for v in velocities])
    assert spread.std() > 0.0
    # Positions along the path are genuinely different, so the field is queried
    # at many points rather than at a single x0.
    assert len({tuple(np.round(p.reshape(-1), 6)) for p in positions}) > 1


class _TimeRecordingModel:
    """Non-rigid constant field, recording every time the sampler queries it at.

    The field varies per atom: a spatially constant one would integrate to a
    pure translation, which any Kabsch-aligned metric correctly reports as zero
    motion.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.times: list[float] = []

    def eval(self):
        return self

    def __call__(self, batch, t=None):
        if t is None:
            raise AssertionError("the sampler must pass an explicit time")
        self.times.append(float(np.asarray(t).reshape(-1)[0]))
        velocity = batch.pos.new_zeros(batch.pos.shape)
        # Atom i moves (i + 1) * scale along x: a shear, not a translation.
        velocity[:, 0] = self.scale * (torch.arange(batch.pos.shape[0]) + 1).to(batch.pos.dtype)
        return {"velocity": velocity}


def test_sampler_integrates_the_field_over_a_time_grid(tmp_path):
    """Sampling must be an ODE: N steps of dt, queried at t = 0, dt, 2dt..."""
    import csv

    from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd
    from basinflow.evaluation.transition_state import run_transition_state_trials

    model = _TimeRecordingModel()
    summary = run_transition_state_trials(
        model=model,
        catalog=_catalog(),
        output_dir=tmp_path,
        cutoff=5.0,
        num_steps=4,
        source_scale=0.0,
        trials_per_basin=1,
        sampling_seed=5,
    )

    assert summary["num_steps"] == 4
    assert summary["evaluation_protocol"] == "transition_state_flow_ode"
    # Four distinct query times, not one repeated t = 0.
    assert model.times == [0.0, 0.25, 0.5, 0.75]

    # Integrating the shear over [0, 1] in four steps displaces atom i by
    # (i + 1) A; a single query at t = 0 would displace it by a quarter of that.
    source = np.asarray(0.5 * (np.array(REACTANT) + np.array(PRODUCT)))
    n_atoms = len(source)
    displacement = np.zeros((n_atoms, 3))
    displacement[:, 0] = np.arange(1, n_atoms + 1)
    reference = StructureRecord("r", ["H"] * n_atoms, source, cell=np.eye(3) * 15.0, pbc=False)

    def moves(scale: float) -> StructureRecord:
        return StructureRecord("m", ["H"] * n_atoms, source + scale * displacement,
                               cell=np.eye(3) * 15.0, pbc=False)

    expected = kabsch_aligned_rmsd(moves(1.0), reference, reference)
    one_step = kabsch_aligned_rmsd(moves(0.25), reference, reference)

    rows = list(csv.DictReader((tmp_path / "sampling" / "trial_metrics.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["total_motion_angstrom"]) == pytest.approx(expected, rel=1e-4)
    # The two regimes are far apart, so this test cannot pass by accident.
    assert one_step < 0.4 * expected


def test_sampler_rejects_an_unknown_time_law(tmp_path):
    from basinflow.evaluation.transition_state import run_transition_state_trials

    with pytest.raises(ValueError, match="time_distribution"):
        run_transition_state_trials(
            model=_TimeRecordingModel(),
            catalog=_catalog(),
            output_dir=tmp_path,
            cutoff=5.0,
            num_steps=2,
            source_scale=0.0,
            time_distribution="cosine",
        )


def test_sampler_writes_every_trial_integration_path(tmp_path):
    """``write_trajectories`` must store x0 ... x_num_steps with the right species."""
    import csv

    from ase.io import read

    from basinflow.evaluation.transition_state import run_transition_state_trials

    model = _TimeRecordingModel()
    summary = run_transition_state_trials(
        model=model,
        catalog=_catalog(),
        output_dir=tmp_path,
        cutoff=5.0,
        num_steps=3,
        source_scale=0.0,
        trials_per_basin=2,
        sampling_seed=11,
        time_distribution="beta",
        write_trajectories=True,
    )

    written = summary["trajectories"]
    assert written["written"] is True
    assert written["format"] == "extxyz"
    assert written["frames_per_trial"] == 4

    rows = list(csv.DictReader((tmp_path / "sampling" / "trial_metrics.csv").open()))
    assert len(rows) == 2
    assert all(row["trajectory"] for row in rows)

    for row in rows:
        atoms = read(tmp_path / "sampling" / row["trajectory"], index=":")
        assert len(atoms) == 4
        assert all(frame.get_chemical_symbols() == ["H", "H", "H"] for frame in atoms)
        # Frame 0 is the drawn source; the field is a shear, so integrating in
        # equal steps moves atom i by (i + 1) * k / num_steps cumulatively.
        k = 1.0 / 3.0
        for step, frame in enumerate(atoms):
            np.testing.assert_allclose(frame.positions[1][0] - atoms[0].positions[1][0],
                                       2.0 * k * step, atol=1e-4)
            np.testing.assert_allclose(frame.positions[2][0] - atoms[0].positions[2][0],
                                       3.0 * k * step, atol=1e-4)


def test_trajectories_are_off_by_default_and_do_not_write_files(tmp_path):
    from basinflow.evaluation.transition_state import run_transition_state_trials

    summary = run_transition_state_trials(
        model=_TimeRecordingModel(),
        catalog=_catalog(),
        output_dir=tmp_path,
        cutoff=5.0,
        num_steps=2,
        source_scale=0.0,
        trials_per_basin=2,
        sampling_seed=3,
    )

    assert summary["trajectories"] == {
        "written": False, "directory": None, "frames_per_trial": None, "format": None,
    }
    assert not (tmp_path / "sampling" / "trajectories").exists()
