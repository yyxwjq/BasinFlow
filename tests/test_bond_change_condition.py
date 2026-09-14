"""The bond-change edge condition: labels, dataset views, and model consumption.

The condition is the discrete part of a reaction.  These tests pin the two
properties that make it admissible: it never carries a product coordinate, and
the basin-level view can only obtain it from the seed.
"""
import numpy as np
import pytest

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.geometry.bonds import (
    BOND_CHANGE_CLASSES,
    BROKEN,
    FORMED,
    UNCHANGED,
    bond_adjacency,
    bond_change_labels,
)
from basinflow.seeds import EventSeed, SeedContext, event_seed_from_context

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.loader import DataLoader

from basinflow.data.pyg import BasinDataset, EventFlowDataset
from basinflow.models.factory import model_spec
from basinflow.seeds import GaussianInit

# H-H bonding starts below 1.3 * (0.31 + 0.31) = 0.806 Å, so 0.74 Å is a bond
# and 2.5 Å is not.  Atom 0 keeps a bond throughout; its partner swaps from
# atom 1 to atom 2.
_REACTANT = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0.0, 0.0, 2.5]]
_PRODUCT = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.74, 0.0]]


def _catalog():
    structures = {
        "r": StructureRecord("r", ["H", "H", "H"], _REACTANT, cell=np.eye(3) * 12.0, pbc=False),
        "p": StructureRecord("p", ["H", "H", "H"], _PRODUCT, cell=np.eye(3) * 12.0, pbc=False),
    }
    events = {"e": EventRecord("e", "r", "p", "b")}
    basins = {"b": BasinRecord("b", "r", ["e"])}
    return EventCatalog(structures, events, basins)


def test_bond_change_labels_mark_the_swapped_partner():
    labels = bond_change_labels(
        bond_adjacency(_REACTANT, [1, 1, 1]),
        bond_adjacency(_PRODUCT, [1, 1, 1]),
    )

    assert labels.shape == (3, 3)
    assert labels[0, 1] == BROKEN and labels[1, 0] == BROKEN
    assert labels[0, 2] == FORMED and labels[2, 0] == FORMED
    assert labels[1, 2] == UNCHANGED
    assert np.all(np.diag(labels) == UNCHANGED)


def test_event_seed_validates_a_proposed_topology_change():
    n_atoms = 3
    zeros = np.zeros((n_atoms, 3))
    movable = np.ones(n_atoms, dtype=bool)

    with pytest.raises(ValueError, match="bond_change must have shape"):
        EventSeed("s", "t", zeros, zeros, movable, movable, bond_change=np.zeros((2, 2), dtype=np.int8))
    with pytest.raises(ValueError, match="labels must be integers"):
        EventSeed("s", "t", zeros, zeros, movable, movable, bond_change=np.full((3, 3), 3))
    # Atoms 0 and 1 are fixed; atom 2 is movable.
    partially_movable = np.array([False, False, True])
    with pytest.raises(ValueError, match="pair of fixed atoms"):
        proposed = np.zeros((3, 3), dtype=np.int8)
        proposed[0, 1] = proposed[1, 0] = FORMED
        EventSeed("s", "t", zeros, zeros, movable, partially_movable, bond_change=proposed)

    # A fixed atom may still change a bond with a movable partner.
    allowed = np.zeros((3, 3), dtype=np.int8)
    allowed[0, 2] = allowed[2, 0] = BROKEN
    seed = EventSeed("s", "t", zeros, zeros, movable, partially_movable, bond_change=allowed)
    assert seed.bond_change[0, 2] == BROKEN


def test_seed_factory_carries_the_proposed_change():
    catalog = _catalog()
    context = SeedContext("e", "b", catalog.structures["r"])
    labels = np.zeros((3, 3), dtype=np.int8)
    labels[0, 2] = labels[2, 0] = FORMED

    seed = event_seed_from_context(
        context, "s", "t", np.zeros((3, 3)), bond_change=labels
    )

    assert seed.bond_change.dtype == np.int8
    assert seed.bond_change[0, 2] == FORMED
    assert event_seed_from_context(context, "s", "t", np.zeros((3, 3))).bond_change is None


def test_supervised_view_emits_padded_bond_change_labels():
    dataset = EventFlowDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                               bond_change_condition=True)

    sample = dataset[0]

    assert sample.bond_change.shape == (1, 3, 3)
    assert sample.bond_change.dtype == torch.int8
    assert sample.bond_change[0, 0, 1].item() == BROKEN
    assert sample.bond_change[0, 0, 2].item() == FORMED
    assert not hasattr(sample, "bond_product")


def test_supervised_view_can_refuse_the_oracle_labels():
    dataset = EventFlowDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                               bond_change_condition=True, bond_change_source="seed")

    assert not dataset[0].bond_change.any()

    with pytest.raises(ValueError, match="bond_change_source"):
        EventFlowDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                         bond_change_condition=True, bond_change_source="product")


def test_basin_view_needs_the_seed_and_defaults_to_no_change():
    plain = BasinDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                         bond_change_condition=True)
    assert not plain[0].bond_change.any()
    assert plain[0].bond_change.shape == (1, 3, 3)

    disabled = BasinDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)])
    assert not hasattr(disabled[0], "bond_change")


class _ProposalInit:
    """Minimal initializer that proposes a topology change and no geometry."""

    seed_type = "stub"
    requires_target = False

    def __init__(self, labels):
        self.labels = labels

    def generate(self, context, seed_id=None):
        return event_seed_from_context(
            context,
            seed_id or "stub",
            self.seed_type,
            np.zeros((context.reactant.n_atoms, 3)),
            bond_change=self.labels,
        )


def test_basin_view_forwards_a_seed_proposal():
    labels = np.zeros((3, 3), dtype=np.int8)
    labels[0, 1] = labels[1, 0] = BROKEN
    dataset = BasinDataset(_catalog(), [_ProposalInit(labels)], bond_change_condition=True)

    assert dataset[0].bond_change[0, 0, 1].item() == BROKEN


def test_model_spec_reads_the_condition_flag_from_ini_strings():
    _, enabled = model_spec({"backend": "painn", "bond_change_condition": "true"})
    _, disabled = model_spec({"backend": "painn", "bond_change_condition": "false"})

    assert enabled["bond_change_condition"] is True
    assert disabled["bond_change_condition"] is False


def test_painn_widens_its_edge_filter_and_consumes_the_channel():
    from basinflow.models.painn import PaiNN

    dataset = EventFlowDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                               bond_change_condition=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    conditioned = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
                        bond_change_condition=True)
    plain = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0)

    assert conditioned.backbone.messages[0].reference_filter.in_features == 6 + BOND_CHANGE_CLASSES
    assert plain.backbone.messages[0].reference_filter.in_features == 6
    assert conditioned(batch)["velocity"].shape == (3, 3)
    assert conditioned.bond_change_condition is True
    assert plain.bond_change_condition is False


def test_conditioned_forward_requires_the_label_tensor():
    from basinflow.models.painn import PaiNN
    from basinflow.models.painn.modules import prepare_flow_input

    dataset = EventFlowDataset(_catalog(), [GaussianInit(scale=0.05, random_seed=1)],
                               bond_change_condition=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    prepared = prepare_flow_input(batch, 12.0)
    assert "bond_change" in prepared

    stripped = batch.clone()
    del stripped.bond_change
    model = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
                  bond_change_condition=True)
    with pytest.raises(KeyError):
        model(stripped)


def test_oracle_sampler_arm_reads_the_recorded_topology():
    from basinflow.sampling import CandidateSampler
    from basinflow.models.painn import PaiNN

    catalog = _catalog()
    model = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
                  bond_change_condition=True)

    def sampler(source):
        return CandidateSampler(model=model, init_generators=[GaussianInit(scale=0.0, random_seed=1)],
                                num_steps=1, bond_change_source=source)

    oracle = sampler("oracle")._oracle_bond_change(catalog, catalog.basins["b"])
    assert oracle[0, 0, 1].item() == BROKEN
    assert oracle[0, 0, 2].item() == FORMED
    assert sampler("seed")._oracle_bond_change(catalog, catalog.basins["b"]) is None


def test_sampler_rejects_an_unknown_condition_source():
    from basinflow.sampling import CandidateSampler

    with pytest.raises(ValueError, match="bond_change_source"):
        CandidateSampler(model=None, init_generators=[GaussianInit(scale=0.0, random_seed=1)],
                         num_steps=1, bond_change_source="product")


def test_kabsch_aligned_rmsd_removes_a_rigid_motion_that_raw_rmsd_keeps():
    from basinflow.data.records import StructureRecord
    from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd, movable_mic_rmsd

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = positions @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T + 4.0
    reference = StructureRecord("r", ["H"] * 3, positions, cell=np.eye(3) * 20.0, pbc=False)
    moved = StructureRecord("m", ["H"] * 3, rotated, cell=np.eye(3) * 20.0, pbc=False)

    assert movable_mic_rmsd(moved, reference, reference) > 3.0
    assert kabsch_aligned_rmsd(moved, reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_fragment_count_and_preservation():
    from basinflow.geometry.bonds import fragment_count, preserves_fragments

    chain = np.zeros((3, 3), dtype=bool)
    chain[0, 1] = chain[1, 0] = True
    chain[1, 2] = chain[2, 1] = True
    split = np.zeros((3, 3), dtype=bool)
    split[0, 1] = split[1, 0] = True

    assert fragment_count(chain) == 1
    assert fragment_count(split) == 2
    assert fragment_count(np.zeros((3, 3), dtype=bool)) == 3
    assert preserves_fragments(chain, chain)
    assert not preserves_fragments(chain, split)
    assert preserves_fragments(split, split.copy())

    with pytest.raises(ValueError, match="square"):
        fragment_count(np.zeros((2, 3), dtype=bool))


def test_reactant_and_product_bond_graphs_of_the_swapped_partner():
    from basinflow.geometry.bonds import preserves_fragments

    reactant = bond_adjacency(_REACTANT, [1, 1, 1])
    product = bond_adjacency(_PRODUCT, [1, 1, 1])

    assert preserves_fragments(reactant, product)



def test_directional_arm_is_opt_in_and_keeps_older_state_dicts_loadable():
    from basinflow.models.painn import PaiNN
    from basinflow.models.factory import model_spec

    plain = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
                  bond_change_condition=True)
    directional = PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
                        bond_change_condition=True, bond_change_direction=True)

    # The plain conditioned arm must not gain parameters, or every checkpoint
    # written before the directional arm would stop loading.
    assert plain.backbone.condition_vector_input is None
    assert directional.backbone.condition_vector_input is not None
    assert set(plain.state_dict()) <= set(directional.state_dict())
    directional.load_state_dict(plain.state_dict(), strict=False)

    with pytest.raises(ValueError, match="requires bond_change_condition"):
        PaiNN(num_features=8, num_layers=1, num_radial_basis=6, r_max=12.0,
              bond_change_direction=True)

    dataset = EventFlowDataset(_catalog(), [GaussianInit(scale=0.0, random_seed=1)],
                               flow_time=0.0, bond_change_condition=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    assert directional(batch)["velocity"].shape == (3, 3)

    _, config = model_spec({"backend": "painn", "bond_change_direction": "true"})
    assert config["bond_change_direction"] is True
    assert model_spec({"backend": "painn"})[1]["bond_change_direction"] is False


def test_bond_change_direction_points_towards_a_forming_bond():
    from basinflow.models.painn.painn import _bond_change_direction

    reference = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    receiver = torch.tensor([0, 0])
    sender = torch.tensor([1, 2])
    one_hot = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 0-1 forms, 0-2 breaks

    direction = _bond_change_direction(reference, receiver, sender, one_hot)

    assert torch.allclose(direction[0], torch.tensor([1.0, -1.0, 0.0]))
    assert torch.allclose(direction[1], torch.tensor([-1.0, 0.0, 0.0]))
    assert torch.allclose(direction[2], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(direction.sum(0), torch.zeros(3), atol=1e-6)  # no net translation


def test_single_event_subset_drops_a_dissociating_product():
    from basinflow.data.catalog import EventCatalog
    from basinflow.data.records import BasinRecord, EventRecord, StructureRecord

    # A connected triangle is one fragment; splitting atom 1 off makes two.
    connected = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0.0, 0.74, 0.0]]
    rearranged = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [0.0, 0.74, 0.0]]
    dissociated = [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 0.74, 0.0]]
    structures = {
        "r": StructureRecord("r", ["H", "H", "H"], connected, cell=np.eye(3) * 20.0, pbc=False),
        "p": StructureRecord("p", ["H", "H", "H"], rearranged, cell=np.eye(3) * 20.0, pbc=False),
        "pd": StructureRecord("pd", ["H", "H", "H"], dissociated, cell=np.eye(3) * 20.0, pbc=False),
    }
    events = {"keep": EventRecord("keep", "r", "p", "b"), "drop": EventRecord("drop", "r", "pd", "bd")}
    basins = {"b": BasinRecord("b", "r", ["keep"]), "bd": BasinRecord("bd", "r", ["drop"])}
    catalog = EventCatalog(structures, events, basins)

    kept = catalog.single_event_subset()

    assert kept.basin_ids == ["b"]
    assert set(kept.events) == {"keep"}

