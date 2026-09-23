import importlib.util
from pathlib import Path

import numpy as np
import pytest

from basinflow.data import EventSplit, Structure, load_events, resolve_split, save_split

SPEC = importlib.util.spec_from_file_location("summarize_benchmark", Path("tools/summarize_benchmark.py"))


def _module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def _structure(displacement):
    return Structure(["H", "H"], [[0, 0, 0], [2 + displacement, 0, 0]], movable_mask=[False, True])


def test_summary_reads_a_single_directory_run(tmp_path):
    import csv
    import configparser
    from ase import Atoms
    from ase.io import write

    events = tmp_path / "events"
    events.mkdir()
    for index in range(2):
        frames = [Atoms('HH', positions=[[0, 0, 0], [distance, 0, 0]])
                  for distance in (2. + 0.1 * index, 2.3 + 0.1 * index)]
        write(events / f'event_{index}.traj', frames)
    save_split(EventSplit(('0',), (), ('1',)), events / 'split.csv')

    catalog = load_events(events)
    split, _ = resolve_split(events, catalog, auto=False)
    run = tmp_path / 'run'
    (run / 'sampling').mkdir(parents=True)
    config = configparser.ConfigParser()
    config.read_dict({
        'data': {'events_dir': str(events)},
        'init': {'gaussian_scale': '.05'}, 'model': {'backend': 'painn'},
        'training': {'epochs': '1'}, 'evaluation': {'split': 'test'},
    })
    with (run / 'config.resolved.ini').open('w') as destination:
        config.write(destination)
    frame = catalog.events['1'].product.to_ase()
    frame.info.update(event_id='1', trial_index=0)
    write(run / 'sampling/generated_before_eam.traj', [frame])
    with (run / 'sampling/trial_metrics.csv').open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=['event_id', 'trial_index', 'init_random_seed'])
        writer.writeheader()
        writer.writerow({'event_id': '1', 'trial_index': 0, 'init_random_seed': 42})

    result = _module().summarize_run(run)

    assert result['methods']['model']['macro_nearest_product_rmsd_angstrom'] == pytest.approx(0.)
    assert result['data_sources']['events_dir'] == str(events)
    assert result['per_event'][0]['known_event_ids'] == ['1']


def test_recall_matches_each_event_independently_and_preserves_frozen_metrics():
    module = _module()
    reactant = _structure(0)
    products = [_structure(.2), _structure(.3)]
    result = module.score_candidates([_structure(.25)], products, reactant)
    assert result["geometric_event_recall"]["0.1"] == 1.0
    assert np.isclose(result["mean_nearest_product_rmsd_angstrom"], .05)
    assert result["max_frozen_atom_drift_angstrom"] == 0.0
    assert result["collision_fraction"] == 0.0


def test_duplicate_fraction_and_zero_baseline_are_geometric():
    module = _module()
    reactant = _structure(0)
    result = module.score_candidates([reactant, reactant], [_structure(.3)], reactant)
    assert result["duplicate_fraction"] == .5
    assert result["geometric_event_recall"]["0.1"] == 0.0
    assert result["geometric_event_recall"]["0.5"] == 1.0


def test_overlap_and_frozen_atom_motion_are_reported_separately():
    module = _module()
    reactant = _structure(0)
    candidate = Structure(["H", "H"], [[.2, 0, 0], [.21, 0, 0]], movable_mask=[False, True])
    result = module.score_candidates([candidate], [_structure(.3)], reactant)
    assert result["collision_fraction"] == 1.0
    assert np.isclose(result["max_frozen_atom_drift_angstrom"], .2)


def test_molecular_alignment_removes_rigid_motion_without_changing_raw_metrics():
    module = _module()
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    product = Structure(["C", "H", "N", "O"], positions)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    candidate = Structure(product.species, positions @ rotation + [4., 5., 6.])
    raw = module.score_candidates([candidate], [product], product)
    aligned = module.molecular_alignment_diagnostics([candidate], [product])
    assert raw["mean_nearest_product_rmsd_angstrom"] > 1.0
    assert "molecular_alignment_diagnostic" not in raw
    assert aligned["mean_nearest_aligned_rmsd_angstrom"] < 1e-12
    assert aligned["mean_nearest_pair_distance_mae_angstrom"] < 1e-12
    assert aligned["aligned_geometric_reference_recall"]["0.1"] == 1.0


def test_molecular_alignment_uses_proper_rotation_not_reflection():
    module = _module()
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    product = Structure(["C", "H", "N", "O"], positions)
    candidate = Structure(product.species, positions * [-1., 1., 1.])
    aligned = module.molecular_alignment_diagnostics([candidate], [product])
    assert aligned["mean_nearest_aligned_rmsd_angstrom"] > .5
    assert aligned["mean_nearest_pair_distance_mae_angstrom"] < 1e-12


def test_molecular_alignment_is_optimal_not_merely_proper():
    """A proper but non-optimal rotation must not pass.

    Swapping the two singular-vector factors of the Kabsch convention still
    yields a determinant +1 rotation, so it satisfies the proper-rotation test
    above while inflating the aligned RMSD by ~12% on non-rigid displacements.
    The returned value is therefore checked against an independent probe: no
    rotation sampled from SO(3) may beat it.
    """
    module = _module()
    candidate_positions = np.array([
        [0.0018452300362238614, 0.4481183062627048, -0.41120678304332636],
        [-1.3358877581359114, -0.6820061777575839, -1.4874698324946936],
        [0.09021540389615773, 2.0103228683318, -0.7383097778269945],
        [-0.9307123497299106, 0.7347630752777974, 0.5353305122400911],
        [0.15812137349684785, -1.395702067062307, -0.043877733694910236],
        [1.0429547916874318, -2.016321820927623, -0.6864236415603273],
    ])
    product_positions = np.array([
        [-2.851834109701266, -1.9343066096774642, -2.7626025566875985],
        [-0.3526366966120219, -1.9011697221655548, 0.4068965382325523],
        [0.23512662993633773, -0.28039641694493156, -3.7751395662307696],
        [-0.8080393437699549, -0.07275141810160798, 0.16996347900496134],
        [-2.29520364825809, -0.7166299140508959, -1.4677786170849592],
        [-1.2132558591383988, 1.591347935079118, -1.2113020129978447],
    ])
    species = ["C"] * 6
    candidate = Structure(species, candidate_positions)
    product = Structure(species, product_positions)
    value = module.molecular_alignment_diagnostics([candidate], [product])[
        "mean_nearest_aligned_rmsd_angstrom"]

    source = candidate_positions - candidate_positions.mean(axis=0)
    target = product_positions - product_positions.mean(axis=0)

    sampled, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(5000, 3, 3)))
    sampled *= np.sign(np.linalg.det(sampled))[:, None, None]
    probe = float(np.min(np.sqrt(np.mean(np.sum((source @ sampled - target) ** 2, axis=2), axis=1))))
    assert value <= probe

    left, _, right = np.linalg.svd(target.T @ source)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    wrong = np.sqrt(np.mean(np.sum(
        (source @ (right @ correction @ left) - target) ** 2, axis=1)))
    assert value < wrong - .1
    assert probe < wrong - .1


def test_gaussian_baseline_reconstructs_sampler_initial_state():
    import pytest
    pytest.importorskip("torch_geometric")
    from basinflow.data import EventProposalDataset, Event, EventCatalog
    from basinflow.seeds import GaussianInit

    module = _module()
    reactant = _structure(0)
    catalog = EventCatalog(
        events={"0": Event("0", reactant, _structure(0.3))},
    )
    baseline = module.gaussian_baseline(reactant, "0", random_seed=42, scale=.05)
    batch = EventProposalDataset(catalog, [GaussianInit(scale=.05, random_seed=42)])[0]
    np.testing.assert_allclose(baseline.positions, batch.pos.numpy(), atol=1e-7)


def test_saved_run_pairs_artifacts_and_enforces_heldout_split(tmp_path):
    import configparser
    import csv
    import json
    from ase.io import write
    import pytest
    from test_stage3_au_scripts import _write_tiny_events

    module = _module()
    events = tmp_path / "events"
    events.mkdir()
    _write_tiny_events(events)
    run_dir = tmp_path / "run"
    sampling = run_dir / "sampling"
    sampling.mkdir(parents=True)
    config = configparser.ConfigParser()
    config.read_dict({"data": {"events_dir": str(events)}, "init": {"gaussian_scale": ".05"}, "model": {"backend": "painn"}, "training": {"max_steps": "2"}, "evaluation": {"split": "test"}})
    with (run_dir / "config.resolved.ini").open("w") as destination:
        config.write(destination)
    catalog = load_events(events)
    frame = catalog.events["1"].product.to_ase()
    frame.info.update({"event_id": "1", "trial_index": 0})
    write(sampling / "generated_before_eam.traj", [frame])
    with (sampling / "trial_metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["event_id", "trial_index", "init_random_seed"])
        writer.writeheader()
        writer.writerow({"event_id": "1", "trial_index": 0, "init_random_seed": 42})
    result = module.summarize_run(run_dir)
    assert result["evaluation_scope"] == "heldout_single_event_molecular_subset"
    assert result["evaluated_pbc_patterns"] == [[False, False, False]]
    assert result["methods"]["model"]["event_weighted_geometric_recall"]["0.1"] == 1.0
    assert result["paired_trials"][0]["model"] == 0.0
    assert result["paired_comparisons"]["zero_motion"]["strict_improvement_fraction"] == 1.0
    # The fixture events are Transition1x-tagged, so the molecular alignment
    # diagnostic is reported alongside the raw metrics for every method.
    assert all(
        "molecular_alignment_diagnostic" in metrics for metrics in result["methods"].values()
    )
    # The tag lives in each frame's info, not in a side table, and it is read per
    # event: dropping it turns the molecular diagnostic off while leaving the
    # candidate budget and the event count untouched.
    import ase.io

    for path in sorted(events.glob("event_*")):
        atoms = ase.io.read(path, index=":")
        for frame in atoms:
            frame.info.pop("dataset_kind", None)
        ase.io.write(path, atoms)
    untagged_result = module.summarize_run(run_dir)
    for method in result["methods"]:
        assert "molecular_alignment_diagnostic" not in untagged_result["methods"][method]
        assert untagged_result["methods"][method]["num_candidates"] == result["methods"][method]["num_candidates"]
        assert untagged_result["methods"][method]["num_events"] == result["methods"][method]["num_events"]
    save_split(EventSplit(("1",), (), ("0",)), events / "split.csv")
    with pytest.raises(ValueError, match="outside evaluation split"):
        module.summarize_run(run_dir)
