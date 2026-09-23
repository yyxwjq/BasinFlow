"""One data-source mode: an event directory plus its ``split.csv``.

There is no side table and no manifest. An event file carries its own identity
(the integer in its name) and its own metadata, and the train/val/test assignment
lives in ``split.csv`` inside the events directory -- either written there by
``auto_split = true`` or supplied by the caller.
"""
import configparser
import json
import subprocess
import sys

import pytest
from ase import Atoms
from ase.io import write

from basinflow.data import (
    EventSplit,
    load_events,
    load_groups,
    load_split,
    resolve_split,
    save_split,
    split_events,
    split_events_by_group,
)
from basinflow.workflows.config import read_config
from basinflow.workflows.semantic import REQUIRED as _SEMANTIC_REQUIRED
from basinflow.workflows.transition1x import REQUIRED as _TRANSITION1X_REQUIRED

# Each workflow declares the config sections it requires; the example scripts and
# the CLI both read configs through this one validator.
REQUIRED_BY_SCRIPT = {
    'benchmark_semantic_flow': _SEMANTIC_REQUIRED,
    'train_transition1x_product_flow': _TRANSITION1X_REQUIRED,
}


def _events(path, reactions):
    """One events directory holding one two-frame event per reaction.

    ``reactions`` names the source reactions; event ids are the integers ``0..N-1``
    in that order, exactly as the converters emit them.
    """
    path.mkdir()
    for index, reaction in enumerate(reactions):
        frames = [
            Atoms('HO', positions=[[0, 0, 0], [1.0 + 0.1 * index, 0, 0]]),
            Atoms('HO', positions=[[0, 0, 0], [1.1 + 0.1 * index, 0, 0]]),
        ]
        for role, frame in zip(('reactant', 'product'), frames):
            frame.info.update(
                source_dataset='Transition1x',
                rxn=reaction,
                dataset_kind='transition1x_single_event_molecule',
                role=role,
            )
        write(path / f'event_{index}.traj', frames)
    return path


def test_a_split_csv_next_to_the_events_defines_the_split(tmp_path):
    events = _events(tmp_path / 'events', ['rxn-a', 'rxn-b', 'rxn-c'])
    save_split(EventSplit(('0',), ('1',), ('2',)), events / 'split.csv')

    catalog = load_events(events)
    split, sources = resolve_split(events, catalog, auto=False)

    assert catalog.event_ids == ['0', '1', '2']
    assert split.train_ids == ('0',)
    assert split.val_ids == ('1',)
    assert split.test_ids == ('2',)
    assert sources['events_dir'] == str(events)
    assert sources['split_file'] == str(events / 'split.csv')
    assert sources['split_mode'] == 'loaded'


def test_auto_split_writes_split_csv_and_reuses_it(tmp_path):
    events = _events(tmp_path / 'events', ['r1', 'r2', 'r3', 'r4'])
    catalog = load_events(events)

    split, sources = resolve_split(
        events, catalog, auto=True, train=0.5, val=0.25, test=0.25, seed=7
    )

    assert sources['split_mode'] == 'created'
    assert (events / 'split.csv').is_file()
    assigned = sorted(split.ids('all'), key=int)
    assert assigned == catalog.event_ids
    assert len(set(assigned)) == len(assigned)

    # A rerun must not silently change which events are held out.
    again, sources_again = resolve_split(events, catalog, auto=True, seed=999)
    assert sources_again['split_mode'] == 'reused'
    assert again == split

    # Regenerating is explicit and needs the caller to ask for it.
    regenerated, sources_regenerated = resolve_split(
        events, catalog, auto=True, train=0.5, val=0.25, test=0.25, seed=999, regenerate=True
    )
    assert sources_regenerated['split_mode'] == 'regenerated'
    assert regenerated != split


def test_missing_split_with_auto_split_off_uses_every_event_for_training(tmp_path):
    events = _events(tmp_path / 'events', ['r1', 'r2', 'r3'])
    catalog = load_events(events)

    split, sources = resolve_split(events, catalog, auto=False)

    assert sources['split_file'] is None
    assert sources['split_mode'] == 'all_train'
    assert split.val_ids == () and split.test_ids == ()
    assert sorted(split.train_ids, key=int) == catalog.event_ids
    assert not (events / 'split.csv').exists()


def test_split_csv_names_events_by_integer_and_rejects_anything_else(tmp_path):
    events = _events(tmp_path / 'events', ['r1', 'r2'])
    catalog = load_events(events)
    (events / 'split.csv').write_text('train,val,test\n0,,\nrxn-b,,\n')

    with pytest.raises(ValueError, match='integer'):
        load_split(events / 'split.csv', catalog)


def test_split_csv_rejects_unknown_ids_and_overlap(tmp_path):
    events = _events(tmp_path / 'events', ['r1', 'r2'])
    catalog = load_events(events)

    (events / 'split.csv').write_text('train,val,test\n0,,\n9,,\n')
    with pytest.raises(ValueError, match='unknown event ids'):
        load_split(events / 'split.csv', catalog)

    (events / 'split.csv').write_text('train,val,test\n0,0,\n')
    with pytest.raises(ValueError, match='overlap'):
        load_split(events / 'split.csv', catalog)


def test_events_left_out_of_the_split_file_are_simply_unassigned(tmp_path):
    """The Au dataset's split covers 13 of its 24 events; the rest are not scored."""
    events = _events(tmp_path / 'events', ['r1', 'r2', 'r3'])
    catalog = load_events(events)
    (events / 'split.csv').write_text('train,val,test\n0,,\n')

    split, sources = resolve_split(events, catalog, auto=False)

    assert split.train_ids == ('0',)
    assert sorted(split.ids('all'), key=int) == ['0']
    assert sources['split_mode'] == 'loaded'
    assert '2' not in split.ids('all')


def test_empty_partition_is_legal_data_but_cannot_be_selected(tmp_path):
    """An empty test partition is legal data; selecting it must raise, not stay silent."""
    events = _events(tmp_path / 'events', ['r1', 'r2'])
    save_split(EventSplit(('0',), ('1',), ()), events / 'split.csv')
    catalog = load_events(events)
    split, _ = resolve_split(events, catalog, auto=False)

    assert split.test_ids == ()
    with pytest.raises(ValueError, match='test event selection is empty'):
        split.select(catalog, 'test')


def test_group_split_keeps_a_reactions_events_together(tmp_path):
    """Two events of one reaction must never land in different partitions."""
    events = _events(tmp_path / 'events', ['r1', 'r1', 'r2', 'r2', 'r3', 'r3'])
    catalog = load_events(events)
    groups = {event_id: f'rxn{index // 2}' for index, event_id in enumerate(catalog.event_ids)}

    split = split_events_by_group(catalog, groups, train=0.34, val=0.33, test=0.33, seed=5)

    for members in ({'0', '1'}, {'2', '3'}, {'4', '5'}):
        placements = [
            name for name in ('train', 'val', 'test') if members & set(split.partition(name))
        ]
        assert len(placements) <= 1, f"{members} spans {placements}"


def test_split_events_is_deterministic_for_a_seed(tmp_path):
    events = _events(tmp_path / 'events', [f'r{i}' for i in range(10)])
    catalog = load_events(events)

    first = split_events(catalog, seed=11)
    second = split_events(catalog, seed=11)
    third = split_events(catalog, seed=12)

    assert first == second
    assert first != third


def test_config_rejects_unknown_keys_instead_of_running_a_missing_option(tmp_path):
    config = configparser.ConfigParser()
    config.read('configs/0910/smoke_transition1x_stable.ini')
    config['data']['active_threshold'] = '0.1'
    path = tmp_path / 'stale.ini'
    with path.open('w') as destination:
        config.write(destination)

    with pytest.raises(ValueError, match=r'unknown keys in \[data\]'):
        read_config(path, REQUIRED_BY_SCRIPT['train_transition1x_product_flow'])


def test_the_removed_manifest_option_is_rejected(tmp_path):
    """A config that still carries the deleted option must fail, not silently ignore it."""
    config = configparser.ConfigParser()
    config.read('configs/0910/smoke_transition1x_stable.ini')
    config['data']['manifest'] = '/external/split_manifest.json'
    path = tmp_path / 'stale.ini'
    with path.open('w') as destination:
        config.write(destination)

    with pytest.raises(ValueError, match=r'unknown keys in \[data\]'):
        read_config(path, REQUIRED_BY_SCRIPT['train_transition1x_product_flow'])


def test_data_config_does_not_require_the_automatic_split_section(tmp_path):
    for script, name in [('benchmark_semantic_flow', 'pt'),
                         ('train_transition1x_product_flow', 'transition1x')]:
        config = configparser.ConfigParser()
        config.read(f'configs/0910/smoke_{name}_stable.ini')
        config.remove_section('split')
        config['data']['auto_split'] = 'false'
        path = tmp_path / f'{name}.ini'
        with path.open('w') as destination:
            config.write(destination)
        loaded = read_config(path, REQUIRED_BY_SCRIPT[script])
        assert loaded['data']['auto_split'] == 'false'


@pytest.mark.parametrize('script,system', [
    ('benchmark_semantic_flow', 'pt'),
    ('train_transition1x_product_flow', 'transition1x'),
])
def test_single_directory_run_trains_validates_samples_and_refuses_overwrite(tmp_path, script, system):
    import torch

    config = configparser.ConfigParser()
    config.read(f'configs/0910/smoke_{system}_stable.ini')
    config.remove_section('split')
    events = _events(tmp_path / 'events', ['rxn-train', 'rxn-val', 'rxn-test'])
    save_split(EventSplit(('0',), ('1',), ('2',)), events / 'split.csv')
    config['data']['events_dir'] = str(events)
    config['data']['auto_split'] = 'false'
    if system == 'pt':
        config['data']['pbc_override'] = 'false,false,false'
    output = tmp_path / 'run'
    config['output']['output_dir'] = str(output)
    config['model'].update(num_features='8', num_layers='1', num_radial_basis='4')
    config['training'].update(epochs='2', max_steps='2', validate_each_epoch='true', save_each_epoch='true')
    config['runtime']['cpu_threads'] = '1'
    config['evaluation']['split'] = 'test'
    config['evaluation']['eval_num_steps'] = '8'
    config['evaluation']['trials_per_test_event' if system == 'pt' else 'trials_per_event'] = '1'
    if config.has_option('evaluation', 'num_test_events'):
        config.remove_option('evaluation', 'num_test_events')
    path = tmp_path / 'run.ini'
    with path.open('w') as destination:
        config.write(destination)
    command = [sys.executable, f'examples/{script}.py', '--config', str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    sources = json.loads((output / 'data_sources.json').read_text())
    assert sources['events_dir'] == str(events)
    assert sources['split_file'] == str(events / 'split.csv')
    history = json.loads((output / 'history.json').read_text())
    assert len(history) == 2
    assert history[-1]['validation_num_items'] == 1
    saved = torch.load(output / 'epoch_checkpoints/epoch_0002.pt', weights_only=False)
    assert saved['completed_epochs'] == 2
    assert saved['optimizer_state_dict']['state']
    assert (output / 'sampling/summary.json').exists()
    before = (output / 'training.log').read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert 'prior run' in repeated.stderr
    assert (output / 'training.log').read_bytes() == before


def test_training_scope_single_event_is_applied_end_to_end(tmp_path):
    """``training_scope = single_event`` must reach the scope filter, not a stale method."""
    config = configparser.ConfigParser()
    config.read('configs/0910/smoke_transition1x_stable.ini')
    config.remove_section('split')
    events = _events(tmp_path / 'events', ['rxn-train', 'rxn-val', 'rxn-test'])
    save_split(EventSplit(('0',), ('1',), ('2',)), events / 'split.csv')
    config['data'].update(events_dir=str(events), auto_split='false', training_scope='single_event')
    config['output']['output_dir'] = str(tmp_path / 'run')
    config['model'].update(num_features='8', num_layers='1', num_radial_basis='4')
    config['training'].update(epochs='1', max_steps='1')
    config['runtime']['cpu_threads'] = '1'
    config['evaluation'].update(split='val', eval_num_steps='8', trials_per_event='1')
    path = tmp_path / 'run.ini'
    with path.open('w') as destination:
        config.write(destination)

    from basinflow.workflows.transition1x import run_transition1x

    report = run_transition1x(path)

    # Both fixture products stay bonded, so the scope keeps every event and the
    # run records the scope it actually applied.
    assert report['training_scope'] == 'single_event'
    assert report['split']['train'] == 1
    sources = json.loads((tmp_path / 'run' / 'data_sources.json').read_text())
    assert sources['training_scope'] == 'single_event'


def test_group_table_round_trips_and_rejects_bad_rows(tmp_path):
    events = _events(tmp_path / 'events', ['rxn-a', 'rxn-a', 'rxn-b'])
    catalog = load_events(events)
    table = events / 'basin_table.csv'
    table.write_text('event_id,basin\n0,rxn-a\n1,rxn-a\n2,rxn-b\n')

    groups = load_groups(table, catalog)

    assert groups == {'0': 'rxn-a', '1': 'rxn-a', '2': 'rxn-b'}

    # A row for an event this catalog does not hold is skipped, so one table can
    # describe a whole dataset while a caller loads a single partition of it.
    table.write_text('event_id,basin\n0,rxn-a\n9,rxn-b\n')
    assert load_groups(table, catalog) == {'0': 'rxn-a'}

    table.write_text('event_id,basin\n0,rxn-a\n0,rxn-b\n')
    with pytest.raises(ValueError, match='twice'):
        load_groups(table, catalog)

    table.write_text('basin,event_id\nrxn-a,0\n')
    with pytest.raises(ValueError, match='header'):
        load_groups(table, catalog)

    with pytest.raises(FileNotFoundError):
        load_groups(events / 'missing.csv', catalog)


def test_the_group_table_is_not_a_training_input(tmp_path):
    """The train/val/test assignment comes from split.csv alone."""
    events = _events(tmp_path / 'events', ['rxn-a', 'rxn-a', 'rxn-b', 'rxn-c'])
    save_split(EventSplit(('0', '1'), ('2',), ('3',)), events / 'split.csv')
    (events / 'basin_table.csv').write_text('event_id,basin\n0,rxn-a\n1,rxn-a\n2,rxn-b\n3,rxn-c\n')
    catalog = load_events(events)

    split, sources = resolve_split(events, catalog, auto=False)

    assert split.train_ids == ('0', '1')
    assert sources['split_file'] == str(events / 'split.csv')
