import configparser
import json
import subprocess
import sys

import pytest
from ase import Atoms
from ase.io import write

from basinflow.data.catalog import BasinSplit
from basinflow.workflows.config import read_config
from basinflow.workflows.semantic import REQUIRED as _SEMANTIC_REQUIRED
from basinflow.workflows.transition1x import REQUIRED as _TRANSITION1X_REQUIRED

# Each workflow declares the config sections it requires; the example scripts and
# the CLI both read configs through this one validator.
REQUIRED_BY_SCRIPT = {
    'benchmark_semantic_flow': _SEMANTIC_REQUIRED,
    'train_transition1x_product_flow': _TRANSITION1X_REQUIRED,
}


def _events(path, basin, source_index):
    path.mkdir()
    frames = [Atoms('HO', positions=[[0, 0, 0], [distance, 0, 0]]) for distance in (1.0, 1.1)]
    for frame in frames:
        frame.info.update(source_dataset='Transition1x', rxn=f'rxn-{source_index}')
    write(path / 'event_0.traj', frames)
    (path / 'basin_table.csv').write_text(f'file,basin,dataset_kind\nevent_0.traj,{basin},transition1x_single_event_molecule\n')
    return str(path)


def test_external_directories_preserve_membership_and_namespace_local_filenames(tmp_path):
    from basinflow.data.partitions import load_data_partitions

    data = {f'{partition}_events_dir': _events(tmp_path / partition, f'basin-{partition}', index)
            for index, partition in enumerate(('train', 'val', 'test'))}
    catalog, split, sources = load_data_partitions(data, {})
    assert split.train_ids == ('basin-train',)
    assert split.val_ids == ('basin-val',)
    assert split.test_ids == ('basin-test',)
    assert set(catalog.event_ids) == {'train:event_0', 'val:event_0', 'test:event_0'}
    assert catalog.events['test:event_0'].metadata['original_event_id'] == 'event_0'
    assert sources['mode'] == 'external_directories'
    assert sources['automatic_split'] is False


@pytest.mark.parametrize('duplicate_kind', ['basin', 'reaction'])
def test_external_directories_reject_cross_partition_identity_leakage(tmp_path, duplicate_kind):
    from basinflow.data.partitions import load_data_partitions

    data = {'train_events_dir': _events(tmp_path / 'train', 'same', 0),
            'val_events_dir': _events(tmp_path / 'val', 'same' if duplicate_kind == 'basin' else 'different',
                                     1 if duplicate_kind == 'basin' else 0)}
    with pytest.raises(ValueError, match='overlap'):
        load_data_partitions(data, {})


def test_manifest_mode_does_not_require_split_fractions(tmp_path):
    from basinflow.data.partitions import load_data_partitions

    events = _events(tmp_path / 'events', 'basin', 0)
    path = tmp_path / 'manifest.json'
    BasinSplit(('basin',), (), (), 19).save(path)
    catalog, split, sources = load_data_partitions({'events_dir': events}, {'manifest': str(path)})
    assert catalog.basin_ids == ['basin']
    assert split.seed == 19
    assert sources['mode'] == 'manifest'
    assert sources['manifest_sha256']


@pytest.mark.parametrize('extra', [{'events_dir': '/unused'}, {'manifest': '/unused'}])
def test_external_directories_reject_conflicting_data_modes(extra):
    from basinflow.data.partitions import load_data_partitions

    data = {'train_events_dir': '/unused'}
    split = {}
    (split if 'manifest' in extra else data).update(extra)
    with pytest.raises(ValueError, match='mutually exclusive'):
        load_data_partitions(data, split)


def test_explicit_data_config_readers_do_not_require_automatic_split_section(tmp_path):
    for script, name in [('benchmark_semantic_flow', 'pt'), ('train_transition1x_product_flow', 'transition1x')]:
        config = configparser.ConfigParser()
        config.read(f'configs/0910/smoke_{name}_stable.ini')
        config.remove_section('split')
        del config['data']['events_dir']
        config['data']['train_events_dir'] = '/external/train'
        config['data']['val_events_dir'] = '/external/val'
        config['data']['test_events_dir'] = '/external/test'
        path = tmp_path / f'{name}.ini'
        with path.open('w') as destination:
            config.write(destination)
        loaded = read_config(path, REQUIRED_BY_SCRIPT[script])
        assert loaded['data']['train_events_dir'] == '/external/train'


@pytest.mark.parametrize('script,system', [('benchmark_semantic_flow', 'pt'), ('train_transition1x_product_flow', 'transition1x')])
def test_external_directories_run_training_validation_sampling_and_refuse_overwrite(tmp_path, script, system):
    import torch

    config = configparser.ConfigParser()
    config.read(f'configs/0910/smoke_{system}_stable.ini')
    config.remove_section('split')
    del config['data']['events_dir']
    for index, partition in enumerate(('train', 'val', 'test')):
        config['data'][f'{partition}_events_dir'] = _events(tmp_path / partition, f'basin-{partition}', index)
    if system == 'pt':
        config['data']['pbc_override'] = 'false,false,false'
    output = tmp_path / 'run'
    config['output']['output_dir'] = str(output)
    config['model'].update(num_features='8', num_layers='1', num_radial_basis='4')
    config['training'].update(epochs='2', max_steps='2', validate_each_epoch='true', save_each_epoch='true')
    config['runtime']['cpu_threads'] = '1'
    config['evaluation']['split'] = 'test'
    config['evaluation']['eval_num_steps'] = '1'
    config['evaluation']['trials_per_test_basin' if system == 'pt' else 'trials_per_basin'] = '1'
    if config.has_option('evaluation', 'num_test_basins'):
        config.remove_option('evaluation', 'num_test_basins')
    path = tmp_path / 'run.ini'
    with path.open('w') as destination:
        config.write(destination)
    command = [sys.executable, f'examples/{script}.py', '--config', str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'split_manifest.json').read_text())['train'] == ['basin-train']
    assert json.loads((output / 'data_sources.json').read_text())['mode'] == 'external_directories'
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


@pytest.mark.parametrize('script,name', [('benchmark_semantic_flow', 'pt'), ('train_transition1x_product_flow', 'transition1x')])
def test_external_partition_training_validation_checkpoint_and_sampling(tmp_path, script, name):
    config = configparser.ConfigParser()
    config.read(f'configs/0910/smoke_{name}_stable.ini')
    config.remove_section('split')
    del config['data']['events_dir']
    for index, partition in enumerate(('train', 'val', 'test')):
        basin = f'transition1x:{index}' if name == 'transition1x' else f'{partition}-basin'
        config['data'][f'{partition}_events_dir'] = _events(tmp_path / partition, basin, index)
    if name == 'pt':
        config['data']['pbc_override'] = 'false,false,false'
    config['output']['output_dir'] = str(tmp_path / 'run')
    config['training'].update(epochs='1', max_steps='1', validate_each_epoch='true', save_each_epoch='true')
    config['model'].update(num_features='8', num_layers='1', num_radial_basis='4')
    config['evaluation']['split'] = 'test'
    config['evaluation']['eval_num_steps'] = '1'
    config['evaluation']['trials_per_test_basin' if name == 'pt' else 'trials_per_basin'] = '1'
    config['runtime']['cpu_threads'] = '1'
    path = tmp_path / 'run.ini'
    with path.open('w') as destination:
        config.write(destination)
    result = subprocess.run([sys.executable, f'examples/{script}.py', '--config', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    run = tmp_path / 'run'
    assert (run / 'checkpoint.pt').is_file()
    assert (run / 'epoch_checkpoints/epoch_0001.pt').is_file()
    assert (run / 'sampling/summary.json').is_file()
    history = json.loads((run / 'history.json').read_text())
    assert history[0]['validation_num_items'] == 1
    assert history[0]['validation_velocity_loss'] >= 0
    sources = json.loads((run / 'data_sources.json').read_text())
    assert sources['automatic_split'] is False
