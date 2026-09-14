"""Explicit dataset sources shared by training and saved-result evaluation."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from basinflow.data.catalog import BasinSplit, EventCatalog


def _namespaced(catalog, partition):
    structure_ids = {identifier: f'{partition}:{identifier}' for identifier in catalog.structures}
    event_ids = {identifier: f'{partition}:{identifier}' for identifier in catalog.events}
    return EventCatalog(
        structures={structure_ids[identifier]: replace(record, structure_id=structure_ids[identifier])
                    for identifier, record in catalog.structures.items()},
        events={event_ids[identifier]: replace(
            record, event_id=event_ids[identifier],
            reactant_structure_id=structure_ids[record.reactant_structure_id],
            product_structure_id=structure_ids[record.product_structure_id],
            transition_state_structure_id=structure_ids.get(record.transition_state_structure_id),
            metadata={**record.metadata, 'original_event_id': identifier, 'source_partition': partition},
        ) for identifier, record in catalog.events.items()},
        basins={identifier: replace(
            record, reactant_structure_id=structure_ids[record.reactant_structure_id],
            known_event_ids=[event_ids[event_id] for event_id in record.known_event_ids],
        ) for identifier, record in catalog.basins.items()},
    )


def load_data_partitions(data, split_config, *, pbc_override=None):
    """Return catalog, explicit basin membership, and source provenance.

    External directories preserve basin identifiers across partitions; reused
    local event filenames receive namespaces. Reaction metadata, when provided,
    is checked across partitions independently of those namespaces.
    """
    directories = {partition: str(data.get(f'{partition}_events_dir', '')).strip()
                   for partition in ('train', 'val', 'test')}
    combined = str(data.get('events_dir', '')).strip()
    manifest = str(split_config.get('manifest', '')).strip()
    seed = int(split_config.get('seed', 42))
    if any(directories.values()):
        if combined or manifest:
            raise ValueError('external directories and events_dir/manifest are mutually exclusive')
        if not directories['train']:
            raise ValueError('external directories require train_events_dir')
        catalogs = {}
        seen_basins = set()
        reaction_partitions = {}
        sources = {}
        for partition, directory in directories.items():
            if not directory:
                catalogs[partition] = EventCatalog({}, {}, {})
                continue
            path = Path(directory).resolve()
            catalog = EventCatalog.from_eon_directory(path, pbc_override=pbc_override)
            overlap = seen_basins & set(catalog.basin_ids)
            if overlap:
                raise ValueError(f'cross-partition basin overlap: {sorted(overlap)}')
            seen_basins.update(catalog.basin_ids)
            for event in catalog.events.values():
                metadata = catalog.structures[event.reactant_structure_id].metadata
                reaction = metadata.get('rxn')
                if reaction is None:
                    continue
                identity = (str(metadata.get('source_dataset', '')), str(reaction))
                previous = reaction_partitions.get(identity)
                if previous is not None and previous != partition:
                    raise ValueError(f'cross-partition reaction overlap: {identity}')
                reaction_partitions[identity] = partition
            sources[partition] = {'events_dir': str(path), 'basin_ids': catalog.basin_ids,
                                  'original_event_ids': catalog.event_ids}
            catalogs[partition] = _namespaced(catalog, partition)
        catalog = EventCatalog(
            {key: value for part in catalogs.values() for key, value in part.structures.items()},
            {key: value for part in catalogs.values() for key, value in part.events.items()},
            {key: value for part in catalogs.values() for key, value in part.basins.items()},
        )
        split = BasinSplit(*(tuple(catalogs[partition].basin_ids) for partition in ('train', 'val', 'test')), seed)
        provenance = {'mode': 'external_directories', 'automatic_split': False, 'partitions': sources}
    else:
        if not combined:
            raise ValueError('specify events_dir or train_events_dir')
        path = Path(combined).resolve()
        catalog = EventCatalog.from_eon_directory(path, pbc_override=pbc_override)
        if manifest:
            manifest_path = Path(manifest).resolve()
            split = BasinSplit.load(manifest_path)
            provenance = {'mode': 'manifest', 'automatic_split': False,
                          'events_dir': str(path), 'manifest': str(manifest_path),
                          'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        else:
            if any(key not in split_config for key in ('train', 'val', 'test')):
                raise ValueError('automatic splitting requires explicit train/val/test fractions')
            split = BasinSplit.create(catalog, **{key: float(split_config[key]) for key in ('train', 'val', 'test')}, seed=seed)
            provenance = {'mode': 'automatic', 'automatic_split': True, 'events_dir': str(path)}
    split.validate(catalog)
    if not split.train_ids:
        raise ValueError('training partition must be nonempty')
    return catalog, split, provenance
