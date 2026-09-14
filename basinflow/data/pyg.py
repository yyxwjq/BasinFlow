"""PyG training and inference views over :mod:`basinflow.data.catalog`."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Iterable

import numpy as np

from basinflow.geometry.bonds import bond_adjacency, bond_change_labels
from basinflow.data.catalog import EventCatalog
from basinflow.seeds.records import SeedContext
from basinflow.seeds import GaussianInit

try:  # Keep raw readers usable without optional model dependencies.
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - exercised in dependency-free installs
    torch = None
    Dataset = object
    Data = object


_SEED_TYPE_IDS = {"zero": 0, "gaussian_movable": 1, "directional": 2, "product_displacement": 3}


def _require_pyg() -> None:
    if torch is None:
        raise ImportError(
            "PyG event datasets require the optional 'models' dependencies: "
            "install basinflow[models]"
        )


def _seed_type_id(seed_type: str) -> int:
    return _SEED_TYPE_IDS.get(str(seed_type), len(_SEED_TYPE_IDS))


class EventData(Data):
    """One event-flow graph with explicit graph-level PBC fields."""

    _GRAPH_FIELDS = {"cell", "pbc", "flow_time", "seed_type_id"}

    def __init__(self, **kwargs) -> None:
        _require_pyg()
        super().__init__(**kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._GRAPH_FIELDS:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class _BaseEventDataset(Dataset):
    def __init__(self, catalog: EventCatalog, initializers: Iterable) -> None:
        _require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        self.catalog = catalog
        self.initializers = tuple(initializers)
        if not self.initializers:
            raise ValueError("at least one initializer is required")

    @staticmethod
    def _seed_context(event_id: str, basin_id: str, reactant) -> SeedContext:
        return SeedContext(event_id=str(event_id), basin_id=str(basin_id), reactant=reactant)

    @staticmethod
    def _base_data(context: SeedContext, seed, *, pos, source_pos, flow_time: float) -> EventData:
        dtype = torch.float32
        return EventData(
            z=torch.as_tensor(context.reactant.atomic_numbers, dtype=torch.long),
            pos=torch.as_tensor(pos, dtype=dtype),
            reactant_pos=torch.as_tensor(context.reactant.positions, dtype=dtype),
            source_pos=torch.as_tensor(source_pos, dtype=dtype),
            movable_mask=torch.as_tensor(seed.movable_mask, dtype=torch.bool),
            active_prior=torch.as_tensor(seed.active_prior, dtype=torch.bool),
            seed_displacement=torch.as_tensor(seed.seed_displacement, dtype=dtype),
            seed_direction=torch.as_tensor(seed.seed_direction, dtype=dtype),
            cell=torch.as_tensor(context.reactant.cell, dtype=dtype),
            pbc=torch.as_tensor(context.reactant.pbc, dtype=torch.bool),
            flow_time=torch.tensor([flow_time], dtype=dtype),
            seed_type_id=torch.tensor([_seed_type_id(seed.seed_type)], dtype=torch.long),
            seed_type=seed.seed_type,
            event_id=context.event_id,
            basin_id=context.basin_id,
            seed_id=seed.seed_id,
            seed_metadata_json=json.dumps(seed.metadata),
            num_nodes=context.reactant.n_atoms,
        )


class EventFlowDataset(_BaseEventDataset):
    """Supervised event-level R-to-P flow samples for PyG training."""

    def __init__(
        self,
        catalog: EventCatalog,
        initializers: Iterable,
        *,
        active_threshold: float = 0.1,
        flow_time: float | None = None,
        seed: int = 42,
        diagnostic_oracle: bool = False,
        time_distribution: str = "uniform",
        beta_alpha: float = 0.8,
        bond_change_condition: bool = False,
        bond_change_source: str = "oracle",
    ) -> None:
        super().__init__(catalog, initializers)
        if flow_time is not None and not 0.0 <= float(flow_time) <= 1.0:
            raise ValueError("flow_time must be in [0, 1]")
        self.active_threshold = float(active_threshold)
        self.fixed_flow_time = None if flow_time is None else float(flow_time)
        self.seed = int(seed)
        self.epoch = 0
        self.diagnostic_oracle = bool(diagnostic_oracle)
        if time_distribution not in {"uniform", "beta"}:
            raise ValueError("time_distribution must be 'uniform' or 'beta'")
        if beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be positive")
        self.time_distribution = time_distribution
        self.beta_alpha = float(beta_alpha)
        self.bond_change_condition = bool(bond_change_condition)
        if bond_change_source not in {"oracle", "seed"}:
            raise ValueError("bond_change_source must be 'oracle' or 'seed'")
        self.bond_change_source = bond_change_source
        self.n_max = max((self.catalog.structures[sid].n_atoms
                          for sid in self.catalog.structures), default=0)
        if any(getattr(initializer, "requires_target", False) for initializer in self.initializers):
            if not self.diagnostic_oracle:
                raise ValueError("target-based initializers require diagnostic_oracle=True")
        self._items = [
            (event_id, initializer_index)
            for event_id in self.catalog.event_ids
            for initializer_index in range(len(self.initializers))
        ]

    def _bond_change_labels(self, seed, target) -> np.ndarray:
        """Per-pair condition, as an explicit topology change rather than a product.

        ``seed`` wins when the seed proposes a change; otherwise ``oracle``
        falls back to the reactant/product difference, which is available only
        because this is the supervised pairwise view.  The basin-level view
        never takes this branch.
        """
        reactant = target.reactant
        if seed.bond_change is not None:
            labels = seed.bond_change
        elif self.bond_change_source == "seed":
            labels = np.zeros((reactant.n_atoms, reactant.n_atoms), dtype=np.int8)
        else:
            labels = bond_change_labels(
                bond_adjacency(reactant.positions, reactant.atomic_numbers,
                               reactant.cell, reactant.pbc),
                bond_adjacency(target.target_positions, reactant.atomic_numbers,
                               reactant.cell, reactant.pbc),
            ).astype(np.int8)
        return _mask_unmovable_pairs(labels, seed.movable_mask)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._items)

    def _flow_time(self, index: int) -> float:
        if self.fixed_flow_time is not None:
            return self.fixed_flow_time
        payload = f"{self.seed}:{self.epoch}:{index}".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        rng = np.random.default_rng(local_seed)
        if self.time_distribution == "beta":
            return float(rng.beta(self.beta_alpha, self.beta_alpha))
        return float(rng.random())

    def __getitem__(self, index: int) -> EventData:
        event_id, initializer_index = self._items[index]
        target = self.catalog.event_target(event_id, active_threshold=self.active_threshold)
        context = self._seed_context(event_id, target.event.basin_id, target.reactant)
        initializer = self.initializers[initializer_index]
        if isinstance(initializer, GaussianInit) and initializer.random_seed is None:
            initializer = replace(initializer, random_seed=self.seed)
        seed_id = (
            f"{event_id}:seed:{initializer_index}:{initializer.seed_type}"
            f":dataset:{self.seed}:epoch:{self.epoch}"
        )
        if getattr(initializer, "requires_target", False):
            seed = initializer.generate(
                context,
                seed_id=seed_id,
                target_displacement=target.displacement,
            )
        else:
            seed = initializer.generate(context, seed_id=seed_id)
        source_pos = seed.initial_positions(target.reactant.positions)
        time = self._flow_time(index)
        pos = (1.0 - time) * source_pos + time * target.target_positions
        target_velocity = target.target_positions - source_pos
        fixed = ~seed.movable_mask
        target_velocity[fixed] = 0.0
        target_direction = target.direction.copy()
        target_direction[fixed] = 0.0
        data = self._base_data(
            context,
            seed,
            pos=pos,
            source_pos=source_pos,
            flow_time=time,
        )
        data.target_pos = torch.as_tensor(target.target_positions, dtype=torch.float32)
        data.target_velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
        if self.bond_change_condition:
            labels = self._bond_change_labels(seed, target)
            padded = torch.zeros((1, self.n_max, self.n_max), dtype=torch.int8)
            padded[0, : labels.shape[0], : labels.shape[1]] = torch.as_tensor(labels, dtype=torch.int8)
            data.bond_change = padded
        data.target_active_mask = torch.as_tensor(target.active_mask, dtype=torch.bool)
        data.target_direction = torch.as_tensor(target_direction, dtype=torch.float32)
        return data


class TransitionStateDataset(Dataset):
    """R+P -> TS flow samples: both endpoints are conditioning inputs.

    ReactOT and MolGEN both condition on the reactant *and* the product, so the
    product frame is an input here rather than the leak it would be in the
    R -> P view.

    This is a flow-matching task, so the interpolant must be supervised over a
    *range* of times, not at a single point:

    ``x0 = (R + P) / 2 + sigma * xi``, ``x_t = (1 - t) * x0 + t * TS``,
    ``v = TS - x0``.

    ``source_scale = 0`` collapses the source to the deterministic midpoint
    bridge (ReactOT's prior) and ``time_distribution = "fixed"`` collapses the
    supervision to one point. Either degeneracy removes the field: with a fixed
    ``t`` the network only ever has to fit ``TS - midpoint`` as a function of
    the endpoints, which is direct regression, and a single sampler step is the
    whole model. MolGEN avoids both by drawing ``t ~ Beta(0.8, 0.8)`` over a
    ``randn * x0std`` source (``mdgen/transport/transport.py:262``), which is
    what makes its sampler stochastic and best-of-N meaningful.

    It does not go through :class:`_BaseEventDataset` because there is no seed:
    both endpoints are given by the task, so no initializer is involved.
    """
    def __init__(
        self,
        catalog: EventCatalog,
        *,
        flow_time: float | None = None,
        seed: int = 42,
        source_scale: float = 0.0,
        time_distribution: str = "fixed",
        beta_alpha: float = 0.8,
        endpoint_condition: bool = True,
        endpoint_equivariant: bool = True,
    ) -> None:
        _require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        self.catalog = catalog
        # The training loop reads this surface off any prepared dataset.
        self.initializers: tuple = ()
        self.diagnostic_oracle = False
        self.active_threshold = 0.0
        if flow_time is not None and not 0.0 <= float(flow_time) <= 1.0:
            raise ValueError("flow_time must be in [0, 1]")
        if source_scale < 0.0:
            raise ValueError("source_scale must be non-negative")
        if time_distribution not in {"fixed", "beta", "uniform"}:
            raise ValueError("time_distribution must be 'fixed', 'beta' or 'uniform'")
        if beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be positive")
        if flow_time is not None and time_distribution != "fixed":
            # Silently ignoring flow_time would reintroduce exactly the
            # single-point supervision this dataset exists to avoid.
            raise ValueError(
                "flow_time pins the supervision to one point and cannot be combined with "
                f"time_distribution={time_distribution!r}; pass flow_time=None instead"
            )
        self.fixed_flow_time = None if flow_time is None else float(flow_time)
        self.seed = int(seed)
        self.source_scale = float(source_scale)
        self.time_distribution = time_distribution
        self.beta_alpha = float(beta_alpha)
        self.endpoint_condition = bool(endpoint_condition)
        self.endpoint_equivariant = bool(endpoint_equivariant)
        self.epoch = 0
        missing = [event_id for event_id in catalog.event_ids
                   if catalog.events[event_id].transition_state_structure_id is None]
        if missing:
            raise ValueError(f"{len(missing)} events have no transition state, e.g. {missing[0]!r}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.catalog.event_ids)

    def _rng(self, index: int) -> np.random.Generator:
        """Deterministic per ``(seed, epoch, index)`` generator.

        Fresh noise and a fresh time per epoch are what make the field a field:
        a constant source would let the network memorise one displacement per
        event instead of learning how to transport from an arbitrary point.
        """
        return np.random.default_rng([self.seed, self.epoch, int(index)])

    def _flow_time(self, index: int, rng: np.random.Generator | None = None) -> float:
        """Sampled training time, or the pinned value when ``time_distribution`` is fixed.

        Training always routes through :meth:`__getitem__`, which shares one
        generator between the source noise and the time. The optional ``rng``
        keeps direct calls (tests, diagnostics) from drawing a different stream.
        """
        if self.time_distribution == "fixed":
            return 0.0 if self.fixed_flow_time is None else self.fixed_flow_time
        rng = self._rng(index) if rng is None else rng
        if self.time_distribution == "beta":
            return float(rng.beta(self.beta_alpha, self.beta_alpha))
        return float(rng.random())

    def __getitem__(self, index: int) -> EventData:
        event_id = self.catalog.event_ids[index]
        target = self.catalog.event_target(event_id)
        reactant, product, state = target.reactant, target.product, target.transition_state
        if reactant.n_atoms != state.n_atoms or product.n_atoms != state.n_atoms:
            raise ValueError("reactant, product and transition state must share an atom count")
        midpoint = 0.5 * (reactant.positions + target.target_positions)
        source = midpoint
        if self.source_scale or self.time_distribution != "fixed":
            # One generator per item so the geometry and the time stay
            # reproducible together for a given (seed, epoch, index).
            rng = self._rng(index)
            if self.source_scale:
                source = midpoint + rng.normal(scale=self.source_scale, size=midpoint.shape)
            time = self._flow_time(index, rng)
        else:
            time = self._flow_time(index)
        source = np.asarray(source, dtype=float)
        pos = (1.0 - time) * source + time * state.positions
        movable = np.asarray(reactant.movable_mask, dtype=bool)
        return EventData(
            z=torch.as_tensor(reactant.atomic_numbers, dtype=torch.long),
            pos=torch.as_tensor(pos, dtype=torch.float32),
            reactant_pos=torch.as_tensor(reactant.positions, dtype=torch.float32),
            endpoint_pos=torch.as_tensor(target.target_positions, dtype=torch.float32),
            source_pos=torch.as_tensor(source, dtype=torch.float32),
            movable_mask=torch.as_tensor(movable, dtype=torch.bool),
            active_prior=torch.zeros(reactant.n_atoms, dtype=torch.bool),
            seed_displacement=torch.zeros((reactant.n_atoms, 3), dtype=torch.float32),
            seed_direction=torch.zeros((reactant.n_atoms, 3), dtype=torch.float32),
            cell=torch.as_tensor(reactant.cell, dtype=torch.float32),
            pbc=torch.as_tensor(reactant.pbc, dtype=torch.bool),
            flow_time=torch.tensor([time], dtype=torch.float32),
            seed_type_id=torch.tensor([0], dtype=torch.long),
            seed_type="midpoint_bridge",
            event_id=event_id,
            basin_id=target.event.basin_id,
            seed_id=f"{event_id}:midpoint",
            seed_metadata_json=json.dumps({
                "bridge": "midpoint",
                "source_scale": self.source_scale,
                "time_distribution": self.time_distribution,
            }),
            target_pos=torch.as_tensor(state.positions, dtype=torch.float32),
            target_velocity=torch.as_tensor(state.positions - source, dtype=torch.float32),
            num_nodes=reactant.n_atoms,
        )


def _mask_unmovable_pairs(labels: np.ndarray, movable_mask) -> np.ndarray:
    """Drop labels on pairs where neither atom can move.

    A bond cannot change without relative motion, so those entries are not a
    legal proposal.  Enforcing it in one place keeps the supervised and the
    basin view identical on constrained structures.
    """
    frozen = ~np.asarray(movable_mask, dtype=bool)
    if not np.any(frozen):
        return labels
    labels = np.array(labels, copy=True)
    labels[np.ix_(frozen, frozen)] = 0
    return labels


class BasinDataset(_BaseEventDataset):
    """Reactant-only basin samples for target-free proposal inference."""

    def __init__(
        self,
        catalog: EventCatalog,
        initializers: Iterable,
        *,
        bond_change_condition: bool = False,
    ) -> None:
        super().__init__(catalog, initializers)
        if any(getattr(initializer, "requires_target", False) for initializer in self.initializers):
            raise ValueError("basin proposal initializers must not require target data")
        self.bond_change_condition = bool(bond_change_condition)
        self.n_max = max((self.catalog.structures[sid].n_atoms
                          for sid in self.catalog.structures), default=0)
        self._items = [
            (basin_id, initializer_index)
            for basin_id in self.catalog.basin_ids
            for initializer_index in range(len(self.initializers))
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> EventData:
        basin_id, initializer_index = self._items[index]
        basin = self.catalog.basins[basin_id]
        reactant = self.catalog.structures[basin.reactant_structure_id]
        initializer = self.initializers[initializer_index]
        context = self._seed_context(
            f"{basin_id}:proposal:{initializer_index}",
            basin_id,
            reactant,
        )
        seed = initializer.generate(
            context,
            seed_id=f"{basin_id}:seed:{initializer_index}:{initializer.seed_type}",
        )
        source_pos = seed.initial_positions(reactant.positions)
        data = self._base_data(
            context,
            seed,
            pos=source_pos,
            source_pos=source_pos,
            flow_time=0.0,
        )
        if self.bond_change_condition:
            # Target-free: the only admissible source is the seed's own
            # proposed topology change, and an absent proposal means "nothing
            # changes here" rather than "consult the product".
            labels = _mask_unmovable_pairs(
                np.zeros((reactant.n_atoms, reactant.n_atoms), dtype=np.int8)
                if seed.bond_change is None
                else seed.bond_change,
                seed.movable_mask,
            )
            padded = torch.zeros((1, self.n_max, self.n_max), dtype=torch.int8)
            padded[0, : labels.shape[0], : labels.shape[1]] = torch.as_tensor(labels, dtype=torch.int8)
            data.bond_change = padded
        return data
