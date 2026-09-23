"""The three sample views the project trains and samples with.

Exactly three, one per path that needs data:

1. :class:`EventFlowDataset` -- **mode A**, R -> P. The product is *supervision*
   (``target_pos``/``target_velocity``); it is never an input. The seed supplies
   the initial geometry and the vector conditions.
2. :class:`TransitionStateDataset` -- **mode B**, R + P -> TS. Here the product is
   a *conditioning input* (``endpoint_pos``) because the task is defined to
   receive it; the TS is supervision. This is not the leak that the same field
   would be in mode A, and the two views are separate classes precisely so that
   stays visible.
3. :class:`EventProposalDataset` -- target-free inference. No ``target_*`` field is
   constructed on this path at all, so an oracle cannot reach the model even by
   accident.

Semantics that are easy to get backwards, stated once here:

* ``pos`` is the *current flow state* ``x_t``, not a product geometry.
* ``movable_mask`` is a **given constraint** (ASE ``FixAtoms``, or all-True when
  the frame carries none), never a learning target; fixed atoms get zero velocity
  and zero seed motion.
* ``active_prior`` is an **input condition** -- which atoms received the initial
  perturbation -- derived from the seed displacement. It is not supervision.
* ``seed_direction`` is an **input** (the seed's unit direction). There is no
  supervised direction field anywhere: the deleted one was the event's
  displacement direction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Iterable

import numpy as np

from basinflow.data.events import EventCatalog, event_target
from basinflow.geometry.bonds import bond_adjacency, bond_change_labels
from basinflow.seeds import GaussianInit, SeedContext, seed_type_id

try:  # Keep the raw readers usable without the optional model dependencies.
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - exercised in dependency-free installs
    torch = None
    Dataset = object
    Data = object


def require_pyg() -> None:
    if torch is None:
        raise ImportError(
            "PyG event datasets require the optional 'models' dependencies: "
            "install basinflow[models]"
        )


class EventData(Data):
    """One event-flow graph, with graph-level fields kept graph-level.

    ``__cat_dim__`` returns ``None`` for the fields that describe *one graph*
    rather than one node (``cell``, ``pbc``, ``flow_time``, ``seed_type_id``), so
    collation stacks them along a new leading axis: ``cell`` ``[3, 3]`` becomes
    ``[B, 3, 3]`` and ``flow_time`` ``[1]`` becomes ``[B, 1]``. That is the exact
    shape the backbones index per graph, and the assertions in
    :func:`_graph_fields` keep it from silently changing. Every other tensor
    (``pos``, ``z``, ``target_velocity``, ...) is node-level and concatenates on
    dim 0, which is PyG's default.

    ``bond_change`` is a third case: per graph it is ``[1, n_max, n_max]`` padded
    to the catalog maximum, so the default dim-0 concatenation yields
    ``[B, n_max, n_max]`` -- a dense per-graph pair label, which is what the PaiNN
    edge condition needs to survive collation.
    """

    _GRAPH_FIELDS = {"cell", "pbc", "flow_time", "seed_type_id"}

    def __init__(self, **kwargs) -> None:
        require_pyg()
        super().__init__(**kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._GRAPH_FIELDS:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def _graph_fields(cell, pbc, flow_time: float, seed_type: str) -> dict:
    """The four graph-level tensors, with the shapes collation depends on."""
    cell_tensor = torch.as_tensor(cell, dtype=torch.float32)
    pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool)
    time_tensor = torch.tensor([float(flow_time)], dtype=torch.float32)
    type_tensor = torch.tensor([seed_type_id(seed_type)], dtype=torch.long)
    if cell_tensor.shape != (3, 3) or pbc_tensor.shape != (3,):
        raise ValueError("cell must be (3, 3) and pbc (3,) to collate as graph-level")
    if time_tensor.shape != (1,) or type_tensor.shape != (1,):
        raise ValueError("flow_time and seed_type_id must be length-1 graph-level tensors")
    return {
        "cell": cell_tensor,
        "pbc": pbc_tensor,
        "flow_time": time_tensor,
        "seed_type_id": type_tensor,
    }


def _seed_context(event_id: str, reactant) -> SeedContext:
    return SeedContext(event_id=str(event_id), reactant=reactant)


def _mask_unmovable_pairs(labels: np.ndarray, movable_mask) -> np.ndarray:
    """Drop labels on pairs where neither atom can move.

    A bond cannot change without relative motion, so those entries are not a legal
    proposal. Enforcing it in one place keeps the supervised and the proposal view
    identical on constrained structures.
    """
    frozen = ~np.asarray(movable_mask, dtype=bool)
    if not np.any(frozen):
        return labels
    labels = np.array(labels, copy=True)
    labels[np.ix_(frozen, frozen)] = 0
    return labels


def _bond_change_tensor(seed, reactant, *, n_max: int, source: str, oracle_positions=None):
    """Per-pair ``{unchanged, formed, broken}`` labels as one padded tensor.

    The seed wins when it proposes a change. Otherwise ``oracle`` falls back to
    the reactant/product bond difference -- available only because the supervised
    pairwise view knows the product. The inference view passes no oracle, so an
    absent seed proposal means "nothing changes here" rather than "consult the
    product". This is the single implementation of the padding convention
    documented on :class:`EventData`.
    """
    if seed.bond_change is not None:
        labels = np.asarray(seed.bond_change)
    elif oracle_positions is None or source == "seed":
        labels = np.zeros((reactant.n_atoms, reactant.n_atoms), dtype=np.int8)
    else:
        labels = bond_change_labels(
            bond_adjacency(reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
            bond_adjacency(
                oracle_positions, reactant.atomic_numbers, reactant.cell, reactant.pbc
            ),
        ).astype(np.int8)
    labels = _mask_unmovable_pairs(labels, seed.movable_mask)
    padded = torch.zeros((1, n_max, n_max), dtype=torch.int8)
    padded[0, : labels.shape[0], : labels.shape[1]] = torch.as_tensor(labels, dtype=torch.int8)
    return padded


def _base_data(
    event_id: str,
    reactant,
    seed,
    *,
    pos,
    source_pos,
    flow_time: float,
) -> EventData:
    """The shared, target-free part of an event sample.

    Mode A and the inference view differ only in what they add on top of this, so
    the seed conditions and the graph-level fields are built once.
    """
    data = EventData(
        z=torch.as_tensor(reactant.atomic_numbers, dtype=torch.long),
        pos=torch.as_tensor(pos, dtype=torch.float32),
        reactant_pos=torch.as_tensor(reactant.positions, dtype=torch.float32),
        source_pos=torch.as_tensor(source_pos, dtype=torch.float32),
        movable_mask=torch.as_tensor(seed.movable_mask, dtype=torch.bool),
        active_prior=torch.as_tensor(seed.active_prior, dtype=torch.bool),
        seed_displacement=torch.as_tensor(seed.seed_displacement, dtype=torch.float32),
        seed_direction=torch.as_tensor(seed.seed_direction, dtype=torch.float32),
        num_nodes=reactant.n_atoms,
        seed_type=seed.seed_type,
        event_id=event_id,
        seed_id=seed.seed_id,
        seed_metadata_json=json.dumps(seed.metadata),
        **_graph_fields(reactant.cell, reactant.pbc, flow_time, seed.seed_type),
    )
    return data


def _catalog_n_max(catalog: EventCatalog) -> int:
    return max((event.reactant.n_atoms for event in catalog.events.values()), default=0)


class EventFlowDataset(Dataset):
    """Supervised event-level R -> P flow samples (mode A)."""

    def __init__(
        self,
        catalog: EventCatalog,
        initializers: Iterable,
        *,
        flow_time: float | None = None,
        seed: int = 42,
        diagnostic_oracle: bool = False,
        time_distribution: str = "uniform",
        beta_alpha: float = 0.8,
        bond_change_condition: bool = False,
        bond_change_source: str = "oracle",
    ) -> None:
        require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        self.catalog = catalog
        self.initializers = tuple(initializers)
        if not self.initializers:
            raise ValueError("at least one initializer is required")
        if flow_time is not None and not 0.0 <= float(flow_time) <= 1.0:
            raise ValueError("flow_time must be in [0, 1]")
        if time_distribution not in {"uniform", "beta"}:
            raise ValueError("time_distribution must be 'uniform' or 'beta'")
        if beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be positive")
        if bond_change_source not in {"oracle", "seed"}:
            raise ValueError("bond_change_source must be 'oracle' or 'seed'")
        self.fixed_flow_time = None if flow_time is None else float(flow_time)
        self.seed = int(seed)
        self.epoch = 0
        self.diagnostic_oracle = bool(diagnostic_oracle)
        self.time_distribution = time_distribution
        self.beta_alpha = float(beta_alpha)
        self.bond_change_condition = bool(bond_change_condition)
        self.bond_change_source = bond_change_source
        if not self.diagnostic_oracle and any(
            getattr(initializer, "requires_target", False) for initializer in self.initializers
        ):
            raise ValueError("target-based initializers require diagnostic_oracle=True")
        self.n_max = _catalog_n_max(catalog)
        self._items = [
            (event_id, initializer_index)
            for event_id in self.catalog.event_ids
            for initializer_index in range(len(self.initializers))
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._items)

    def _flow_time(self, index: int) -> float:
        """Deterministic per ``(seed, epoch, index)`` time.

        ``hash`` is unusable here: it is salted per process, so the same
        ``(seed, epoch, index)`` would give a different time in a different
        worker and a run could not be reproduced from its config.
        """
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
        target = event_target(self.catalog.events[event_id])
        context = _seed_context(event_id, target.reactant)
        initializer = self.initializers[initializer_index]
        if isinstance(initializer, GaussianInit) and initializer.random_seed is None:
            initializer = replace(initializer, random_seed=self.seed)
        seed_id = (
            f"{event_id}:seed:{initializer_index}:{initializer.seed_type}"
            f":dataset:{self.seed}:epoch:{self.epoch}"
        )
        if getattr(initializer, "requires_target", False):
            seed = initializer.generate(
                context, seed_id=seed_id, target_displacement=target.displacement
            )
        else:
            seed = initializer.generate(context, seed_id=seed_id)
        source_pos = seed.initial_positions(target.reactant.positions)
        time = self._flow_time(index)
        pos = (1.0 - time) * source_pos + time * target.target_positions
        target_velocity = target.target_positions - source_pos
        target_velocity[~seed.movable_mask] = 0.0
        data = _base_data(
            event_id,
            target.reactant,
            seed,
            pos=pos,
            source_pos=source_pos,
            flow_time=time,
        )
        data.target_pos = torch.as_tensor(target.target_positions, dtype=torch.float32)
        data.target_velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
        if self.bond_change_condition:
            data.bond_change = _bond_change_tensor(
                seed,
                target.reactant,
                n_max=self.n_max,
                source=self.bond_change_source,
                oracle_positions=target.target_positions,
            )
        return data


class EventProposalDataset(Dataset):
    """Reactant-only event samples for target-free proposal inference."""

    def __init__(
        self,
        catalog: EventCatalog,
        initializers: Iterable,
        *,
        bond_change_condition: bool = False,
    ) -> None:
        require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        self.catalog = catalog
        self.initializers = tuple(initializers)
        if not self.initializers:
            raise ValueError("at least one initializer is required")
        if any(
            getattr(initializer, "requires_target", False) for initializer in self.initializers
        ):
            raise ValueError("event proposal initializers must not require target data")
        self.bond_change_condition = bool(bond_change_condition)
        self.n_max = _catalog_n_max(catalog)
        self._items = [
            (event_id, initializer_index)
            for event_id in self.catalog.event_ids
            for initializer_index in range(len(self.initializers))
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> EventData:
        event_id, initializer_index = self._items[index]
        reactant = self.catalog.events[event_id].reactant
        initializer = self.initializers[initializer_index]
        context = _seed_context(f"{event_id}:proposal:{initializer_index}", reactant)
        seed = initializer.generate(
            context, seed_id=f"{event_id}:seed:{initializer_index}:{initializer.seed_type}"
        )
        source_pos = seed.initial_positions(reactant.positions)
        data = _base_data(
            event_id,
            reactant,
            seed,
            pos=source_pos,
            source_pos=source_pos,
            flow_time=0.0,
        )
        if self.bond_change_condition:
            # Target-free: no oracle positions are passed, so an absent seed
            # proposal means "nothing changes here".
            data.bond_change = _bond_change_tensor(
                seed, reactant, n_max=self.n_max, source="seed"
            )
        return data


class TransitionStateDataset(Dataset):
    """R + P -> TS flow samples: both endpoints are conditioning inputs (mode B).

    ReactOT and MolGEN both condition on the reactant *and* the product, so the
    product frame is an input here rather than the leak it would be in mode A.

    This is a flow-matching task, so the interpolant must be supervised over a
    *range* of times, not at a single point:

    ``x0 = (R + P) / 2 + sigma * xi``, ``x_t = (1 - t) * x0 + t * TS``,
    ``v = d x_t / dt``.

    ``source_scale = 0`` collapses the source to the deterministic midpoint bridge
    (ReactOT's prior) and ``time_distribution = "fixed"`` collapses supervision to
    one point. Either degeneracy removes the field: with a fixed ``t`` the network
    only has to fit ``TS - midpoint`` as a function of the endpoints, which is
    direct regression, and a single sampler step is the whole model. MolGEN avoids
    both by drawing ``t ~ Beta(0.8, 0.8)`` over a ``randn * x0std`` source
    (``mdgen/transport/transport.py:262``).
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
        time_epsilon: float = 0.0,
        path: str = "linear",
    ) -> None:
        require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        if flow_time is not None and not 0.0 <= float(flow_time) <= 1.0:
            raise ValueError("flow_time must be in [0, 1]")
        if source_scale < 0.0:
            raise ValueError("source_scale must be non-negative")
        if time_distribution not in {"fixed", "beta", "uniform"}:
            raise ValueError("time_distribution must be 'fixed', 'beta' or 'uniform'")
        if beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be positive")
        if not 0.0 <= float(time_epsilon) < 0.5:
            raise ValueError("time_epsilon must be in [0, 0.5)")
        if path not in {"linear", "trigonometric", "decaying"}:
            raise ValueError("path must be 'linear', 'trigonometric' or 'decaying'")
        if flow_time is not None and time_distribution != "fixed":
            # Silently ignoring flow_time would reintroduce exactly the
            # single-point supervision this dataset exists to avoid.
            raise ValueError(
                "flow_time pins the supervision to one point and cannot be combined with "
                f"time_distribution={time_distribution!r}; pass flow_time=None instead"
            )
        missing = [
            event_id
            for event_id, event in catalog.events.items()
            if event.transition_state is None
        ]
        if missing:
            raise ValueError(f"{len(missing)} events have no transition state, e.g. {missing[0]!r}")
        self.catalog = catalog
        # The training loop reads this surface off any prepared dataset.
        self.initializers: tuple = ()
        self.diagnostic_oracle = False
        self.fixed_flow_time = None if flow_time is None else float(flow_time)
        self.seed = int(seed)
        self.source_scale = float(source_scale)
        self.time_distribution = time_distribution
        self.beta_alpha = float(beta_alpha)
        self.time_epsilon = float(time_epsilon)
        self.path = path
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.catalog.event_ids)

    def _rng(self, index: int) -> np.random.Generator:
        """Deterministic per ``(seed, epoch, index)`` generator.

        Fresh noise and a fresh time per epoch are what make the field a field: a
        constant source would let the network memorise one displacement per event
        instead of learning how to transport from an arbitrary point.
        """
        return np.random.default_rng([self.seed, self.epoch, int(index)])

    def _flow_time(self, index: int, rng: np.random.Generator | None = None) -> float:
        """Sampled training time, or the pinned value when the distribution is fixed.

        ``time_epsilon`` keeps the sampled time inside ``[eps, 1 - eps]``, the
        standard flow-matching interval: the endpoints are where the linear path is
        degenerate (``t = 0`` gives the source, ``t = 1`` the target).
        """
        if self.time_distribution == "fixed":
            return 0.0 if self.fixed_flow_time is None else self.fixed_flow_time
        rng = self._rng(index) if rng is None else rng
        if self.time_distribution == "beta":
            raw = float(rng.beta(self.beta_alpha, self.beta_alpha))
        else:
            raw = float(rng.random())
        epsilon = self.time_epsilon
        if epsilon == 0.0:
            return raw
        return epsilon + (1.0 - 2.0 * epsilon) * raw

    def _interpolate(
        self, source: np.ndarray, target_positions: np.ndarray, time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x_t, dx_t/dt)`` for the configured probability path.

        ``linear`` is ``x_t = (1 - t) x0 + t x1``, whose tangent ``x1 - x0`` does
        not depend on ``t``: the field has no reason to depend on time either, and
        the trained object degenerates into a one-step map (``docs/25``).

        ``decaying`` keeps the same straight conditional path but reparametrises it
        in time, ``x_t = x0 + (x1 - x0)(2t - t^2)``, whose tangent
        ``2(1 - t)(x1 - x0)`` vanishes at ``t = 1``. The trajectory is identical to
        the linear one while the field is self-limiting, and the integral is exact
        by construction: ``int_0^1 2(1 - t)(x1 - x0) dt = x1 - x0``.
        """
        delta = target_positions - source
        if self.path == "decaying":
            return source + delta * (2.0 * time - time * time), 2.0 * (1.0 - time) * delta
        if self.path == "trigonometric":
            # Parameterises the same straight line by angle.
            angle = 0.5 * np.pi * time
            return (
                source + delta * float(np.sin(angle)),
                0.5 * np.pi * float(np.cos(angle)) * delta,
            )
        return (1.0 - time) * source + time * target_positions, delta

    def __getitem__(self, index: int) -> EventData:
        event_id = self.catalog.event_ids[index]
        target = event_target(self.catalog.events[event_id])
        reactant, product, state = target.reactant, target.product, target.transition_state
        midpoint = 0.5 * (reactant.positions + target.target_positions)
        source = midpoint
        if self.source_scale or self.time_distribution != "fixed":
            # One generator per item so geometry and time stay reproducible
            # together for a given (seed, epoch, index).
            rng = self._rng(index)
            if self.source_scale:
                source = midpoint + rng.normal(scale=self.source_scale, size=midpoint.shape)
            time = self._flow_time(index, rng)
        else:
            time = self._flow_time(index)
        source = np.asarray(source, dtype=float)
        pos, velocity = self._interpolate(source, np.asarray(state.positions, dtype=float), time)
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
            num_nodes=reactant.n_atoms,
            seed_type="midpoint_bridge",
            event_id=event_id,
            seed_id=f"{event_id}:midpoint",
            seed_metadata_json=json.dumps(
                {
                    "bridge": "midpoint",
                    "source_scale": self.source_scale,
                    "time_distribution": self.time_distribution,
                }
            ),
            target_pos=torch.as_tensor(state.positions, dtype=torch.float32),
            target_velocity=torch.as_tensor(velocity, dtype=torch.float32),
            **_graph_fields(reactant.cell, reactant.pbc, time, "midpoint_bridge"),
        )


class BridgeFlowDataset(Dataset):
    """R + P -> TS Schrodinger-bridge samples: the pair ReactOT actually fits.

    One item is the reference's training pair (``reactot/diffusion/en_sb.py``):

    ``x_1 = (R + P) / 2`` (the prior), ``x_0 = TS`` (the target),
    ``x_t = mu_x0 * x_0 + mu_x1 * x_1`` at a uniformly drawn bridge time, and the
    supervised residual ``(x_t - x_0) / sigma_t``.

    It is a separate class rather than a flag on :class:`TransitionStateDataset`
    because the two emit different supervised keys (``target_scaled`` here,
    ``target_velocity`` there); a flag would let a run end up half-switched, and
    ``loss.flow_loss`` selects the objective from the key that is present.

    This view is target-aware by construction -- the recorded TS *is* the
    regression target -- so it is a training-only view. Inference for this task
    builds no target and is driven by
    :func:`basinflow.models.sb.make_bridge_ode_field`.

    ``timesteps``, ``beta_max``, ``power`` and ``inv_power`` default to the
    reference configuration (``train_rpsb_ts1x.py:128-140``). The schedule is
    imported lazily so ``basinflow.data`` stays importable without the optional
    model dependencies.
    """

    def __init__(
        self,
        catalog: EventCatalog,
        *,
        timesteps: int | None = None,
        beta_max: float | None = None,
        power: float | None = None,
        inv_power: float | None = None,
        seed: int = 42,
    ) -> None:
        require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        from basinflow.models.sb import (
            DEFAULT_BETA_MAX,
            DEFAULT_INV_POWER,
            DEFAULT_POWER,
            DEFAULT_TIMESTEPS,
            SBSchedule,
        )

        missing = [
            event_id
            for event_id, event in catalog.events.items()
            if event.transition_state is None
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} events have no transition state, e.g. {missing[0]!r}"
            )
        self.catalog = catalog
        # The training loop reads this surface off any prepared dataset.
        self.initializers: tuple = ()
        self.diagnostic_oracle = False
        self.fixed_flow_time = None
        self.seed = int(seed)
        self.schedule = SBSchedule(
            timesteps=DEFAULT_TIMESTEPS if timesteps is None else int(timesteps),
            beta_max=DEFAULT_BETA_MAX if beta_max is None else float(beta_max),
            power=DEFAULT_POWER if power is None else float(power),
            inv_power=DEFAULT_INV_POWER if inv_power is None else float(inv_power),
        )
        self.timesteps = int(self.schedule.timesteps)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.catalog.event_ids)

    def _rng(self, index: int) -> np.random.Generator:
        """Deterministic per ``(seed, epoch, index)`` generator."""
        return np.random.default_rng([self.seed, self.epoch, int(index)])

    def __getitem__(self, index: int) -> EventData:
        import torch

        from basinflow.models.sb import compute_label, q_sample

        event_id = self.catalog.event_ids[index]
        target = event_target(self.catalog.events[event_id])
        reactant = target.reactant
        state = target.transition_state
        ts = np.asarray(state.positions, dtype=float)
        midpoint = 0.5 * (reactant.positions + target.target_positions)

        step = int(self._rng(index).integers(0, self.timesteps))
        # The mask keeps the reference's per-sample centring in the code path; it
        # is a no-op for the centred frames this layer stores.
        xt = q_sample(
            self.schedule,
            torch.tensor(step),
            torch.as_tensor(ts, dtype=torch.float32),
            torch.as_tensor(midpoint, dtype=torch.float32),
            ot_ode=True,
            mask=torch.zeros(reactant.n_atoms, dtype=torch.long),
        )
        scaled = compute_label(
            self.schedule, torch.tensor(step), torch.as_tensor(ts, dtype=torch.float32), xt
        )
        time = step / self.timesteps
        return EventData(
            z=torch.as_tensor(reactant.atomic_numbers, dtype=torch.long),
            pos=xt,
            reactant_pos=torch.as_tensor(reactant.positions, dtype=torch.float32),
            endpoint_pos=torch.as_tensor(target.target_positions, dtype=torch.float32),
            source_pos=torch.as_tensor(midpoint, dtype=torch.float32),
            movable_mask=torch.as_tensor(reactant.movable_mask, dtype=torch.bool),
            active_prior=torch.zeros(reactant.n_atoms, dtype=torch.bool),
            seed_displacement=torch.zeros((reactant.n_atoms, 3), dtype=torch.float32),
            seed_direction=torch.zeros((reactant.n_atoms, 3), dtype=torch.float32),
            num_nodes=reactant.n_atoms,
            seed_type="midpoint_bridge",
            event_id=event_id,
            seed_id=f"{event_id}:bridge",
            seed_metadata_json=json.dumps(
                {
                    "seed_type": "midpoint_bridge",
                    "bridge_step": step,
                    "bridge_timesteps": self.timesteps,
                }
            ),
            target_pos=torch.as_tensor(ts, dtype=torch.float32),
            target_scaled=scaled,
            **_graph_fields(reactant.cell, reactant.pbc, time, "midpoint_bridge"),
        )
