"""The sample views the project trains and samples with.

:class:`TransitionStateDataset` -- R + P -> TS velocity-flow samples (the PaiNN /
EGNN baseline).
:class:`BridgeFlowDataset` -- ReactOT's Schrodinger-bridge training pairs; the
midpoint ``(R + P) / 2`` is the prior and the recorded transition state is the
target.
"""
from __future__ import annotations

import json

import numpy as np

from basinflow.data.events import EventCatalog, event_target

#: The single source of seed-type ids, shared with the models. Only
#: ``midpoint_bridge`` is used by the two remaining sample views, but keeping the
#: table here keeps the historical checkpoint ids stable.
_SEED_TYPE_IDS = {
    "zero": 0,
    "gaussian_movable": 1,
    "directional": 2,
    "product_displacement": 3,
    "midpoint_bridge": 4,
}


def seed_type_id(seed_type: str) -> int:
    name = str(seed_type)
    try:
        return _SEED_TYPE_IDS[name]
    except KeyError:
        raise ValueError(f"unknown seed type {name!r}") from None

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
    ``[B, 3, 3]`` and ``flow_time`` ``[1]`` becomes ``[B, 1]``. Every other tensor
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


class TransitionStateDataset(Dataset):
    """R + P -> TS flow samples: both endpoints are conditioning inputs (mode B).

    Supervised over a range of times:

    ``x0 = (R + P) / 2 + sigma * xi``, ``x_t = (1 - t) * x0 + t * TS``,
    ``v = d x_t / dt``.

    ``source_scale = 0`` collapses the source to the deterministic midpoint bridge
    and ``time_distribution = "fixed"`` collapses supervision to one point.
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
        """Deterministic per ``(seed, epoch, index)`` generator."""
        return np.random.default_rng([self.seed, self.epoch, int(index)])

    def _flow_time(self, index: int, rng: np.random.Generator | None = None) -> float:
        """Sampled training time, or the pinned value when the distribution is fixed.

        The sampled time lies in ``[time_epsilon, 1 - time_epsilon]``.
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

        ``linear``: ``x_t = (1 - t) x0 + t x1``, tangent ``(x1 - x0)``.

        ``decaying``: ``x_t = x0 + (x1 - x0)(2t - t^2)``, tangent
        ``2(1 - t)(x1 - x0)``, which vanishes at ``t = 1``.

        ``trigonometric``: the same straight line parameterised by angle.
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
    """R + P -> TS Schrodinger-bridge samples.

    One item carries:

    ``x_1 = (R + P) / 2`` (the prior), ``x_0 = TS`` (the target),
    ``x_t = mu_x0 * x_0 + mu_x1 * x_1`` at a uniformly drawn bridge time, and the
    supervised residual ``(x_t - x_0) / sigma_t`` under the key ``target_scaled``.

    Training only: the recorded transition state is the regression target.
    Inference for this task is driven by
    :func:`basinflow.models.sb.make_bridge_ode_field`.

    ``timesteps``, ``beta_max``, ``power`` and ``inv_power`` fall back to
    :mod:`basinflow.models.sb` defaults, imported lazily.
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
        lowest_step: int = 0,
        swap_reactant_product: bool = False,
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
        self.lowest_step = int(lowest_step)
        if not 0 <= self.lowest_step < self.timesteps:
            raise ValueError("lowest_step must be in [0, timesteps)")
        # R/P swap is a training augmentation: the midpoint and the TS target are
        # unchanged, only which endpoint is the "reactant" side of the model input.
        self.swap_reactant_product = bool(swap_reactant_product)
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
        if self.swap_reactant_product:
            reactant_positions = target.target_positions
            endpoint_positions = reactant.positions
        else:
            reactant_positions = reactant.positions
            endpoint_positions = target.target_positions

        step = int(self._rng(index).integers(self.lowest_step, self.timesteps))
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
            reactant_pos=torch.as_tensor(reactant_positions, dtype=torch.float32),
            endpoint_pos=torch.as_tensor(endpoint_positions, dtype=torch.float32),
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
