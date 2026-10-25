"""The single training view for the ``R + P -> TS`` bridge.

Every backend (LEFTNet, PaiNN, EGNN) consumes :class:`EventData` items from
:class:`BridgeFlowDataset` and predicts the same residual
``(x_t - x_0) / sigma_t``. The prior is the deterministic midpoint
``(R + P) / 2``; the recorded transition state is the target. There is no second
dataset, objective or sampler per backbone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from basinflow.data.events import EventCatalog, event_target

if TYPE_CHECKING:
    from basinflow.models.sb import SBSchedule

try:  # Keep the raw event readers usable without the optional model dependencies.
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
    """One event graph; ``cell``, ``pbc`` and ``flow_time`` stay graph-level.

    ``__cat_dim__`` returns ``None`` for those fields so PyG's collation stacks
    them along a new leading axis (``cell`` ``[3, 3]`` -> ``[B, 3, 3]``,
    ``flow_time`` ``[1]`` -> ``[B, 1]``). Everything else is node-level and
    concatenates on dim 0.
    """

    _GRAPH_FIELDS = {"cell", "pbc", "flow_time"}

    def __init__(self, **kwargs) -> None:
        require_pyg()
        super().__init__(**kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._GRAPH_FIELDS:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def _graph_fields(cell, pbc, flow_time: float) -> dict:
    """The three graph-level tensors, with the shapes collation depends on."""
    cell_tensor = torch.as_tensor(cell, dtype=torch.float32)
    pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool)
    time_tensor = torch.tensor([float(flow_time)], dtype=torch.float32)
    if cell_tensor.shape != (3, 3) or pbc_tensor.shape != (3,):
        raise ValueError("cell must be (3, 3) and pbc (3,) to collate as graph-level")
    return {"cell": cell_tensor, "pbc": pbc_tensor, "flow_time": time_tensor}


class BridgeFlowDataset(Dataset):
    """One bridge training pair per event.

    An item carries the sampled state ``x_t``, the prior ``(R + P) / 2``, both
    endpoints, the recorded transition state, the normalised bridge time and the
    regression target ``(x_t - TS) / sigma_t``. ``lowest_step=1`` reproduces the
    reference's validation-time range.
    """

    def __init__(
        self,
        catalog: EventCatalog,
        schedule: "SBSchedule",
        *,
        seed: int = 42,
        lowest_step: int = 0,
        swap_reactant_product: bool = False,
    ) -> None:
        require_pyg()
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        missing = [
            event_id
            for event_id, event in catalog.events.items()
            if event.transition_state is None
        ]
        if missing:
            raise ValueError(f"{len(missing)} events have no transition state, e.g. {missing[0]!r}")
        if not 0 <= int(lowest_step) < int(schedule.timesteps):
            raise ValueError("lowest_step must be in [0, timesteps)")
        self.catalog = catalog
        self.schedule = schedule
        self.seed = int(seed)
        self.lowest_step = int(lowest_step)
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
        from basinflow.models.sb import compute_label, q_sample

        outcome = event_target(self.catalog.events[self.catalog.event_ids[index]])
        reactant = outcome.reactant
        target = np.asarray(outcome.transition_state.positions, dtype=float)
        midpoint = 0.5 * (reactant.positions + outcome.target_positions)
        reactant_positions = (
            outcome.target_positions if self.swap_reactant_product else reactant.positions
        )
        endpoint_positions = (
            reactant.positions if self.swap_reactant_product else outcome.target_positions
        )
        step = int(self._rng(index).integers(self.lowest_step, self.schedule.timesteps))
        sampled = q_sample(
            self.schedule,
            torch.tensor(step),
            torch.as_tensor(target, dtype=torch.float32),
            torch.as_tensor(midpoint, dtype=torch.float32),
            ot_ode=True,
            mask=torch.zeros(reactant.n_atoms, dtype=torch.long),
        )
        return EventData(
            z=torch.as_tensor(reactant.atomic_numbers, dtype=torch.long),
            pos=sampled,
            reactant_pos=torch.as_tensor(reactant_positions, dtype=torch.float32),
            endpoint_pos=torch.as_tensor(endpoint_positions, dtype=torch.float32),
            source_pos=torch.as_tensor(midpoint, dtype=torch.float32),
            movable_mask=torch.as_tensor(reactant.movable_mask, dtype=torch.bool),
            num_nodes=reactant.n_atoms,
            event_id=self.catalog.event_ids[index],
            target_pos=torch.as_tensor(target, dtype=torch.float32),
            target_scaled=compute_label(
                self.schedule,
                torch.tensor(step),
                torch.as_tensor(target, dtype=torch.float32),
                sampled,
            ),
            **_graph_fields(
                reactant.cell, reactant.pbc, step / self.schedule.timesteps
            ),
        )
