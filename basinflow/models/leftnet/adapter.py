"""``EventData`` -> LEFTNet adapter for the ``R + P -> TS`` bridge.

This is the BasinFlow side of the port. It rebuilds the reference's
multi-fragment tensor layout from BasinFlow's graph batch:

    ReactOT                                    here
    ----------------------------------------- --------------------------------
    ``xh_t = [R | TS | P]``, each ``[n_atom, 9]``   ``h = [feat | feat | feat]``
    ``hs = stack([r_other, t_other, p_other])``     same, one column block each
    ``combined_mask`` (sample id per node)          ``batch.batch`` triplicated
    ``n_frag_switch`` (0/1/2 per fragment)          ``[0]*N + [1]*N + [2]*N``
    ``get_edges_index(combined_mask)``              same-sample pairs (pre-cutoff)
    ``get_subgraph_mask(ei, n_frag_switch)``        block equality per edge
    ``net_eps_xh[self.idx][:, :3]``                 TS block of ``pos_out - pos``
    ``self.remove_mean_batch(vel, combined_mask)``  :func:`remove_mean_batch`

Only the TS block is returned, and it is mean-removed per sample exactly as
``reactot/dynamics/egnn_dynamics.py:206-218`` does; ``fixed_idx = [0, 2]``
discards the R/P blocks, so they are never read out here.

Equivalence note (must be stated with any result): ReactOT first builds *every*
same-sample edge and then masks the ones beyond ``cutoff`` to zero inside
``LEFTNet``; this adapter drops those edges up front. Masked edges contribute
zero messages, and ``LEFTNet`` computes distances after a per-component mean
removal (translation only), so the two are identical for the same cutoff.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from basinflow.models.leftnet.leftnet import LEFTNet
from basinflow.models.sb import remove_mean_batch

#: Element vocabulary of the reference (H, C, N, O, F); ``run_model.py`` rejects
#: anything else, and Transition1x contains only H/C/N/O.
DEFAULT_ELEMENT_VOCAB: tuple[int, ...] = (1, 6, 7, 8, 9)
_MAX_ATOMIC_NUMBER = 118

#: Fragment order of the reference ``EnSB`` for ``R+P->TS`` (fragment_names).
FRAGMENT_ORDER = ("reactant", "transition_state", "product")


class LeftNetBridge(nn.Module):
    """``R + P -> TS`` scorer: LEFTNet over the reference 3-fragment layout.

    ``forward`` returns ``{"scaled": net_out}`` where ``net_out`` is the bridge
    regression target the reference fits, ``(x_t - x_0) / sigma_t``
    (``reactot/diffusion/en_sb.py:245-250``). The ``R + P -> TS`` task has no
    seed and no velocity field, so this backend deliberately does **not** emit
    the ``{"velocity": ...}`` key that the flow backends use.

    ``t`` is the normalised bridge time ``t_sb in (0, 1]`` (``1`` = the
    ``(R + P) / 2`` prior, ``-> 0`` = the transition state), matching
    ``EnSB``. :func:`basinflow.models.sb.make_bridge_ode_field` performs the
    change of variable needed by ``sample_ode``'s increasing interval.
    """

    #: Read by the samplers: this module rebuilds its own graph every call.
    builds_own_graph = True
    #: Read by the loss selector: the objective is the bridge residual, not a velocity.
    bridge = True
    #: Read by the datasets/workflow: both endpoints are conditioning inputs.
    endpoint_condition = True
    endpoint_equivariant = True

    def __init__(
        self,
        *,
        hidden_channels: int = 196,
        num_layers: int = 6,
        num_radial: int = 96,
        cutoff: float = 10.0,
        num_elements: int = 5,
        reflect_equiv: bool = True,
        legacy: bool = True,
        update: bool = True,
        single_layer_output: bool = True,
        object_aware: bool = True,
        element_vocab: Sequence[int] = DEFAULT_ELEMENT_VOCAB,
    ) -> None:
        super().__init__()
        vocab = tuple(int(value) for value in element_vocab)
        if len(vocab) != int(num_elements):
            # The one-hot width is the reference's ``node_nfs`` category count.
            raise ValueError(
                f"num_elements ({num_elements}) must equal len(element_vocab) ({len(vocab)})"
            )
        if any(z < 1 or z > _MAX_ATOMIC_NUMBER for z in vocab):
            raise ValueError(f"element_vocab must hold valid atomic numbers, got {vocab}")
        self.num_elements = int(num_elements)
        self.element_vocab = vocab
        self.cutoff = float(cutoff)
        self.r_max = float(cutoff)
        # Node features: one-hot(Z) + Z (the reference's "charge" column, which is
        # the atomic number) + bridge time + condition placeholder.
        self.in_hidden_channels = self.num_elements + 3

        lookup = torch.zeros(_MAX_ATOMIC_NUMBER + 1, dtype=torch.long)
        for index, atomic_number in enumerate(vocab):
            lookup[atomic_number] = index
        self.register_buffer("_vocab_lookup", lookup, persistent=False)

        self.leftnet = LEFTNet(
            cutoff=self.cutoff,
            num_layers=int(num_layers),
            hidden_channels=int(hidden_channels),
            num_radial=int(num_radial),
            in_hidden_channels=self.in_hidden_channels,
            reflect_equiv=bool(reflect_equiv),
            legacy=bool(legacy),
            update=bool(update),
            single_layer_output=bool(single_layer_output),
            object_aware=bool(object_aware),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _categories(self, z: Tensor) -> Tensor:
        """Atomic numbers -> vocabulary index, rejecting anything outside it."""
        if int(z.max()) > _MAX_ATOMIC_NUMBER or int(z.min()) < 1:
            raise ValueError("atomic numbers out of range")
        mapped = self._vocab_lookup[z]
        inside = torch.zeros_like(z, dtype=torch.bool)
        for atomic_number in self.element_vocab:
            inside = inside | (z == atomic_number)
        if not bool(inside.all()):
            unknown = sorted({int(v) for v in z[~inside].detach().cpu()})
            raise ValueError(
                f"LEFTNet bridge supports only {self.element_vocab}; got unsupported Z={unknown}"
            )
        return mapped

    def _node_features(self, z: Tensor, time_nodes: Tensor) -> Tensor:
        categories = self._categories(z)
        one_hot = F.one_hot(categories, num_classes=self.num_elements).to(torch.float32)
        charge = z.to(torch.float32).unsqueeze(-1)
        condition = torch.zeros_like(time_nodes)
        return torch.cat([one_hot, charge, time_nodes, condition], dim=-1)

    def _edges(self, pos: Tensor, combined: Tensor, fragment: Tensor):
        """Same-sample edges within ``cutoff``, plus the same-fragment mask."""
        same_sample = combined[:, None] == combined[None, :]
        distance = torch.cdist(pos, pos)
        adjacency = same_sample & (distance <= self.cutoff)
        adjacency.fill_diagonal_(False)
        edge_index = torch.stack(torch.where(adjacency), dim=0)
        same_fragment = (fragment[edge_index[0]] == fragment[edge_index[1]]).to(torch.float32)
        return edge_index, same_fragment.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, batch, t: Optional[Tensor] = None) -> dict[str, Tensor]:
        z = batch.z.to(torch.long)
        reactant = batch.reactant_pos
        state = batch.pos
        endpoint = batch.endpoint_pos
        if endpoint is None:
            raise ValueError("the LEFTNet bridge requires endpoint_pos (the product frame)")
        index = batch.batch.to(torch.long)
        atoms = int(z.shape[0])

        if t is None:
            time = batch.flow_time
        else:
            time = t
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device).reshape(-1)
        if time.numel() == 1:
            time_nodes = time.expand(atoms).unsqueeze(-1)
        else:
            time_nodes = time[index].unsqueeze(-1)

        features = self._node_features(z, time_nodes)
        h = torch.cat([features, features, features], dim=0)
        pos = torch.cat([reactant, state, endpoint], dim=0)
        combined = torch.cat([index, index, index], dim=0)
        fragment = torch.cat(
            [
                torch.zeros(atoms, dtype=torch.long, device=z.device),
                torch.ones(atoms, dtype=torch.long, device=z.device),
                torch.full((atoms,), 2, dtype=torch.long, device=z.device),
            ],
            dim=0,
        )

        edge_index, subgraph_mask = self._edges(pos, combined, fragment)
        _, pos_out, _ = self.leftnet(h, pos, edge_index, subgraph_mask=subgraph_mask)
        displacement = (pos_out - pos)[atoms : 2 * atoms]
        return {"scaled": remove_mean_batch(displacement, index)}
