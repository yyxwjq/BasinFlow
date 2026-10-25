"""Initialise a model from a saved checkpoint before training.

Supports this repository's ``checkpoint.pt`` / ``training_state.pt`` format and
the released ReactOT Lightning checkpoint (via ``models.reference``). The paper
fine-tunes from an OA-ReactDiff checkpoint; that file is not distributed, so this
is the generic hook for a future warm start.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch


def load_initial_weights(model, path: str | Path, *, reference_src: str | Path | None = None) -> str:
    """Load ``path`` into ``model`` and return the detected format."""
    source = reference_src or os.environ.get("REACTOT_SRC")
    if source is not None:
        # ``torch.load`` unpickles the reference classes, which import
        # ``torch_scatter`` at module import time.
        sys.path.insert(0, str(source))
        from basinflow.models.reference import _install_torch_scatter_stub

        _install_torch_scatter_stub()
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
        return "basinflow"
    if "state_dict" in state:
        from basinflow.models.reference import load_reactot_checkpoint

        reference, _ = load_reactot_checkpoint(
            path, reference_src=reference_src or os.environ.get("REACTOT_SRC")
        )
        model.load_state_dict(reference.state_dict(), strict=True)
        return "reactot"
    raise ValueError(f"{path} is neither a BasinFlow nor a ReactOT checkpoint")
