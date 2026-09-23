"""Ported object-aware LEFTNet backbone (ReactOT reference architecture).

Public surface:
    * :class:`LEFTNet` -- the tensor-in/tensor-out backbone.
    * ``layers`` -- the building blocks, kept name-compatible with
      ``reactot/model/leftnet.py`` for auditing.

The ``EventData`` adapter lives in :mod:`basinflow.models.leftnet.adapter`.
"""
from basinflow.models.leftnet.leftnet import LEFTNet

__all__ = ["LEFTNet"]
