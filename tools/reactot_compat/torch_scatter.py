"""PyG-based drop-in replacement for :mod:`torch_scatter`.

Why this file exists
--------------------
The reference repo ``react-ot`` imports ``torch_scatter`` in a handful of
places, but macOS arm64 + ``torch==2.2.1`` has no ``torch-scatter`` wheel and
the source build fails in an isolated build environment. Every symbol react-ot
actually uses (``scatter``/``scatter_sum``/``scatter_add``/``scatter_mean``,
``segment_coo``, ``segment_csr``) is a plain segment reduction that
``torch.Tensor.index_add_`` / ``torch.Tensor.scatter_reduce_`` already provide.

Scope and provenance
--------------------
* This module is **not** part of BasinFlow and must never be imported by
  ``basinflow.*``. It is used only to run the *reference* repository inside the
  local ``reactot`` conda environment, by putting this directory on
  ``PYTHONPATH``::

      PYTHONPATH=<repo>/tools/reactot_compat conda run -n reactot python ...

* It does **not** modify the reference repository or its environment; no file
  under ``/Users/wx/Desktop/yyxwjq/react-ot`` is touched.
* Numerical note (must be stated next to any reference number): ``sum``/``add``
  are computed with ``index_add_`` and ``mean`` as ``sum / count``; both agree
  with ``torch_scatter`` reductions up to floating-point summation order, which
  is not bit-identical to the CUDA/C++ kernels the reference build would use.
  ``max``/``min`` are provided for completeness but are unused by react-ot.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__version__ = "2.1.2+pyg-shim"

__all__ = [
    "scatter",
    "scatter_sum",
    "scatter_add",
    "scatter_mean",
    "scatter_mul",
    "scatter_max",
    "scatter_min",
    "segment_coo",
    "segment_csr",
]

_REDUCERS = {"sum", "add", "mean", "mul", "max", "min"}


def _check(reduce: str) -> str:
    if reduce not in _REDUCERS:
        raise ValueError(f"unexpected reduce {reduce!r}; expected one of {sorted(_REDUCERS)}")
    return "sum" if reduce == "add" else reduce


def _as_index(index: Tensor) -> Tensor:
    return index.to(torch.long).reshape(-1)


def _segment_sum(src: Tensor, index: Tensor, dim: int, dim_size: int) -> Tensor:
    moved = src.movedim(dim, 0)
    out = torch.zeros((dim_size,) + tuple(moved.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, moved)
    return out.movedim(0, dim)


def _segment_mean(src: Tensor, index: Tensor, dim: int, dim_size: int) -> Tensor:
    moved = src.movedim(dim, 0)
    out = torch.zeros((dim_size,) + tuple(moved.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, moved)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts.index_add_(0, index, torch.ones(moved.shape[0], dtype=src.dtype, device=src.device))
    counts = counts.clamp(min=1.0)
    out = out / counts.reshape((dim_size,) + (1,) * (moved.dim() - 1))
    return out.movedim(0, dim)


def _segment_extremum(src: Tensor, index: Tensor, dim: int, dim_size: int, reduce: str) -> Tensor:
    moved = src.movedim(dim, 0)
    fill = torch.finfo(src.dtype).min if reduce == "max" else torch.finfo(src.dtype).max
    out = torch.full(
        (dim_size,) + tuple(moved.shape[1:]), fill, dtype=src.dtype, device=src.device
    )
    idx = index.reshape((-1,) + (1,) * (moved.dim() - 1)).expand_as(moved)
    out.scatter_reduce_(0, idx, moved, reduce="amax" if reduce == "max" else "amin", include_self=True)
    return out.movedim(0, dim)


def _reduce(src: Tensor, index: Tensor, dim: int, dim_size: Optional[int], reduce: str) -> Tensor:
    reduce = _check(reduce)
    index = _as_index(index)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    dim_size = int(dim_size)
    if reduce == "sum":
        return _segment_sum(src, index, dim, dim_size)
    if reduce == "mean":
        return _segment_mean(src, index, dim, dim_size)
    if reduce in ("max", "min"):
        return _segment_extremum(src, index, dim, dim_size, reduce)
    # reduce == "mul": fall back to a product via log/exp is wrong for zeros, so
    # use an explicit scatter_reduce with a ones-initialised output.
    moved = src.movedim(dim, 0)
    out = torch.ones((dim_size,) + tuple(moved.shape[1:]), dtype=src.dtype, device=src.device)
    idx = index.reshape((-1,) + (1,) * (moved.dim() - 1)).expand_as(moved)
    out.scatter_reduce_(0, idx, moved, reduce="prod", include_self=True)
    return out.movedim(0, dim)


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """torch_scatter.scatter: generic segment reduction (``reduce`` in sum/mean/mul/max/min)."""
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, reduce)


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
) -> Tensor:
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "sum")


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
) -> Tensor:
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "sum")


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
) -> Tensor:
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "mean")


def scatter_mul(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 1.0,
) -> Tensor:
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "mul")


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
):
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "max"), None


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
):
    dim = dim if dim >= 0 else src.dim() + dim
    return _reduce(src, index, dim, dim_size, "min"), None


def segment_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """torch_scatter.segment_coo for a 1-D ``src`` and a 1-D ``index``."""
    if out is not None:
        dim_size = out.shape[0]
    return _reduce(src, index, 0, dim_size, reduce)


def segment_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
    reduce: str = "sum",
) -> Tensor:
    """torch_scatter.segment_csr: segment ids rebuilt from the CSR indptr."""
    indptr = _as_index(indptr)
    num_segments = int(indptr.numel()) - 1
    if num_segments < 0:
        raise ValueError("indptr must have at least one element")
    lengths = (indptr[1:] - indptr[:-1]).clamp(min=0)
    index = torch.repeat_interleave(
        torch.arange(num_segments, device=indptr.device, dtype=torch.long), lengths
    )
    if index.numel() != src.shape[0]:
        raise ValueError(
            f"segment_csr: indptr implies {int(index.numel())} entries but src has {src.shape[0]}"
        )
    if out is not None:
        num_segments = out.shape[0]
    return _reduce(src, index, 0, num_segments, reduce)
