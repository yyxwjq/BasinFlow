"""The EGNN backbone.

Layout mirrors :mod:`basinflow.models.painn`: ``layers`` holds the equivariant
message-passing layer and ``egnn`` the assembled module.
"""

from basinflow.models.egnn.egnn import EGNNFlow
from basinflow.models.egnn.layers import EGNNLayer

__all__ = ["EGNNFlow", "EGNNLayer"]
