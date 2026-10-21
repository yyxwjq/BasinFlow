"""BasinFlow: a reproduction of ReactOT on an event catalog.

Subpackages follow the pipeline stages:

- ``geometry``   coordinate-space operations and graph construction
- ``data``       event catalogs, splits and the two sample views
- ``models``     LEFTNet bridge, PaiNN/EGNN baselines, model factory
- ``training``   the shared training loop
- ``sampling``   ODE integration for the bridge and the velocity field
- ``evaluation`` reference/strict RMSD protocols and samplers
"""

__all__ = [
    "data",
    "evaluation",
    "geometry",
    "models",
    "sampling",
    "training",
]
