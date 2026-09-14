"""BasinFlow: learning-assisted KMC event proposal.

Subpackages follow the pipeline stages, not the frameworks they happen to use:

- ``geometry``   coordinate-space operations and graph construction
- ``data``       event catalogs, splits and PyG datasets
- ``seeds``      ``EventSeed`` definitions and initialization generators
- ``models``     flow backbones and the model factory
- ``training``   the training loop
- ``sampling``   candidate proposal from a basin
- ``evaluation`` benchmark metrics and summaries
- ``relaxation`` optional structure relaxation
"""

__all__ = [
    "data",
    "evaluation",
    "geometry",
    "models",
    "relaxation",
    "sampling",
    "seeds",
    "training",
]
