"""Build seed generators from a workflow's ``[init]`` config section."""
from __future__ import annotations

import configparser

from basinflow.seeds.generators import GaussianInit, ZeroInit

SUPPORTED_INIT_TYPES = ("zero", "gaussian")


def init_generators(config: configparser.ConfigParser, seed: int) -> list:
    """Return the generators named by ``[init] types``, in config order.

    Only the target-free initializers are accepted here: a workflow may not
    silently enable an initializer that needs product information.
    """
    kinds = [kind.strip().lower() for kind in config["init"]["types"].split(",") if kind.strip()]
    unknown = set(kinds) - set(SUPPORTED_INIT_TYPES)
    if unknown:
        raise ValueError(
            f"init types must be one of {SUPPORTED_INIT_TYPES}, got {sorted(unknown)}"
        )
    generators: list = []
    if "zero" in kinds:
        generators.append(ZeroInit())
    if "gaussian" in kinds:
        generators.append(
            GaussianInit(scale=config["init"].getfloat("gaussian_scale"), random_seed=seed)
        )
    if not generators:
        raise ValueError("init types cannot be empty")
    return generators
