"""Reading and validating a workflow INI config.

Validation is two-sided: required keys must be present, and every key present must
be declared here.
"""
from __future__ import annotations

import configparser
from collections.abc import Mapping
from pathlib import Path

# Every section and key a workflow config may contain. Sections are shared across
# the three workflows; a workflow's own ``required`` table narrows what it needs.
ALLOWED: dict[str, frozenset[str]] = {
    "data": frozenset({
        "events_dir", "auto_split", "regenerate_split",
    }),
    "output": frozenset({"output_dir", "log_to_stdout"}),
    "split": frozenset({"train", "val", "test", "seed"}),
    "model": frozenset({
        "backend", "hidden_dim", "num_features", "num_layers", "cutoff", "r_max",
        "num_rbf", "num_radial_basis", "num_radial", "num_elements", "time_dim",
        "radial_basis", "envelope", "stability_mode", "bond_change_condition",
        "bond_change_direction", "bond_change_source", "endpoint_condition",
        "endpoint_equivariant", "time_scale", "time_per_layer", "max_atomic_number",
        "seed_type_vocab",
        # LEFTNet (ReactOT reference backbone) construction switches.
        "reflect_equiv", "legacy", "update", "single_layer_output", "object_aware",
    }),
    "init": frozenset({
        "types", "gaussian_scale", "time_distribution", "beta_alpha", "flow_time",
        "source_scale", "time_epsilon", "path",
    }),
    "training": frozenset({
        "epochs", "max_steps", "lr", "batch_size", "optimizer", "amsgrad",
        "weight_decay", "max_grad_norm", "scheduler", "warmup_fraction",
        "min_lr_fraction", "validate_each_epoch", "save_each_epoch", "seed",
        "resume_from", "ema_decay", "batch_sampler", "max_num", "grad_clip",
        "limit_train_batches", "validation_split", "val_max_num", "limit_val_batches",
        "early_stopping_patience", "swap_reactant_product", "init_checkpoint",
        "save_every_n_epochs",
    }),
    "bridge": frozenset({"timesteps", "beta_max", "power", "inv_power", "t_eps"}),
    "loss": frozenset({"velocity_weight", "norm"}),
    "runtime": frozenset({"device", "cpu_threads", "gpu_ids", "gpu_count"}),
    "evaluation": frozenset({
        "eval_num_steps", "trials_per_event", "trials_per_test_event",
        "num_test_events", "num_events", "sampling_seed",
        "selection_seed", "split", "bond_change_source", "write_trajectories",
        "integrator", "solver_atol", "solver_rtol",
    }),
    "relaxation": frozenset({
        "enabled", "eam_potential", "eam_form", "relax_fmax", "relax_steps",
    }),
}


def read_config(
    path: Path | str,
    required: Mapping[str, tuple[str, ...]],
) -> configparser.ConfigParser:
    """Read a workflow INI, rejecting unknown sections/keys and missing required ones."""
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"config file not found: {path}")
    for section in config.sections():
        if section not in ALLOWED:
            raise ValueError(f"unknown config section [{section}] in {path}")
        unknown = sorted(set(config[section]) - ALLOWED[section])
        if unknown:
            raise ValueError(f"unknown keys in [{section}] of {path}: {', '.join(unknown)}")
    for section, keys in required.items():
        if not config.has_section(section):
            raise ValueError(f"config is missing [{section}]")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"config [{section}] is missing keys: {', '.join(missing)}")
    return config


def optional_section(config: configparser.ConfigParser, name: str) -> Mapping[str, str]:
    """A section, or an empty mapping when the config does not have it."""
    return config[name] if config.has_section(name) else {}


def parse_bool(value: str) -> bool:
    """Parse the boolean spellings accepted in workflow INI files."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")
