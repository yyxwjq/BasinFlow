"""Reading and validating a workflow INI config.

Validation is two-sided: required keys must be present, and every key present must
be declared here.
"""
from __future__ import annotations

import configparser
from collections.abc import Mapping
from pathlib import Path

# Every section and key a workflow config may contain. One workflow reads all of
# them: the choice of backend changes the network, not the sections.
ALLOWED: dict[str, frozenset[str]] = {
    "data": frozenset({"events_dir", "auto_split", "regenerate_split"}),
    "output": frozenset({"output_dir", "log_to_stdout"}),
    "split": frozenset({"train", "val", "test", "seed"}),
    "model": frozenset({
        "backend", "hidden_dim", "num_features", "num_layers", "cutoff", "r_max",
        "num_rbf", "num_radial_basis", "num_radial", "num_elements", "time_dim",
        "radial_basis", "envelope", "stability_mode", "time_scale", "time_per_layer",
        "max_atomic_number",
        # LEFTNet (ReactOT reference backbone) construction switches.
        "reflect_equiv", "legacy", "update", "single_layer_output", "object_aware",
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
    "loss": frozenset({"norm", "velocity_weight"}),
    "runtime": frozenset({"device", "cpu_threads"}),
    "evaluation": frozenset({
        "eval_num_steps", "trials_per_event", "num_events", "sampling_seed",
        "split", "integrator", "solver_atol", "solver_rtol",
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
