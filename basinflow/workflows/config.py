"""Reading and validating a workflow INI config."""
from __future__ import annotations

import configparser
from collections.abc import Mapping
from pathlib import Path


def read_config(
    path: Path | str,
    required: Mapping[str, tuple[str, ...]],
) -> configparser.ConfigParser:
    """Read a workflow INI file and check that ``required`` sections and keys exist.

    ``required`` maps a section name to the keys it must contain, so a typo in a
    config fails here instead of deep inside training.
    """
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"config file not found: {path}")
    for section, keys in required.items():
        if not config.has_section(section):
            raise ValueError(f"config is missing [{section}]")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"config [{section}] is missing keys: {', '.join(missing)}")
    return config


def parse_bool(value: str) -> bool:
    """Parse the boolean spellings accepted in workflow INI files."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")
