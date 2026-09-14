"""Entry points must make ``basinflow`` importable on their own.

The package uses a ``src/`` layout and is not assumed to be installed, so every
script under ``examples/`` and ``tools/`` that imports ``basinflow`` has to put
``REPO_ROOT/src`` on ``sys.path`` first. One tool shipped without that preamble
and died with ``ModuleNotFoundError`` on its first run, which is the regression
these tests pin down.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIRS = ("examples", "tools")
CANONICAL = "sys.path.insert(0, str(REPO_ROOT))"


def _entry_points():
    for directory in ENTRY_DIRS:
        yield from sorted((REPO_ROOT / directory).glob("*.py"))


def _imports_package(text):
    return any(line.startswith(("from basinflow", "import basinflow"))
               for line in text.splitlines())


def test_entry_points_importing_the_package_bootstrap_sys_path():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _entry_points()
        if _imports_package(path.read_text()) and CANONICAL not in path.read_text()
    ]
    assert offenders == [], f"entry points missing the sys.path bootstrap: {offenders}"


def test_entry_points_do_not_reinvent_the_bootstrap():
    """A second convention (e.g. a local SRC_DIR) must not creep back in."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _entry_points()
        if "sys.path.insert" in path.read_text() and "SRC_DIR" in path.read_text()
    ]
    assert offenders == [], f"entry points using a non-canonical bootstrap: {offenders}"
