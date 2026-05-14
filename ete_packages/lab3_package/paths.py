"""
Project-root discovery and file-location utilities.

These helpers are intentionally filesystem-only (no geo or analysis imports)
so they can run as part of the notebook bootstrap before the analysis stack
is imported.
"""
from __future__ import annotations

from pathlib import Path

from .config import GPKG_FILENAME

_PROJECT_MARKERS: list[str] = ["README.md"]


def candidate_roots(start: Path) -> list[Path]:
    current = start.resolve()
    return [current, *current.parents]


def find_project_root(start: Path) -> Path:
    """Walk upward from *start* until a directory containing all project markers is found."""
    for root in candidate_roots(start):
        if all((root / marker).exists() for marker in _PROJECT_MARKERS):
            return root
    for root in candidate_roots(start):
        if (root / "data" / GPKG_FILENAME).exists():
            return root
    return start.resolve()


def find_file(filename: str, project_root: Path, preferred_subdir: str | None = None) -> Path:
    """Locate *filename* relative to *project_root*, checking preferred_subdir first."""
    if preferred_subdir:
        candidate = project_root / preferred_subdir / filename
        if candidate.exists():
            return candidate
    direct = project_root / filename
    if direct.exists():
        return direct
    matches = list(project_root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename!r} below {project_root}")
