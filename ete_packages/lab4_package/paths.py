"""Path discovery and output-directory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from . import config
from .utils import as_path, require_file


def find_project_root(start: str | Path | None = None, markers: Iterable[str] = ("data", ".git", "pyproject.toml")) -> Path:
    """Find a plausible project root by walking upward from ``start``.

    If no marker is found, the current working directory is returned.
    """

    current = as_path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


def discover_data_file(
    filename: str = config.DEFAULT_GPKG_NAME,
    search_roots: Iterable[str | Path] | None = None,
    required: bool = True,
) -> Path | None:
    """Search common teaching locations for a data file."""

    roots = list(search_roots or [Path.cwd() / "data", Path.cwd(), Path("/mnt/data")])
    for root in roots:
        candidate = as_path(root) / filename
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Could not find {filename}. Searched: {[str(as_path(r)) for r in roots]}"
        )
    return None


def ensure_output_dir(path: str | Path = config.DEFAULT_OUTPUT_DIR) -> Path:
    """Create and return an output directory."""

    p = as_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_file_availability(paths: Mapping[str, str | Path]) -> pd.DataFrame:
    """Return a notebook-friendly table showing which expected files are available."""

    rows: list[dict[str, object]] = []
    for key, raw_path in paths.items():
        p = as_path(raw_path)
        rows.append(
            {
                "key": key,
                "path": str(p),
                "exists": p.exists(),
                "is_file": p.is_file(),
                "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
                "suffix": p.suffix if p.exists() else None,
            }
        )
    return pd.DataFrame(rows)
