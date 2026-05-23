"""Small validation and display helpers used across the package."""

from __future__ import annotations

import importlib
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class LabDataError(RuntimeError):
    """Raised when required lab data are missing or invalid."""


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required for a requested operation."""


def as_path(path: str | Path) -> Path:
    """Return ``path`` as an expanded :class:`~pathlib.Path` without requiring existence."""

    return Path(path).expanduser()


def require_file(path: str | Path, label: str = "file") -> Path:
    """Validate that a required file exists and return it as a Path."""

    p = as_path(path)
    if not p.exists():
        raise LabDataError(
            f"Required {label} not found: {p}. "
            "Check DATA_DIR or pass the correct path to the package function."
        )
    if not p.is_file():
        raise LabDataError(f"Expected {label} to be a file, but got: {p}")
    return p


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str = "data") -> None:
    """Raise a concise error if ``frame`` lacks any required columns."""

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise LabDataError(f"{label} is missing required columns: {missing}")


def optional_import(module_name: str, purpose: str | None = None) -> Any:
    """Import an optional dependency or raise an actionable error."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"Optional package '{module_name}' is required"
        if purpose:
            msg += f" for {purpose}"
        msg += ". Install it in the lab environment or skip this optional output."
        raise OptionalDependencyError(msg) from exc


def module_version(module_name: str) -> str:
    """Return a module version string, or a readable availability status."""

    if module_name == "python":
        return platform.python_version()
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "installed; version unavailable")


def safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric while preserving NaN for invalid values."""

    return pd.to_numeric(series, errors="coerce")


def finite_mask(values: Sequence[float]) -> np.ndarray:
    """Boolean mask for finite numeric values."""

    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr)


def dataframe_overview(df: pd.DataFrame, label: str = "data") -> pd.DataFrame:
    """Compact notebook-friendly column overview."""

    return pd.DataFrame(
        {
            "dataset": label,
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def warn_if(condition: bool, message: str) -> None:
    """Emit a warning when ``condition`` is true."""

    if condition:
        warnings.warn(message, stacklevel=2)


def tidy_metric_rows(
    records: list[dict[str, Any]],
    sort_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a consistently sorted metrics table."""

    out = pd.DataFrame.from_records(records)
    if sort_by and not out.empty:
        sort_cols = [c for c in sort_by if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def clip_n_splits(n_requested: int, n_observations: int, n_groups: int | None = None) -> int:
    """Choose a valid number of CV splits for the available observations/groups."""

    upper = n_observations if n_groups is None else min(n_observations, n_groups)
    return max(2, min(int(n_requested), int(upper)))


def safe_divide(num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float) -> Any:
    """Divide while returning NaN for zero or invalid denominators."""

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.asarray(num, dtype=float) / np.asarray(den, dtype=float)
    result = np.where(np.isfinite(result), result, np.nan)
    return result


def short_status(ok: bool, detail: str) -> str:
    """Return a compact status label for QA tables."""

    return ("OK: " if ok else "CHECK: ") + detail


def normalise_bool_column(series: pd.Series) -> pd.Series:
    """Coerce common boolean encodings to bool while preserving missing values."""

    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
    }
    return series.astype("string").str.lower().map(mapping).astype("boolean")
