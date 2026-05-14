"""
Display utilities: rich-output helpers and package availability checks.
"""
from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json


def package_status(package_names: list[str]) -> list[dict]:
    rows = []
    for name in package_names:
        available = importlib.util.find_spec(name) is not None
        try:
            version = importlib_metadata.version(name) if available else None
        except importlib_metadata.PackageNotFoundError:
            version = "installed, version unknown" if available else None
        rows.append({"package": name, "available": available, "version": version})
    return rows


def display_table(rows, columns=None) -> None:
    """Display rows as a DataFrame when pandas/IPython are available, otherwise print JSON."""
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(rows, columns=columns))
    except Exception:
        print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))


def show_object(obj) -> None:
    """Display rich objects in notebooks, with a plain-text fallback."""
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        print(obj)
