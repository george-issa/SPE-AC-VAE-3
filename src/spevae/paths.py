"""Repo-anchored paths, so scripts and tests work from any working directory.

``DATA_DIR`` honours ``$SPEVAE_DATA`` first, then the repository's ``Data``
symlink/directory (two levels above this file's package when running from a
source checkout).  An installed package has no repo root, so consumers must
treat a missing directory as "no bundled data", not as an error.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ROOT", "DATA_DIR", "OUT_DIR", "data_path"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SPEVAE_DATA", REPO_ROOT / "Data"))
OUT_DIR = Path(os.environ.get("SPEVAE_OUT", REPO_ROOT / "out"))


def data_path(*parts: str) -> Path:
    """Join under ``DATA_DIR`` without asserting existence."""
    return DATA_DIR.joinpath(*parts)
