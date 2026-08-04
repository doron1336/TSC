"""
Central path configuration for the TSC pipeline.

All scripts and modules in this repo read their filesystem layout from this
file. Override the defaults by setting the environment variables listed below
before running any script.

Environment variables
---------------------
TSC_CODE_DIR        Directory that contains the `TSC/` Python package.
                    Default: parent of this file + '/code'.

TSC_DATA_DIR        Directory that contains the UCR archive, handMovement
                    raw .mat files, and any other read-only input data.
                    Default: parent of TSC_CODE_DIR + '/data' (i.e. one
                    level above `code/`, sitting next to it).

TSC_RESULTS_DIR     Directory where pipeline outputs (per-dataset pickles,
                    averaged plots, generated tables) are written.
                    Default: parent of TSC_CODE_DIR + '/results'.

Defaults are resolved lazily so this module is safe to import from any
working directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------
# paths.py lives at the repo root. The Python package lives at
# <REPO_ROOT>/code/TSC/ and externalized data lives at <REPO_ROOT>/data and
# <REPO_ROOT>/results. We treat REPO_ROOT as the parent of the directory
# containing this file.
#
# If you check the repo out elsewhere, set the three env vars to absolute
# paths and the rest of the pipeline will follow.
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent

_DEFAULT_CODE_DIR = _REPO_ROOT / "code"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"


def _resolve(var: str, default: Path) -> Path:
    """Read an env var or fall back to the default path, expanding ~."""
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


CODE_DIR: Path = _resolve("TSC_CODE_DIR", _DEFAULT_CODE_DIR)
DATA_DIR: Path = _resolve("TSC_DATA_DIR", _DEFAULT_DATA_DIR)
RESULTS_DIR: Path = _resolve("TSC_RESULTS_DIR", _DEFAULT_RESULTS_DIR)


# Add CODE_DIR to sys.path so `from TSC import ...` works whether the user
# runs scripts from the repo root or from inside CODE_DIR. Don't worry about
# idempotency — Python's path manipulation is a list anyway.
import sys  # noqa: E402

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ---------------------------------------------------------------------------
# Concrete paths the pipeline reads & writes
# ---------------------------------------------------------------------------
# Inside DATA_DIR you should have:
#   - ucr_archive/                 # the UCR Time Series Classification Archive
#   - hand_movement/               # raw .mat files for hand-movement experiments
#
# Inside RESULTS_DIR the pipeline creates:
#   - per_dataset/<dataset>/<fold>/{algo_manager, selected_<k>_features, ...}
#   - figures/                     # averaged HAR plots
#   - tables/                      # LaTeX tables
# ---------------------------------------------------------------------------

UCR_DIR: Path = DATA_DIR / "ucr_archive"
HAND_MOVEMENT_DIR: Path = DATA_DIR / "hand_movement"

PER_DATASET_DIR: Path = RESULTS_DIR / "per_dataset"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def ensure_results_subdirs() -> None:
    """Create the standard output subdirectories if they don't exist."""
    for d in (PER_DATASET_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset_name: str, fold: Optional[int] = None) -> Path:
    """Return the per-dataset scratch directory, optionally for a specific fold."""
    if fold is None:
        return PER_DATASET_DIR / dataset_name
    return PER_DATASET_DIR / dataset_name / str(fold)


def ucr_dataset_dir(dataset_name: str) -> Path:
    """Return the location of the UCR TSV files for a given dataset."""
    return UCR_DIR / dataset_name


# ---------------------------------------------------------------------------
# Diagnostic print on import (helpful in CI / fresh clones)
# ---------------------------------------------------------------------------
def _summarize() -> str:
    return (
        f"TSC paths: code={CODE_DIR}  data={DATA_DIR}  results={RESULTS_DIR}\n"
        f"           ucr={UCR_DIR}  hand_movement={HAND_MOVEMENT_DIR}"
    )


if __name__ == "__main__":
    print(_summarize())
    print()
    print("Override with env vars:")
    print("  TSC_CODE_DIR, TSC_DATA_DIR, TSC_RESULTS_DIR")
