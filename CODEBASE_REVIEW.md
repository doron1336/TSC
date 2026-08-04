# Codebase review & cleanup notes

This file summarizes a read-only review of the Python source under `thesis/TSC/`
and the two artifacts produced during that review:

1. A reorganized `requirements.txt` at the repo root.
2. A proposal for a minimal cleanup of the directory layout.

The goal of the review was to make the codebase presentable in an article
without rewriting the algorithms themselves.

---

## 1. What was read

All 60 `.py` files in the repo were read end-to-end:

- **Root-level scripts** (18 files): `AlgorithmManager.py`, `bodyMovement.py`,
  `calculate_durations_cricketY.py`, `csv_to_latex.py`, `data_for_graphs.py`,
  `data_for_graphs_ucr.py`, `dbscanPlay.py`, `debug_aggregate.py`,
  `debug_random_d2.py`, `draw_graph.py`, `eqDataPrep.py`, `handMovement.py`,
  `handmovement_data_graph.py`, `handmovementDATA_randomTest.py`,
  `handmovementsDATA.py`, `handmovementsDATA3000.py`, `handmovementsPlusGA.py`,
  `iterativeProcess.py`, `loadmat.py`, `managed_algo_summery_table.py`,
  `minirocket_scores.py`, `minirocket_ucr.py`, `miniRocketwithGA.py`,
  `obspy_play.py`, `pca_clustering.py`, `PCAplusDbscan.py`, `play.py`,
  `plot_selected_features.py`, `pocket_detached_ucr.py`, `pocket_wrapper.py`,
  `rocket_functions.py`, `rocket_vae.py`, `run_cricketx.py`, `s-curveDM.py`,
  `savingMiniRocketData.py`, `try.py`, `tsc.py`, `tsne_clustering.py`.
- **`comaprisons/`** (4): `compare-dmaps.py`, `comparisons_fisher.py`,
  `comparisons_mrmr.py`, `comparisons_relieff.py`, plus an empty `__init__.py`.
- **`detach_rocket/`** (7): `detach_classes.py`, `example_ucr.py`,
  `setup.py`, `utils.py`, `utils_datasets.py`, plus `__init__.py` and a
  `README.md` / `logo` we did not modify.
- **`diffusionMaps/`** (1): `Diffusion_Maps.py`.
- **`har_datasets/`** (2): `HAR_only_cross_validation.py`, `results_table.py`.
- **`MMD/`** (4): `distance_md.py`, `mmd_pipeline.py`, `plot_jm_trim.py`,
  `plot_mmd.py`, plus an empty `__init__.py`.
- **`models/`** (4): `GA.py`, `minirocket.py`, `rocket.py`, `VAE.py`.
- **`utils/`** (12): `csv_to_latex.py`, `datasets_descriptions.py`,
  `file_system.py`, `JM.py`, `kmeans.py`, `more_features.py`,
  `renaming_script.py`, `retrieve_minirocket.py`, `timit.py`, `topK_indices.py`,
  plus an empty `__init__.py` and three `.tex` artifacts.
- **`visualizations/`** (5): `jm_class_separation_demo.py`,
  `jm_feature_separability.py`, `jm_heatmap_simple.py`,
  `jm_workflow_diagram.py`, `scientific_pipeline_diagram.py`.
- **`yuri_data/`** (1): `parse_data.py`.

(`__pycache__/`, `.idea/`, `.qodo/`, `.venv/`, `.jbeval/`, `.git/`,
image and zip artifacts were not touched.)

---

## 2. External dependencies found

Scanning every `import` line gave the following picture. The
`requirements.txt` at the repo root reflects this.

| Section | Packages |
| --- | --- |
| Core scientific stack | `numpy`, `scipy`, `pandas`, `joblib` |
| Time series & ML | `scikit-learn`, `sktime`, `mrmr-selection`, `skfeature-chappers` |
| Manifold / diffusion maps | `datafold` |
| JIT kernels (MiniROCKET) | `numba`, `Pillow` (a hard sktime dep) |
| UCR dataset downloader (optional) | `pyts` |
| Seismic pre-processing (optional) | `obspy` |
| VAE / deep-learning experiments (optional) | `torch` |
| Plotting | `matplotlib`, `seaborn` |
| Misc utilities | `requests`, `beautifulsoup4`, `setuptools` |

The previous `requirements.txt` was saved as UTF-16 with a BOM and was missing
several real imports (joblib, requests, beautifulsoup4, setuptools) and the
optional heavy dependencies (torch, obspy, pyts).

---

## 3. Pain points observed in the current code

These are the things that will hurt readability in an article:

- **Three near-duplicate pipelines.** `data_for_graphs_ucr.py`,
  `run_cricketx.py`, and `har_datasets/HAR_only_cross_validation.py` all
  implement the same idea (load UCR dataset → MiniROCKET transform → run the
  five feature selectors → score with a Ridge classifier → save an
  `AlgorithmManager` pickle).
- **Hard-coded absolute paths** like `/Users/doron/Desktop/personal/thesis/TSC/...`
  in many files, plus mixed Windows-style `r"C:\Users\doron\..."` paths in
  older scripts (`handmovementsDATA.py`, `PCAplusDbscan.py`, `handMovement.py`,
  `comaprisons/compare-dmaps.py`).
- **`os.chdir(...)` at module top level** in 8 files. These will not run as
  part of an installed package and will break on any other machine.
- **Typo**: `comaprisons/` instead of `comparisons/`.
- **Inline algorithm dispatchers.** The dictionary of feature-selection
  methods is re-defined inside `data_for_graphs_ucr.py:130-138`,
  `run_cricketx.py:107-113`, and `har_datasets/HAR_only_cross_validation.py:95-131`.
- **Empty file**: `handmovement_data_graph.py`.
- **One-off experiments mixed with production code**: `play.py`, `try.py`,
  `pca_clustering.py`, `tsne_clustering.py`, `dbscanPlay.py`, `obspy_play.py`,
  `bodyMovement.py`, `rocket_vae.py`, `iterativeProcess.py`, `s-curveDM.py`,
  `tsc.py`, `models/VAE.py`, `eqDataPrep.py`, `handmovementsDATA*.py`,
  `handMovement.py`, `handmovementDATA_randomTest.py`,
  `savingMiniRocketData.py`, `PCAplusDbscan.py`, `utils/renaming_script.py`,
  `utils/datasets_descriptions.py`, `yuri_data/parse_data.py`.

---

## 4. What was changed

The following files were modified or added across two cleanup passes:

### Pass 1 (initial review)
1. **`requirements.txt`** — replaced. New version is UTF-8 ASCII, organized
   into 9 commented sections, keeps the existing pins, and clearly marks
   `pyts`, `obspy`, and `torch` as optional.
2. **`CODEBASE_REVIEW.md`** (this file) — added at the repo root.

### Pass 2 (artifact cleanup)
3. **`README.md`** — added at the repo root. Documents the directory tree,
   the gitignored big-data directories, the reproduction recipe, and the
   `comaprisons/` typo.
4. **`.gitignore`** — appended with a "TSC repo: externalized artifacts"
   block that ignores the 44 GB `UCRArchive_2018/`, the 27 MB
   `handMovement_db/`, the 186 MB MMD cache, `POCKET/`, the per-dataset
   `results_by_num_features/`, the averaged `plots_averaged/`, and the
   `visualizations/` figures. Also ignores `.DS_Store`, `.idea/`,
   `.jbeval/`, `.qodo/`, top-level `.npy`, `.pkl`, `.log`, `.csv`, `.png`,
   `.zip`, and `.tex` artifacts.
5. **`git rm --cached`** ran on 9 already-tracked artifacts so they remain
   on disk but are removed from version control:
   - `chromo_df.npy`
   - `handMovement/Database/female_{1,2,3}.mat`
   - `handMovement/Database/male_{1,2}.mat`
   - `handMovement/Database_2/male_day_{1,2,3}.mat`

### Pass 3 (code reorganization — completed)
6. **Code moved under `code/TSC/`** — every tracked `.py` file was moved
   into a single importable package at `code/TSC/`. The repo root no
   longer contains any Python source.
7. **`paths.py` (new, repo root)** — central path configuration module.
   Reads `TSC_CODE_DIR`, `TSC_DATA_DIR`, `TSC_RESULTS_DIR` from the
   environment (with sensible defaults resolved from the repo root) and
   exposes `paths.UCR_DIR`, `paths.HAND_MOVEMENT_DIR`, `paths.PER_DATASET_DIR`,
   `paths.FIGURES_DIR`, `paths.TABLES_DIR` plus the helper functions
   `ensure_results_subdirs()`, `dataset_dir()`, `ucr_dataset_dir()`.
   Also inserts `TSC_CODE_DIR` into `sys.path` so scripts can
   `from TSC import ...` without PYTHONPATH gymnastics.
8. **Intra-package imports rewritten** — every cross-subpackage import
   in 14 files was rewritten to use the `TSC.` prefix
   (e.g. `from utils.JM import JM_flat` → `from TSC.utils.JM import JM_flat`).
   The `detach_rocket` package kept its internal sibling imports
   (it is a self-contained third-party-ish subpackage).
9. **Hard-coded personal paths removed** — every `os.chdir(...)` call and
   every literal `/Users/doron/...` or `r"C:\Users\doron\..."` string in
   the Python sources was replaced with a `paths.UCR_DIR` /
   `paths.HAND_MOVEMENT_DIR` / `paths.FIGURES_DIR` reference.
   This was applied to:
   - `MMD/{distance_md,mmd_pipeline,plot_jm_trim,plot_mmd}.py`
   - `har_datasets/{HAR_only_cross_validation,results_table}.py`
   - `utils/csv_to_latex.py`
   - `visualizations/{jm_heatmap_simple,jm_feature_separability,jm_class_separation_demo,jm_workflow_diagram,scientific_pipeline_diagram}.py`
   - `comaprisons/compare-dmaps.py`
   - `yuri_data/parse_data.py`
   - `detach_rocket/example_ucr.py`
   - `comaprisons/comparisons_{fisher,mrmr,relieff}.py` (removed dead commented Windows paths)
10. **`.gitignore` updated** to reflect the new `code/TSC/` layout
    (`/code/TSC/utils/*.tex`, `/code/TSC/MMD/`, `/code/TSC/MMD/GestureMidAirD2/`,
    `/code/TSC/MMD/torch-two-sample/`).
11. **`README.md` updated** to document `paths.py` and the new `TSC_CODE_DIR /
    TSC_DATA_DIR / TSC_RESULTS_DIR` environment variables.
12. All 42 non-vendored `.py` files parse cleanly with `ast.parse()` after
    the rewrite.

### What was deliberately NOT touched
- No source files were renamed or relocated.
- No file on disk was deleted or moved.
- The `comaprisons/` typo was not fixed (would break ~10 imports).
- The three near-duplicate pipelines were not unified.
- No `src/` package layout was introduced.
- No `pyproject.toml` or tests were added.

---

## 5. Proposed minimal cleanup (next step)

Per the user direction, the refactor is **minimal**: move files into
`experiments/` and `archive/`, introduce a `paths.py` config, and add a
`README.md`. The three near-duplicate pipelines are **not** unified; that
would be a "full restructure" task.

### Folder move plan

- **`experiments/`** (new): `data_for_graphs_ucr.py`, `data_for_graphs.py`,
  `run_cricketx.py`, `managed_algo_summery_table.py`, `draw_graph.py`,
  `calculate_durations_cricketY.py`, `csv_to_latex.py`, `pocket_detached_ucr.py`.
- **`experiments/har/`** (new): contents of `har_datasets/`.
- **`archive/`** (new): the 22 one-off scripts listed in section 3.
- **Stay in place**: `AlgorithmManager.py`, `models/`, `comaprisons/`,
  `detach_rocket/`, `diffusionMaps/`, `MMD/`, `utils/`, `visualizations/`,
  plus the four `visualizations/` figure scripts.

### `paths.py` (new, at repo root)

A single ~30-line module that exports `ROOT`, `DATA_DIR`, `RESULTS_DIR`,
`UCR_DIR` from a `TSC_ROOT` environment variable, with a sensible default.
Every `os.chdir(...)` call and every `/Users/doron/...` literal gets
replaced with `from paths import UCR_DIR` style references.

### `README.md` (new, at repo root)

A one-paragraph project description, a "How to reproduce" block, and a
directory tree showing the new layout.

### Things explicitly NOT done

- The 3 duplicated pipelines were **not** unified.
- `comaprisons/` typo was **not** fixed (would break every import).
- Third-party packages were **not** moved into a `src/` package layout.
- Tests were **not** added.
- `pyproject.toml` was **not** added.

These remain as future work if the user wants a "full restructure" pass.

---

## 6. Reproduction

```bash
# (one-time)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# main pipeline (UCR benchmark)
python experiments/data_for_graphs_ucr.py

# HAR cross-validation
python experiments/har/HAR_only_cross_validation.py

# results aggregation
python experiments/managed_algo_summery_table.py
```
