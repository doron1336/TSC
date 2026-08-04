# TSC — Time Series Classification with MiniROCKET + JM-KMeans Feature Selection

This repository contains the implementation behind a thesis on time series
classification. Five feature-selection methods (Fisher, mRMR, ReliefF, random
baseline, and the proposed **JM-KMeans** method) are compared on the UCR Time
Series Archive and on seven Human Activity Recognition (HAR) datasets.

All Python code lives in a single importable package at `code/TSC/`. Every
filesystem location (input data, results cache, generated figures, LaTeX
tables) is configured through `paths.py` at the repo root — set the three
environment variables to override the defaults and you can move the data
directory anywhere.

---

## Repository layout

```
TSC/
├── paths.py                       # Central path config (env-var driven, see below)
├── AlgorithmManager.py            # (legacy module — kept at root for import compat)
├── CODEBASE_REVIEW.md             # Read-only review of every .py file + cleanup history
├── LICENSE
├── README.md                      # ← this file
├── requirements.txt               # Pinned pip dependencies (organized by section)
│
├── code/                          # All Python source, organized as a package
│   └── TSC/
│       ├── __init__.py            # Marks the inner package
│       ├── AlgorithmManager.py    # Pickle container for per-algorithm scores + durations
│       ├── models/                # MiniROCKET, ROCKET, GA, VAE
│       ├── comaprisons/           # Baseline feature selectors (fisher, mrmr, relieff, dm)
│       │                          # (typo "comaprisons" is historical — do NOT rename)
│       ├── detach_rocket/         # Uribarri & Barone's DetachROCKET subpackage
│       ├── diffusionMaps/         # Diffusion Maps ranking
│       ├── MMD/                   # Maximum Mean Discrepancy experiments
│       ├── utils/                 # JM distance, K-Means, file IO, scoring, dataset IO
│       ├── visualizations/        # JM/JM-trim figures used in the article
│       ├── har_datasets/          # HAR stratified 5-fold cross-validation pipeline
│       └── yuri_data/             # Legacy hand-movement data parsing
│
├── data/                          # Input data — set TSC_DATA_DIR (default: ../data)
│   ├── ucr_archive/               #   UCR Time Series Classification Archive
│   └── hand_movement/             #   Raw .mat files for handMovement experiments
│
└── results/                       # Pipeline outputs — set TSC_RESULTS_DIR (default: ../results)
    ├── per_dataset/               #   per_dataset/<dataset>/<fold>/algo_manager
    ├── figures/                   #   averaged HAR plots
    └── tables/                    #   LaTeX tables
```

---

## Path configuration

`paths.py` at the repo root is the single source of truth for filesystem
locations. The defaults resolve to a sibling-directory layout:

| Variable          | Default                  | What it controls                          |
| ----------------- | ------------------------ | ----------------------------------------- |
| `TSC_CODE_DIR`    | `<repo>/code`            | Where the importable `TSC` package lives  |
| `TSC_DATA_DIR`    | `<repo>/../data`         | Read-only input data (UCR archive, etc.)  |
| `TSC_RESULTS_DIR` | `<repo>/../results`      | Pipeline outputs (figures, tables, caches)|

Override any of them by exporting before running a script:

```bash
export TSC_DATA_DIR=/scratch/ucr-archive
export TSC_RESULTS_DIR=/scratch/tsc-results
python code/TSC/MMD/mmd_pipeline.py
```

`paths.py` also adds `TSC_CODE_DIR` to `sys.path` automatically, so scripts
can `from TSC.utils import ...` without any PYTHONPATH gymnastics.

### Helper functions exposed by `paths`

- `paths.UCR_DIR`             — points at `data/ucr_archive`
- `paths.HAND_MOVEMENT_DIR`   — points at `data/hand_movement`
- `paths.PER_DATASET_DIR`     — points at `results/per_dataset`
- `paths.FIGURES_DIR`         — points at `results/figures`
- `paths.TABLES_DIR`          — points at `results/tables`
- `paths.ensure_results_subdirs()` — creates the three results subdirectories
- `paths.dataset_dir(name, fold=...)` — per-dataset scratch directory
- `paths.ucr_dataset_dir(name)` — location of UCR TSV files for one dataset

---

## Big directories (kept on disk, **not** tracked by git)

These are present on disk after `git clone` (or you can recreate them); they
are excluded via `.gitignore` so `git status` stays clean.

| Directory on disk                  | Size    | Source / how to recreate                                                  |
| ---------------------------------- | ------- | ------------------------------------------------------------------------- |
| `UCRArchive_2018/` (sibling)       | ~44 GB  | Download from <https://www.timeseriesclassification.com/Downloads/Archive.zip> |
| `handMovement_db/` (sibling)       | ~27 MB  | Raw `.mat` files for the hand-movement experiments                        |
| `code/TSC/MMD/torch-two-sample/`   | small   | `pip install torch-two-sample`                                            |
| `POCKET/` (top-level)              | ~1 MB   | POCKET pruning baselines — third-party code                              |

The MMD cache (`code/TSC/MMD/GestureMidAirD2/`) is regenerated by
`code/TSC/MMD/mmd_pipeline.py`.

---

## Reproduction

These commands assume the gitignored big directories are in place (see
"Big directories" above). The pipeline has three stages; each is independent
and writes only into the gitignored output directories.

### 1. Set up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the UCR benchmark (per-dataset scores + durations)

```bash
# The pipeline driver lives under code/TSC/. From the repo root:
python -m TSC.har_datasets.HAR_only_cross_validation     # or wherever the UCR runner is
```

### 3. Run the HAR cross-validation (5-fold, stratified)

```bash
python -m TSC.har_datasets.HAR_only_cross_validation
# → writes results/per_dataset/<dataset>/<fold>/algo_manager
```

### 4. Aggregate results into CSV + LaTeX

```bash
# Per-dataset summary across folds for HAR datasets
python -m TSC.har_datasets.results_table
# → writes results/tables/har_scores_dataframe.csv (and into ucr/HAR_datasets/...)

# CSV → LaTeX for the article
python -m TSC.utils.csv_to_latex
# → writes results/tables/HAR_table.tex and HAR_table_duration.tex
```

### 5. Generate figures

```bash
python -m TSC.visualizations.scientific_pipeline_diagram
python -m TSC.visualizations.jm_workflow_diagram
python -m TSC.visualizations.jm_feature_separability
```

---

## Optional: cleaning a fresh checkout

After `git clone`:

```bash
# 1. Get the data — point TSC_DATA_DIR at a parent directory that contains:
#    UCRArchive_2018/   handMovement_db/   (or place them under data/ inside the repo)

# 2. Optional outputs — fully regenerable, so just delete to start clean:
rm -rf results/ plots_averaged/ code/TSC/MMD/GestureMidAirD2/ code/TSC/MMD/torch-two-sample/

# 3. Run the pipeline as above
```

---

## See also

- `CODEBASE_REVIEW.md` — review of every Python file in the repo, the pain
  points that were observed, and what cleanup was already performed.
- `paths.py` — the path configuration module; every script imports it.
- `requirements.txt` — pinned dependencies, organized into nine labeled sections.
