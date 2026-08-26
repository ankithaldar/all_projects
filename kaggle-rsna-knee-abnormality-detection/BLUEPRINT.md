# RSNA Knee Abnormalities Detection — MVP Blueprint

Multimodal (MRI + reports) detection of 12 knee abnormalities per study.
Metric: macro-averaged ROC-AUC across the 12 targets.

## 1. Objective and Hard Constraints

Produce a valid `submission.csv` from an offline Kaggle notebook using
PyTorch + Lightning + timm, targeting 2x T4 GPUs.

| Resource | Limit | Design consequence |
|---|---|---|
| Raw data | ~570 GB | Never copied. Mounted read-only under `/kaggle/input/`. Stream-decode per batch; only headers are indexed once |
| RAM | 30 GB | Streaming datasets; no whole-series in-RAM buffers |
| Disk (writable) | 20 GB | Only checkpoints / OOF / submission (<2 GB); pushed to a Kaggle Dataset after each session |
| Kernel runtime | 12 h/session | Training split across sessions via checkpoint-resume protocol (Section 6) |
| GPU quota | 30 h T4/week | MVP sized at 7-12 GPU-h (Section 8); every compute knob lives in YAML |
| Submission runtime | <= 9 h (competition rule) | Inference target < 45 min |

## 2. Measured Facts (from `notebooks/01_EDA.ipynb`)

| Finding | Design impact |
|---|---|
| 4,407 studies / 24,371 series (~5.5 series/study) | Cap 4 series x K=24 slices -> ~423k slice-forwards/epoch (~20-25 min on 2xT4). Full 5-fold x 4-epoch MVP = 7-12 GPU-h |
| 0% missing reports | Rule-based pseudo-labels cover every training study |
| Median voxel anisotropy 9.92x | Pure-3D convolutions ruled out; 2.5D confirmed |
| Oblique IOP vectors exist; some series unsorted or missing IPP | Slice order precomputed in `index.parquet` via projection onto slice normal; `InstanceNumber` fallback. Filename sort is unsafe |
| `PatientSex` present ~94% (M/F/O), `PatientAge` absent | Sex one-hot enters metadata vector; age dropped |
| 1000 sampled studies map to 1000 unique patients | GroupKFold by StudyUID; PatientID captured during header scan to re-group if multi-study patients appear |
| "200/200 corrupted" audit result was an EDA logic bug (rate over unchecked files) | Corrected audit moved to notebook 02; decoder registry keeps zeros-frame fallback regardless |
| Domain-shift analysis crashed before producing numbers | `index.parquet` stores `ScanningSequence` / `MagneticFieldStrength` so folds may stratify by protocol proxy; DANN deferred |
| Label-vs-text contradiction rates not yet quantified | Fixed contradiction-rate table becomes part of notebook 02 QC; label smoothing strength calibrated from it |
| 73 high-frequency DICOM tags discovered | Header-only index captures all tags per series (first slice) plus per-slice minimal set (IPP, IOP, InstanceNumber) |

## 3. MVP Scope

| In MVP | Deferred (backlog Section 11) |
|---|---|
| Header-only DICOM index (geometry, spacing, ordering) | Physical isotropic resampling |
| On-the-fly decode (gdcm/pylibjpeg fallback registry) | Decoded-pixel cache datasets |
| Percentile normalization + autocrop inside Dataset | MONAI bias-field / ghosting augmentations |
| Rule-based English pseudo-labels (negation-aware) | XLM-R multilingual labeler, soft-label distillation |
| 2.5D sampling: K slices/series -> attention pooling | Per-plane encoders, CLIP-style image-report pretraining |
| Shared backbone + plane/sequence embedding + FiLM(metadata) | TTA, snapshot ensembles |
| Weighted BCE (YAML-swappable Asymmetric Focal) | ONNX Runtime export, INT8/TensorRT |
| StratifiedGroupKFold 5 folds, OOF macro-AUC | Fold-count reduction experiments |
| Session-resume training via Kaggle Dataset versioning | Efficiency-prize micro-tuning |

## 4. Architecture Overview

```
StudyInstanceUID
 └── Series 1..N (plane, Fluid_Sensitive, Fat_Suppression, Sex from index.parquet)
      ├── ordered slice list (precomputed; zero DICOM parsing at train time)
      └── sample K=24 evenly spaced slices -> worker decodes JPEG/JP2
           -> rescale -> percentile normalize -> autocrop -> resize img_size
           └── shared Backbone (tf_efficientnetv2_s) -> slice embeddings
                -> AttentionPool -> series token (+ plane/seq embedding)
                 ▼
     StudyAggregator (learnable query cross-attends series tokens)
     ⊕ FiLM(metadata MLP: plane ⊕ FS ⊕ FL ⊕ sex ⊕ log counts ⊕ spacing stats)
                 ▼
          trunk MLP -> 12 sigmoid heads
```

Reports exist only at training time: gold labels where present,
rule-derived pseudo-labels elsewhere (Section 7).

## 5. Configuration System (zero hardcoded configuration)

### 5.1 Rules

- Every hyperparameter, path, seed, size, and class selection lives in `configs/*.yaml`.
- Any object is specified as:

```yaml
class_path: package.module.ClassName   # fully qualified import path
init_params:                           # kwargs for __init__, resolved recursively
  key: value
```

- Lists of such specs are allowed (augmentation pipelines, metrics).

### 5.2 Files

```
configs/
├── config.yaml          # experiment name, seed, paths (mounts, ckpt dataset slug,
│                        #   kaggle secret names), integrations, folds to run
├── data.yaml            # n_slices, img_size, series-selection policy,
│                        #   normalization percentiles, decode backend order
├── folds.yaml           # CV class_path + init_params (StratifiedGroupKFold)
├── augmentations.yaml   # albumentations pipelines: train / valid
├── model.yaml           # KneeNet tree: backbone, pools, aggregator, heads
├── loss.yaml            # criterion class_path + init_params
├── optimizer.yaml       # optimizer/scheduler class_paths, lr groups, warmup_epochs
├── train.yaml           # Trainer args (epochs, precision, devices),
│                        #   session_time_budget_h, checkpoint_every_n_epochs
├── datamodule.yaml      # DataModule/Dataset class_paths, batch_size, workers
├── infer.yaml           # ckpt fold selection, fp16, batch_size, submission_path
└── experiments/
    ├── mvp_efnv2s_384_k24_5f.yaml   # main experiment: defaults list + override
    └── smoke_ci.yaml                # CPU smoke run for CI/dry runs
```

Each experiment file deep-merges its `defaults:` stems in order, applies its
`override:` section last, and the fully resolved dictionary is dumped to
`artifact_dir/resolved_<name>.yaml` at every run start — one self-contained
file per experiment guarantees traceback.

### 5.3 Loader API (`src/knee/config_params/loader.py`)

```python
def load_config(path, overrides=None) -> dict:
    """Load YAML + dot-path overrides + ${a.b} interpolation."""

def load_experiment(path, overrides=None) -> dict:
    """Compose defaults list -> override -> interpolation (Section 5.2)."""

def dump_config(config, path) -> None:
    """Persist a resolved configuration for run traceability."""

def instantiate(spec) -> Any:
    """Recursively resolve class_path/init_params specs into live objects."""
```

Implementation uses `yaml.safe_load` + `importlib`; no third-party config framework.

## 6. Storage and Resume Protocol

Training cannot finish one 12 h session; kernels are ephemeral; GPU quota resets weekly.

Artifacts (all tiny; total < 2 GB):

| Artifact | Location | Content |
|---|---|---|
| `index.parquet`, `labels_pseudo.csv`, `folds.csv` | Kaggle Dataset `<slug>-index` | ordered SOP lists, spacing, plane/FS/FL, sex, protocol tags; supervision; splits |
| `fold{k}/last.ckpt`, `fold{k}/done`, `run_state.json` | Kaggle Dataset `<slug>-ckpt` (versioned) | model+optimizer+scheduler+epoch+step+best metric; fold completion markers |

Flow (`src/knee/helpers/kaggle_io.py`):

1. Credentials resolve through `knee.helpers.secrets`:
   process env -> project `.env` -> Kaggle `UserSecretsClient`; secret names
   are read from `config.kaggle_secrets.*`.
2. `pull_latest(slug, dest)` downloads newest version if it exists.
3. Engine start: per fold — `done` marker -> skip; `last.ckpt` -> resume via
   `Trainer.fit(ckpt_path=...)`.
4. `TimeBudgetCallback` stops fit when wall clock exceeds
   `session_time_budget_h - time_margin_min`; checkpoints also written every
   `checkpoint_every_n_epochs`.
5. Session end: `push_version(slug, folder)` creates a new immutable dataset
   version (rollback history).
6. Push retries x3 with backoff; on failure the ckpt remains in
   `/kaggle/working` with logged manual recovery path.
7. Inference consumes whichever folds have `done` markers (partial ensembles
   degrade gracefully).

## 6b. Experiment Tracking and Notifications

All three channels run simultaneously and independently; none can crash a
scoring run:

| Channel | Implementation | Content |
|---|---|---|
| CSV | `loggers/csv_logger.py` (Lightning `CSVLogger`) | per-fold metrics under `<output_dir>/logs/<exp>/fold{k}/` |
| W&B | `loggers/wandb_logger.py` | same metrics online when `WANDB_API_KEY` resolves; auto-offline otherwise |
| Discord | `loggers/discord_logger.py` | fit start, per-epoch macro-AUC, fold completion + duration, crash tracebacks via webhook |

Credentials resolve by *name* from YAML through
`helpers/secrets.get_secret(name)`: env -> `.env` -> Kaggle secrets
(see `.env.example`: `DISCORD_WEBHOOK_URL`, `WANDB_API_KEY`, `KAGGLE_*`).

## 7. NLP Pseudo-Labeling (MVP rules)

- Per-target lexicon regexes on lowercased report text.
- Negation triggers (`no|without|intact|unremarkable|absent`) within a +/-6-token window flip to negative.
- Uncertain cues (`rule out|cannot exclude|question`) emit `-1` mask (excluded from that target's loss).
- Output `labels_pseudo.csv`: hard labels + `source in {gold, rules}`.
- QC in notebook 02: agreement kappa vs gold subset per class, prevalence sanity table,
  language census over all reports (sizes the future XLM-R backlog item),
  fixed contradiction-rate table (calibrates label smoothing).

## 8. Compute Budget (fits 30 GPU-h/week)

```
per epoch : 4407 studies x 4 series x 24 slices ~= 423k slices @384 px
            ~= 300-350 img/s (2xT4, bf16-mixed)  ~= 20-25 min (+decode overlap)
MVP total : 5 folds x 4 epochs                  ~= 7-12 GPU-h
inference : ~1300 studies                       ~= < 15 min
headroom  : >= 15 GPU-h for iteration/backlog
```

Levers if measurements disagree: `n_slices`, `img_size`, `max_series_per_study`,
backbone name — all in YAML. Notebook 01 records measured decode throughput.

## 9. Repository Layout

```
main.py                              # CLI: build-index | build-labels | build-folds |
                                     #   train | infer  (+ --experiment/--override)
kaggle_run.sh                        # staged shell driver: setup|index|labels|folds|
                                     #   train|infer|all; sources .env; offline wheels
kaggle_cell.py                       # notebook cell entrypoint wrapping the driver
requirements.txt                     # pinned deps incl. DICOM codec wheels
configs/                             # Section 5.2 (bases + experiments/)
src/knee/
├── config_params/loader.py          # load_config, load_experiment, dump_config,
│                                    #   deep_merge, recursive instantiate
├── helpers/
│   ├── dicom_io.py                  # DecoderRegistry: native->gdcm->pylibjpeg
│   ├── header_scan.py               # parallel header-only scan -> index.parquet
│   ├── geometry.py                  # normal-projection ordering; InstanceNumber fallback
│   ├── intensity.py                 # rescale, percentile normalize, autocrop
│   ├── nlp_labeling.py              # bidirectional negation window (sentence-scoped),
│   │                                #   OA anchor x compartment co-occurrence, -1 mask
│   ├── folds.py                     # strata builder + grouped fold assignment
│   ├── kaggle_io.py                 # retrying CLI client: create/version/pull
│   ├── secrets.py                   # env -> .env -> Kaggle UserSecrets resolution
│   └── utils.py                     # seeding, timer, logger
├── datasets/
│   ├── series_dataset.py            # even-index K-slice sampling from ordered SOP lists
│   └── study_dataset.py             # priority series selection, metadata builder,
│                                    #   collate matching KneeNet's flat contract
├── datamodules/study_datamodule.py  # fold-aware LightningDataModule
├── augmentations/factory.py         # YAML specs -> Compose (resize/totensor bookends)
├── layers/pooling.py                # AttentionPool2d, StudyAggregator, FiLM
├── models/
│   ├── backbones.py                 # timm wrapper with offline checkpoint support
│   ├── series_encoder.py            # single flattened backbone pass + pooling
│   └── knee_net.py                  # full net + differential-LR parameter groups
├── losses/asymmetric_focal.py       # WeightedBCE default; AsymmetricFocal; -1 masking
├── metrics/auc.py                   # per-class + macro ROC-AUC accumulator
├── engines/
│   ├── assembly.py                  # config -> reader/datasets/datamodule/model
│   ├── train_module.py              # KneeModule: loss, warmup+cosine, OOF dumps
│   └── inferencer.py                # fold-ensemble fp16 predictor + schema asserts
├── callbacks/session.py             # TimeBudgetCallback + PeriodicPushCallback
├── loggers/
│   ├── csv_logger.py                # per-fold Lightning CSVLogger
│   ├── discord_logger.py            # webhook notifier + lifecycle callback
│   └── wandb_logger.py              # W&B logger (offline fallback, key via secrets)
notebooks/
├── 01_EDA.ipynb                     # done
├── 02_build_index_labels_folds.ipynb  # pending (or use kaggle_run.sh stages)
├── 03_model_training.ipynb          # pending thin wrapper over kaggle_cell.py --stage train
└── 04_inference.ipynb               # pending thin wrapper over kaggle_cell.py --stage infer
tests/                               # 45 tests across geometry/NLP/loader/net/resume
```

Style contract for all Python: shebang + coding header, Google docstrings,
2-space indentation, single quotes, SOLID boundaries
(Registry/Factory/Adapter/Strategy/Template Method/Facade as annotated above).

## 10. Verification Gates

- `pylint --rcfile=../.pylintrc` >= 9.5 (currently 10.00) and `ruff check`
  clean on every Python file under src/, tests/, main.py, kaggle_cell.py.
- `pytest tests/` green.
- Every config file instantiates through `load_config` + `instantiate` without error.
- KneeNet forward passes shape tests for variable series/slice counts.
- Notebook 03 prints per-fold/per-class OOF AUC; sanity gate >= 0.80 macro on gold+pseudo CV.
- Notebook 04 asserts exact header, row count, and UID set equality before writing `submission.csv`.

## 11. Iteration Backlog (post-MVP)

1. XLM-R multilingual pseudo-labeler -> soft-target distillation.
2. ONNX Runtime fp16 export + parity tests (< 0.001 OOF delta).
3. TTA and snapshot ensembles.
4. MONAI MRI-specific augmentations (bias field, ghosting).
5. Per-plane encoder specialization.
6. Efficiency tuning: K ablation, fold pruning, INT8 quantization.
7. Contrastive image-report pretraining.
8. Protocol-stratified folds once domain-shift analysis is rerun cleanly.

## 12. Usage

### 12.1 Kaggle session flow

`kaggle_cell.py` self-bootstraps: inside a checkout it refreshes the
committed `repo_meta.json` from `.git` (remote URL + branch, worktree-aware);
outside one (pasted cell / dataset mount) it shallow-clones that branch into
`$KNEE_REPO_DIR`, resolves the project directory within the monorepo, and
re-executes your command there - zero manual code.

```bash
# Cell 1 (fresh kernel): install pinned deps; WHEELS_DIR enables offline mode
!bash kaggle_run.sh setup

# Cell 2..n: any stage; training resumes automatically from pulled last.ckpt
%run kaggle_cell.py --stage index
%run kaggle_cell.py --stage labels
%run kaggle_cell.py --stage folds
%run kaggle_cell.py --stage train --fold 0     # or omit --fold for run.folds list
%run kaggle_cell.py --stage infer              # writes submission.csv
```

Environment overrides: `EXPERIMENT` (YAML under configs/experiments/),
`WHEELS_DIR` (offline wheel directory), `PIP_EXTRA`,
`KNEE_REPO_DIR` (clone location), `GIT_TOKEN`/`GITHUB_TOKEN`/`GH_TOKEN`
(https credentials for private upstreams).

### 12.2 Local development

```bash
pip install -r requirements.txt
PYTHONPATH=src python main.py train \
  --experiment configs/experiments/smoke_ci.yaml   # CPU smoke run
pytest tests
ruff check src tests main.py && ruff format --check .
pylint --rcfile=../.pylintrc src/knee main.py
```

### 12.3 Experiment traceback

Every run dumps its fully composed configuration to
`artifact_dir/resolved_<experiment>.yaml`; new experiments are a single YAML
under `configs/experiments/` listing `defaults` + an `override` block.

## 13. Implementation Status

| Milestone | State |
|---|---|
| EDA + decoder forensics (`01_EDA.ipynb`) | done |
| Config system (bases, experiments, loader, resolved-dump) | done |
| Helpers: geometry/intensity/dicom_io/header_scan/nlp/folds/kaggle_io/secrets | done |
| Model stack: pooling layers, backbone, encoders, KneeNet | done |
| Engines: assembly, Lightning module, inferencer, submission asserts | done |
| Callbacks: time budget, periodic push | done |
| Logging: CSV + W&B + Discord | done |
| Kaggle drivers (kaggle_run.sh, kaggle_cell.py) | done |
| Tests: 45 passing; pylint 10.00/10 on all files; ruff clean | done |
| Notebooks 02-04 as thin wrappers | pending (stages already runnable via drivers) |
| First real-data end-to-end run (index -> folds -> train fold0) | pending |
| Backlog items 1-8 | pending |

Commit trail: `21def73` blueprint -> `fd41d80` configs -> `8a04e11`
loader+helpers -> `e15f02e` model stack -> `9f6facc` engines/CLI ->
`4fa02c4`+`f09bac0` tests & fixes -> `52659d8` style enforcement ->
`bed9ceb` logging + drivers.
