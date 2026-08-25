# RSNA Knee Abnormality Detection

Multimodal weak-supervision pipeline for the RSNA AI Challenge: detect 12
knee findings per MRI study, scored by **macro ROC-AUC**. Reports exist for
all training studies but are withheld at test time, so the architecture is a
**text teacher -> image student distillation** system (see
[BLUEPRINT.md](BLUEPRINT.md) for the full strategy).

## Layout

```
configs/            YAML experiments (omegaconf + pydantic-validated)
notebooks/          kaggle_cell.py -- paste-ready first cell for kernels
src/knee/
  activations/      activation factory shared by layers/models
  augmentations/    2.5D stack transforms + GPU differentiable ops
  callbacks/        EMA, Discord updates, progressive unfreezing,
                    AMP watcher, batch/accum tuners, online HP, swaps
  config_params/    ComponentSpec schema + loader/instantiate factory
  datamodules/      fold-scoped LightningDataModule (stream-decoding)
  datasets/         volume_store (stream decode + LRU), volume_builder
                    (optional npz shards), study datasets, balanced
                    multilabel sampler, frozen folds
  engines/          KneeStudyLitModule, trainer factory, text teacher,
                    rule/NLI labelers, OpenRouter LLM labeler,
                    weak-label builder, TTA predictor, greedy blender
  helpers/          secrets (.env/Kaggle), input-mount discovery,
                    seeding, logging
  layers/           attention-MIL / transformer aggregators,
                    metadata encoder (+ projection into model fusion)
  loggers/          Discord notifier transport, W&B io helpers
  losses/           ASL/focal/soft-BCE + curriculum/DWA weighting
  metrics/          macro ROC-AUC with NaN-class accounting
  models/           2.5D timm + MONAI 3D series encoders,
                    composite study model
  optimizers/       param groups, warmup-cosine, Lookahead,
                    clipping strategies, gradient noise
scripts/            kernel entrypoints (dispatcher: kaggle_run.sh;
                    artifact publishing: publish_dataset.py)
tests/              dependency-light smoke tests
```

## Stages and the files they touch

State lives in `/kaggle/working`; `publish` mirrors it to one private
Kaggle dataset, and every kernel's bootstrap restores it. Files under
`$DATA_ROOT` (train.csv, DICOMs, test.csv) come from the read-only
input mount.

| Stage | Needs in `WORK` | Writes |
|---|---|---|
| `folds` | -- | `train_folds.csv` |
| `volumes` *(opt.)* | -- | `volumes_cache/*.npz` + manifest |
| `weak-labels` | ○ `text_teacher/oof_probs.parquet` | `weak_labels.parquet` |
| `weak-labels-llm` | secret `OPENROUTER_API_KEY`; ○ cache | `weak_labels.parquet`, `llm_label_cache.parquet` |
| `teacher` | ✓ `train_folds.csv`; ○ own ckpts (resume) | `text_teacher/ckpts/*`, `oof_probs.parquet` |
| `student` | ✓ `train_folds.csv`, ✓ `weak_labels.parquet`, `$EXP`; ○ ckpts/OOF (resume/skip) | `checkpoints/fold{N}/*`, `predictions/*_oof_fold{N}.parquet`, merged `*_oof.parquet` |
| `self-train` | ✓ `predictions/<name>_oof.parquet` | `weak_labels_round2.parquet` |
| `infer` | ✓ ≥1 fold checkpoint | `submission_<exp>.csv` |
| `blend` | ✓ OOF frames, ✓ `train_folds.csv`, ✓ submissions | `submission_blended.csv` |
| `publish` | all of the above | remote dataset |

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env            # fill DISCORD_WEBHOOK_URL / NEPTUNE_* if desired
export DATA_ROOT=/path/to/rsna
bash scripts/run_all.sh         # stages with resumable artifacts
```

## Kaggle workflow (30 GB disk, 12 h/kernel, ~30 GPU-hours)

The DICOM tree is ~570 GB -- a full volumes cache CANNOT fit and is not
required. Volumes **stream-decode** from the read-only `/kaggle/input`
mount through a bounded LRU RAM cache
(`src/knee/datasets/volume_store.StreamingVolumeStore`); any mounted npz
shards are used as a read-only accelerator when present.

### Multi-kernel lifecycle (fresh container every kernel)

Every kernel boots an empty `/kaggle/working`; all resumable state
therefore lives in **one private Kaggle dataset** (default:
`ah2022-rsna-knee-abnormality-detection`).

One-time per kernel setup:

- Add-ons -> Secrets -> `KAGGLE_USERNAME`, `KAGGLE_KEY` (dataset
  publishing), `GITHUB_TOKEN` (private repo), and
  `OPENROUTER_API_KEY` (only for `weak-labels-llm`).
- After the dataset exists once: Add Data -> Your Datasets -> attach it
  (any mount layout works -- the bootstrap discovers the slug at
  `/kaggle/input/<slug>` or nested `datasets/<owner>/<slug>`).

Then:

1. **Kernel N**: run stages; push state with
   `bash scripts/kaggle_run.sh publish` (or `export AUTO_PUBLISH=1` to
   push after every successful stage -- a 12 h kill then costs at most
   one stage of work).
2. **Kernel N+1**: point the bootstrap at the mounted dataset before
   dispatching:

   ```bash
   export PREV_OUTPUT=/kaggle/input/ah2022-rsna-knee-abnormality-detection
   export FOLDS_LIST='2,3'          # this kernel's fold shard
   bash /kaggle/working/repo/scripts/kaggle_run.sh student
   ```

   The bootstrap copies checkpoints/OOF/folds/labels forward into the
   writable `/kaggle/working`: finished folds skip instantly,
   interrupted folds resume epoch-level from
   `checkpoints/fold<N>/last.ckpt` (`RESUME=1` default), and new folds'
   artifacts land in `$WORK` ready for the next `publish`.

Budget mechanics baked in:

- **Fold sharding**: `FOLDS_LIST='0,1'` then `'2,3'` ... per kernel;
  per-fold OOF parquets make re-running idempotent.
- **Time budget**: `train.time_budget_hours` -> Lightning `max_time`,
  sized to remaining wall clock (default 11 h), so kernels end with
  valid checkpoints instead of being hard-killed.
- Deps install is import-checked: the image's CUDA-matched torch /
  lightning / timm / transformers stack is reused, only genuinely
  missing packages are pip-installed (~1 min, `FORCE_PIP=1` overrides).
- GPU quota is consumed per kernel, so skipped/resumed folds cost
  nothing; Save Version remains an optional belt-and-braces backup
  alongside the dataset handoff.

First cell of every kernel: copy
[notebooks/kaggle_cell.py](notebooks/kaggle_cell.py) verbatim (edit
`REPO_URL` once). It pulls secrets, clones the repo, auto-restores the
dataset when attached, dispatches the stage and streams output:

```python
REPO_URL = 'https://github.com/<user>/<repo>.git'  # <-- EDIT ONCE
STAGE = 'student'                                  # per kernel
FOLDS_LIST = '2,3'                                 # per kernel
AUTO_PUBLISH = 1                                   # push artifacts after
```

Secrets resolve in order: env vars -> `.env` -> Kaggle Secrets
(`src/knee/helpers/env.py`). Discord updates and Neptune tracking
activate automatically when their keys exist; both are silent no-ops
otherwise.

## LLM weak-label tier (optional)

`weak-labels-llm` replaces the trained XLM-R teacher with an OpenRouter
chat model (default `stealth/ox-alpha`; override via `LLM_MODEL`):
every training report is labeled zero-shot in any language, replies are
hard-constrained to JSON (`response_format`, prompt contract, and a
finding-only parser that strips rationales/wrappers), results cache to
disk so 12 h kills never re-bill, and fusion still runs through
`WeakLabelBuilder` (gold > rule seeds > LLM). Concurrency defaults to 2
(`LLM_CONCURRENCY`) to respect rate limits.

## Engineering standards

- Every swappable component is declared as `{target: class.path, params: {}}`
  in YAML and instantiated by one factory — no behavior changes require
  Python edits.
- PyTorch Lightning owns all training/prediction loops; Trainer, callbacks
  (`ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor`, EMA, Discord)
  and loggers (CSVLogger + NeptuneLogger) are config-declared.
- Style is machine-enforced: Google docstrings, 2-space indent, single
  quotes, file headers (`ruff format && ruff check`).
- CV: frozen iterative-stratification folds; gold-OOF macro AUC is the only
  metric used to accept changes.
