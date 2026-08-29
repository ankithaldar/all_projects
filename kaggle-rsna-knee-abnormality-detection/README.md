# RSNA Knee Abnormalities Detection

Multimodal detection of 12 knee abnormalities (ACL/MCL tears, medial/lateral
meniscus tears, three OA compartments, effusion, synovitis, Baker's cyst,
contusion, fracture) from knee MRI studies. Scored by macro-averaged ROC-AUC.

Design blueprint with measured constraints and rationale:
[BLUEPRINT.md](BLUEPRINT.md).

## Approach (MVP)

```
StudyInstanceUID
 └── Series 1..N
      ├── ordered slice list precomputed in index.parquet (geometry-based)
      └── sample K=24 evenly spaced slices -> stream-decode JPEG/JP2
           -> percentile normalize -> autocrop -> resize 384 px
           └── shared timm backbone (tf_efficientnetv2_s) -> slice embeddings
                -> AttentionPool -> series token (+ plane/FS/FL embedding)
                 ▼
     StudyAggregator (learnable query cross-attends series tokens)
     ⊕ FiLM metadata conditioning (plane ⊕ sex ⊕ flags ⊕ counts ⊕ spacing)
                 ▼
          trunk MLP -> 12 sigmoid heads
```

- Reports are training-time supervision only: gold labels where present,
  negation-aware rule-derived pseudo-labels elsewhere (`-1` = uncertain is
  masked out of the loss).
- StratifiedGroupKFold(5) grouped by study/patient; per-class + macro OOF AUC.
- The train loader optionally uses a frequency-aware study sampler
  (tempered inverse prevalence) so rare positives still get gradient
  updates under the macro-AUC metric; validation stays uniform.
- DICOMs are decoded ONCE into sharded HDF5 volumes (`--stage cache`),
  pushed as Kaggle datasets, and mounted read-only afterwards: training,
  selftest and inference stream pixels from local SSD, never re-decoding.
- Training survives Kaggle's 12 h kernels via versioned Kaggle-Dataset
  checkpoints; inference ensembles every completed fold.
- A noise-floor harness (`--stage sweep`) trains the same config at
  several seeds and reports the keep/drop gate (mean + 2*std of OOF
  macro-AUC) that every experiment change must beat.

## Quickstart

### Local

```bash
pip install -r requirements.txt
pytest tests                                   # 173 tests
ruff check src tests main.py kaggle_cell.py    # style gate
pylint --rcfile=../.pylintrc src/knee main.py  # 10.00/10 required
PYTHONPATH=src python main.py train \
  --experiment configs/experiments/smoke_ci.yaml   # CPU smoke run
```

### Kaggle notebook session

Paste nothing but one line - the cell bootstraps itself. `kaggle_cell.py`
carries its repository coordinates in the committed `repo_meta.json`
(auto-refreshed from `.git` on every local run): outside a checkout it
shallow-clones the recorded branch into `$KNEE_REPO_DIR`
(default `/kaggle/working/repo`) and re-executes your command from there.

```bash
!bash kaggle_run.sh setup                      # once per kernel image
%run kaggle_cell.py --stage index              # header scan -> index.parquet
%run kaggle_cell.py --stage labels             # rule pseudo-labels
%run kaggle_cell.py --stage folds              # CV assignment
%run kaggle_cell.py --stage cache              # decode once -> HDF5 datasets
%run kaggle_cell.py --stage selftest           # preflight: 2 real steps + ckpt
%run kaggle_cell.py --stage train --fold 0     # resumes if ckpt exists
%run kaggle_cell.py --stage sweep              # noise floor: seeds x folds
%run kaggle_cell.py --stage infer              # submission.csv
```

Private repos: set `GIT_TOKEN` (env/.env/Secrets) before the first call.
Stages: `setup | index | labels | folds | cache | selftest | train | infer | all`.
Overrides: `EXPERIMENT=<yaml>` env var, `WHEELS_DIR` for offline pip,
`KNEE_REPO_DIR` for the clone location.

## Configuration

Everything tunable lives in `configs/*.yaml`; code contains no literals.
Objects follow the `class_path` / `init_params` convention resolved
recursively:

```yaml
slice_pool:
  class_path: knee.layers.pooling.AttentionPool2d
  init_params: {embed_dim: 1280, num_heads: 8}
```

Each experiment is one self-contained file under `configs/experiments/`
(a `defaults:` list deep-merged from the base files plus an explicit
`override:` block). Every run dumps its fully composed configuration to
`artifact_dir/resolved_<experiment>.yaml` for traceback. CLI dot-path
overrides work anywhere:

```bash
python main.py train --experiment <yaml> --fold 0 --override data.n_slices=16
```

## Secrets

Referenced by name in YAML, resolved by `knee/helpers/secrets.py`:
process env -> `.env` -> Kaggle User Secrets. Copy `.env.example` locally or
register the same names under Add-ons > Secrets on Kaggle:
`DISCORD_WEBHOOK_URL`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`.

## Logging (production-style)

Every stage writes ONE flat file capturing python logging AND mirrored
stdout/stderr (tqdm bars, Lightning prints, raw tracebacks):

```
/kaggle/working/logs/knee_<stage>_<experiment>_<UTC>.log
```

Batch/epoch progress lines land in the same file every
`trainer.log_every_n_steps` batches:

```
epoch 0 | batch 25/236 | step 25 | train_loss 0.3123 | lr 0.0003 | gpu 9.8/14.5 GiB
```

Knobs live under the `logging:` config block (level, capture_streams,
progress.{enabled, refresh_rate, gpu_mem}).

## Notifications (Discord)

| Event | Message |
|---|---|
| fit start / finish (+duration) | per fold |
| first optimizer step | `pacing OK` - instant pace signal |
| every `every_n_steps` (50) | `step N (epoch E), train_loss, lr` |
| validation epoch end | macro-AUC |
| crash | truncated traceback |

The cache stage adds: progress every 10k decoded frames (with % + ETA),
a line per dataset push, and a final summary naming every pushed
dataset with a ready-to-paste `KNEE_HDF5_CACHE_DIRS` line.

## Imbalanced Sampling (train loader only)

Targets are skewed but scored by macro-AUC, so the MVP experiment
enables a weighted study sampler: per-target weight
`prevalence**-tempering` (normalized), aggregated per study with `max`
over the study's positive targets. `-1` (uncertain) labels never
influence prevalence; studies without positives get `baseline`.

```yaml
datamodule:
  train_sampler:                    # null = uniform (base default)
    class_path: knee.samplers.weighted.StudyWeightedRandomSampler
    init_params: {tempering: 0.5, aggregation: max, baseline: 1.0}
```

Draws are with replacement and `num_samples = len(dataset)`: epoch
size, step counts, and the resume protocol are unchanged. Under DDP
Lightning wraps the sampler so both T4s see disjoint streams. Changing
the sampling policy mid-fold restarts that fold's distribution - do
not resume a fold checkpoint across the change.

## Volume Cache (HDF5 shards)

`--stage cache` decodes every indexed series once (all slices,
percentile-normalized, autocropped, resized, uint8) into rolling
`volume_shard_NNN.h5` files capped at 10 GiB uncompressed, each shard
pushed as its own Kaggle dataset `<base>-NNN` (base:
`ah2002-rsna-knee-abnormality-detection-cache`). Resume-safe: UIDs in
completed shards are skipped; partial tails from killed sessions are
dropped and re-decoded. Every pushed dataset carries a manifest fragment
+ generation stamp; readers merge fragments across ALL attached datasets
(auto-discovered - no env vars needed) and fall back to live DICOM
decoding on any cache miss. Generation mismatches across mounts are
logged at ERROR. Optional local mirror: `volume_cache.
copy_mounts_to_working: true` + `copy_dir: /kaggle/tmp/cache_roots`
copies mounts to local scratch once, removing FUSE read latency.

## Resume Protocol

`fold{k}/last.ckpt` (+ `done` markers) live in a versioned Kaggle Dataset.
Session start pulls the newest version; finished folds are skipped, partial
folds resume via `Trainer.fit(ckpt_path=...)`; a time-budget callback stops
fitting before the kernel limit and pushes a new immutable dataset version.
Inference ensembles whichever folds carry `done` markers.
Cache-session resume mirrors this: already-pushed shards (read from the
attached fragment manifests) are never re-decoded, and new shards
continue the `-NNN` sequence instead of version-bumping old ones.

## Repository Map

```
main.py            # build-index | build-labels | build-folds | train | infer
kaggle_run.sh      # staged shell driver (setup..all, offline wheels)
kaggle_cell.py     # notebook cell wrapper around the driver
configs/           # base YAMLs + experiments/<name>.yaml
src/knee/          # config_params, helpers, datasets, datamodules,
                   # augmentations, layers, models, losses, metrics,
                   # engines, callbacks, loggers
tests/             # geometry, NLP rules, config loader, net shapes, resume
notebooks/         # 01_EDA (done); 02-04 thin wrappers over the stages
```

## Status

Implementation complete through logging/drivers, the HDF5 volume cache,
the noise-floor `sweep` stage, and imbalanced-target sampling; notebooks
02-04 remain as thin wrappers, the noise-floor gate is unmeasured until
`sweep` runs on the real data, and the first full 5-fold run is
outstanding. Detailed status table and iteration backlog:
[BLUEPRINT.md](BLUEPRINT.md).
