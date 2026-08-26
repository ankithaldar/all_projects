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
- Training survives Kaggle's 12 h kernels via versioned Kaggle-Dataset
  checkpoints; inference ensembles every completed fold.

## Quickstart

### Local

```bash
pip install -r requirements.txt
pytest tests                                   # 45 tests
ruff check src tests main.py kaggle_cell.py    # style gate
pylint --rcfile=../.pylintrc src/knee main.py  # 10.00/10 required
PYTHONPATH=src python main.py train \
  --experiment configs/experiments/smoke_ci.yaml   # CPU smoke run
```

### Kaggle notebook session

```bash
!bash kaggle_run.sh setup                      # once per kernel image
%run kaggle_cell.py --stage index              # header scan -> index.parquet
%run kaggle_cell.py --stage labels             # rule pseudo-labels
%run kaggle_cell.py --stage folds              # CV assignment
%run kaggle_cell.py --stage train --fold 0     # resumes if ckpt exists
%run kaggle_cell.py --stage infer              # submission.csv
```

Stages: `setup | index | labels | folds | train | infer | all`.
Overrides: `EXPERIMENT=<yaml>` env var, `WHEELS_DIR` for offline pip.

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

## Logging

All channels run simultaneously and can never crash a scoring run:

| Channel | Notes |
|---|---|
| CSV | per-fold metrics at `logs/<experiment>/fold{k}/` |
| Weights & Biases | online when a key resolves, auto-offline otherwise |
| Discord | fold start/finish, epoch macro-AUC, crash tracebacks |

## Resume Protocol

`fold{k}/last.ckpt` (+ `done` markers) live in a versioned Kaggle Dataset.
Session start pulls the newest version; finished folds are skipped, partial
folds resume via `Trainer.fit(ckpt_path=...)`; a time-budget callback stops
fitting before the kernel limit and pushes a new immutable dataset version.
Inference ensembles whichever folds carry `done` markers.

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

Implementation complete through logging/drivers; notebooks 02-04 remain as
thin wrappers and the first real-data end-to-end run is outstanding.
Detailed status table and iteration backlog: [BLUEPRINT.md](BLUEPRINT.md).
