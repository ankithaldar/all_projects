# RSNA Knee Abnormality Detection

Multimodal weak-supervision pipeline for the RSNA AI Challenge: detect 12
knee findings per MRI study, scored by **macro ROC-AUC**. Reports exist for
all training studies but are withheld at test time, so the architecture is a
**text teacher -> image student distillation** system (see
[BLUEPRINT.md](BLUEPRINT.md) for the full strategy).

## Layout

```
configs/            YAML experiments (omegaconf + pydantic-validated)
src/knee/
  activations/      activation factory shared by layers/models
  augmentations/    2.5D stack transforms + GPU differentiable ops
  callbacks/        EMA, Discord updates, progressive unfreezing,
                    AMP watcher, batch/accum tuners, online HP, swaps
  config_params/    ComponentSpec schema + loader/instantiate factory
  datamodules/      fold-scoped LightningDataModule
  datasets/         DICOM decode, volume cache, study datasets,
                    balanced multilabel sampler, frozen folds
  engines/          KneeStudyLitModule, trainer factory, text teacher,
                    rule/NLI labelers, weak-label builder, TTA
                    predictor, greedy blender
  helpers/          secrets (.env/Kaggle), seeding, logging
  layers/           attention-MIL / transformer aggregators,
                    metadata encoder
  loggers/          Discord notifier transport, W&B io helpers
  losses/           ASL/focal/soft-BCE + curriculum/DWA weighting
  metrics/          macro ROC-AUC with NaN-class accounting
  models/           2.5D timm + MONAI 3D series encoders,
                    composite study model
  optimizers/       param groups, warmup-cosine, Lookahead,
                    clipping strategies, gradient noise
scripts/            kernel entrypoints (see run_all.sh)
tests/              dependency-light smoke tests
```

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env            # fill DISCORD_WEBHOOK_URL / NEPTUNE_* if desired
export DATA_ROOT=/path/to/rsna
bash scripts/run_all.sh         # kernels 1->7 with resumable stages
```

## Kaggle workflow

Each `scripts/*.py` stage maps to one notebook kernel; push artifacts
(volume cache, folds, weak labels, checkpoints) as versioned datasets and
point `paths.*` overrides at them via `--set` dotlists:

```bash
python scripts/train_image_student.py --config configs/experiment/$EXP.yaml \
    --set paths.volumes_cache=/kaggle/input/knee-volumes-cache-v1 \
          paths.weak_labels_parquet=/kaggle/input/knee-weak-labels-v1/weak_labels.parquet
```

Secrets resolve in order: env vars -> `.env` -> Kaggle Secrets
(`src/knee/utils/env.py`). Discord updates and Neptune tracking activate
automatically when their keys exist; both are silent no-ops otherwise.

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
