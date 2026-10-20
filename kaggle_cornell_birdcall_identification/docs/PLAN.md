# Documentation Plan — Cornell Birdcall 1st Place Solution

Status tracker for the multi-agent documentation build. Notion-style, in `docs/`.

## Deliverables

### Module chapters → `docs/modules/`

| # | File | Covers | Agent |
|---|------|--------|-------|
| 00 | `_TEMPLATE.md` | shared style + heading skeleton | written |
| 01 | `01-data-pipeline.md` | `sed_dataset.py`, `imbalanced_dataset_sampler.py` | dispatched |
| 02 | `02-audio-frontend.md` | `helpers/sed_audio_utils.py`, `audio_utils.py` | dispatched |
| 03 | `03-augmentations.md` | `augmentations/*` (7 files) | dispatched |
| 04 | `04-model-architecture.md` | `models/sed_models.py` | dispatched |
| 05 | `05-training-engine.md` | `engine/base/base_engine.py`, `sed_engine.py`, `sed_train.py` | dispatched |
| 06 | `06-checkpointing-and-resuming.md` | ignite Checkpoint/DiskSaver, resume path | dispatched |
| 07 | `07-losses.md` | `loss/*` (16 files), `loss_keys` contract | dispatched |
| 08 | `08-inference-and-submission.md` | `metrics/sed_test.py`, 13-model vote ensemble | dispatched |
| 09 | `09-config-and-cross-validation.md` | `config_params/*`, fold strategy | dispatched |
| 10 | `10-metrics-and-tracking.md` | `metrics/*`, `logger/*`, Neptune + Slack | dispatched |
| 11 | `11-legacy-research-paths.md` | `train_vN`, `main_engine_vN`, `audio_dataset_vN`, legacy models | dispatched |

### Framework comparison → `docs/framework-comparisons/`

Kept deliberately separate. No merging.

| File | Covers | Agent |
|------|--------|-------|
| `pytorch-lightning-port.md` | same 10 phases expressed in PyTorch Lightning 2.x | dispatched |
| `tensorflow-keras-port.md` | same 10 phases expressed in TF/Keras 3, incl. counterfactual verdict | dispatched |

### Master synthesis → `docs/` (author writes, after agents return)

- `README.md` — navigation, the 1st-place summary, reading order
- `00-overview-and-secret-sauce.md` — the reverse-engineered thesis
- `12-kaggle-constraints.md` — RAM/VRAM, 9h limit, 150×10min files, 2 submissions/day
- `13-experiment-log.md` — ensemble/threshold decision table + LB outcomes
- `frameworks/comparison-matrix.md` — Ignite vs Lightning vs Keras structural table

## Known facts to fold in (verified by direct read)

- PyTorch Ignite `0.4.0.post1` is the only training framework. **TensorFlow/Keras is absent from the repo entirely.**
- `requirements.txt` pins ignite, neptune-client, audiomentations, tqdm, psutil, SoundFile.
  **torch / torchvision / torchaudio are NOT pinned** — assumed preinstalled.
- Loss consumes **only** `y_pred["clipwise_output"]`; inference thresholds **both** heads.
- `USE_AMP = False` in both branches of the apex try/except — AMP detection is vestigial.
- Checkpoint saver attaches to the **evaluator** at `COMPLETED`, not the trainer.
- `ConcatScheduler` built with a `durations` list of length 1 for 2 schedulers.
- `os.getenv(XLA_USE_BF16)` called with no default, swallowed by a bare `except`.
- Training data was pre-resampled to 32kHz WAV on disk; `.mp3` → `.wav` rewritten at load.
- Layout: `/data/{first_letter}/{ebird_code}/{filename}.wav`, plus `/pinknoise`.

## Reproduction gaps (stated honestly in the docs)

- The per-fold train/valid CSVs are **not in the repo**. The CV split is not reproducible
  from this code alone; it was produced offline.
- No training/validation loss or LR curves committed — Neptune runs are external.
- Inference/voting code lives in `metrics/sed_test.py` and emits per-model files;
  the cross-model vote is a separate, largely undocumented step.
