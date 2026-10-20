# Cornell Birdcall Identification — PyTorch Lightning & TensorFlow/Keras

Two working implementations of the Cornell Birdcall 1st-place solution
(PANNs-style sound event detection with a DenseNet121 backbone, `tanh`
attention, and a 13-model vote), plus the **13-model vote and
`submission.csv` generation that actually won the competition**.

The source of truth for every behaviour is the technical documentation in
[`docs/`](docs/README.md) — a per-file reference of the original solution
([`docs/src-reference/`](docs/src-reference/README.md)), a 112-row feature
parity matrix ([`docs/ports/feature-parity-matrix.md`](docs/ports/feature-parity-matrix.md)),
a file-by-file porting worklist
([`docs/ports/porting-worklist.md`](docs/ports/porting-worklist.md)) and
step-by-step implementation guides for
[Lightning](docs/ports/pytorch-lightning-guide.md) and
[Keras](docs/ports/tensorflow-keras-guide.md).

---

## The one-paragraph answer

A single architecture — PANNs-style SED with a DenseNet121 backbone and `tanh`
attention — trained 13 times (4-fold no-mixup, 5-fold no-mixup, 4-fold mixup)
with plain BCE on weak, file-level labels. At inference a species is accepted
for a 5-second row when **≥ 4 of 13 models** fire with **clip ≥ 0.3 AND
frame ≥ 0.3**. That ensemble was 2nd on public (0.616) and **1st on private
(0.681)**, moving 7th → 1st.

## What is in this repository

| Path | What it is |
|---|---|
| `docs/` | The technical documentation of the original solution. **Read this first** — it is the specification this code implements. |
| `src/birdcall/` | The implementation. Two backends over one shared, framework-agnostic core. |
| `tests/` | pytest suite: unit, parity and end-to-end checks. |
| `tasks/plan.md` | The implementation plan: architecture decisions, risks, open questions. |
| `tasks/todo.md` | The task list, with acceptance criteria and verification commands per task. |

## The two backends

| | PyTorch Lightning | TensorFlow / Keras 3 |
|---|---|---|
| Prefix | `pl_*` | `tf_*` |
| Entry point | `birdcall-train-pl` | `birdcall-train-tf` |
| What is shared | the label space, the config schema, the mel matrix, the waveform augmentations, the mixup, the dataset core, the LR-schedule maths, the score artefact and the vote | |
| What is rewritten | the control plane only | the control plane **and** the numeric core |
| Numerical risk | low — the model and loss maths are untouched | higher — the mel scale, the dB formula, the reflect padding, the softmax axis and the frame padding all change silently if wrong |

The shared modules import neither framework. That is enforced by a test
(`tests/test_shared_is_framework_free.py`), because a shared module that
quietly imports torch would make the Keras port untestable.

## Install

```bash
cd kaggle_cornell_birdcall_identification
python3.12 -m venv .venv          # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -e '.[tf,augment]'    # both backends + the augmentation stack
pip install -e '.[build]'         # only needed to rebuild the mel matrix
```

**Python 3.12 or 3.13.** TensorFlow has no stable wheel for Python 3.14.

`librosa` is a **build-time-only** dependency. The mel matrix is precomputed
once and committed as `src/birdcall/assets/mel_slaney_*.npy`, so neither
backend needs librosa at runtime. This also removes the
librosa/numba/numpy version triangle that otherwise breaks
`librosa.filters` at first use.

The waveform augmentation stack is pinned to `audiomentations==0.11.0`
because its `AddGaussianSNR` treats `min_SNR`/`max_SNR` as a **linear
amplitude ratio**, not dB. A different version changes the experiment.

## Configuration

Every experiment is **one named YAML file**, validated by a pydantic model.
The configuration is composed from three layers:

| layer | where | holds |
|---|---|---|
| **module files** | `src/birdcall/config_params/modules/*.yaml` | one file per functional module — model, augmentation, loss, optimizer, scheduler, data, metrics, logging, checkpoint, runtime — holding the values every experiment shares |
| **experiment files** | `src/birdcall/config_params/experiments/**/*.yaml` | one file per experiment, holding only what that experiment changes |
| **backend selection** | `src/birdcall/config_params/run/backend.yaml` | whether the PyTorch Lightning backend, the Keras backend, or both run |

```bash
birdcall-train-pl --config src/birdcall/config_params/experiments/final_sed/final_sed_dense121_mix_fold0.yaml
```

An experiment file is short, because everything it does not change comes
from the module files:

```yaml
name: final_sed_dense121_mix_fold0
fold: 0
modules: [model, augmentation, loss, optimizer, scheduler, data, metrics,
           logging, checkpoint, runtime]
overrides:
  apply_mixup: true
  train_bs: 32
  train_ds_params:
    csv_dir: /folds/fold_0_train.csv
```

The schema lives in `src/birdcall/config_params/parameters.py` and is the
only place a config attribute is named. It sets `extra='forbid'`, so a typo
is a validation error rather than a silently ignored key, and every
enumerated name (`model_name`, `criterion_name`, `scheduler_name`,
`track_metric`, …) is a `Literal`, so an unknown value is rejected with the
allowed values listed.

`src/birdcall/config_params/run/backend.yaml` selects the backend:

```yaml
backend: both        # pl | tf | both
backends:
  pl: {enabled: true, entry_point: birdcall-train-pl}
  tf: {enabled: true, entry_point: birdcall-train-tf}
```

The `--backend` flag on either entry point overrides it.

The 21 experiment configs mirror the original's layout: `final_sed/` (15),
`final_sed_5_fold/` (5) and `example_config.yaml`, plus two additions — an
inference config (`experiments/inference/final_5sec_sed.yaml`) and the
ensemble manifest (`config_params/ensembles/ensemble_13.yaml`).

## Running

```bash
# train one member
birdcall-train-pl --config src/birdcall/config_params/experiments/final_sed/final_sed_dense121_mix_fold0.yaml

# the same experiment on the Keras backend
birdcall-train-tf --config src/birdcall/config_params/experiments/final_sed/final_sed_dense121_mix_fold0.yaml

# score a model, keeping the frame axis
birdcall-train-pl --config <cfg>.yaml --mode infer --checkpoint <ckpt>.ckpt

# the 13-model vote -> submission.csv
birdcall-vote --ensemble src/birdcall/config_params/ensembles/ensemble_13.yaml \
              --scores <dir> --output submission.csv

# replay an experiment from its manifest
birdcall-reproduce run_manifest.json
```
## Reproducibility

Every experiment is described by its configuration alone, and every run
records what it did.

- **All seeds are config fields.** `seed`, `mixup_seed` and `deterministic`
  are part of the schema, so no seed hides in a default argument.
- **One seeding module.** `helpers/seed.py` (shared), `helpers/pl_seed.py`
  and `helpers/tf_seed.py` (per backend) are the only places allowed to
  seed an RNG; a test greps the tree to keep it that way.
- **Per-item randomness** is derived from `(seed, epoch, index)`, so a crop
  does not change when the number of dataloader workers changes.
- **Every run writes `manifest.json`**: the resolved config, the seeds, the
  git commit (and a hash of a dirty diff), the package versions, and the
  SHA-256 of the fold CSVs, the audio, the label space, the mel matrix and
  the score artefacts.
- **`birdcall-reproduce`** takes a manifest, re-checks every input hash and
  re-runs the recorded command.

The fold CSVs and the label space are generated by
`scripts/make_folds.py` and `scripts/build_bird_codes.py`, both seeded, so
the ensemble can be re-derived.

## Code style

The project's style contract is the repository root
[`.pylintrc`](../.pylintrc) (Google-derived), which is **read-only** — code
is refactored to fit it, never the other way round.

- 2-space indentation, 80-column lines, 4-space hanging indent
- single quotes for every string literal
- Google-style docstrings on every module, class, function and method
- every file starts with:

  ```python
  #!/usr/bin/env python
  # -*- coding: utf-8 -*-

  '''<module doc string>'''
  ```

- one public class per file, named after it
- `pl_<name>.py` = PyTorch Lightning, `tf_<name>.py` = TensorFlow/Keras,
  no prefix = shared and framework-agnostic

```bash
pylint src/birdcall tests
```

## Tests

```bash
pytest -q                     # the whole suite
pytest -q -m parity           # cross-backend parity only
```

The parity tests are the ones that matter: they check the log-mel to 1e-4,
the `clipwise_output` to 1e-4, the three metrics to 1e-6, the LR curve
step-for-step, and that both backends produce a **byte-identical
`submission.csv`**.

## What this repository cannot reproduce

Stated plainly, because the original solution's own README is vague about it:

| Gap | Consequence |
|---|---|
| **The original fold CSVs are gone.** They were produced offline and lived at `/folds/`, which is not in the upstream repository either. | `scripts/make_folds.py` produces a **new, reproducible** split (stratified on the primary `ebird_code` only). The original's exact split is unrecoverable. |
| **No competition audio, no pretrained checkpoints, no GPU here.** | Every test is synthetic. A real training run is an operator step, not a test. |
| The original's tracked experiment used **Neptune**; this implementation tracks to **wandb**, keeping the original's metric series names. | The dashboards are comparable; the historical runs are not in wandb. |

## Where the original source was available

The documentation in `docs/` is a *reverse-engineered* spec. The original
source is public, and this implementation was checked against it:

<https://github.com/ryanwongsa/kaggle-birdsong-recognition>

That closed the two gaps the documentation could not close on its own:

- **The 264 species codes** are recovered from the original
  `src/config_params/configs.py` and committed as
  `src/birdcall/config_params/bird_codes.json`. The index order is the
  label order everywhere, so it is data, not code.
- **The 21 experiment configurations** are generated from the original
  Python modules by `scripts/build_configs.py`, so the YAML tree is exact
  rather than reconstructed from a summary table.

### Corrections the original source settled

Three claims in the documentation turned out to be wrong, and the original
source is the authority. This implementation follows the **source**:

| claim | the source says |
|---|---|
| the `_augd` criterion drops 70 % of the primary positions | it drops **30 %**: the mask is 1 on a primary position with probability `primary_drop_out_factor` and the loss is multiplied by `(1 - mask)`, so a mask of 1 zeroes the position |
| `fc1` is a 1×1 convolution | it is `nn.Linear(1024, 1024)`, applied to a transposed tensor |
| the mel matrix is a frozen buffer | it is an `nn.Parameter` with `requires_grad=False`, and it is in the checkpoint under that name |

Two more came from the competition's own data page rather than the
repository: the `row_id` format (`site_1_<audio_id>_<end_second>`, and
`site_3_<audio_id>` for the file-level site 3 rows), and the fact that the
competition's sample submission calls its second column **`birds`**. This
project writes **`labels`**, as its own brief specified; rename the column
before uploading to Kaggle.

## Licence and attribution

The PANNs architecture is vendored from
[qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
(MIT). The imbalanced sampler follows
[ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler).
The original solution is
[1st place, Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition).
