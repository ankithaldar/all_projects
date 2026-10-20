# Agent contract — read this first

Every agent working on this project reads this file. It is the shared
context: the style rules, the file naming, the shared APIs that already
exist, and the invariants that must not be broken.

## 0. Where the truth is

This project **reimplements** a documented solution. The original source
code is **not** in the repository. The specification is `docs/`:

| what | file |
|---|---|
| the original's per-file behaviour | `docs/src-reference/*.md` |
| the feature parity matrix (112 rows, with a "verify" column) | `docs/ports/feature-parity-matrix.md` |
| the file-by-file worklist (traps + acceptance checks) | `docs/ports/porting-worklist.md` |
| the Lightning guide | `docs/ports/pytorch-lightning-guide.md` |
| the Keras guide | `docs/ports/tensorflow-keras-guide.md` |
| the ranked list of 23 issues in the original | `docs/ports/code-issues-and-framework-wins.md` |
| the inferred voting semantics | `docs/modules/08-inference-and-submission.md` |
| the measured findings | `docs/FINDINGS.md` |

**Read the relevant doc file before writing code.** The behaviours below are
a summary, not a substitute.

## 1. Style contract (non-negotiable)

The repository root `.pylintrc` (`competitions_main/.pylintrc`) is
**read-only**. Do not edit it, and do not add a `[tool.pylint]` section to
`pyproject.toml`.

- **2 spaces** for indentation. Never tabs, never 4.
- **80 columns** maximum.
- **4 spaces** for a hanging indent after an open paren.
- **Single quotes** for every string literal, consistently within a file.
- **Google-style docstrings** on every module, class, function and method.
  A Google docstring has a one-line summary, a blank line, an `Args:`
  section, a `Returns:` or `Raises:` section, and a `YAML`/`Example`
  section only if it helps.
- **Every file starts with exactly this header:**

  ```python
  #!/usr/bin/env python
  # -*- coding: utf-8 -*-

  '''<module doc string>'''
  ```

  The module docstring uses **triple single quotes**.
- One **public class per file**, named after it (the file name rule below).
- No `print` in library code. Use `logging` if a message is needed.

## 2. File naming

| prefix | meaning |
|---|---|
| `pl_<name>.py` | the **PyTorch Lightning** implementation |
| `tf_<name>.py` | the **TensorFlow/Keras 3** implementation |
| no prefix | **shared**: importable from both, and must import neither torch nor tensorflow |

A shared module that imports a framework breaks the cross-backend parity
tests. `tests/test_shared_is_framework_free.py` enforces this.

The shared surface is exactly:

```
config_params/          helpers/seed.py       helpers/mel_matrix.py
augmentations/          datasets/sed_dataset_core.py
metrics/scoring.py      engines/schedules.py  voting/vote.py
provenance.py
```

## 3. What already exists (use it, do not re-create it)

```python
# birdcall.helpers.seed  (shared)
derive_seed(base_seed: int, *parts: int) -> int
make_generator(base_seed: int, *parts: int) -> np.random.Generator
seed_worker(worker_id: int, base_seed: int, epoch: int = 0) -> int

# birdcall.helpers.pl_seed  (torch)
seed_everything(base_seed: int, rank: int = 0, deterministic: bool = True) -> int
make_torch_generator(base_seed: int, *parts: int) -> torch.Generator
worker_init_fn(worker_id: int, base_seed: int, epoch: int = 0) -> None

# birdcall.helpers.tf_seed  (tensorflow)
seed_everything(base_seed: int, rank: int = 0, deterministic: bool = True) -> int
worker_init_fn(worker_id: int, base_seed: int, epoch: int = 0) -> None

# birdcall.helpers.mel_matrix  (shared)
build_mel_matrix(sr, n_fft, n_mels, fmin, fmax) -> np.ndarray   # needs librosa
load_mel_matrix(sr, n_fft, n_mels, fmin, fmax) -> np.ndarray       # float32
save_mel_matrix(matrix, sr, n_fft, n_mels, fmin, fmax) -> Path
mel_matrix_path(sr, n_fft, n_mels, fmin, fmax) -> Path

# birdcall.engines.schedules  (shared)
schedule_multiplier(name, index, epoch_length, num_epochs, scale_factor) -> float
linear_multiplier(index, cycle_size, start, end) -> float
cosine_multiplier(index, cycle_size, start, end) -> float
#   names: 'none' | 'warmup_with_cosine' | 'one_cycle_cosine'

# birdcall.config_params  (shared)
Parameters(**yaml_dict)             # the validated config; see parameters.py
load_parameters(path) -> Parameters # reads YAML and validates it
read_config_document(path) -> dict
get_bird_code() / get_inv_ebird_label() / get_inv_bird_code()

# birdcall.metrics.scoring  (shared) — the score artefact
save_score_artefact(path, audio_id, clip, frames, frame_rate) -> Path
load_score_artefact(path) -> dict
write_json_dump(path, filename, predicted_frames, predicted_clip, targets,
                lraps_score_clip, lraps_score_frames) -> Path
merge_json_dumps(path) -> Path
```

The `Parameters` model (`config_params/parameters.py`) is the **only** place
a configuration attribute is named. Read it before writing any code that
consumes a config.

## 4. The interfaces the parallel agents code against

These are fixed. If you need to change one, do not — say so in your final
report instead.

### Model (torch)

```python
# models/pl_model_att_block.py
class AttBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int,
               activation: str = 'linear', temperature: float = 1.0) -> None
  def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]
  # x: (B, in_features, T) -> (clipwise, norm_att, cla)

# models/pl_model_panns_dense121_att.py
class PANNsDense121Att(nn.Module):
  def __init__(self, sample_rate: int, window_size: int, hop_size: int,
               mel_bins: int, fmin: int, fmax: int, classes_num: int,
               apply_aug: bool, top_db: float | None = None) -> None
  def forward(self, inputs: tuple[Tensor, Tensor | None]) -> dict[str, Tensor]
  # inputs: (waveform (B, C, S), mixup_lambda | None)
  # returns {'clipwise_output': (B, C, O), 'framewise_output': (B, C, T, O)}

# models/pl_registry.py
MODEL_REGISTRY: dict[str, type] = {'sed_dense121att': PANNsDense121Att}
def build_model(name: str, **kwargs) -> nn.Module
```

### Front-end (torch) — `helpers/pl_frontend_*.py`

```python
class STFT(nn.Module):
  def __init__(self, n_fft=2048, hop_length=None, win_length=None,
               window='hann', center=True, pad_mode='reflect',
               freeze_parameters=True)
  def forward(self, input: Tensor) -> tuple[Tensor, Tensor]   # (B, 513, T)

class Spectrogram(nn.Module):
  def __init__(self, n_fft=2048, hop_length=None, win_length=None,
               window='hann', center=True, pad_mode='reflect', power=2.0,
               freeze_parameters=True)
  def forward(self, input: Tensor) -> Tensor                    # (B, 1, T, 513)

class LogmelFilterBank(nn.Module):
  def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000,
               is_log=True, ref=1.0, amin=1e-10, top_db=80.0,
               freeze_parameters=True)
  def forward(self, input: Tensor) -> Tensor                    # (B,C,T) -> (B,T,M)

class SpecAugmentation(nn.Module):
  def __init__(self, time_drop_width, time_stripes_num,
               freq_drop_width, freq_stripes_num)
  def forward(self, input: Tensor) -> Tensor
```

### Data (torch) — `datasets/pl_dataset_sed.py`, `datasets/pl_sampler_imbalanced.py`

```python
class SedDataset(Dataset):
  def __init__(self, root_dir, csv_dir, period, bird_code, inv_ebird_label,
               isTraining, num_test_samples, transform=None,
               background_audio_dir=None)
  def get_label(self, dataset, idx) -> str
  def set_epoch(self, epoch: int) -> None
  def __getitem__(self, idx) -> dict
  # {'filename', 'waveforms', 'all_labels', 'primary_labels',
  #  'secondary_labels'}
  # waveforms: (C, S) float32, all_labels: (C, 264) float32

class ImbalancedDatasetSampler(Sampler):
  def __init__(self, dataset, get_label)
  def set_epoch(self, epoch: int) -> None
```

### Augmentations (shared) — `augmentations/`

```python
# sed_augment_default.py
def get_transforms(bckgrd_aug_dir=None) -> dict[str, Callable]
# sed_augment_background.py
def get_transforms(bckgrd_aug_dir=None, secondary_bckgrd_aug_dir=None) \
    -> dict[str, Callable]
# mixup.py
class Mixup:
  def __init__(self, mixup_alpha: float, random_seed: int = 1234)
  def get_lambda(self, batch_size: int, device) -> Tensor
def do_mixup(x: Tensor, mixup_lambda: Tensor) -> Tensor
# registry.py
AUGMENTATIONS: dict[str, Callable]
def build_transforms(name, bckgrd_aug_dir=None,
                      secondary_bckgrd_aug_dir=None) -> dict[str, Callable]
```

### Losses (torch) — `losses/pl_loss_*.py`, one class per file

```python
class SedScaledPosNegFocalLoss(nn.Module):
  loss_keys: list[str]
  def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0,
               secondary_factor=0.1)
  def forward(self, y_pred: Tensor, y_target: dict) \
      -> tuple[Tensor, dict[str, Tensor]]
```

`y_pred` is the **clipwise** head, already sigmoided. `y_target` is the
label dict with `all_labels` and `secondary_labels`. The return is
`(scalar, dict)`.

The five classes, their files and their `loss_keys`:

| file | class | `loss_keys` |
|---|---|---|
| `pl_loss_focal_standard.py` | `FocalLossStandard` | `bce_loss, F_loss` |
| `pl_loss_removed_focal.py` | `SedRemovedFocalLoss` | `bce_loss, F_loss, FRemoved_loss` |
| `pl_loss_scaled_pos_neg_focal.py` | `SedScaledPosNegFocalLoss` | `bce_loss, F_loss, FScaled_loss, F_loss_0, F_loss_1` |
| `pl_loss_scaled_pos_neg_focal_augd.py` | `SedScaledPosNegFocalLossAugd` | same five |
| `pl_loss_bce_controlled_standard.py` | `BCEControlledStandardLoss` | *(empty)* |

`losses/pl_registry.py` exposes `CRITERION_REGISTRY: dict[str, type]` and
`build_criterion(name, **params) -> nn.Module`.

### Metrics (torch) — `metrics/pl_metric_*.py`, one class per file

```python
class SedLRAPS:
  def __init__(self, output_transform=lambda x: x,
               with_background_branch: bool = False)
  def reset(self) -> None
  def update(self, output: dict) -> None
  def compute(self) -> float

class SedF1ScoreClip:
  def __init__(self, threshold: float = 0.5, output_transform=lambda x: x)
class SedF1ScoreFrame:
  def __init__(self, threshold: float = 0.5, output_transform=lambda x: x)
```

`output` is `{'y_pred': {'clipwise_output': ..., 'framewise_output': ...},
'y': {'all_labels': ...}}`. The metrics are **epoch-wise accumulators**:
they accumulate a sum and a count and divide at `compute()` time.

`metrics/pl_registry.py` exposes `METRIC_REGISTRY: dict[str, type]` and
`build_metric(name, **kwargs)`.

### The Keras counterparts

Same names, same constructor signatures, same `forward`/`call` semantics,
`tf_` prefix:

| torch | keras |
|---|---|
| `pl_model_att_block.py` | `tf_model_att_block.py` |
| `pl_model_panns_dense121_att.py` | `tf_model_panns_dense121_att.py` |
| `pl_loss_*.py` | `tf_loss_*.py` |
| `pl_metric_*.py` | `tf_metric_*.py` |
| `pl_dataset_sed.py` | `tf_dataset_sed.py` |

The Keras front-end is a single layer in `helpers/tf_frontend_logmel.py`:

```python
class LogmelFrontEnd(keras.layers.Layer):
  def __init__(self, sample_rate, n_fft, hop_size, mel_bins, fmin, fmax,
               top_db=None)
  def call(self, inputs, training=False) -> Tensor
```

## 5. Invariants that must not be broken

| # | rule |
|---|---|
| 1 | The focal loss is **inert** at the shipped parameters: `gamma=0`, `alpha_0=alpha_1=1`, `secondary_factor=1` makes it **bit-identical to mean BCE**. Reproduce the collapse; do not "fix" it. |
| 2 | No loss or metric may apply a second sigmoid. The `AttBlock` already sigmoided. |
| 3 | Only `clipwise_output` ever reaches a loss. `framewise_output` is unsupervised by design. |
| 4 | The `0.3` clip gate in the frame metric stays a **literal**, and the two gates stay **independent**. |
| 5 | `frames_num == 3001`, the DenseNet time axis is `93`, the tail pad is `25` frames, a 5-second row is `500` frames, a 10-minute file is `120` rows. Measure, do not assume. |
| 6 | The Keras `AttBlock` softmax is over **time** (`axis=-2`). `axis=-1` is a silent wrong answer. |
| 7 | Keep librosa's **Slaney, area-normalised** mel matrix. `tf.signal`'s is HTK and is a different matrix. |
| 8 | Feed **power** (`real² + imag²`), not magnitude, to the dB formula. |
| 9 | The **reflect pad** before framing is explicit. Without it `frames_num` is 2977, not 3001. |
| 10 | The `bn0` → SpecAugment ordering is load-bearing. |
| 11 | The waveform augmentation runs on the **whole file, before slicing**. |
| 12 | The label is a **264-vector** one-hot broadcast to every snippet, not a per-frame matrix. |
| 13 | The LR schedule is numerically identical to Ignite's. The golden fixture is in `engines/schedules.py`'s docstring. |
| 14 | Every RNG is seeded from a config field, and per-item randomness is derived from `(seed, epoch, index)`. |
| 15 | The Keras port ships **only** `sed_dense121att`. The other four variants have no `keras.applications` equivalent and no shipped config selects them. |
| 16 | One public class per file. |

## 6. Environment

```bash
cd /run/media/helder/Code_Dump/all_projects_all_branches/competitions_main/kaggle_cornell_birdcall_identification
source .venv/bin/activate
pip install -e .          # already done, but re-run if imports fail
pytest -q tests/<your test file>
```

- Python 3.12, CPU only, no GPU.
- `torch`, `torchvision`, `lightning`, `torchmetrics`, `tensorflow`, `keras`,
  `pydantic`, `pyyaml`, `librosa` are installed.
- `audiomentations` is **not** installed (it pins `librosa<=0.7.2`, which
  conflicts with the mel-matrix build). If your code needs it, import it
  lazily inside the function and let the test skip when it is missing:
  `pytest.importorskip('audiomentations')`.
- There is **no competition audio and no GPU**. Every test must use
  synthetic data with a fixed seed, and short waveforms (2 seconds is
  plenty). A 30-second, 960000-sample waveform is only needed to assert
  the frame arithmetic, and that can be done with the analytic constants.
- `pylint` is not installed; do not spend time trying to run it. Keep to
  the style rules above.

## 7. Rules of engagement

- **Only create or modify the files your brief lists.** Other agents are
  working in the same tree at the same time.
- If a file you need does not exist yet because another agent is writing
  it, code against the interface in section 4 and do not create it.
- When you are done, run your tests and make them pass.
- In your final report, state: the files you created, the test command you
  ran and its result, and anything you could not implement or verify.

## 8. Tests: one per file, and detailed

**Every Python file you create gets its own test file, named after it:**

```
src/birdcall/models/pl_model_att_block.py  ->  tests/test_pl_model_att_block.py
src/birdcall/losses/pl_loss_focal_standard.py
                                            ->  tests/test_pl_loss_focal_standard.py
```

`tests/test_module_test_coverage.py` walks `src/birdcall/` and asserts that
every module has a matching test file, so this is enforced, not requested.

**A test file must be detailed, not a smoke test.** For the module under
test, assert:

1. **The public entry points** return the documented types and shapes.
2. **The documented error paths** raise the documented exception — an
   unknown name, a bad value, a missing file.
3. **The edge cases the documentation calls out** — the traps, the
   degenerate inputs, the off-by-one boundaries.
4. **The behaviour that is easy to get subtly wrong** — the axis, the
   reduction order, the threshold, the seed.

A test that only asserts "it imports" is not a test. Aim for the number of
assertions that would catch the failure the documentation warns about; for
the numeric modules that is usually five to ten assertions per behaviour.

Use `pytest.raises(...)` with a `match=` for error messages. Use
`pytest.approx` for floats. Fix the seed of every random draw with
`numpy.random.default_rng(seed)` or a module level seed, so the tests are
reproducible.

If a behaviour cannot be tested here (no GPU, no competition data, a
dependency that is not installed), test the part that can be and say so in
your report — do not delete the test.

