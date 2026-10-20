# `src/augmentations/` — 7 files

Two live waveform augmenters, the mixup helper, and four legacy/dead files.

| file | LOC | status |
|---|---|---|
| [`sed_default_augment.py`](#sed_default_augmentpy) | 32 | **LIVE** |
| [`sed_background_augment.py`](#sed_background_augmentpy) | 41 | **LIVE** |
| [`mixup.py`](#mixuppy) | 59 | **LIVE** |
| [`base_augment.py`](#base_augmentpy) | 19 | LEGACY |
| [`default_augment.py`](#default_augmentpy) | 54 | LEGACY |
| [`spec_augment.py`](#spec_augmentpy) | 74 | LEGACY + **BROKEN** (L8) |
| [`noise_maker.py`](#noise_makerpy) | 119 | **DEAD** |

**Two different augmentation stages** exist and must not be confused:

| stage | where | domain |
|---|---|---|
| Waveform | `src/augmentations/` — runs in the DataLoader worker | raw samples |
| Spectrogram | `src/helpers/sed_audio_utils.py::SpecAugmentation` — runs **inside the model** | log-mel |

---

## `sed_default_augment.py`

**Status: LIVE** — used when `aug_name == "default"`: 15 of the 21 configs
(the `final_sed` family).

```python
def get_transforms(bckgrd_aug_dir=None) -> {"train": fn, "valid": fn}
```

**Behaviour.** The training stack (L22-28):

| transform | parameters | p |
|---|---|---|
| `AddGaussianNoise` | `min_amplitude=0.001`, `max_amplitude=0.015` | 0.3 |
| `AddGaussianSNR` | library defaults | 0.3 |
| `AddBackgroundNoise` | `bckgrd_aug_dir` | 0.5 — only if the dir is not `None` |

The validation transform is the identity (L16-18). The transform signature is
**`transform(x, sr)`** — the waveform and its sample rate.

**Gotchas**

- ⚠️ `AddGaussianSNR`'s `min_SNR`/`max_SNR` are **linear amplitude ratios,
  not dB** (docs/modules/03:145). The default range spans `0.001 → 1.0`, i.e.
  `1:1000 … 1:1`, so `−60 dB … 0 dB` SNR. At the top of that range the white
  noise has the same standard deviation as the whole recording.
- The `AddBackgroundNoise` path is only added when `bckgrd_aug_dir` is not
  `None`; the shipped configs set it to `"/pinknoise"`, which is **not in the
  repository** — so the path must exist or audiomentations raises at
  construction.
- The transform is applied to the **whole file** before it is sliced into
  snippets (`dataloaders/sed_dataset.py:99`).

---

## `sed_background_augment.py`

**Status: LIVE** — used when `aug_name == "secondary_default"`: the
`example_config` and all five `final_sed_5_fold` configs.

```python
def get_transforms(bckgrd_aug_dir=None, secondary_bckgrd_aug_dir=None) -> {"train": fn, "valid": fn}
```

**Behaviour.** A **palindromic bracket** (L22-37) — 6, 7 or 8 entries
depending on which directories are given:

| # | transform | parameters | p |
|---|---|---|---|
| 1 | `AddGaussianNoise` | library defaults | 0.2 |
| 2 | `AddGaussianSNR` | library defaults | 0.2 |
| 3 | `Gain` | `min_gain_in_db=-15`, `max_gain_in_db=15` | 0.3 |
| 4 | `AddBackgroundNoise` | `bckgrd_aug_dir` | 0.2 |
| 5 | `AddShortNoises` | `secondary_bckgrd_aug_dir`, `min_time_between_sounds=0.0`, `max_time_between_sounds=15.0`, `burst_probability=0.5` | 0.6 |
| 6 | `AddGaussianNoise` | *(repeat)* | 0.2 |
| 7 | `AddGaussianSNR` | *(repeat)* | 0.2 |
| 8 | `Gain` | *(repeat)* | 0.3 |

Entries 4 and 5 are conditional. A commented-out
`AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3)` sits at
L23.

**Gotchas**

- ⚠️ **The bracket means the same transform is applied twice.** A `Gain` of
  +15 dB followed by another +15 dB yields up to **+30 dB with no clipping
  guard**, and the SNR is perturbed twice.
- ⚠️ `PitchShift` is imported (L7) and never used.
- The directory arguments are **not validated** — audiomentations raises at
  construction if the path does not exist.
- `transform(x, sr)` — the same two-argument signature as the default stack.

---

## `mixup.py`

**Status: LIVE** — the mixup coefficient generator and the mixing kernel.
Used by 9 of the 21 configs (`apply_mixup=True`).

```python
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234)
    def get_lambda(self, batch_size, device) -> Tensor

def do_mixup(x, mixup_lambda)
```

**Behaviour.** `get_lambda` (L34-47) draws `lam ~ Beta(alpha, alpha)` once per
**pair** and emits `[lam, 1-lam]` — so even and odd indices are paired.

`do_mixup` (L49-59) mixes them:

```
out = x[0::2] * mixup_lambda[0::2] + x[1::2] * mixup_lambda[1::2]
```

**Two call sites, and they must agree**

- The **labels** are mixed in `engine/sed_engine.py:28-30`, and the batch is
  halved to `(B·C)//2, 1, 264`.
- The **spectrogram** is mixed inside the model, `models/sed_models.py:390-392`.

**Gotchas**

- ⚠️ `random_seed=1234` is a **default argument** and the engine constructs
  `Mixup(mixup_alpha=1.)` without overriding it — so **every run, every epoch
  and every DDP rank draws the same sequence** of `Beta(1,1)` lambdas. The
  mixup is far less random than it looks, and in a multi-rank run every rank
  mixes the same pairs with the same weights.
- ⚠️ `do_mixup` assumes an **even** batch length; the halved label tensor
  relies on that.
- The MIT header (L1-23) credits `qiuqiangkong/audioset_tagging_cnn`.

---

## `base_augment.py`

**Status: LEGACY.** `get_transforms()` → `{"train": fn, "valid": fn}`.

**Behaviour.** `transform(x)` = `torch.tensor(x/255).permute(0,3,1,2).float()`
(L6/L12) — an `(S,H,W,3)` uint8 stack becomes a `(S,3,H,W)` float tensor.
**No resize, no crop, no augmentation at all**, and the train and valid
transforms are byte-identical.

**Gotchas**

- ⚠️ The train transform is a **no-op** — it is not an augmentation.
- `x/255` on a uint8 array produces float64 first, then `.float()` casts back
  to float32 — precision is lost twice.
- `permute(0,3,1,2)` requires a 4-D input; the v2/v3 engines feed 3-D arrays
  and would raise `RuntimeError: input.dim() = 3 …`.
- `from torchvision import transforms` (L1) is unused and shadowed by the
  local name `transforms` in `get_transforms` (L16).

---

## `default_augment.py`

**Status: LEGACY.** Imported by `engine/main_engine_v4.py:206`.

**Public API.** `random_crop(x)`, `get_training_augmentation()`,
`get_validation_augmentation()`, `get_transforms()`.

**Behaviour.** Each mel image becomes three channels — the image plus
`librosa.feature.delta` at order 1 and order 2 — stacked to `(3,H,W)`, scaled
by `/30`, cropped, and resized to 256×256.

`random_crop` (L8-16) fires when `np.random.random() > 0.66` (p ≈ 0.34) with
insets `x ∈ [1,9]` and `y ∈ [1,24]`. The validation transform is identical
**minus** the crop.

**Gotchas**

- ⚠️ `librosa.feature.delta` is applied to a 2-D `(H,W)` mel image, so it is
  interpreted as `(D=128, n=128)` and differentiates along the **width** axis,
  not along time.
- The `/30` divisor and the `256,256` size are bare literals.
- `F.interpolate` defaults to **nearest-neighbour**, not bilinear.
- `transform(x)` takes one argument — incompatible with the SED waveform
  contract `transform(x, sr)`.
- `import numpy as np` appears twice (L3, L6); `from torchvision import
  transforms` (L1) is unused.

---

## `spec_augment.py`

**Status: BROKEN** — fails at import time.

```python
def spec_augment(spec, num_mask=2, freq_masking=0.15, time_masking=0.20, value=0)
def random_crop(x)
def get_transforms()
```

**⚠️ Broken at L8**: `from helpers.audio_utils import mono_to_color` →
`ImportError: cannot import name 'mono_to_color' from 'helpers.audio_utils'`.
`mono_to_color` is **defined nowhere in the repository**.

**Behaviour if it ran.** `spec_augment` (L15) sets
`num_mask = np.random.randint(1, num_mask)` — and `randint`'s high bound is
**exclusive**, so with the default `num_mask=2` it is **always 1** iteration.
Each iteration zeroes a full-width horizontal band
(`freq_percentage ~ U(0, 0.15)`) and then a full-height vertical band
(`time_percentage ~ U(0, 0.20)`), in that order. The training transform fills
the bands with **`x_i.min()`**, not the mean and not 0 — the `value=0` default
is never used by the shipped transform. The pipeline then does
`mono_to_color(x_i)/255.0` → `random_crop` → `F.interpolate(size=(256,512))`.

**Gotchas**

- ⚠️ The band fill is the per-example **minimum**.
- `import cv2` (L5) and `from skimage.transform import resize` (L6) are
  **unused**, and neither is in `requirements.txt`.
- The resize is `(256,512)` here vs `(256,256)` in `default_augment`.
- The only importers are `main_engine_v2.py:153` and `main_engine_v3.py:153`.

---

## `noise_maker.py`

**Status: DEAD** — imported by nothing (verified by repo-wide grep: only its
own definition and a docstring example match).

**Public API.** `powerlaw_psd_gaussian(exponent, size, fmin=0)`.

**Behaviour.** A verbatim vendored copy of
[felixpatzelt/colorednoise](https://github.com/felixpatzelt/colorednoise)
(MIT, L1-23). Generates Gaussian **1/f^β** noise using the Timmer & Koenig
(1995) algorithm: `rfftfreq` → flatten below `fmin` → amplitude scaling
`f**(-exponent/2)` → random real+imaginary Fourier components → DC and
Nyquist forced real → `irfft` → divided by the theoretical σ, so the output
is **unit variance**. `exponent=1` is pink/flicker, `exponent=2` is brown.

**Gotchas**

- ⚠️ It is a **library snippet, not a script** — there is no `main()`, no
  `__main__` block and no file writer. Running it produces nothing on disk.
  The actual pink-noise WAVs came from an external Kaggle kernel
  (`README.md:127`); `sed_background_augment` only *reads* that directory.
- `size` may be an int or an iterable; the **last** dimension is the time
  axis and all others are independent realisations.
- `size` is copied before being mutated (L70-73), so a caller's list is safe.

---

## See also

- [modules/03 — Augmentations](../modules/03-augmentations.md) — the topic chapter
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/feature-parity-matrix.md](../ports/feature-parity-matrix.md) §D — the porting checklist for this package
