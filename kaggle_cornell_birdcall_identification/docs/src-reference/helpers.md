# `src/helpers/` — 2 files

One live file holds the entire in-graph audio front-end; the other is the
legacy librosa helper set.

| file | LOC | status |
|---|---|---|
| [`sed_audio_utils.py`](#sed_audio_utilspy) | 271 | **LIVE** |
| [`audio_utils.py`](#audio_utilspy) | 55 | LEGACY |

---

## `sed_audio_utils.py`

**Status: LIVE.** The front-end runs **inside the model**: raw waveform in,
log-mel out, every step. There is no mel caching.

Vendored from [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
(ISC/librosa-derived).

**Public API**

```python
class DFTBase(nn.Module):
    def __init__(self)
    def dft_matrix(self, n)        # (n,n) unitary DFT
    def idft_matrix(self, n)

class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect',
                 freeze_parameters=True)
    def forward(self, input)       # (B, L) -> (real, imag), each (B, 513, T)

class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', power=2.0,
                 freeze_parameters=True)
    def forward(self, input)       # (B, L) -> (B, 1, T, 513)

class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000,
                 is_log=True, ref=1.0, amin=1e-10, top_db=80.0,
                 freeze_parameters=True)
    def forward(self, input)       # (B, C, T) -> (B, T, 64)
    def power_to_db(self, input)

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num)
    def forward(self, input)
    def transform_slice(self, e, total_width)

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num,
                 freq_drop_width, freq_stripes_num)
    def forward(self, input)
```

**Behaviour**

- `STFT` is implemented as **two `Conv1d` layers** with `bias=False` whose
  weights are the dense DFT matrix times the window, frozen when
  `freeze_parameters=True` (`requires_grad = False`). `hop_length` defaults
  to `win_length // 4`. With `center=True` the input is **reflect-padded** by
  `n_fft // 2` first.
- `Spectrogram` computes `real**2 + imag**2` and, for `power != 2.0`, raises it
  to `power/2`. Output `(B, 1, T, n_fft//2+1)`.
- `LogmelFilterBank` holds `melW = librosa.filters.mel(sr, n_fft, n_mels,
  fmin, fmax).T` — shape `(513, 64)`, frozen — and computes `input @ melW`.
  `power_to_db` is `10*log10(clamp(x, min=amin)) - 10*log10(max(amin, ref))`,
  then (if `top_db` is not `None`) clamps to `max - top_db`.
- `DropStripes` zeroes `stripes_num` random bands along `dim` (2 = time,
  3 = frequency) of a `(B,C,T,F)` tensor, **in place, in a Python loop over
  the batch**. It is a **no-op when `self.training` is False**.
- `SpecAugmentation` runs the time-dropper (dim 2) and then the
  freq-dropper (dim 3).

**Configuration in the live models** — every one of the five classes in
`models/sed_models.py` builds the same front-end:

```
Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
                  fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=top_db,
                  freeze_parameters=True)
SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                 freq_drop_width=8, freq_stripes_num=2)
```

With the shipped config (`window_size=1024`, `hop_size=320`, `period=30` s at
32 kHz): `S = 960000` samples, `n_fft//2+1 = 513` linear bins,
**`frames_num = 960000//320 + 1 = 3001`**.

**Gotchas**

- ⚠️ **L58 is a hard blocker on modern librosa.**
  `fft_window = librosa.util.pad_center(fft_window, n_fft)` passes `size`
  positionally, but librosa **0.10.0** made it keyword-only:
  `pad_center(data, *, size, axis=-1, **kwargs)`. Verified on librosa 0.10.2:

  ```
  TypeError: pad_center() takes 1 positional argument but 2 were given
  ```

  So on any librosa ≥ 0.10, `PANNsDense121Att.__init__` **raises** and the
  model cannot be constructed. **The original does not run today.** The fix
  is one line — `pad_center(fft_window, size=n_fft)` — and a port should make
  it.
- ⚠️ **`amin = 1e-10` is the AMP killer.** Under fp16, `1e-10` underflows to
  `0.0`, so `log10(0) = -inf` → NaN. This is the root cause the repo's
  `USE_AMP = False` is working around, and the `amin` fix is described but
  **untested** (docs/modules/02).
- ⚠️ **`top_db` is `None` in every shipped config**, so the clamp is dead code
  in the final path.
- ⚠️ `power_to_db` is `log10` of **power**, not magnitude. Feeding magnitude
  gives a factor-of-2 error that no test catches.
- ⚠️ `LogmelFilterBank.forward` expects `(B, C, T)` and returns `(B, T, M)` —
  the opposite axis order to `Spectrogram`, which returns `(B, 1, T, F)`.
- ⚠️ `DropStripes` mutates in place and skips work when not training. A Keras
  port must thread the `training` flag explicitly or SpecAugment will corrupt
  `framewise_output` at inference — the tensor the 13-model vote consumes.
- ⚠️ `DropStripes` draws an **independent band per batch item**; a shared mask
  would weaken the augmentation invisibly.
- The mel matrix is librosa's — **Slaney** mel, area-normalised.
  `tf.signal.linear_to_mel_weight_matrix` is HTK with unit-peak triangles and
  is a *different matrix*.

---

## `audio_utils.py`

**Status: LEGACY** — the pre-SED librosa helpers. Imported by
`dataloaders/audio_dataset.py`, `audio_dataset_v2.py`, `audio_dataset_v3.py`
(all via `from helpers.audio_utils import *`) and `audio_dataset_v4.py:14`.
The shipped SED path uses `sed_audio_utils.py` instead.

**Public API** — exactly five functions

```python
def read_audio(pathname, conf)                 # (np.float32 array, duration)
def audio_to_melspectrogram(audio, config)     # (n_mels, T)
def show_melspectrogram(mels, config, title='Log-frequency power spectrogram')
def count_parameters(model)
def normalise(arrs, min_val=None, max_val=None)
```

**⚠️ `mono_to_color` and `get_snippets` are NOT here** — and are not defined
anywhere in the repository. Both are called anyway:

| symbol | called at |
|---|---|
| `get_snippets` | `dataloaders/audio_dataset.py:102`, `audio_dataset_v2.py:145,150,160,191`, `audio_dataset_v3.py:142,147,157,186` |
| `mono_to_color` | `augmentations/spec_augment.py:8,49,63`, `dataloaders/audio_dataset.py:112` |

Note the difference in failure mode: `spec_augment.py` uses a **named** import
so it is an `ImportError` at import time; the datasets use `import *`, which
succeeds and so the failure is a `NameError` at first use.

**Behaviour**

- `read_audio` (L9-20): `sf.read` for `.wav`, otherwise
  `librosa.load(sr=conf.sampling_rate, mono=True, res_type="kaiser_fast")`;
  pads to `conf.samples` with `np.pad(..., conf.padmode)` centred at
  `offset = padding // 2`. Returns `(np.float32 array, duration)` where
  **`duration` is a sample count, not seconds**.
- `audio_to_melspectrogram` (L22-32): `librosa.feature.melspectrogram` with
  the config's `sampling_rate`/`hop_length`/`n_mels`/`n_fft`/`fmin`/`fmax`,
  then `librosa.power_to_db` (default `top_db=80`) and `.astype(np.float32)`.
  Output `(n_mels, T)`.
- `normalise` (L47-56): scales each array to 0-255, then
  `np.stack(arr2, axis=2).astype('uint8')` — because `axis=2` inserts the
  *list* axis last, the result is `(H, W, K)`, RGB-style.
- `count_parameters` (L44-45): sums `p.numel()` **only over
  `requires_grad=True`** parameters, so the frozen PANNs front-end counts as
  zero.
- `show_melspectrogram` (L34-42): `librosa.display.specshow` + colorbar +
  `plt.show()` — blocking.

**Gotchas**

- ⚠️ **L10 is a live bug**: `pathname.suffix==".wav"` is missing the leading
  dot, so it is **always False**. Every file — including `.wav` — goes through
  `librosa.load` (resampled to 32 kHz, mono) and the `sf.read` branch is dead
  code.
- ⚠️ **`normalise` has a sticky-range bug.** `min_val`/`max_val` are captured
  in the loop scope and never reset, so every array after the first is
  normalised with the **first** array's range. Verified numerically:
  `normalise([[0,1],[2,3]], [[10,20],[30,40]])` → the second array becomes
  `[[82,164],[246,72]]` (uint8 wrap-around) instead of `[[0,85],[170,255]]`.
- ⚠️ `read_audio` **never truncates** — an input longer than `conf.samples` is
  returned longer than the window.
- `normalise` truncates rather than rounds when casting to `uint8`, and
  negative values wrap.
- `show_melspectrogram` re-imports `matplotlib.pyplot` locally (L35),
  shadowing the module-level import (L4), and calls `plt.show()` — never call
  it headless.
- `count_parameters` ignores frozen parameters, so its number is **not** the
  model's parameter count.
- `import numpy as np` is duplicated (L2, L5).

---

## See also

- [modules/02 — Audio front-end](../modules/02-audio-frontend.md) — the topic chapter
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/tensorflow-keras-guide.md](../ports/tensorflow-keras-guide.md) §2a–2g — the Keras front-end and its four numerical traps
