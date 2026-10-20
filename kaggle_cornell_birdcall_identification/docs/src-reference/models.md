# `src/models/` — 7 files

One live file holds the entire winning architecture; the other six are a
superseded model zoo.

| file | LOC | status |
|---|---|---|
| [`sed_models.py`](#sed_modelspy) | 809 | **LIVE** |
| [`aux_densenet121.py`](#aux_densenet121py) | 161 | LEGACY |
| [`aux_densnet161.py`](#aux_densnet161py) | 156 | DEAD + **BROKEN** |
| [`classifier_dcase.py`](#classifier_dcasepy) | 173 | LEGACY |
| [`classifier_m_series.py`](#classifier_m_seriespy) | 110 | **DEAD** |
| [`densenet121.py`](#densenet121py) | 18 | LEGACY |
| [`densenet161.py`](#densenet161py) | 18 | LEGACY |

---

## `sed_models.py`

**Status: LIVE** — the whole solution. Reached from
`engine/sed_engine.py:177-198`.

> The MIT License, Copyright (c) 2018-2020 Qiuqiang Kong — vendored from
> [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
> (`README.md:136`).

**Purpose.** Five PANNs-style sound-event-detection models that take a raw
waveform and return a clip-level and a frame-level score over 264 species.

**Public API**

```python
# module-level helpers
def init_layer(layer)                      # xavier_uniform_ + zero bias
def init_bn(bn)                            # bias 0, weight 1
def interpolate(x, ratio)                  # (B,T,C) -> (B,T*ratio,C)
def pad_framewise_output(framewise_output, frames_num)   # repeat the last frame

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels)
    def init_weight(self)
    def forward(self, input, pool_size=(2,2), pool_type='avg')

class AttBlock(nn.Module):
    def __init__(self, in_features, out_features, activation="linear", temperature=1.0)
    def init_weights(self)
    def forward(self, x)                    # -> (x, norm_att, cla)
    def nonlinear_transform(self, x)

class PANNsCNN14Att(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, classes_num, apply_aug)
class PANNsDense121Att(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, classes_num, apply_aug, top_db=None)
class PANNsDense161Att(nn.Module):    # same + top_db
class PANNsDense169Att(nn.Module):    # same + top_db
class PANNsDense201Att(nn.Module):    # same + top_db
```

Each model's `forward(input_data)` takes the **tuple** `(input_x, mixup_lambda)`
and returns the dict `{"framewise_output", "clipwise_output"}`.

**The shape trace** (train: B=32, C=1, S=960000, 30 s @ 32 kHz)

| stage | shape |
|---|---|
| `input_x` | (B, C, S) |
| `input_x.reshape(b*c, s)` | (B·C, S) — the snippet axis folds into the batch |
| `Spectrogram` | (B·C, 1, 3001, 513) |
| `LogmelFilterBank` | (B·C, 1, 3001, 64) |
| transpose 1↔3 → `bn0` → transpose 1↔3 | (B·C, 1, 3001, 64) |
| `expand` to 3 channels | (B·C, 3, 3001, 64) — a view, zero copy |
| `densenet121.features` | (B·C, 1024, 93, 2) — **measured** |
| mean over the frequency axis | (B·C, 1024, 93) |
| max_pool1d + avg_pool1d (k=3, s=1, p=1) | (B·C, 1024, 93) |
| `fc1` 1024→1024 | (B·C, 1024, 93) |
| `att_block` | (B·C, 264, 93) |
| `clipwise_output` = sum over time | (B·C, 264) |
| `segmentwise_output` | (B·C, 93, 264) |
| `interpolate` ×32 | (B·C, 2976, 264) |
| `pad_framewise_output` to 3001 | (B·C, 3001, 264) — **25 frames of tail repeat** |

**Behaviour worth knowing**

- `AttBlock.forward` (L164-169) is `softmax(tanh(att(x)), dim=-1) * sigmoid(cla(x))`
  summed over time. The `tanh` **replaces** upstream PANNs' `clamp(-10, 10)`;
  `README.md:22` states this was done to avoid overfitting.
- `bn_att` (L156) and `temperature` (L164) are **dead parameters** — never
  used in `forward`.
- `interpolate_ratio = 32` (L191, L329, L451, L580, L703) and the DenseNet's
  93 frames give 2976 < 3001, so the tail pad is exactly 25 frames.
- `preprocess` (L379-393) applies SpecAugment **after** `bn0`, and mixup
  **after** SpecAugment, both only when `self.training and self.apply_aug`.
- Mixup is applied to the **spectrogram**, inside the model (L390-392); the
  labels are mixed separately in `prepare_batch`.
- When `mixup_lambda is not None`, `forward` sets `b = (b*c)//2; c = 1`
  (L405-407) — **the batch axis changes meaning mid-forward**, and
  `framewise_output.reshape(b, c, T, O)` (L312) undoes it.
- `fc1` widths: 1024→1024 (CNN14, Dense121), 2208→2048 (Dense161),
  1664→1024 (Dense169), 1920→1024 (Dense201).
- The DenseNet backbones load ImageNet weights (`pretrained=True`, L367, L492,
  L615, L738).
- Only `PANNsCNN14Att` accepts `top_db`; the four Dense variants do too, the
  CNN14 does not (L182) — a config with `sed_ccnn14att` **and** `top_db` raises
  `TypeError`.

**Gotchas**

- ⚠️ **The checkpoint is 744 tensors.** For `PANNsDense121Att` with
  `classes_num=264`: 9,713,195 total elements, 9,628,768 trainable
  parameters, 1,083,456 frozen front-end elements and 7,037,625 DenseNet
  backbone elements across 725 tensors. **Only 19 tensors are not backbone** —
  those are the ones a port must remap by hand, and they are listed in
  [ports/porting-worklist.md §12](../ports/porting-worklist.md#12-weight-conversion).
- ⚠️ `apply_aug=False` **breaks mixup**: the model's `b = (b*c)//2` branch still
  fires while the spectrogram is not mixed, so the batch halves against
  unmixed data.
- ⚠️ `init_weight` deliberately skips the backbone — the DenseNet's own
  initialisation is preserved.
- The `AttBlock` softmax is over **time**. In a Keras port the axis must be
  `-2`; `axis=-1` is a silent wrong answer.
- `helpers.sed_audio_utils` is imported with `from helpers.sed_audio_utils import *`
  (L177), so the front-end's names are in this module's namespace.

---

## `aux_densenet121.py`

**Status: LEGACY** — the deep-supervision architectural ancestor. Imported by
`engine/main_engine_v3.py:145` and `engine/main_engine_v4.py:201`.

**Purpose.** DenseNet-121 feature extractor with CBAM attention, three
auxiliary heads and skip-connection fusion, for snippet-level classification.

**Public API**

```python
class SkipBlock(nn.Module):          # __init__(self, in_channels, out_channels, scale_factor=2)
class AuxBlock(nn.Module):           # __init__(self, last_fc, num_classes, base_size, dropout)
class ChannelAttention(nn.Module):   # __init__(self, in_planes, ratio=16)
class SpatialAttention(nn.Module):   # __init__(self, kernel_size=7)
class _Transition(nn.Sequential):    # __init__(self, num_input_features, num_output_features)
class ConvolutionalBlockAttentionModule(nn.Module):
                                      # __init__(self, in_planes, ratio=16, kernel_size=7)
class AUXDense121(torch.nn.Module):
    def __init__(self, base_size=64, last_fc=8, num_classes=264, dropout=0.2,
                 ratio=16, kernel_size=7, last_filters=8)
```

**Behaviour.** Input `(bs, s, c, h, w)` → reshaped to `(bs*s, c, h, w)`
(L136-137). Output is a **list** `[x, aux3, aux2, aux1]`, each
`(bs, s, num_classes)` (L157-161) — the four heads that `main_engine_v3` and
`_v4` iterate over.

**Gotchas**

- `AuxBlock.__init__`'s first `Linear` hard-codes `base_size*8` in-features
  (L30), so changing `base_size` is safe but changing the
  `last_filters`/`base_size` ratio is not.
- `AuxBlock.__init__`'s argument order is `(last_fc, num_classes, base_size,
  dropout)` and is called **positionally** (L115-117) — correct but fragile.
- `densenet121(pretrained=True)` (L105) downloads ImageNet weights at
  construction.
- The skip-channel indices (idx4→256, idx6→512, idx8→1024, final→1024) were
  verified correct against torchvision, and `torch.cat` yields 2048 =
  `base_size*4*8` (L119).

---

## `aux_densnet161.py`

**Status: DEAD + BROKEN** — imported by nothing in the repository.

**Public API.** Identical class set to `aux_densenet121.py`, with
`AUXDense161.__init__(self, base_size=64, last_fc=2, num_classes=264,
dropout=0.2, ratio=16, kernel_size=7, last_filters=8)`.

**Broken at** — `NameError: name 'F' is not defined` at **L17** (in
`SkipBlock.forward`) and **L149**. The module imports only `torch`,
`torch.nn as nn`, `torchvision.models` and `collections.namedtuple` — **there
is no `import torch.nn.functional as F`**, yet `F.avg_pool2d` and
`F.adaptive_avg_pool2d` are called. The module *imports* fine; it dies on the
first forward.

**Gotchas**

- The output is a **tuple** `(x, aux3, aux2, aux1)` (L156), whereas
  `aux_densenet121` returns a **list** — an inconsistency between siblings.
- `last_fc` defaults to **2** here vs **8** in the 121 variant, so
  `base_size*last_fc` is 128 vs 512.
- The channel arithmetic was verified correct for torchvision's densenet161
  (idx4→384, idx6→768, idx8→2112, final→2208). The missing import is the only
  defect.

---

## `classifier_dcase.py`

**Status: LEGACY** — the DCASE audio-tagging CNN port. Imported by
`engine/main_engine.py:98` and `engine/main_engine_v2.py:139`.

**Public API**

```python
def init_layer(layer, nonlinearity='leaky_relu')
def init_bn(bn)                                  # dead - never called
class SpatialAttention2d(nn.Module):  # __init__(self, channel)
class GAB(nn.Module):                 # __init__(self, input_dim, reduction=4)
class SCse(nn.Module):                # __init__(self, dim)
class ConvBlock(nn.Module):           # __init__(self, in_channels, out_channels, reduction=16)
class Classifier_DCase(nn.Module):     # __init__(self, num_classes=80)
```

**Behaviour.** Consumes `(bs, seq, c, h, w)`, flattens to `(bs*seq, c, h, w)`
(L144-145), returns `(bs, seq, num_classes)` (L172). Five `ConvBlock`s are
hard-wired `3→32→64→128→256→512` (L114-118) and the head is
`Linear(1536, 512)` (L123), where 1536 = 3×512 from the pooled branches
(L156-167).

**Gotchas**

- ⚠️ The head's 1536 is **not parameterised** — changing the block widths
  breaks it.
- `ConvBlock.__init__` accepts `reduction=16` and **never uses it**; SCSE is
  hard-wired to `reduction=4` inside `GAB` (L38-39).
- ⚠️ `ConvBlock.forward` falls into `import pdb; pdb.set_trace()` (L105-106)
  on an unknown `pool_type` — a debugger trap inside a dataloader worker.
- Each 2×2 conv with padding (1,1) **grows** H and W by 1, so the input must
  be at least H≥5, W≥8 for the `max_pool2d((5,8))` / `(2,4)` at L156/L160.
- `init_bn` (L14) is never called.

---

## `classifier_m_series.py`

**Status: DEAD** — imported by nothing.

**Public API**

```python
def adaptive_concat_pool2d(x, sz=(1,1))
class ConvBlock(nn.Module):      # __init__(self, in_channels, out_channels, kernel_size=3, pool=True)
class Classifier_M3(nn.Module):   # __init__(self, num_classes=6)
class Classifier_M2(nn.Module):   # __init__(self, num_classes=6)
```

**Behaviour.** EfficientNet-ish (MBConv-style) residual dense blocks with
pyramid pooling. `ConvBlock.conv2` takes `out_channels + in_channels` inputs
(L23) — the concatenation of `x` and `x1` (L42). `pool=False` only on the last
block of each net (L54, L87), so the final spatial size is input/16. The head
in-features are hard-coded: `BatchNorm1d(3840)` for M3 = 2×(128+256+512+1024)
(L57) and `BatchNorm1d(1792)` for M2 = 2×(128+256+512) (L90).

**Gotchas**

- ⚠️ The `num_classes=6` defaults are **wrong for this task** (264 species),
  and no engine instantiates these classes.
- `Classifier_M3.forward` assigns `x1` (L69) and never uses it.
- `ConvBlock.pool` halves **both** H and W (L43), so a mel spectrogram's
  aspect ratio is squashed.

---

## `densenet121.py` / `densenet161.py`

**Status: LEGACY** — thin wrappers. Imported by `engine/main_engine_v2.py:142`
and `:145` respectively.

**Public API**

```python
class DenseNet121(nn.Module):   # __init__(self, pretrained=True, num_classes=6)
class DenseNet161(nn.Module):   # __init__(self, pretrained=True, num_classes=6)
```

**Behaviour.** `models.densenet121(pretrained=pretrained)` (L8) — the
`pretrained` default is **True**, so ImageNet weights download at construction.
The classifier head is replaced with `Linear(1024, num_classes)` (121) or
`Linear(2208, num_classes)` (161). Input `(bs, seq, c, h, w)` → `(bs*seq, c,
h, w)` → output `(bs, seq, num_classes)` (L13-18).

**Gotchas**

- The `num_classes=6` default is a placeholder; the engine always overrides it
  with `hparams.num_classes` — **which no config defines** (see
  [engine.md](engine.md)).
- `pretrained=` is deprecated in torchvision ≥ 0.13 (warns, still functional).
- The file is named `densenet161.py` while its sibling is misspelled
  `aux_densnet161.py` (missing "e") — the import paths differ accordingly.

---

## See also

- [modules/04 — Model architecture](../modules/04-model-architecture.md)
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/tensorflow-keras-guide.md](../ports/tensorflow-keras-guide.md) §2 — the Keras shape trace and its traps
