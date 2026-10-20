# `src/dataloaders/` — 6 files

The dataset layer. One live class, one live sampler, and four superseded
generations that no longer work.

| file | LOC | status |
|---|---|---|
| [`sed_dataset.py`](#sed_datasetpy) | 133 | **LIVE** |
| [`imbalanced_dataset_sampler.py`](#imbalanced_dataset_samplerpy) | 25 | **LIVE** |
| [`audio_dataset.py`](#audio_datasetpy) | 129 | LEGACY + **BROKEN** |
| [`audio_dataset_v2.py`](#audio_dataset_v2py) | 225 | LEGACY + **BROKEN** |
| [`audio_dataset_v3.py`](#audio_dataset_v3py) | 219 | LEGACY + **BROKEN** |
| [`audio_dataset_v4.py`](#audio_dataset_v4py) | 198 | LEGACY + **BROKEN** (L144) |

---

## `sed_dataset.py`

**Status: LIVE** — the only dataset the shipped path uses. Reached from
`engine/sed_engine.py:214-224`.

**Purpose.** Reads a fold CSV, loads the referenced WAV, and returns fixed-length
waveform snippets with three label tensors.

**Public API**

```python
class SedDataset(Dataset):
    def __init__(self, root_dir, csv_dir, period, bird_code, inv_ebird_label,
                 isTraining, num_test_samples, transform=None, background_audio_dir=None)
    def get_label(self, dataset, idx) -> str        # the primary ebird code
    def init_workers_fn(self, worker_id)             # per-worker numpy seed
    def get_audio(self, idx) -> dict
    def convert_labels_to_coded(self, num_images, labels) -> Tensor
    def __len__(self) -> int
    def __getitem__(self, idx) -> dict
```

**Behaviour**

- `__init__` (L27-29) reads the CSV with `pd.read_csv(csv_dir)` and applies
  **`eval()` to the `secondary_labels` column** — arbitrary Python execution
  on CSV content. A NaN cell raises `TypeError`.
- Rows are kept as `{"filename", "ebird_code", "secondary_labels"}` dicts
  (L29). `self.sampler = ImbalancedDatasetSampler` (L25) stores the **class**;
  the engine instantiates it.
- `background_audio_dir` (L34-41) appends every `**/*.wav` under it as a
  label-less row `{"filename": bk}` — hard negatives on all 264 classes.
- `get_audio` (L55-85) rewrites `.mp3` → `.wav` (L67, L69) and supports two
  `root_dir` shapes: a `Path` (`root_dir/ebird_code/name.wav`) or a `dict`
  keyed by the **first letter** of the ebird code (the 5-archive sharding
  from `configs.get_dict_value`, which nothing calls).
- `convert_labels_to_coded` (L87-92) produces a **264-vector** one-hot, not a
  per-frame matrix. The weak file-level label is broadcast to every snippet.
- `__getitem__` (L94-133) applies the waveform `transform` to the **entire
  file** (L99) *before* slicing into 30 s snippets.
  - **Training** (L103-115): a short clip is placed at a **random** offset
    inside the window (zero-padded); a long clip is **randomly cropped**.
  - **Validation** (L117-125): deterministic contiguous slicing, tail zero-pad,
    long files truncated.
- The returned dict is `{"filename", "waveforms", "all_labels",
  "primary_labels", "secondary_labels"}` (L132-133).

**Gotchas**

- ⚠️ `eval()` on CSV content (L28) — use `ast.literal_eval`.
- ⚠️ `item["duration"]` is a **sample count**, not seconds (L83) — the name
  lies. `effective_length = sr * period` (L101) is in samples too, so the
  arithmetic is self-consistent.
- ⚠️ `secondary_labels` are *all* labels minus the primary (L81) — the
  "secondary" set can therefore contain a species that is also the primary of
  another file.
- The `root_dir` dict branch (L66-68) is dead: `get_dict_value` is imported by
  every config and called by none.
- `additional_loader_params` (L22-24) is `{"worker_init_fn": self.init_workers_fn}`.

---

## `imbalanced_dataset_sampler.py`

**Status: LIVE** — used by every shipped training run.

**Purpose.** Inverse-frequency sampling so rare species are seen as often as
common ones.

**Public API**

```python
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, get_label)
    def __iter__(self)
    def __len__(self) -> int
```

**Behaviour.** Counts each label (L11-17), builds weights `1/count` (L19-20),
and samples `len(dataset)` indices **with replacement** via
`torch.multinomial` (L23-24).

Vendored from [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)
(`README.md:138`).

**Gotchas**

- ⚠️ **Not a `DistributedSampler`.** Under DDP every rank can see the same
  samples unless the framework shards it.
- `get_label` is passed as `dataset.get_label` and called as
  `get_label(dataset, idx)` — the label is the **primary** ebird code.
- `self.indices` is a plain list; `__iter__` returns a generator, so the
  sampler is single-pass per epoch.
- A fresh `multinomial` draw happens every `__iter__`, so the epoch's
  ordering and selection change each time.

---

## `audio_dataset.py`

**Status: LEGACY + BROKEN** — v1 of the pre-SED mel-*image* line. Imported by
`engine/main_engine.py:107,111,115`.

**Purpose.** Reads a birdsong CSV, cuts fixed-length snippets, converts to a
mel spectrogram and a 3-channel image, one-hot/multi-hot labels, optional
memory-buffer mixup.

**Public API**

```python
class AudioDataset(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code, mem_size=32,
                 file_type="mp3", num_splits=5, apply_mix_aug=False,
                 isTraining=True, transform=None)
    def get_label(self, dataset, idx)
    def init_workers_fn(self, worker_id)
    def mix_aug(self, y_snippets, label)
    def __len__(self)
    def __getitem__(self, idx) -> {"filenames", "images", "coded_labels"}
```

**Broken at**

- **L102** `NameError: name 'get_snippets' is not defined` — first failure.
- **L112** `NameError: name 'mono_to_color' is not defined` — after 102 is fixed.

Both symbols are called but **defined nowhere in the repository**. Note the
import at L12 is `from helpers.audio_utils import *`, which *succeeds* — the
names are simply absent, so the failure is a `NameError` at call time, not an
`ImportError` at import time.

**Gotchas**

- `conf` is a **required positional** argument (L18) that no config supplies —
  every config's `*_ds_params` carries `period` and `num_test_samples` instead.
- `mix_aug` (L63) **mutates** `labels[index]` with a memory-buffer label, so a
  row can become multi-label.
- The `file_type != "mp3"` branch (L86) uses `label[0]` — the first character
  of the ebird code as a directory letter.
- `read_audio` returns `duration` in **samples**, and `snip_duration` is
  likewise in samples, so the arithmetic is self-consistent.

---

## `audio_dataset_v2.py`

**Status: LEGACY + BROKEN** — adds secondary labels, optional Xeno-canto
audio, a label mixer, a noisy/clean flag, and noise injection. Imported by
`engine/main_engine_v2.py:157,161,165`.

**Public API**

```python
class AudioDatasetV2(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code, inv_ebird_label,
                 background_audio_dir=None, xeno_csv=None, xeno_dir=None,
                 file_type="mp3", num_splits=5, apply_mixer=False,
                 isTraining=True, transform=None)
    def get_label(self, dataset, idx)
    def init_workers_fn(self, worker_id)
    def mix_aug(self, y_snippets, original_labels)
    def get_audio(self, idx)
    def __getitem__(self, idx) -> {"filenames", "images", "coded_labels", "clean"}
```

**Broken at** — `NameError: name 'get_snippets' is not defined` at **L145**
(also L150, L160, L191).

**Gotchas**

- ⚠️ **Shared-list aliasing in `mix_aug`.** L82 builds
  `labels = [original_labels] * len(y_snippets)` — every element is the *same*
  list object — and L96 does `labels[index].extend(labels_secondary)`. The
  extension therefore mutates **every** snippet's label list.
- ⚠️ `self.selection_index = list(range(0, 5*sampling_rate, sampling_rate))`
  (L69) = `[0, 32000, 64000, 96000, 128000]`, but a snippet is only 32000
  samples long. Indices ≥ 32000 slice past the end → empty arrays →
  `np.array(temp).mean(axis=0)` on an empty array → a shape-mismatch
  assignment. *(Unverified — requires a run.)*
- The `apply_mixer` path builds `ebird_counts` with
  `df.groupby("ebird_code").agg(len)` (L47). In pandas ≥ 2 that yields a
  DataFrame whose single column is named `filename`, so `ebird_counts["filename"]`
  (L48) is the **group count**, not a filename column.
- `df.secondary_labels.apply(eval)` (L21, L40) — again `eval` on CSV text.

---

## `audio_dataset_v3.py`

**Status: LEGACY + BROKEN** — v2 minus the noise injection, with a
division-based mixing scheme. Imported by `engine/main_engine_v3.py:157,161,165`.

**Public API** — identical to `AudioDatasetV2`, class name `AudioDatasetV3`.

**Broken at** — `NameError: name 'get_snippets' is not defined` at **L142**
(also L147, L157, L186).

**Gotchas**

- Same shared-list aliasing in `mix_aug` as v2 (L80, L94).
- v3 has **no** `self.selection_index`, so v2's index bug does not apply; it
  uses a local `selection_index = range(0, snip_duration, division_length)`
  (L165) instead.
- Mixing probability is 0.75 (L162) vs v2's 0.5.
- `indices` are `range` objects rather than lists (L154, 156, 183, 185) —
  `get_snippets` was presumably written to accept a range, since
  `np.array(y_snippet).mean(axis=0)` (L159) assumes a 2-D array of snippets.
- Unlike v1, v2 and v3 do **not** call `mono_to_color`.

---

## `audio_dataset_v4.py`

**Status: LEGACY** — the "final" pre-SED dataset. Importable and free of
undefined names; the only legacy dataset that does not reference
`get_snippets`. Imported by `engine/main_engine_v4.py:210,214,218`.

**Public API**

```python
class AudioDatasetV4(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code, inv_ebird_label,
                 num_test_samples=10, bckgrd_aug_dir=None,
                 background_audio_dir=None, file_type="mp3", isTraining=True,
                 transform=None, apply_mixer=False)
    def get_label(self, dataset, idx)
    def init_workers_fn(self, worker_id)
    def collate_fn(self, samples)
    def get_audio(self, idx)
    def convert_labels_to_coded(self, num_images, labels)
    def add_augmentation(self, y, primary_codes, all_labels)
    def __getitem__(self, idx) -> {"filenames", "images", "coded_labels", "primary_codes"}
```

**Behaviour**

- Uses `conf.samples` — not `conf.duration` — as the window length (L160-161).
- Imports the audiomentations stack at module level (L12) and builds
  `self.augmenter` **only when `apply_mixer=True`** (L50-68).
- The shared mix-augmentation code is commented out (L134-142).
- `primary_codes` is `None` in eval mode (L188) and a list of class indices in
  train mode (L152, L169).
- `collate_fn` (L78-86) stacks `images`/`coded_labels` and passes
  `primary_codes` through untouched.

**Gotchas**

- ⚠️ **Latent `AttributeError`.** `add_augmentation` calls
  `self.augmenter(...)` **unconditionally** (L144), but the attribute is only
  created when `apply_mixer=True`. Training with `apply_mixer=False` → 
  `AttributeError: 'AudioDatasetV4' object has no attribute 'augmenter'`.
- ⚠️ **Latent `KeyError`.** Background rows are appended as `{"filename": bk}`
  with **no `ebird_code` key** (L42-44), so `get_label` (L72) raises
  `KeyError: 'ebird_code'` for them when the imbalanced sampler is built.
- `convert_labels_to_coded` sets **all** rows of a class to 1 (L130) — a
  secondary label applies to every snippet.
- `df.secondary_labels.apply(eval)` (L36) again.

---

## See also

- [modules/01 — Data pipeline](../modules/01-data-pipeline.md) — the topic chapter
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/feature-parity-matrix.md](../ports/feature-parity-matrix.md) §C — the porting checklist for this package
