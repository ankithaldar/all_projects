# `src/config_params/` — 22 files

One shared module of constants and **21 experiment configs**. Every config is
a plain Python class named `Parameters`, instantiated with no arguments, and
resolved by `importlib.import_module` from a dotted `--config` path.

| file | LOC | status |
|---|---|---|
| [`configs.py`](#configspy) | 351 | **LIVE** — shared constants |
| [`example_config.py`](#example_configpy) | 138 | **LIVE** — the documented example |
| [`final_sed/` (15 files)](#final_sed) | ~137 each | **LIVE** — the `final_sed` family |
| [`final_sed_5_fold/` (5 files)](#final_sed_5_fold) | 138 each | **LIVE** — the 5-fold family |

**21 attributes vary across the 21 configs; the other 18 are identical.**

---

## `configs.py`

**Status: LIVE.** The shared constant module, imported by every config.

**Public API**

```python
class AUDIO_CONFIG:
    sampling_rate = 32000; duration = 1; hop_length = 251
    fmin = 500; fmax = 15000; n_mels = 128; n_fft = n_mels * 20
    samples = sampling_rate * duration; padmode = 'reflect'

def fill_range(letter_start, letter_end, path_fill, dict_val={}) -> dict
def get_dict_value(input_dir) -> dict

BIRD_CODE: dict[str, int]      # 264 entries, species code -> class index
EBIRD_LABEL: dict[str, str]    # species code -> "Genus species_Common Name"
INV_EBIRD_LABEL: dict[str, str]  # the exact inverse of EBIRD_LABEL
```

**Behaviour**

- `BIRD_CODE` (L30+) maps 264 ebird codes to class indices **0..263**. The
  index order is the label order everywhere in the codebase.
- `fill_range` (L15-19) fills a dict for an alphabet range; `get_dict_value`
  (L21-28) maps the first letter of an ebird code to one of five archive
  directories (`birdsong-resampled-train-audio-00` … `-04`).

**Gotchas**

- ⚠️ `AUDIO_CONFIG` is **defined but never instantiated** anywhere — it is
  legacy. The live `SedDataset` takes `period`, not `duration`.
- ⚠️ `get_dict_value` is **imported by every config and called by none**. It
  backs the `root_dir`-as-dict branch in `dataloaders/sed_dataset.py:66-68`,
  which is therefore dead. The 5-archive sharding was never used.
- ⚠️ `fill_range` has a **mutable default** `dict_val={}` — safe here only
  because every call passes a fresh dict.
- `INV_EBIRD_LABEL` is the exact inverse of `EBIRD_LABEL`, keyed on
  `"Genus species_Common Name"`.

---

## `example_config.py`

**Status: LIVE** — the only config documented in the README
(`README.md:113`: `python sed_train.py --config config_params.example_config`).

**Its 39 attributes**, grouped:

| group | attributes and defaults |
|---|---|
| identity | `fold = 0`, `name = "example_config"` |
| model | `model_name = "sed_dense121att"`, `model_config` = `{sample_rate: 32000, window_size: 1024, hop_size: 320, mel_bins: 64, fmin: 50, fmax: 14000, classes_num: 264, apply_aug: True, top_db: None}`, `pretrained_path = None` |
| augmentation | `aug_name = "secondary_default"`, `apply_mixup = False`, `bckgrd_aug_dir = "pinknoise"`, `secondary_bckgrd_aug_dir = "pinknoise"` |
| loss | `criterion_name = "sed_scaled_pos_neg_focal_loss"`, `criterion_params` = `{gamma: 0.0, alpha_0: 1.0, alpha_1: 1.0, secondary_factor: 1.0}` |
| optimiser | `optimizer_name = "adamw"`, `weight_decay = 0.01`, `lr = 0.001` |
| scheduler | `scheduler_name = "warmup_with_cosine"`, `lr_scale_factor = 0.01` |
| data | `PERIOD = 30`, `train_ds_params`, `valid_ds_params`, `test_ds_params`, `train_bs = 28`, `train_num_workers = 8`, `valid_bs = 32`, `valid_num_workers = 4` |
| metrics | `metrics = ["lraps", "f1score_clip", "f1score_frame"]`, `track_metric = "f1score_clip"`, `metric_factor = 1` |
| logging | `logger_name = "neptune"`, `logger_params` = `{project_name: "NEPTUNE_PROJECT_NAME", log_every: 10, name, prefix_name, tags: [fold, name, model_name, criterion_name], params: {...}}` |
| checkpointing | `checkpoint_params` = `{save_dir: "saved_models/{name}", n_saved: 10, prefix_name: name}`, `checkpoint_dir = None`, `load_model_only = True` |
| runtime | `run_params` = `{max_epochs: 50, epoch_length: None}`, `dist_params = {}`, `val_length = None`, `eval_every = 2`, `add_pbar = True`, `accumulation_steps = 1`, `gradient_clip_val = 0` |

The `train_ds_params` / `valid_ds_params` / `test_ds_params` dictionaries each
carry `root_dir`, `csv_dir`, `period`, `bird_code`, `inv_ebird_label`,
`isTraining`, `num_test_samples` and (for valid/test)
`background_audio_dir`. The example uses **relative** paths
(`Path("data/")`, `Path("fold_0_train.csv")`); the `final_sed` family uses
**absolute** ones (`/data/`, `/folds/fold_{n}_train.csv`).

**Gotchas**

- ⚠️ `logger_params["project_name"]` is the literal placeholder
  `"NEPTUNE_PROJECT_NAME"`.
- ⚠️ `n_saved = 10` here; the `final_sed` family uses 5.
- ⚠️ `logger_params["params"]` is a hand-built flat dict that re-lists
  `**model_config` and `**criterion_params` — it must be updated by hand when
  either changes, and nothing validates it.

---

## The attribute that varies

Measured by instantiating all 21 configs and comparing attribute by attribute.
**21 attributes vary; 18 do not.**

| attribute | values across the 21 configs |
|---|---|
| `fold` | `0`…`3` (4-fold), `0`…`4` (5-fold), **`'all'`** (the two all-data retrains) |
| `name` | the file stem |
| `aug_name` | `default` (final_sed family) · `secondary_default` (5-fold family, example) |
| `apply_mixup` | `True` (the `mix` family) · `False` (the `nomix` family, 5-fold, example) |
| `bckgrd_aug_dir` | `"/pinknoise"` (final_sed) · `"pinknoise"` (example) |
| `secondary_bckgrd_aug_dir` | `"/pinknoise"` (5-fold) · `"pinknoise"` (example) · **`None`** (final_sed) |
| `criterion_name` | `sed_scaled_pos_neg_focal_loss` (all but two) · `sed_scaled_pos_neg_focal_loss_augd` (`nomix_fold2_augd`, `nomix_fold3_augd`) |
| `criterion_params` | adds `primary_drop_out_factor: 0.3` in the `_augd` configs |
| `lr` | `0.001` · **`0.0005`** in the four `_2` stage-2 fine-tunes |
| `weight_decay` | `0.01` · **`0.0001`** in `nomix_fold0` · **`0.001`** in the two `_augd` configs |
| `train_bs` | `32` (mix family) · `24` (nomix family) · `28` (5-fold family, example) |
| `eval_every` | `2` · **`10`** in the three all-data retrains |
| `load_model_only` | `True` · **`False`** in the three all-data retrains |
| `checkpoint_dir` | `None` · a path in the four `_2` fine-tunes and the three all-data retrains |
| `track_metric` | `f1score_clip` · **`f1score_frame`** (5-fold folds 1-4) · **`lraps`** (`mix_2`) |
| `checkpoint_params` | `save_dir`, `n_saved` (5 or 10), `prefix_name` |
| `logger_params` | `project_name`, `log_every`, `name`, `prefix_name`, `tags`, `params` |
| `train_ds_params` / `valid_ds_params` / `test_ds_params` | the fold CSVs and the `num_test_samples` |
| `model_config` | identical in all 21 except `top_db` |

**⚠️ Two findings worth naming**

1. **`track_metric` is inconsistent across the ensemble.** The 4-fold
   `final_sed` configs track `f1score_clip`; the 5-fold configs track
   `f1score_frame` — **except fold 0, which tracks `f1score_clip`** — and
   `mix_2` tracks `lraps`. So "the best checkpoint" means a different thing
   in different members of the same 13-model ensemble.
2. **`weight_decay` spans 100×** across the ensemble: `0.0001`
   (`nomix_fold0`), `0.001` (the `_augd` pair), `0.01` (everything else).
   `docs/modules/09` flags this as the 100× weight-decay spread.

---

## `final_sed/` — 15 files

**Status: LIVE.** The 4-fold family.

### Every file, explicitly

| # | file | fold | family | `apply_mixup` | `criterion_name` | `lr` | `weight_decay` | `train_bs` | `eval_every` | `load_model_only` | `checkpoint_dir` | `track_metric` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `final_sed_dense121_mix_fold0.py` | 0 | 4-fold mixup | `True` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 32 | 2 | `True` | `None` | `f1score_clip` |
| 2 | `final_sed_dense121_mix_fold1.py` | 1 | 4-fold mixup | `True` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 32 | 2 | `True` | `None` | `f1score_clip` |
| 3 | `final_sed_dense121_mix_fold2.py` | 2 | 4-fold mixup | `True` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 32 | 2 | `True` | `None` | `f1score_clip` |
| 4 | `final_sed_dense121_mix_fold3.py` | 3 | 4-fold mixup | `True` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 32 | 2 | `True` | `None` | `f1score_clip` |
| 5 | `final_sed_dense121_mix_fold0_2.py` | 0 | stage-2 fine-tune of 1 | `True` | `sed_scaled_pos_neg_focal_loss` | **0.0005** | 0.01 | 32 | 2 | `True` | **set** | `f1score_clip` |
| 6 | `final_sed_dense121_mix_fold1_2.py` | 1 | stage-2 fine-tune of 2 | `True` | `sed_scaled_pos_neg_focal_loss` | **0.0005** | 0.01 | 32 | 2 | `True` | **set** | `f1score_clip` |
| 7 | `final_sed_dense121_mix_fold2_2.py` | 2 | stage-2 fine-tune of 3 | `True` | `sed_scaled_pos_neg_focal_loss` | **0.0005** | 0.01 | 32 | 2 | `True` | **set** | `f1score_clip` |
| 8 | `final_sed_dense121_mix_fold3_2.py` | 3 | stage-2 fine-tune of 4 | `True` | `sed_scaled_pos_neg_focal_loss` | **0.0005** | 0.01 | 32 | 2 | `True` | **set** | `f1score_clip` |
| 9 | `final_sed_dense121_nomix_fold0.py` | 0 | 4-fold no-mixup | `False` | `sed_scaled_pos_neg_focal_loss` | 0.001 | **0.0001** | 24 | 2 | `True` | `None` | `f1score_clip` |
| 10 | `final_sed_dense121_nomix_fold1.py` | 1 | 4-fold no-mixup | `False` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 24 | 2 | `True` | `None` | `f1score_clip` |
| 11 | `final_sed_dense121_nomix_fold2_augd.py` | 2 | 4-fold no-mixup + drop-out | `False` | **`..._augd`** | 0.001 | **0.001** | 24 | 2 | `True` | `None` | `f1score_clip` |
| 12 | `final_sed_dense121_nomix_fold3_augd.py` | 3 | 4-fold no-mixup + drop-out | `False` | **`..._augd`** | 0.001 | **0.001** | 24 | 2 | `True` | `None` | `f1score_clip` |
| 13 | `final_sed_dense121_nomix_1.py` | **`'all'`** | all-data retrain | `False` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 24 | **10** | **`False`** | **set** | **`lraps`** |
| 14 | `final_sed_dense121_nomix_3.py` | **`'all'`** | all-data retrain | `False` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 24 | **10** | **`False`** | **set** | `f1score_clip` |
| 15 | `final_sed_dense121_mix_2.py` | 0 | all-data **mixup** retrain | `True` | `sed_scaled_pos_neg_focal_loss` | 0.001 | 0.01 | 24 | **10** | **`False`** | **set** | **`lraps`** |

Every one of the 15 also sets `aug_name = "default"`,
`bckgrd_aug_dir = "/pinknoise"`, `secondary_bckgrd_aug_dir = None`,
`criterion_params = {gamma: 0.0, alpha_0: 1.0, alpha_1: 1.0,
secondary_factor: 1.0}` (plus `primary_drop_out_factor: 0.3` in the two `_augd`
files), `optimizer_name = "adamw"`, `scheduler_name = "warmup_with_cosine"`,
`lr_scale_factor = 0.01`, `metrics = ["lraps", "f1score_clip",
"f1score_frame"]`, `metric_factor = 1`, `logger_name = "neptune"`,
`n_saved = 5`, `run_params = {max_epochs: 50, epoch_length: None}`,
`dist_params = {}`, `val_length = None`, `add_pbar = True`,
`accumulation_steps = 1`, `gradient_clip_val = 0`, `PERIOD = 30`,
`model_name = "sed_dense121att"`, and the shared `model_config`.

**Gotchas**

- ⚠️ **20 of the 22 files in `src/config_params/` are CRLF** — `configs.py`
  plus 19 of the 20 `final_sed*` files. `example_config.py` and
  `final_sed/final_sed_dense121_mix_2.py` are LF. Line-anchored `grep`
  silently misses the CRLF files (docs/README.md:61).
- ⚠️ The `_2` files are **stage-2 fine-tunes**: they halve the LR and set
  `checkpoint_dir` to resume from the stage-1 checkpoint. They are *not*
  independent folds.
- ⚠️ The three all-data retrains set `fold='all'` and `load_model_only=False`
  — a **full** resume, including the optimiser state.
- ⚠️ `final_sed_dense121_nomix_fold0.py` carries `weight_decay=0.0001` while
  its three siblings carry `0.01` — a **100× spread** inside one family, and
  `0.001` in the `_augd` pair.

---

## `final_sed_5_fold/` — 5 files

**Status: LIVE.** The 5-fold no-mixup family — five members of the 13-model
ensemble.

### Every file, explicitly

| # | file | fold | `track_metric` | `train_bs` | `eval_every` | `aug_name` | `apply_mixup` |
|---|---|---|---|---|---|---|---|
| 1 | `final_5fold_sed_dense121_nomix_fold0.py` | 0 | **`f1score_clip`** | 28 | 2 | `secondary_default` | `False` |
| 2 | `final_5fold_sed_dense121_nomix_fold1.py` | 1 | `f1score_frame` | 28 | 2 | `secondary_default` | `False` |
| 3 | `final_5fold_sed_dense121_nomix_fold2.py` | 2 | `f1score_frame` | 28 | 2 | `secondary_default` | `False` |
| 4 | `final_5fold_sed_dense121_nomix_fold3.py` | 3 | `f1score_frame` | 28 | 2 | `secondary_default` | `False` |
| 5 | `final_5fold_sed_dense121_nomix_fold4.py` | 4 | `f1score_frame` | 28 | 2 | `secondary_default` | `False` |

**Shared settings:** `aug_name="secondary_default"`, `apply_mixup=False`,
`bckgrd_aug_dir=secondary_bckgrd_aug_dir="/pinknoise"`,
`criterion_name="sed_scaled_pos_neg_focal_loss"`, `lr=0.001`,
`weight_decay=0.01`, `train_bs=28`, `eval_every=2`, `load_model_only=True`,
`checkpoint_dir=None`, `n_saved=5`, `metrics=["lraps","f1score_clip",
"f1score_frame"]`, `metric_factor=1`, `logger_name="neptune"`,
`run_params={max_epochs: 50, epoch_length: None}`, `dist_params={}`,
`val_length=None`, `add_pbar=True`, `accumulation_steps=1`,
`gradient_clip_val=0`, `PERIOD=30`, `model_name="sed_dense121att"`.

**⚠️ The `track_metric` inconsistency** — fold 0 tracks `f1score_clip` while
folds 1-4 track `f1score_frame`. Within one ensemble, the five members rank
their checkpoints by different metrics. This is the "track_metric
Inconsistency in 5-Fold Family" the knowledge graph flags.

**Gotcha** — ⚠️ `aug_name="secondary_default"` has **no branch** in
`engine/main_engine_v4.py:205`, so these configs cannot drive the v4 engine
at all; they are for `SedEngine`, which does have the branch.

---

## How a config is loaded

```
sed_train.py --config config_params.final_sed.final_sed_dense121_mix_fold0
   │
   ├─ importlib.import_module("config_params.final_sed.final_sed_dense121_mix_fold0")
   ├─ module.Parameters()               # no arguments
   └─ SedEngine(local_rank, hyperparams)
```

`Parameters.__init__(self, hparams=None)` takes an optional argument that
every config **ignores** — the object is constructed from its own literals and
the fold number.

**Gotcha** — ⚠️ the dotted path is a **Python module path**, not a file path,
so it only resolves from `src/` on `sys.path`. There is no `__init__.py`
anywhere; the tree works through implicit namespace packages.

---

## See also

- [modules/09 — Config and cross-validation](../modules/09-config-and-cross-validation.md)
- [modules/02 — Audio front-end](../modules/02-audio-frontend.md) — the `AUDIO_CONFIG` values
- [ports/feature-parity-matrix.md](../ports/feature-parity-matrix.md) §I — the porting checklist for this package
