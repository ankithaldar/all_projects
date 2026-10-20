# `src/` file reference

A per-file reference for every file under `src/` — 84 Python files plus
`startup.sh`, 9,105 lines total. Each entry states what the file does, its
public API, its exact behaviour, and whether it is **live**, **legacy**, **dead**
or **broken**.

This complements the topic chapters in [`../modules/`](../modules/), which
explain *how the pieces fit together*. These pages are the *file-level index*:
they answer "what is this file, and can I trust it?".

| Page | Files | Contents |
|---|---|---|
| [dataloaders.md](dataloaders.md) | 6 | `SedDataset`, the imbalanced sampler, and the four legacy `AudioDataset` generations |
| [models.md](models.md) | 7 | the five PANNs SED models, the legacy classifier zoo |
| [engine.md](engine.md) | 6 | `BaseEngine`, `SedEngine`, and the four legacy engines |
| [loss.md](loss.md) | 16 | every criterion, its `loss_keys`, and which are inert |
| [metrics.md](metrics.md) | 7 | LRAP, the two F1 metrics, and the inference dump |
| [augmentations.md](augmentations.md) | 7 | the two live waveform stacks and the legacy image pipeline |
| [helpers.md](helpers.md) | 2 | the vendored PANNs front-end and the legacy librosa helpers |
| [logger.md](logger.md) | 6 | Neptune, print, Slack, and the shared base |
| [config-params.md](config-params.md) | 22 | the 21 experiment configs and the shared `configs.py` |
| [entrypoints.md](entrypoints.md) | 6 | `sed_train.py`, the four legacy trainers, `startup.sh` |

---

## Status legend

| Status | Meaning |
|---|---|
| **LIVE** | On the shipped path: reachable from `sed_train.py` → `SedEngine` with a shipped config. |
| **LEGACY** | Importable and was once the shipped path, but superseded. Nothing shipped reaches it. |
| **DEAD** | Imported by nothing in the repository. Not broken, just unreachable. |
| **BROKEN** | Fails at import time or at first use. The exact line and error are given. |

---

## Coverage matrix — all 85 files

**LIVE: 48** · **LEGACY: 33** · **DEAD: 4** · **BROKEN: 13** — the tags overlap:
10 files are both LEGACY and BROKEN, one is both DEAD and BROKEN, and
`sed_test.py` is LIVE-but-BROKEN, so the tags sum to 98 over 85 files.
**37 of the 85 files are not on the live path.**

### `src/dataloaders/` — 6 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_dataset.py` | 133 | **LIVE** | [modules/01](../modules/01-data-pipeline.md) |
| `imbalanced_dataset_sampler.py` | 25 | **LIVE** | [modules/01](../modules/01-data-pipeline.md) |
| `audio_dataset.py` | 129 | LEGACY + **BROKEN** (L102, L112) | [modules/11](../modules/11-legacy-research-paths.md) |
| `audio_dataset_v2.py` | 225 | LEGACY + **BROKEN** (L145, L150, L160, L191) | [modules/11](../modules/11-legacy-research-paths.md) |
| `audio_dataset_v3.py` | 219 | LEGACY + **BROKEN** (L142, L147, L157, L186) | [modules/11](../modules/11-legacy-research-paths.md) |
| `audio_dataset_v4.py` | 198 | LEGACY + **BROKEN** (L144) | [modules/11](../modules/11-legacy-research-paths.md) |

### `src/models/` — 7 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_models.py` | 809 | **LIVE** | [modules/04](../modules/04-model-architecture.md) |
| `aux_densenet121.py` | 161 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `aux_densnet161.py` | 156 | DEAD + **BROKEN** (L17, L149) | [modules/11](../modules/11-legacy-research-paths.md) |
| `classifier_dcase.py` | 173 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `classifier_m_series.py` | 110 | **DEAD** | [modules/11](../modules/11-legacy-research-paths.md) |
| `densenet121.py` | 18 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `densenet161.py` | 18 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |

### `src/engine/` — 6 files
| file | LOC | status | documented in |
|---|---|---|---|
| `base/base_engine.py` | 375 | **LIVE** | [modules/05](../modules/05-training-engine.md), [modules/06](../modules/06-checkpointing-and-resuming.md) |
| `sed_engine.py` | 226 | **LIVE** | [modules/05](../modules/05-training-engine.md) |
| `main_engine.py` | 118 | LEGACY + **BROKEN** (L32) | [modules/11](../modules/11-legacy-research-paths.md) |
| `main_engine_v2.py` | 168 | LEGACY + **BROKEN** (L35, L153) | [modules/11](../modules/11-legacy-research-paths.md) |
| `main_engine_v3.py` | 168 | LEGACY + **BROKEN** (L153) | [modules/11](../modules/11-legacy-research-paths.md) |
| `main_engine_v4.py` | 221 | LEGACY + **BROKEN** (L43, L48) | [modules/11](../modules/11-legacy-research-paths.md) |

### `src/loss/` — 16 files
| file | LOC | status | `loss_keys` |
|---|---|---|---|
| `sed_scaled_pos_neg_focal_loss.py` | 48 | **LIVE** | `bce_loss, F_loss, FScaled_loss, F_loss_0, F_loss_1` |
| `sed_scaled_pos_neg_focal_loss_augd.py` | 55 | **LIVE** (2 configs) | same 5 |
| `bce_controlled_standard_loss.py` | 37 | **LIVE** | *(empty)* |
| `focal_loss_standard.py` | 34 | **LIVE** (no config) | `bce_loss, F_loss` |
| `sed_removed_focal_loss.py` | 41 | **LIVE** (no config) | `bce_loss, F_loss, FRemoved_loss` |
| `bce_loss.py` | 16 | LEGACY | *(none — AttributeError risk)* |
| `smooth_bce_loss.py` | 25 | LEGACY | *(none)* |
| `bce_controlled_loss.py` | 49 | LEGACY | `primary_loss, bce_loss` |
| `focal_controlled_standard_loss.py` | 45 | LEGACY | `bce_loss, F_loss` |
| `focal_mse_loss.py` | 43 | LEGACY | `mse_loss, F_loss` |
| `f1_controlled_loss.py` | 37 | LEGACY | `precision, recall, f1` |
| `noisy_loss.py` | 39 | LEGACY + **BROKEN** | declares 4, returns 2 |
| `noisy_controlled_loss.py` | 47 | LEGACY | 4 (all returned) |
| `custom_losses.py` | 79 | LEGACY | *(none)* |
| `custom_losses_for_targets.py` | 83 | LEGACY + **BROKEN** | *(none; `"lq"` branch is `pass`)* |
| `sed_scaled_focal_loss.py` | 42 | **DEAD** | `bce_loss, F_loss, FScaled_loss` |

### `src/metrics/` — 7 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_lraps.py` | 33 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `sed_f1score_clip.py` | 43 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `sed_f1score_frame.py` | 45 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `sed_test.py` | 82 | **LIVE** (conditional) + **BROKEN** (L59) | [modules/08](../modules/08-inference-and-submission.md) |
| `custom_f1score.py` | 42 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `custom_lraps.py` | 33 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `custom_test.py` | 79 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |

### `src/augmentations/` — 7 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_default_augment.py` | 32 | **LIVE** | [modules/03](../modules/03-augmentations.md) |
| `sed_background_augment.py` | 41 | **LIVE** | [modules/03](../modules/03-augmentations.md) |
| `mixup.py` | 59 | **LIVE** | [modules/03](../modules/03-augmentations.md) |
| `base_augment.py` | 19 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `default_augment.py` | 54 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `spec_augment.py` | 74 | LEGACY + **BROKEN** (L8) | [modules/11](../modules/11-legacy-research-paths.md) |
| `noise_maker.py` | 119 | **DEAD** | [modules/03](../modules/03-augmentations.md) |

### `src/helpers/` — 2 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_audio_utils.py` | 271 | **LIVE** | [modules/02](../modules/02-audio-frontend.md) |
| `audio_utils.py` | 55 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |

### `src/logger/` — 6 files
| file | LOC | status | documented in |
|---|---|---|---|
| `base/base_logger.py` | 35 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `base/utils.py` | 12 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `neptune/neptune_logger.py` | 61 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `neptune/neptune_utils.py` | 48 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `print/print_logger.py` | 51 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |
| `print/print_utils.py` | 50 | **LIVE** | [modules/10](../modules/10-metrics-and-tracking.md) |

### `src/config_params/` — 22 files
| file | LOC | status | documented in |
|---|---|---|---|
| `configs.py` | 351 | **LIVE** | [modules/09](../modules/09-config-and-cross-validation.md) |
| `example_config.py` | 138 | **LIVE** | [modules/09](../modules/09-config-and-cross-validation.md) |
| `final_sed/*.py` (15) | ~137 each | **LIVE** | [modules/09](../modules/09-config-and-cross-validation.md) |
| `final_sed_5_fold/*.py` (5) | 138 each | **LIVE** | [modules/09](../modules/09-config-and-cross-validation.md) |

### Entry points — 6 files
| file | LOC | status | documented in |
|---|---|---|---|
| `sed_train.py` | 25 | **LIVE** | [modules/05](../modules/05-training-engine.md) |
| `train.py` | 25 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `train_v2.py` | 25 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `train_v3.py` | 25 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `train_v4.py` | 25 | LEGACY | [modules/11](../modules/11-legacy-research-paths.md) |
| `../startup.sh` | 8 | **LIVE** | [modules/12](../modules/12-kaggle-constraints.md) |

---

## The seventeen broken *failures*, in thirteen files

Every one of these fails at import time or on first use. **None of them is on
the shipped path** — which is exactly why nobody noticed. Seventeen failure
modes across thirteen distinct files:

| # | file | fails at | error |
|---|---|---|---|
| 1 | `metrics/sed_test.py` | import | `SyntaxError` — missing comma after `"filename": filename` (L59) |
| 2 | `augmentations/spec_augment.py` | import | `ImportError: cannot import name 'mono_to_color' from 'helpers.audio_utils'` (L8) |
| 3 | `models/aux_densnet161.py` | first forward | `NameError: name 'F' is not defined` (L17, L149) — missing `import torch.nn.functional as F` |
| 4 | `dataloaders/audio_dataset.py` | first `__getitem__` | `NameError: name 'get_snippets' is not defined` (L102) |
| 5 | `dataloaders/audio_dataset.py` | ditto | `NameError: name 'mono_to_color' is not defined` (L112) |
| 6 | `dataloaders/audio_dataset_v2.py` | first `__getitem__` | `NameError: name 'get_snippets' is not defined` (L145, 150, 160, 191) |
| 7 | `dataloaders/audio_dataset_v3.py` | first `__getitem__` | `NameError: name 'get_snippets' is not defined` (L142, 147, 157, 186) |
| 8 | `engine/main_engine.py` | first training step | `AttributeError: 'MainEngine' object has no attribute 'ls_fn'` (L32) |
| 9 | `engine/main_engine_v2.py` | first training step | `AttributeError: 'MainEngineV2' object has no attribute 'ls_fn'` (L35) |
| 10 | `engine/main_engine_v2.py` | import of the aug | `ImportError` (L153) → the `spec_augment` failure above |
| 11 | `engine/main_engine_v3.py` | import of the aug | `ImportError` (L153) → the `spec_augment` failure above |
| 12 | `engine/main_engine_v4.py` | first training step | `AttributeError: 'Parameters' object has no attribute 'loss_factors'` (L43) |
| 13 | `engine/main_engine_v4.py` | first training step | `TypeError` — `BCEControlledStandardLoss.forward` takes 2 args, the engine calls with 4 (L48) |
| 14 | `engine/main_engine_v4.py` | `_init_augmentation` | `AttributeError` — `aug_name="secondary_default"` has no branch (L205) |
| 15 | `loss/noisy_loss.py` | first metric update | `KeyError: 'primary_loss_0'` — `loss_keys` declares 4, `forward` returns 2 |
| 16 | `loss/custom_losses_for_targets.py` | first forward | `AttributeError: 'NoisyCuratedLossTargetted' object has no attribute 'noisy_loss'` — the `"lq"` branch is `pass` (L46) |
| 17 | `dataloaders/audio_dataset_v4.py` | first `__getitem__` | `AttributeError: 'AudioDatasetV4' object has no attribute 'augmenter'` — created only when `apply_mixer=True` (L144) |

**Two of them are latent rather than fatal**, and are more interesting because
they fire on a path that *does* run:

| file | defect |
|---|---|
| `helpers/audio_utils.py:47` | `normalise` captures `min_val`/`max_val` in the loop scope and never resets them, so every array after the first is normalised with the **first** array's range — silently producing wrapped `uint8` values. |
| `helpers/audio_utils.py:10` | `pathname.suffix==".wav"` is missing the leading dot, so it is **always False** and every file — including `.wav` — goes through `librosa.load` (resampled to 32 kHz, mono). The `sf.read` branch is dead code. |

---

## Line endings — 79 of 84 files are CRLF

The repository is overwhelmingly **CRLF**, not just the configs. `docs/README.md:61`
records the config subset (20 of the 22 files in `src/config_params/`); the
true figure across the whole tree is larger:

| | count |
|---|---|
| CRLF | **79** of 84 `.py` files |
| LF | **5** |

The five LF files — the only ones a line-anchored `grep` will see without
special handling:

```
augmentations/default_augment.py
config_params/example_config.py
config_params/final_sed/final_sed_dense121_mix_2.py
engine/base/base_engine.py
loss/sed_scaled_pos_neg_focal_loss.py
```

**Why it matters:** a `grep -n "criterion_name" src/` over `src/loss/` returns
**one** hit (`sed_scaled_pos_neg_focal_loss.py`, the one LF file in that
directory) and silently misses the other fifteen. Same for `src/engine/`,
where `base_engine.py` is LF but `sed_engine.py` is not. Any audit, codemod or
sed-style edit over this tree must handle CRLF explicitly.

---

## The two missing symbols

`get_snippets` and `mono_to_color` and `get_label`… no — two symbols are
called but defined nowhere in the repository:

| symbol | called at | defined at |
|---|---|---|
| `get_snippets` | `dataloaders/audio_dataset.py:102`, `audio_dataset_v2.py:145,150,160,191`, `audio_dataset_v3.py:142,147,157,186` | **nowhere** |
| `mono_to_color` | `augmentations/spec_augment.py:8,49,63`, `dataloaders/audio_dataset.py:112` | **nowhere** |

`helpers/audio_utils.py` today defines exactly: `read_audio`,
`audio_to_melspectrogram`, `show_melspectrogram`, `count_parameters`,
`normalise`. It was rewritten for the SED line and the legacy callers were
never updated.

---

## Where the documentation still has gaps

| what | status |
|---|---|
| 72 of 84 `src` files are referenced by ≥1 topic chapter | ✅ |
| the 12 unreferenced files | all individual `config_params/final_sed*` configs — covered by their directory's chapter ([modules/09](../modules/09-config-and-cross-validation.md)), not cited individually |
| 4 legacy files with no chapter reference | `models/aux_densnet161.py`, `models/classifier_m_series.py`, `train_v2.py`, `train_v3.py` — now covered by [models.md](models.md) and [entrypoints.md](entrypoints.md) |
| broken links in `docs/` | **0** — all 32 were repaired. They were a `modules/` prefix doubling up inside `docs/modules/`, plus four links from `docs/README.md` that pointed at files living in `docs/modules/`. |
