# 13 — Experiment Log (what the repo actually records)

> [!warning] What exists and what does not
> The repo contains **no training curves, no Neptune exports, no dated experiment
> notes**. Git history is three commits (2020-09-16 → 2020-09-17), all *after* the
> competition. The only experiment evidence is: (a) the two leaderboard tables in
> `README.md`, (b) validation scores baked into checkpoint filenames inside configs,
> (c) the diffs between config files. Everything below comes from those three
> sources. Nothing is reconstructed or estimated.

## 1. Final ensemble selection — README tables (verbatim)

**13 models** (4-fold nomix + 5-fold nomix + 4-fold mix):

| Votes | Clip thr | Frame thr | Public LB | Private LB | Selected |
|---|---|---|---|---|---|
| 4 | 0.3 | 0.3 | 0.616 | **0.681** | ✅ |
| 4 | 0.3 | 0.5 | 0.615 | 0.679 | |
| 5 | 0.3 | 0.3 | 0.609 | 0.679 | |
| 5 | 0.3 | 0.5 | 0.606 | 0.676 | |
| 3 | 0.3 | 0.3 | **0.617** | 0.679 | |
| 3 | 0.3 | 0.5 | 0.614 | 0.679 | |

**9 models** (the no-mixup models only):

| Votes | Clip thr | Frame thr | Public LB | Private LB | Selected |
|---|---|---|---|---|---|
| 3 | 0.3 | 0.5 | 0.613 | 0.676 | ✅ |
| 2 | 0.3 | 0.5 | 0.614 | 0.669 | |
| 4 | 0.3 | 0.5 | 0.610 | 0.675 | |

Facts the author states: the selected 13-model ensemble was **not** the best public
score (3 votes was); it was chosen because "3 votes felt too low for 13 models";
result **7th public → 1st private**. The 9-model selection would have placed 3rd.
Frame threshold 0.3 was never tried on the 9-model ensemble (out of daily
submissions).

### Reading the tables *[derived from the numbers above]*

- Public-LB spread across all 9 rows is 0.606–0.617 (0.011); private spread is 0.669–0.681 (0.012). Public rank order does **not** predict private order: the public-best row (3 votes, 0.617) is private 0.679, below the selected row.
- On 13 models, frame 0.3 ≥ frame 0.5 on private for every vote count (0.681 vs 0.679, 0.679 vs 0.676, 0.679 vs 0.679).
- On 13 models, private peaks at 4 votes; on 9 models, at 3 votes — roughly the same *fraction* (≈31 % and 33 %) of the ensemble agreeing.
- Adding the 4 mixup models (9 → 13) at comparable settings (3 votes, 0.3/0.5) moves private 0.676 → 0.679.

## 2. Per-model validation scores recorded in configs

Only checkpoints that a later config *loads* have their score in a filename:

| Checkpoint (loaded by) | Epoch | Score | Metric (that run's `track_metric`) | Validation data |
|---|---|---|---|---|
| `final_sed_dense121_mix_fold0` (by `mix_fold0_2`) | 50 | 0.6328 | `f1score_clip` | fold 0 held-out |
| `final_sed_dense121_mix_fold1` (by `mix_fold1_2`) | 50 | 0.6134 | `f1score_clip` | fold 1 held-out |
| `final_sed_dense121_mix_fold2` (by `mix_fold2_2`) | 50 | 0.6305 | `f1score_clip` | fold 2 held-out |
| `final_sed_dense121_mix_fold3` (by `mix_fold3_2`) | 50 | 0.6370 | `f1score_clip` | fold 3 held-out |
| `final_sed_dense121_nomix_1` (by itself) | 30 | 0.9117 | `lraps` | `/folds/all_data.csv` = **training data** → not a generalisation estimate |

These are **file-level** in-repo metrics (see [10](10-metrics-and-tracking.md)), not
the competition's 5-second-row F1, so they are not comparable to the LB numbers.

## 3. Experiment structure visible in the config diffs

| Experiment axis | Evidence | Chapter |
|---|---|---|
| Two CV schemes (4-fold `/folds`, 5-fold `/folds5`) | `csv_dir` paths | [09](09-config-and-cross-validation.md) |
| Mixup on/off | `apply_mixup`, `train_bs` 32 vs 24/28 | [03](03-augmentations.md) |
| Stage-2 fine-tune of mix models | `_2` configs: `lr 1e-3 → 5e-4`, warm-start from epoch-50 checkpoint, `load_model_only=True` | [06](06-checkpointing-and-resuming.md) |
| Weight decay per fold (4-fold nomix) | fold0 `1e-4`, fold1 `1e-2`, fold2/3 `1e-3` | [09](09-config-and-cross-validation.md) |
| Loss variant | fold2/3 nomix use `sed_scaled_pos_neg_focal_loss_augd` (`primary_drop_out_factor=0.3`); all others plain-BCE-equivalent | [07](07-losses.md) |
| Augmentation set | 5-fold uses `aug_name="secondary_default"`, all others `"default"` | [03](03-augmentations.md) |
| Checkpoint selection metric | 5-fold folds 1-4 `f1score_frame`; everything else `f1score_clip` (all-data: `lraps` for `nomix_1`/`mix_2`) | [06](06-checkpointing-and-resuming.md), [10](10-metrics-and-tracking.md) |
| Full-data retrains | `nomix_1` (3rd stage, resumes own epoch-30), `nomix_3`, `mix_2` | [09](09-config-and-cross-validation.md) |
| Earlier research generations | `train.py`…`train_v4.py`, image/melspec models | [11](11-legacy-research-paths.md) |

## 4. Not recorded — do not fill in from memory

- Per-model or per-fold LB scores; the order in which families were added.
- Training time per model, GPU memory, number of failed runs.
- Which of `mix_foldN` vs `mix_foldN_2` entered the 13.
- The inference `period` / snippet length and the exact vote implementation (external kernel).
