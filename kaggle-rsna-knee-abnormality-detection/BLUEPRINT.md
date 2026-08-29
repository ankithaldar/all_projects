# RSNA Knee Abnormalities Detection — MVP Blueprint

Multimodal (MRI + reports) detection of 12 knee abnormalities per study.
Metric: macro-averaged ROC-AUC across the 12 targets.

## 1. Objective and Hard Constraints

Produce a valid `submission.csv` from an offline Kaggle notebook using
PyTorch + Lightning + timm, targeting 2x T4 GPUs.

| Resource | Limit | Design consequence |
|---|---|---|
| Raw data | ~570 GB | Never copied. Mounted read-only under `/kaggle/input/`. Stream-decode per batch; only headers are indexed once |
| RAM | 30 GB | Streaming datasets; no whole-series in-RAM buffers |
| Disk (writable) | 20 GB | Only checkpoints / OOF / submission (<2 GB); pushed to a Kaggle Dataset after each session |
| Kernel runtime | 12 h/session | Training split across sessions via checkpoint-resume protocol (Section 6) |
| GPU quota | 30 h T4/week | MVP sized at 7-12 GPU-h (Section 8); every compute knob lives in YAML |
| Submission runtime | <= 9 h (competition rule) | Inference target < 45 min |

## 2. Measured Facts (from `notebooks/01_EDA.ipynb`)

| Finding | Design impact |
|---|---|
| 4,407 studies / 24,371 series (~5.5 series/study) | Cap 4 series x K=24 slices -> ~423k slice-forwards/epoch (~20-25 min on 2xT4). Full 5-fold x 4-epoch MVP = 7-12 GPU-h |
| 0% missing reports | Rule-based pseudo-labels cover every training study |
| Median voxel anisotropy 9.92x | Pure-3D convolutions ruled out; 2.5D confirmed |
| Oblique IOP vectors exist; some series unsorted or missing IPP | Slice order precomputed in `index.parquet` via projection onto slice normal; `InstanceNumber` fallback. Filename sort is unsafe |
| `PatientSex` present ~94% (M/F/O), `PatientAge` absent | Sex one-hot enters metadata vector; age dropped |
| 1000 sampled studies map to 1000 unique patients | GroupKFold by StudyUID; PatientID captured during header scan to re-group if multi-study patients appear |
| "200/200 corrupted" audit result was an EDA logic bug (rate over unchecked files) | Corrected audit moved to notebook 02; decoder registry keeps zeros-frame fallback regardless |
| Domain-shift analysis crashed before producing numbers | `index.parquet` stores `ScanningSequence` / `MagneticFieldStrength` so folds may stratify by protocol proxy; DANN deferred |
| Label-vs-text contradiction rates not yet quantified | Fixed contradiction-rate table becomes part of notebook 02 QC; label smoothing strength calibrated from it |
| 73 high-frequency DICOM tags discovered | Header-only index captures all tags per series (first slice) plus per-slice minimal set (IPP, IOP, InstanceNumber) |

## 3. MVP Scope

| In MVP | Deferred (backlog Section 11) |
|---|---|
| Header-only DICOM index (geometry, spacing, ordering) | Physical isotropic resampling |
| On-the-fly decode (gdcm/pylibjpeg fallback registry) | Decoded-pixel cache datasets |
| Percentile normalization + autocrop inside Dataset | MONAI bias-field / ghosting augmentations |
| Rule-based English pseudo-labels (negation-aware) | XLM-R multilingual labeler, soft-label distillation |
| 2.5D sampling: K slices/series -> attention pooling | Per-plane encoders, CLIP-style image-report pretraining |
| Shared backbone + plane/sequence embedding + FiLM(metadata) | TTA, snapshot ensembles |
| Weighted BCE (YAML-swappable Asymmetric Focal) | ONNX Runtime export, INT8/TensorRT |
| StratifiedGroupKFold 5 folds, OOF macro-AUC | Fold-count reduction experiments |
| Session-resume training via Kaggle Dataset versioning | Efficiency-prize micro-tuning |

## 4. Architecture Overview

```
StudyInstanceUID
 └── Series 1..N (plane, Fluid_Sensitive, Fat_Suppression, Sex from index.parquet)
      ├── ordered slice list (precomputed; zero DICOM parsing at train time)
      └── sample K=24 evenly spaced slices -> worker decodes JPEG/JP2
           -> rescale -> percentile normalize -> autocrop -> resize img_size
           └── shared Backbone (tf_efficientnetv2_s) -> slice embeddings
                -> AttentionPool -> series token (+ plane/seq embedding)
                 ▼
     StudyAggregator (learnable query cross-attends series tokens)
     ⊕ FiLM(metadata MLP: plane ⊕ FS ⊕ FL ⊕ sex ⊕ log counts ⊕ spacing stats)
                 ▼
          trunk MLP -> 12 sigmoid heads
```

Reports exist only at training time: gold labels where present,
rule-derived pseudo-labels elsewhere (Section 7).

## 5. Configuration System (zero hardcoded configuration)

### 5.1 Rules

- Every hyperparameter, path, seed, size, and class selection lives in `configs/*.yaml`.
- Any object is specified as:

```yaml
class_path: package.module.ClassName   # fully qualified import path
init_params:                           # kwargs for __init__, resolved recursively
  key: value
```

- Lists of such specs are allowed (augmentation pipelines, metrics).

### 5.2 Files

```
configs/
├── config.yaml          # experiment name, seed, paths (mounts, ckpt/cache dataset
│                        #   slugs, kaggle secret names), integrations (discord
│                        #   transport + cadence), logging/progress, selftest scope,
│                        #   folds to run
├── data.yaml            # n_slices, img_size, series-selection policy,
│                        #   normalization percentiles, decode backend order,
│                        #   target_columns (output schema), crop threshold,
│                        #   resize interpolation, fallback shape, scan chunksize
├── folds.yaml           # CV class_path + init_params (StratifiedGroupKFold)
├── augmentations.yaml   # albumentations pipelines: train / valid
├── model.yaml           # KneeNet tree: backbone, pools, aggregator, heads
├── loss.yaml            # criterion class_path + init_params
├── optimizer.yaml       # optimizer/scheduler class_paths, lr groups, warmup_epochs
├── train.yaml           # Trainer args (epochs, precision, devices, benchmark),
│                        #   session_time_budget_h, checkpoint_every_n_epochs
├── datamodule.yaml      # DataModule/Dataset class_paths, batch_size, workers,
│                        #   train_sampler spec (imbalanced-target sampling;
│                        #   null = uniform)
├── (volume_cache)       # cache policy in config.yaml: split_mode, shard_gib_cap,
│                        #   gzip_level, scan_workers, pool_chunksize, copy_dir
├── infer.yaml           # ckpt fold selection, fp16, batch_size, submission_path
└── experiments/
    ├── mvp_efnv2s_384_k24_5f.yaml   # main experiment: defaults list + override
    └── smoke_ci.yaml                # CPU smoke run for CI/dry runs
```

Each experiment file deep-merges its `defaults:` stems in order, applies its
`override:` section last, and the fully resolved dictionary is dumped to
`artifact_dir/resolved_<name>.yaml` at every run start — one self-contained
file per experiment guarantees traceback.

### 5.3 Loader API (`src/knee/config_params/loader.py`)

```python
def load_config(path, overrides=None) -> dict:
    """Load YAML + dot-path overrides + ${a.b} interpolation."""

def load_experiment(path, overrides=None) -> dict:
    """Compose defaults list -> override -> interpolation (Section 5.2)."""

def dump_config(config, path) -> None:
    """Persist a resolved configuration for run traceability."""

def instantiate(spec) -> Any:
    """Recursively resolve class_path/init_params specs into live objects."""
```

Implementation uses `yaml.safe_load` + `importlib`; no third-party config framework.

## 6. Storage and Resume Protocol

Training cannot finish one 12 h session; kernels are ephemeral; GPU quota resets weekly.

Artifacts (all tiny; total < 2 GB):

| Artifact | Location | Content |
|---|---|---|
| `index.parquet`, `labels_pseudo.csv`, `folds.csv` | Kaggle Dataset `<slug>-index` | ordered SOP lists, spacing, plane/FS/FL, sex, protocol tags; supervision; splits |
| `fold{k}/last.ckpt`, `fold{k}/done`, `run_state.json` | Kaggle Dataset `<slug>-ckpt` (versioned) | model+optimizer+scheduler+epoch+step+best metric; fold completion markers |
| `volume_shard_NNN.h5`, `cache_manifest.parquet`, `cache_meta.json` | one Kaggle Dataset per shard: `<base>-NNN` | decoded pixel volumes (uint8, all slices) keyed by SeriesInstanceUID; uid->shard manifest fragment; generation stamp |

Flow (`src/knee/helpers/kaggle_io.py`):

1. Credentials resolve through `knee.helpers.secrets`:
   process env -> project `.env` -> Kaggle `UserSecretsClient`; secret names
   are read from `config.kaggle_secrets.*`.
2. `pull_latest(slug, dest)` downloads newest version if it exists.
3. Engine start: per fold — `done` marker -> skip; `last.ckpt` -> resume via
   `Trainer.fit(ckpt_path=...)`.
4. `TimeBudgetCallback` stops fit when wall clock exceeds
   `session_time_budget_h - time_margin_min`; checkpoints also written every
   `checkpoint_every_n_epochs`.
5. Session end: `push_version(slug, folder)` creates a new immutable dataset
   version (rollback history).
6. Push retries x3 with backoff; on failure the ckpt remains in
   `/kaggle/working` with logged manual recovery path.
7. Inference consumes whichever folds have `done` markers (partial ensembles
   degrade gracefully).
8. Volume-cache resume mirrors the checkpoint protocol: fragment manifests
   inside every attached `cache-*` dataset describe remote coverage, so
   reruns skip pushed series entirely; new shards CONTINUE the `-NNN`
   sequence (ordinal floor from `remote_cache_state`) instead of
   version-bumping old shards - the overwrite regression is guarded by a
   session-level duplicate-slug assertion.
9. Every pushed cache dataset stamps `cache_meta.json` (img_size, split
   mode, cap, UTC); `load_manifest` logs at ERROR when attached mounts mix
   preprocessing generations.

## 6b. Experiment Tracking and Notifications

All three channels run simultaneously and independently; none can crash a
scoring run:

| Channel | Implementation | Content |
|---|---|---|
| CSV | `loggers/csv_logger.py` (Lightning `CSVLogger`) | per-fold metrics under `<output_dir>/logs/<exp>/fold{k}/` |
| W&B | `loggers/wandb_logger.py` | same metrics online when `WANDB_API_KEY` resolves; auto-offline otherwise |
| Discord | `loggers/discord_logger.py` | fit start, per-epoch macro-AUC, fold completion + duration, crash tracebacks via webhook |

Credentials resolve by *name* from YAML through
`helpers/secrets.get_secret(name)`: env -> `.env` -> Kaggle secrets
(see `.env.example`: `DISCORD_WEBHOOK_URL`, `WANDB_API_KEY`, `KAGGLE_*`).

## 6c. HDF5 Volume Cache (decode once, read forever)

Decoding 819k DICOM frames per epoch starved both T4s (the CPU was the
bottleneck even at 4 vCPUs of pure decode). The `cache` stage collapses
that cost into a one-time pass:

* `ShardWriter` rolls `volume_shard_NNN.h5` under an UNCOMPRESSED byte
  cap (`volume_cache.shard_gib_cap`, default 10 GiB -> ~2-4 GiB files
  after per-chunk gzip-4). Datasets are keyed by SeriesInstanceUID with
  per-slice chunks; completion sentinels (`*.h5.complete`) drive resume
  and pipelined per-shard pushes.
* `H5SeriesReader` serves pixels with the EXACT `SeriesReader.read()`
  contract, including the repeat-sampling rule for series shorter than
  `n_slices` (regression-tested), and falls back to live decoding on
  any miss so training can never hard-fail on a partial cache.
* Root resolution is automatic: attached Kaggle datasets containing a
  `cache_manifest.parquet` fragment qualify (artifact datasets do not);
  `KNEE_HDF5_CACHE_DIRS` overrides, `volume_cache.copy_mounts_to_working`
  + `copy_dir` optionally mirror mounts onto local scratch
  (`/kaggle/tmp/cache_roots` in the MVP - FUSE chunk round-trips were
  the dominant step-time cost).
* Failures never fossilize: a series is cached only when EVERY frame
  decodes; a single corrupt frame leaves the series on the live path.
* Offline equivalent: `build_volume_cache.py` runs the identical
  pipeline without experiment config (CLI flags, optional Discord
  progress).

## 6d. Observability: Flat Logs, Progress, Discord

* `helpers/logging_setup.setup_logging()` runs in `main()` for every
  stage: one flat file `paths.log_dir/knee_<stage>_<experiment>_<UTC>.log`
  receives the ROOT logger AND mirrored stdout/stderr (tqdm, Lightning
  prints, child-process output, raw tracebacks). Eager flush; full-disk
  safe; idempotent per process (tee rebinds across stages).
* `loggers/progress.ProgressLogCallback` mirrors batch/epoch progress
  into that file every `trainer.log_every_n_steps` batches:
  `epoch E | batch b/B | step S | train_loss | lr | gpu A/B GiB`,
  plus validation metrics and epoch durations. Rank-0 only.
  Native `TQDMProgressBar(refresh_rate)` runs alongside.
* Discord (`loggers/discord_logger`): fit start/finish, heartbeat every
  `integrations.discord.every_n_steps` optimizer steps, a step-1
  `pacing OK` ping (`first_step_ping`), per-push dataset announcements
  during cache builds, final cache summary naming every pushed dataset
  with a ready-to-paste `KNEE_HDF5_CACHE_DIRS` line, and truncated
  tracebacks on crashes. Resolution failures are LOUD (config-off logs
  INFO; unresolved webhook logs WARNING naming the secret).
* All knobs live in config: `logging.*`, `integrations.discord.*`,
  `volume_cache.{log_every_series, pool_chunksize, discord_files_every}`.

## 6e. Imbalanced-Target Sampling (train only)

The 12 targets are heavily skewed while the metric is MACRO-AUC, so
uniform study sampling starves rare positives (fracture, contusion,
Baker's cyst, synovitis) of gradient updates. An optional frequency-
aware sampler tilts the TRAIN loader only:

* Per-target weight `prevalence**-tempering` normalized to mean 1.0,
  where prevalence counts only UNMASKED entries (`-1` never votes).
  `tempering: 0` reproduces uniform sampling exactly; `1.0` is full
  inverse frequency; the MVP ships `0.5` (sqrt-tempered).
* Per-study weight = `max` (default) or `mean` over the weights of the
  study's POSITIVE targets; studies with no positive target get
  `baseline` (1.0). The `max` default gives rare-positive studies the
  strongest pull.
* Draws are WITH replacement and `num_samples` defaults to the dataset
  size: epoch size, optimizer-step counts, cosine schedule math, and
  the checkpoint-resume protocol are all unchanged.
* Under DDP Lightning wraps the sampler in
  `DistributedSamplerWrapper`, so 2xT4 ranks draw disjoint streams.
* Validation loaders stay uniform; OOF/macro-AUC remains comparable
  across experiments.

Wiring honours the zero-hardcode contract: `train_sampler` is a
SIBLING key of `class_path/init_params` in the datamodule spec (the
factory needs the fold split, so it builds the sampler at loader
time), resolved by `samplers/factory.py` exactly like the
augmentations factory. Enabling it mid-fold changes the train
distribution: do not resume a partially trained fold across the
change - restart the fold.

## 7. NLP Pseudo-Labeling (MVP rules)

- Per-target lexicon regexes on lowercased report text.
- Negation triggers (`no|without|intact|unremarkable|absent`) within a +/-6-token window flip to negative.
- Uncertain cues (`rule out|cannot exclude|question`) emit `-1` mask (excluded from that target's loss).
- Output `labels_pseudo.csv`: hard labels + `source in {gold, rules}`.
- QC in notebook 02: agreement kappa vs gold subset per class, prevalence sanity table,
  language census over all reports (sizes the future XLM-R backlog item),
  fixed contradiction-rate table (calibrates label smoothing).

## 8. Compute Budget (fits 30 GPU-h/week)

```
per epoch : 4407 studies x 4 series x 24 slices ~= 423k slices @384 px
            ~= 300-350 img/s (2xT4, bf16-mixed)  ~= 20-25 min (+decode overlap)
MVP total : 5 folds x 4 epochs                  ~= 7-12 GPU-h
inference : ~1300 studies                       ~= < 15 min
headroom  : >= 15 GPU-h for iteration/backlog
```

Levers if measurements disagree: `n_slices`, `img_size`, `max_series_per_study`,
backbone name — all in YAML. Notebook 01 records measured decode throughput.

As-built updates (kernel-measured, 2026-08):

* Decode moved OFF the training path (section 6c): streaming DICOM
  decode was starving both T4s at any batch size; shards + local
  `/kaggle/tmp` mirror removed it entirely.
* Effective step preserved under tuning: batch_size 6 x 2 GPUs x
  accumulate 2 = 24 studies, with `grad_checkpointing: true` +
  `chunk_size: 48` bounding activations (7.6/15 GiB observed at batch 3
  -> headroom deliberately consumed).
* `deterministic: false`, `benchmark: true` (fixed slice shapes),
  `num_workers: 2` per DDP process (2 processes x 2 = Kaggle's 4 vCPUs
  exactly - 8 decoders thrashed).
* Pace measurement discipline: Discord heartbeat gaps / 50 optimizer
  steps, or the flat log's batch lines; no more estimating from kernel
  wall-clock.

## 9. Repository Layout

```
main.py                              # CLI: build-index | build-labels | build-folds |
                                     #   train | infer  (+ --experiment/--override)
kaggle_run.sh                        # staged shell driver: setup|index|labels|folds|
                                     #   train|infer|all; sources .env; offline wheels
kaggle_cell.py                       # notebook cell entrypoint wrapping the driver
requirements.txt                     # pinned deps incl. DICOM codec wheels
configs/                             # Section 5.2 (bases + experiments/)
src/knee/
├── config_params/loader.py          # load_config, load_experiment, dump_config,
│                                    #   deep_merge, recursive instantiate
├── helpers/
│   ├── dicom_io.py                  # DecoderRegistry: native->gdcm->pylibjpeg
│   ├── header_scan.py               # parallel header-only scan -> index.parquet
│   ├── geometry.py                  # normal-projection ordering; InstanceNumber fallback
│   ├── intensity.py                 # rescale, percentile normalize, autocrop
│   ├── nlp_labeling.py              # bidirectional negation window (sentence-scoped),
│   │                                #   OA anchor x compartment co-occurrence, -1 mask
│   ├── folds.py                     # strata builder + grouped fold assignment
│   ├── kaggle_io.py                 # retrying CLI client: create/version/pull
│   ├── secrets.py                   # env -> .env -> Kaggle UserSecrets resolution
│   └── utils.py                     # seeding, timer, logger
├── datasets/
│   ├── series_dataset.py            # even-index K-slice sampling from ordered SOP lists
│   └── study_dataset.py             # priority series selection, metadata builder,
│                                    #   collate matching KneeNet's flat contract
├── datamodules/study_datamodule.py  # fold-aware LightningDataModule
├── augmentations/factory.py         # YAML specs -> Compose (resize/totensor bookends)
├── layers/pooling.py                # AttentionPool2d, StudyAggregator, FiLM
├── models/
│   ├── backbones.py                 # timm wrapper with offline checkpoint support
│   ├── series_encoder.py            # single flattened backbone pass + pooling
│   └── knee_net.py                  # full net + differential-LR parameter groups
├── losses/asymmetric_focal.py       # WeightedBCE default; AsymmetricFocal; -1 masking
├── metrics/auc.py                   # per-class + macro ROC-AUC accumulator
├── engines/
│   ├── assembly.py                  # config -> reader/datasets/datamodule/model
│   ├── train_module.py              # KneeModule: loss, warmup+cosine, OOF dumps
│   └── inferencer.py                # fold-ensemble fp16 predictor + schema asserts
├── callbacks/session.py             # TimeBudgetCallback + PeriodicPushCallback
├── loggers/
│   ├── csv_logger.py                # per-fold Lightning CSVLogger
│   ├── discord_logger.py            # webhook notifier + lifecycle callback
│   └── wandb_logger.py              # W&B logger (offline fallback, key via secrets)
notebooks/
├── 01_EDA.ipynb                     # done
├── 02_build_index_labels_folds.ipynb  # pending (or use kaggle_run.sh stages)
├── 03_model_training.ipynb          # pending thin wrapper over kaggle_cell.py --stage train
└── 04_inference.ipynb               # pending thin wrapper over kaggle_cell.py --stage infer
tests/                               # geometry, NLP rules, config loader,
                                     # net shapes, resume, cache, selftest,
                                     # noise floor, imbalanced sampler, OOF
                                     # metric isolation
```

Style contract for all Python: shebang + coding header, Google docstrings,
2-space indentation, single quotes, SOLID boundaries
(Registry/Factory/Adapter/Strategy/Template Method/Facade as annotated above).

## 10. Verification Gates

- `pylint --rcfile=../.pylintrc` >= 9.5 (currently 10.00) and `ruff check`
  clean on every Python file under src/, tests/, main.py, kaggle_cell.py.
- `pytest tests/` green.
- Every config file instantiates through `load_config` + `instantiate` without error.
- KneeNet forward passes shape tests for variable series/slice counts.
- Notebook 03 prints per-fold/per-class OOF AUC; sanity gate >= 0.80 macro on gold+pseudo CV.
- Notebook 04 asserts exact header, row count, and UID set equality before writing `submission.csv`.

## 11. Iteration Backlog (post-MVP)

Each entry states the experiment and the signal that decides keep/drop.
Order within a group is by expected value; run noise-floor study first.

### 11.0 Measurement discipline (run before chasing any gain)

1. **Noise floor**: 3 seeds x best config on fold 0; record macro-AUC
   std. Any change worth shipping must beat mean + 2 sigma.
   IMPLEMENTED as the `sweep` stage (engines/noise_floor.py): per-seed
   isolated checkpoint dirs + dataset slugs, resumable JSON state,
   final-epoch OOF scoring, gate = mean + 2*std announced on Discord.
   NOTE: run it with the sampling policy intended for the follow-up
   experiments (the MVP experiment ships the 6e sampler enabled).
2. **Single-knob sweeps via experiments/*.yaml** (img_size, n_slices,
   max_series, lr, epochs) logged to W&B for automatic comparison.
3. **Per-class attribution notebook**: which of the 12 targets move with
   each knob (macro can hide rare-class regressions).

### 11.1 Supervision & labels

1. XLM-R multilingual pseudo-labeler trained on gold + rule-labeled
   English; replaces rules for non-English reports. Signal: kappa vs
   gold per class, then OOF delta.
2. Soft-target distillation: replace hard pseudo-labels with teacher
   probabilities (weighted BCE on soft targets). Signal: OOF macro-AUC,
   especially on rule-uncertain (`-1`) studies now recovered.
3. Severity auxiliary heads from report modifiers (partial/complete
   tear, grade I-III meniscal, mild/moderate/severe OA - EDA found all
   present): multi-task regularization. Signal: main-target AUC delta.
4. Pseudo-label round 2: retrain labeler with image-model OOF as an
   additional feature (co-training); iterate once. Signal: kappa gain.
5. Dependency-parse negation scope (spaCy/stanza bundled offline)
   replacing the token window; measure precision/recall vs current
   rules on the gold subset before adopting.
6. OPUS-MT translation backfill -> apply English rules to every
   language; compare against XLM-R route (cheaper inference-time
   nothing, but training-only either way).
7. Curriculum: epoch 1-2 on gold subset only, then add pseudo data.
   Signal: stability of rare-class AUC across seeds.
8. Confidence-weighted sampling: studies with high-agreement pseudo
   labels sampled more often; uncertain ones only through soft loss.
9. Near-duplicate detection via MIP perceptual hashing; drop or
   down-weight duplicates to reduce train/test leakage risk.
10. Confident-learning denoising (cleanlab, offline wheel) on the merged
    gold+pseudo matrix; flag likely label errors using out-of-fold
    predictions from a text-only model; review flagged gold labels too.
    Signal: kappa uplift after pruning suspected noise.
11. Report-section auto-segmentation (classify lines into Impression /
    Findings / Clinical history) applied before all rule matching;
    boosts precision for targets whose lexicon overfires in history
    sections.
12. Laterality normalization: resolve left/right knee wording and
    anatomical-side mentions before matching compartment terms; guards
    against Medial/Lateral OA confusion introduced by contralateral
    phrasing ("left knee: lateral compartment narrowing").
13. Noise-injection robustness training: flip a controlled fraction of
    pseudo-labels during training at the measured contradiction rate;
    keep if OOF AUC becomes less seed-sensitive.
14. Semi-supervised consistency on unlabeled test volumes (Mean-Teacher
    style): legality check against competition rules first; accept only
    if OOF proxy gain > noise floor AND organizers confirm permitted.

### 11.2 Architecture

1. Backbone zoo sweep at fixed budget: ConvNeXt-V2-Tiny/Nano,
   MaxViT-Tiny (windowed attention fits slice grids), EfficientNetV2-M,
   Swin-V2-Tiny, DINOv2 ViT-S frozen+linear. Signal: OOF macro per
   GPU-hour.
2. Per-plane encoder specialization: plane-specific backbones vs shared
   trunk with plane tokens vs shared+plane FiLM (current).
3. Series-token transformer: replace AttentionPool2d with tiny
   transformer over slice embeddings incl. positional encodings along
   the through-plane axis (order-aware pooling).
4. Cross-series attention with plane-specific queries in the aggregator
   (current query is protocol-agnostic).
5. Two-stage ROI pipeline: cheap localization (U-Net or threshold/
   morphology) -> aligned crop -> classifier; motivated by EDA CoM
   variance being low but nonzero.
6. Atlas registration normalization instead of autocrop (deterministic
   anatomy alignment; costs CPU at index time if precomputed).
7. Class-decoupled experts for the four rare targets sharing the
   frozen common trunk. Signal: Fracture/Contusion/Baker's/Synovitis
   AUC without macro regression elsewhere.
8. Ordinal heads for OA severity where grade text is extractable.
9. MixUp/CutMix on slices and manifold-mixup on series embeddings;
   flip-augmentation variant that swaps Medial<->Lateral logits.
10. Model souping: uniform/linear interpolation of fold checkpoints;
    free ensemble compression (one weight set, near-ensemble AUC).
11. Deep-MIL gated attention pooling (Ilse et al.) with learnable
    temperature as an alternative to AttentionPool2d; also yields
    interpretable slice-importance maps.
12. Multi-scale fusion: FPN-style laterals over backbone strides for
    small findings (fracture lines, meniscal roots) before pooling.
13. Dual-resolution branches: low-res full series (context) + high-res
    center crops (detail) fused at the study token; targets subtle
    tears lost at 384 px while keeping slice counts low.
14. Geometry-aware cross-plane attention: use indexed IPP/IOP to give
    each series token a spatial prior so the aggregator knows where in
    knee-space each plane sampled; pure-data grounding, zero labels.
15. Vision-Mamba (state-space) slice encoder: linear-time long-context
    alternative to the slice transformer for K>32 long-tail series.
16. Stochastic slice dropout during training (drop 10-20% of slices per
    series) -> robustness to short/padded series and a natural TTA axis.
17. Token merging (ToMe) inside ViT backbones: throughput experiment
    feeding 11.4 efficiency levers at iso-accuracy.

### 11.3 Training recipe

1. Loss bake-off at fixed schedule: WeightedBCE (incumbent),
   AsymmetricFocal, Focal-T, pairwise AUC-style losses
   (e.g. differentiable rank hinge); pick per-class winners.
2. LLRD (layer-wise lr decay) for transformer backbones.
3. EMA vs SWA vs plain weights; soup-of-EMA-snapshots.
4. Progressive resizing 256->384->448 with cosine restarts.
5. Optimizer alternates: LAMB / Lookahead / AdamW+schedule-free.
6. bf16 vs fp16 vs tf32 throughput/accuracy matrix on T4 (feeds
   efficiency score directly).
7. Multi-task gradient surgery: PCGrad or GradVac on the 12 heads;
   adopt when head-gradient cosine conflicts correlate with rare-class
   plateaus.
8. Uncertainty-weighted multi-task loss (Kendall et al. learned
   sigmas) replacing the uniform loss sum across targets.
9. Optuna ASHA sweep over lr/wd/drop_path/dropout inside a fixed
   GPU-hour envelope; results land in W&B for the experiment registry.
10. Per-class checkpoint selection: keep per-target best EMA snapshots
    instead of one global best (macro-AUC assembly at inference).

### 11.4 Inference & ensembling

1. TTA set definition: hflip-with-logit-swap, +/-5% scale, center
   slice-subset re-run; accept only if OOF gain > 2x noise floor and
   runtime stays under efficiency budget.
2. Stacking meta-model on OOF probabilities (per-class logistic on
   [image logits, metadata]) - watch overfit via nested CV.
3. ONNX Runtime fp16 export + parity tests (< 0.001 OOF delta);
   INT8 static quantization calibrated on 200 studies; TensorRT EP
   build script for the submission kernel.
4. Fold pruning: rank folds by OOF, greedily drop while ensemble AUC
   holds (efficiency prize lever).
5. Early-exit cascade: lightweight pass (small backbone @224px) gates
   the full model only when uncertainty is high; tune gate for the
   efficiency score's AUC/runtime trade-off curve.
6. Snapshot ensembles from one training run (cyclic lr snapshots)
   replacing multi-fold compute.
7. Fold weight-sharing: store one shared backbone plus per-fold LoRA
   deltas -> smaller checkpoint dataset, faster session resume, near
   full-ensemble AUC.
8. Structured channel pruning of the backbone followed by brief
   fine-tune (prune-then-distill); measure against 11.4-3 INT8 as the
   cheaper efficiency lever.
9. Kornia GPU-side transform pipeline at inference to overlap
   resize/normalize with compute; drop if kernel launch overhead eats
   the CPU savings on T4.

### 11.5 Data engineering

1. MONAI MRI-specific augmentations: bias-field, ghosting, k-space
   spike artifacts; bundle wheels offline.
2. Protocol-stratified folds once domain-shift analysis is rerun
   cleanly (ScanningSequence/MagneticFieldStrength are already indexed).
3. Sequence clustering from DICOM tags (TR/TE/flip angle) as extra
   metadata features and stratifiers.
4. Motion-corruption scoring at index time (inter-slice variance
   heuristic) -> quality filter or down-weight flag.
5. Isotropic resampling ablation (index-time, cached in parquet) -
   revisit after 11.2-6 registration decision.
6. K-slice *placement* ablation: uniform vs center-weighted vs
   learned slice scorer; interacts with 11.4-5 early-exit design.
7. Histogram matching to a reference scanner (site-shift reduction)
   as a dataset-level intensity normalizer; compare against per-series
   percentile normalization on the domain-shift proxy metrics.
8. Metadata interaction features: sex x plane, FS x plane, log-counts
   ratios; cheap FiLM-vector extensions with a leakage-safe origin.
9. Multi-slice RGB channel variant: 3 channels = 3 adjacent slices
   (classic 2.5D encoding) instead of replicated grayscale; halves
   backbone passes vs current flat scheme at equal coverage.

### 11.6 Pretraining & transfer

1. Contrastive image-report pretraining (CLIP-style ConVIRT) on all
   4.4k studies before supervised fine-tune.
2. Public-medical checkpoint init: RSNA breast/spine competition
   weights, RadImageNet (license check), REMEDIS if downloadable.
3. Self-supervised rotation/slice-order prediction pretraining on the
   unlabeled test-series volumes (test-time adaptation lite).

### 11.7 Robustness & shortcut auditing

1. Metadata-only probe: train a GBM/logistic on [plane, FS, FL, sex,
   counts, spacing] alone; its OOF AUC per target is a *floor* any
   image model must clearly beat - guards against learning site/
   protocol shortcuts instead of pathology.
2. Saliency QC notebook (Grad-CAM / attention rollout on series
   tokens): verify activations sit on menisci/cartilage/ligament
   regions rather than background or coil artifacts.
3. Error taxonomy: cluster top-loss OOF studies by language, scanner
   field strength, plane availability, and pseudo-label source;
   surfaces systematic blind spots (e.g. non-English reports
   mislabeled by rules).
4. Cross-site adaptation probes: test-time BN recalibration and TENT
   on the strongest domain-shift proxy split; adopt only with an
   offline-legal, deterministic recipe.
5. Determinism audit: identical seed across two sessions must
   reproduce OOF to < 1e-4 AUC; catches nondeterministic kernels that
   silently poison ablation readings.
6. Slice-position sensitivity: shuffle test at inference (ordered vs
   random slices) quantifies how much the model truly uses 2.5D order
   versus treating slices as a bag - validates 11.2-3 investment.

### 11.8 Process & infrastructure

1. Experiment registry: append-only runs.csv (config hash, resolved
   yaml path, W&B url, OOF macro/per-class) written automatically by
   `main.py train`; enables instant best-config lookup.
2. Checkpoint-dataset garbage collection policy: keep last N versions
   per slug (kaggle CLI delete) to stay under storage quotas during
   multi-week iteration.
3. GPU utilization logging callback (nvidia-smi sampler -> CSV):
   separates decode-bound from compute-bound epochs and justifies
   worker/prefetch settings with data instead of guesses.
4. DDP vs single-GPU scaling profile at fixed epoch count; documents
   whether devices:2 is actually delivering ~2x before relying on it
   in the budget math.
5. Smoke-CI gate in git hooks/CI runner: ruff + pylint + pytest +
   smoke_ci experiment dry-run (CPU, tiny tensors) on every push.

## 12. Usage

### 12.1 Kaggle session flow

`kaggle_cell.py` self-bootstraps: inside a checkout it refreshes the
committed `repo_meta.json` from `.git` (remote URL + branch, worktree-aware);
outside one (pasted cell / dataset mount) it shallow-clones that branch into
`$KNEE_REPO_DIR`, resolves the project directory within the monorepo, and
re-executes your command there - zero manual code.

```bash
# Cell 1 (fresh kernel): install pinned deps; WHEELS_DIR enables offline mode
!bash kaggle_run.sh setup

# Cells 2..n: data stages are resume-safe and cache-aware; training
# resumes automatically from pulled last.ckpt, pixels from shards
%run kaggle_cell.py --stage index
%run kaggle_cell.py --stage labels
%run kaggle_cell.py --stage folds
%run kaggle_cell.py --stage cache              # one-time: HDF5 shards -> datasets
%run kaggle_cell.py --stage selftest           # preflight: 2 real steps + ckpt
%run kaggle_cell.py --stage train --fold 0     # or omit --fold for run.folds list
%run kaggle_cell.py --stage sweep              # noise floor (11.0-1): seeds x folds
%run kaggle_cell.py --stage infer              # writes submission.csv

# Production log (flat, everything incl tracebacks):
!tail -f /kaggle/working/logs/knee_train_*.log
```

Environment overrides: `EXPERIMENT` (YAML under configs/experiments/),
`WHEELS_DIR` (offline wheel directory), `PIP_EXTRA`,
`KNEE_REPO_DIR` (clone location), `GIT_TOKEN`/`GITHUB_TOKEN`/`GH_TOKEN`
(https credentials for private upstreams), `KNEE_HDF5_CACHE_DIRS`
(colon-separated cache roots; auto-discovery usually makes this
unnecessary), `KNEE_INPUT_ROOTS`, `KNEE_LOG_DIR`, `KNEE_CACHE_COPY`.

Environment overrides: `EXPERIMENT` (YAML under configs/experiments/),
`WHEELS_DIR` (offline wheel directory), `PIP_EXTRA`,
`KNEE_REPO_DIR` (clone location), `GIT_TOKEN`/`GITHUB_TOKEN`/`GH_TOKEN`
(https credentials for private upstreams).

### 12.2 Local development

```bash
pip install -r requirements.txt
PYTHONPATH=src python main.py train \
  --experiment configs/experiments/smoke_ci.yaml   # CPU smoke run
pytest tests
ruff check src tests main.py && ruff format --check .
pylint --rcfile=../.pylintrc src/knee main.py
```

### 12.3 Experiment traceback

Every run dumps its fully composed configuration to
`artifact_dir/resolved_<experiment>.yaml`; new experiments are a single YAML
under `configs/experiments/` listing `defaults` + an `override` block.

## 13. Implementation Status

| Milestone | State |
|---|---|
| EDA + decoder forensics (`01_EDA.ipynb`) | done |
| Config system (bases, experiments, loader, resolved-dump) | done |
| Helpers: geometry/intensity/dicom_io/header_scan/nlp/folds/kaggle_io/secrets | done |
| Model stack: pooling layers, backbone, encoders, KneeNet | done |
| Engines: assembly, Lightning module, inferencer, submission asserts | done |
| Callbacks: time budget, periodic push | done |
| Logging: CSV + W&B + Discord + flat production log + progress lines | done |
| Kaggle drivers (kaggle_run.sh, kaggle_cell.py) | done |
| HDF5 volume cache (shards, manifest, local mirror, resume) | done |
| Preflight selftest stage (artifacts/mount/cache/model/2 real steps) | done |
| Perf engineering: batch 6/accum 2, checkpointing+chunking, local reads | done |
| Tests: 121 passing; pylint 10.00/10 on all files; ruff clean | done |
| Notebooks 02-04 as thin wrappers | pending (stages already runnable via drivers) |
| First full 5-fold train on shards (fold0 partially trained) | in progress |
| Backlog items 1-8 | pending |

Blueprint-implementation branch (post-MVP, one feature per commit):

| Item | State |
|---|---|
| Noise-floor harness (`sweep` stage, 11.0-1): per-seed isolated ckpt dirs/slugs, resumable state, final-epoch OOF scoring, gate mean+2*std via Discord | done |
| Imbalanced-target study sampler (6e): tempering/aggregation/baseline in YAML, train-only, epoch size preserved | done |
| Bugfix: per-epoch AUC accumulator reset (OOF/val metrics previously accumulated across sanity check + epochs) | done |
| Experiment registry (11.8-1), then 11.0-2/3 sweeps + attribution | pending |

Commit trail (abridged): `21def73` blueprint -> `fd41d80` configs ->
`8a04e11` loader+helpers -> `e15f02e` model stack -> `9f6facc`
engines/CLI -> `bed9ceb` logging + drivers -> `b2f9e07`..`f810e02`
(2026-08 hardening + volume cache + selftest + observability series:
slug qualification, config-schema fixes, OOM/memory levers, discord
heartbeats, HDF5 shards, local mirror, production logging) ->
`a59c93f` noise-floor sweep stage -> `38cc093` imbalanced study
sampler -> `23a0547` per-epoch AUC reset fix.
