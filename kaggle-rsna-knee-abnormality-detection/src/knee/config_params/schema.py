#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pydantic v2 schemas validating every experiment configuration.

Design notes
------------
* Every swappable component (encoder, aggregator, loss, optimizer, ...)
  is declared as a ``ComponentSpec`` holding a dotted class path plus
  constructor params. This keeps implementations decoupled from wiring
  (Dependency Inversion): experiments change behaviour by editing YAML,
  never Python.
* Schemas are strict about structure but permissive about component
  params (validated lazily at instantiation time by the target class).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The twelve competition targets, in submission-column order.
TARGETS: tuple[str, ...] = (
  'ACL',
  'MCL',
  'Medial Meniscus',
  'Lateral Meniscus',
  'Medial OA',
  'Lateral OA',
  'PF OA',
  'Effusion',
  'Synovitis',
  "Baker's",
  'Contusion',
  'Fracture',
)


class StrictModel(BaseModel):
  """Base model: forbid unknown keys to catch YAML typos early."""

  model_config = ConfigDict(extra='forbid', validate_assignment=True)


class ComponentSpec(StrictModel):
  """Reference to an injectable implementation: class path + init args."""

  target: str = Field(
    ...,
    description='Dotted import path to the implementation class.',
  )
  params: dict[str, Any] = Field(default_factory=dict)


class PathsConfig(StrictModel):
  """Filesystem locations for data, caches, and artifacts.

  Attributes:
      data_root: Competition dataset root (Kaggle mount or local path).
      train_csv: Study-level labels + reports CSV.
      train_series_csv: Per-series descriptor table.
      train_series_dir: DICOM directory layout root.
      test_csv: Test study ids.
      test_series_csv: Test per-series descriptors.
      test_series_dir: Test DICOM root.
      volumes_cache: Pre-decoded npz cache dir (optional).
      folds_csv: Frozen fold assignments.
      weak_labels_parquet: Fused weak-label table (optional).
      checkpoints_dir: Fold checkpoints (under output_dir).
      predictions_dir: OOF/submission frames (under output_dir).
      output_dir: Kernel working output root.
  """

  data_root: str = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
  train_csv: str = 'train.csv'
  train_series_csv: str = 'train_series.csv'
  train_series_dir: str = 'train_series'
  test_csv: str = 'test.csv'
  test_series_csv: str = 'test_series.csv'
  test_series_dir: str = 'test_series'
  volumes_cache: str | None = Field(
    default=None,
    description='Pre-decoded npz volume dir; None -> decode DICOMs directly.',
  )
  folds_csv: str = 'train_folds.csv'
  weak_labels_parquet: str | None = None
  checkpoints_dir: str = 'checkpoints'
  predictions_dir: str = 'predictions'
  output_dir: str = '/kaggle/working'


class DataConfig(StrictModel):
  """Volume construction and tensor-shaping parameters.

  Attributes:
      num_slices: Slices sampled per series.
      image_size: In-plane resize target (multiple of 32).
      percentile_clip: p1-p99 windowing bounds for MR intensities.
      in_chans: Channel-stacked slices fed to 2.5D encoders.
      min_series_slices: Series shorter than this are localizers -> drop.
      cache_format: npz (compressed) or npy cache payload format.
  """

  num_slices: int = Field(
    default=32, ge=4, le=256, description='Slices sampled per series'
  )
  image_size: int = Field(default=384, multiple_of=32)
  percentile_clip: tuple[float, float] = (1.0, 99.0)
  in_chans: int = Field(
    default=32, description='Channel-stacked slices fed to 2.5D encoders'
  )
  min_series_slices: int = Field(
    default=5, description='Discard localizer series shorter than this'
  )
  cache_format: Literal['npz', 'npy'] = 'npz'


class AugmentItem(StrictModel):
  """One albumentations transform, declared declaratively."""

  target: str = Field(
    ...,
    description='Albumentations class path (e.g. RandomResizedCrop).',
  )
  params: dict[str, Any] = Field(default_factory=dict)


class AugmentConfig(StrictModel):
  train: list[AugmentItem] = Field(default_factory=list)
  valid: list[AugmentItem] = Field(default_factory=list)
  diff_ops: list[str] = Field(
    default_factory=list,
    description='GPU diff-augment subset: color/translation/cutout.',
  )


class ModelConfig(StrictModel):
  encoder: ComponentSpec
  aggregator: ComponentSpec
  metadata_encoder: ComponentSpec
  n_targets: int = Field(default=12, frozen=True)
  head_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class LossConfig(StrictModel):
  criterion: ComponentSpec
  distill_weight: float = Field(default=0.5, ge=0.0, le=1.0)
  distill_temperature: float = Field(default=2.0, gt=0.0)
  dynamic_weights: bool = Field(
    default=False,
    description='DWA auto-balancing of supervised vs distillation terms.',
  )


class OptimizerConfig(StrictModel):
  """Optimizer/scheduler specs plus differential-LR and wrapping options.

  Attributes:
      optimizer: Head (or single) optimizer ComponentSpec.
      scheduler: Optional warmup-cosine spec stepped by epoch fraction.
      scheduler_step_on: Granularity hint for manual stepping.
      backbone_lr_scale: Multiplier applied to encoder param-group LRs.
      backbone_optimizer: Optional separate encoder optimizer (manual mode).
      lookahead: Optional Lookahead wrapper spec around built optimizers.
  """

  optimizer: ComponentSpec
  scheduler: ComponentSpec | None = None
  scheduler_step_on: Literal['epoch', 'batch'] = 'epoch'
  backbone_lr_scale: float = Field(
    default=0.25,
    description='Backbone lr multiplier vs. head lr (differential LRs)',
  )
  backbone_optimizer: ComponentSpec | None = Field(
    default=None,
    description='Separate encoder optimizer (manual optimization).',
  )
  lookahead: ComponentSpec | None = Field(
    default=None,
    description='Optional Lookahead wrapper around every built optimizer.',
  )


class SamplerConfig(StrictModel):
  sampler: ComponentSpec | None = None


class DataModuleConfig(StrictModel):
  """Wiring for knee.datamodules.knee_datamodule.KneeDataModule.

  With no volumes cache (Kaggle's 30 GB disk cannot hold one for ~570 GB
  of raw DICOM), volumes stream from the read-only mount; the ``lru_*``
  fields bound how much decoded data each worker keeps in RAM. Host-RAM
  budget rule of thumb: batch_bytes ~= batch_size x max_series x
  in_chans x image_size^2 x 4 B (e.g. 4x6x32x384^2x4 ~ 450 MB), and the
  loader holds ~(num_workers x prefetch_factor + 1) of those plus the
  main-process copy -- stay inside Kaggle's ~13-15 GB.
  """

  batch_size: int = Field(default=4, ge=1, description='Studies per step')
  num_workers: int = Field(
    default=3,
    ge=0,
    description='Loader workers; each carries its own LRU + prefetch.',
  )
  max_series_per_study: int = Field(default=6, ge=1)
  pin_memory: bool = Field(
    default=False,
    description=(
      'Pinned staging doubles large-batch host copies; enable only '
      'when input feeding is the bottleneck.'
    ),
  )
  lru_max_volumes: int = Field(
    default=64,
    ge=1,
    description='Per-worker LRU capacity in decoded series volumes.',
  )
  lru_max_gb: int = Field(
    default=2,
    ge=1,
    description=(
      'Per-worker LRU capacity in GiB of uint8 voxels; total host RAM '
      'is roughly num_workers x this value -- keep it small on Kaggle.'
    ),
  )
  prefetch_factor: int = Field(
    default=2,
    ge=1,
    description='Batches pre-fetched per worker (ignored at 0 workers).',
  )
  persistent_workers: bool = Field(
    default=True,
    description=(
      'Keep workers alive between epochs: preserves the warm LRU and '
      'avoids fork storms on the 12 h wall.'
    ),
  )


class TrainConfig(StrictModel):
  """Fold policy + label policy + fully-declarative Lightning wiring.

  The Trainer, its callbacks and the logger are ComponentSpecs so the
  entire training engine (precision, accumulation, EMA, early stop)
  is controlled from YAML without touching Python (Open/Closed).
  """

  fold_scheme: Literal['iterative_multilabel', 'group_iterative'] = (
    'iterative_multilabel'
  )
  n_folds: int = Field(default=5, ge=2, le=15)
  train_folds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
  label_source: Literal['gold', 'weak', 'mixed'] = 'mixed'
  epochs: int | None = Field(
    default=None,
    ge=1,
    description='Shorthand overriding train.trainer.params.max_epochs.',
  )
  resume: bool = Field(
    default=False,
    description='Auto-resume each fold from its latest checkpoint.',
  )
  time_budget_hours: float | None = Field(
    default=None,
    gt=0.0,
    description=(
      'Wall-clock cap per Trainer.fit (Kaggle 12 h kernel limit); maps '
      'to Lightning max_time so folds stop cleanly and checkpoints '
      'survive for the next kernel.'
    ),
  )
  grad_clip: ComponentSpec | None = Field(
    default=None,
    description='Gradient clipping strategy; overrides the Trainer clip value.',
  )
  grad_noise: float = Field(
    default=0.0, ge=0.0, description='GradientNoiseInjector eta; 0 disables.'
  )
  curriculum: ComponentSpec | None = Field(
    default=None,
    description='Controller producing per-epoch weight floors.',
  )
  amp_fallback: bool = Field(
    default=False, description='Restart fit in fp32 when AMP diverges.'
  )
  trainer: ComponentSpec = Field(
    default_factory=lambda: ComponentSpec(
      target='lightning.pytorch.Trainer',
      params={
        'max_epochs': 12,
        # Single device by default: DDP doubles every host-side
        # pipeline and Kaggle pods share ~13-15 GB across ranks.
        'devices': 1,
        'precision': 'bf16-mixed',
        'gradient_clip_val': 10.0,
      },
    )
  )
  callbacks: list[ComponentSpec] = Field(default_factory=list)
  loggers: list[ComponentSpec] = Field(
    default_factory=list,
    description='Multiple loggers allowed; cloud ones auto-drop.',
  )


class TTAConfig(StrictModel):
  enabled: bool = True
  vertical_flip: bool = True
  slice_reversal: bool = True
  multi_scale: list[int] = Field(default_factory=lambda: [384])


class ExperimentConfig(StrictModel):
  """Root schema every YAML must satisfy after merge."""

  name: str
  seed: int = 42
  paths: PathsConfig = Field(default_factory=PathsConfig)
  data: DataConfig = Field(default_factory=DataConfig)
  datamodule: DataModuleConfig = Field(default_factory=DataModuleConfig)
  augment: AugmentConfig = Field(default_factory=AugmentConfig)
  model: ModelConfig
  loss: LossConfig
  optimizer: OptimizerConfig
  sampler: SamplerConfig = Field(default_factory=SamplerConfig)
  train: TrainConfig = Field(default_factory=TrainConfig)
  tta: TTAConfig = Field(default_factory=TTAConfig)

  @field_validator('name')
  @classmethod
  def _name_nonempty(cls, v: str) -> str:
    if not v.strip():
      raise ValueError('experiment name must be non-empty')
    return v
