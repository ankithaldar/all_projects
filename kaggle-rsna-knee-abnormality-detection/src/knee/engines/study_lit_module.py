#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LightningModule for study-level knee abnormality classification.

Wraps :class:`knee.models.study_model.KneeStudyModel` and implements the
BLUEPRINT section 5 recipe plus the optimization-enhancement suite:

- confidence-weighted supervision + epoch-ramped temperature-scaled
  distillation against teacher probabilities,
- optional DWA dynamic weighting of the two loss terms,
- optional curriculum weight floors (easy -> hard label schedule),
- custom gradient clipping strategies and decaying gradient noise,
- differential-LR AdamW, optional separate backbone optimizer (manual
  optimization), optional Lookahead wrapping, warmup-cosine schedule,
- optional GPU differentiable augmentation inside the training step.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch

from knee.augmentations.diff_augment import DiffAugment
from knee.config_params.loader import instantiate, resolve_target
from knee.config_params.schema import (
  AugmentConfig,
  LossConfig,
  ModelConfig,
  OptimizerConfig,
  TrainConfig,
)
from knee.losses.classification import SoftBCEWithLogits
from knee.losses.curriculum_weights import (
  CurriculumController,
  DynamicLossWeighter,
)
from knee.metrics.macro_auc import macro_auc
from knee.models.study_model import KneeStudyModel
from knee.optimizers.gradients import ClipStrategy, GradientNoiseInjector
from knee.optimizers.lookahead import Lookahead
from knee.optimizers.param_groups_scheduler import (
  WarmupCosineScheduler,
  build_param_groups,
)


class KneeStudyLitModule(pl.LightningModule):
  """Lightning wrapper implementing training and validation logic.

  Args:
      model_cfg: Encoder/aggregator/metadata/head specification.
      loss_cfg: Supervised criterion spec + distillation settings.
      optimizer_cfg: Optimizer/scheduler specs and LR scaling.
      train_cfg: Fold/label policy, gradient control, curriculum.
      augment_cfg: Augment section; ``diff_ops`` enables GPU augmentation.
  """

  def __init__(
    self,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    optimizer_cfg: OptimizerConfig,
    train_cfg: TrainConfig,
    augment_cfg: AugmentConfig | None = None,
  ) -> None:
    """Build the composite model and every injectable training helper.

    Args:
        model_cfg: Validated model section.
        loss_cfg: Validated loss section.
        optimizer_cfg: Validated optimizer section.
        train_cfg: Validated train section.
        augment_cfg: Validated augment section (optional for back-compat).
    """
    super().__init__()
    self.save_hyperparameters()
    self.model = KneeStudyModel(
      encoder=instantiate(model_cfg.encoder),
      aggregator=instantiate(model_cfg.aggregator),
      metadata_encoder=instantiate(model_cfg.metadata_encoder),
      n_targets=model_cfg.n_targets,
      head_dropout=model_cfg.head_dropout,
    )
    self.criterion = instantiate(loss_cfg.criterion)
    self.distill_criterion = SoftBCEWithLogits(
      temperature=loss_cfg.distill_temperature
    )
    self.distill_weight = loss_cfg.distill_weight
    self.optimizer_cfg = optimizer_cfg
    self.train_cfg = train_cfg

    # --- optimization enhancements (all default-off) ------------------ #
    self.automatic_optimization = optimizer_cfg.backbone_optimizer is None
    self.grad_clip: ClipStrategy | None = (
      instantiate(train_cfg.grad_clip)
      if train_cfg.grad_clip is not None
      else None
    )
    self.grad_noise: GradientNoiseInjector | None = (
      GradientNoiseInjector(eta=train_cfg.grad_noise)
      if train_cfg.grad_noise > 0
      else None
    )
    self.curriculum: CurriculumController | None = (
      instantiate(train_cfg.curriculum)
      if train_cfg.curriculum is not None
      else None
    )
    ops = augment_cfg.diff_ops if augment_cfg is not None else []
    self.diff_augment: DiffAugment | None = (
      DiffAugment(ops=ops) if ops else None
    )
    self.dynamic_weighter: DynamicLossWeighter | None = (
      DynamicLossWeighter(initial_weights=(1.0, max(self.distill_weight, 1e-3)))
      if loss_cfg.dynamic_weights
      else None
    )
    self._term_sums = [0.0, 0.0]
    self._term_batches = 0
    self._term_weights = (
      1.0,
      self.distill_weight if self.dynamic_weighter is None else 1.0,
    )

    self._scheduler: WarmupCosineScheduler | None = None
    self._val_probs: list[np.ndarray] = []
    self._val_targets: list[np.ndarray] = []
    self._val_gold: list[np.ndarray] = []

  # ------------------------------------------------------------------ #
  def forward(
    self,
    images: torch.Tensor,
    meta: torch.Tensor,
    series_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Predict per-study logits.

    Args:
        images: Batched series stacks ``(B, S, C, H, W)``.
        meta: Series metadata ``(B, S, 5)``.
        series_mask: Optional validity mask ``(B, S)``.

    Returns:
        Raw logits of shape ``(B, n_targets)``.
    """
    return self.model(images, meta, series_mask)

  def _distill_lambda(self) -> float:
    """Ramp the distillation weight linearly over the first half of training.

    Returns:
        Current effective distillation weight in ``[0, distill_weight]``.
    """
    total = max(1, int(self.trainer.max_epochs))
    ramp_point = max(1, total // 2)
    progress = min(self.current_epoch / ramp_point, 1.0)
    return self.distill_weight * progress

  def _compute_terms(
    self, batch: dict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the supervised and distillation term scalars for a batch.

    Args:
        batch: Collated batch dict from the DataModule.

    Returns:
        Tuple ``(sup_term, distill_term, total)``; ``total`` already
        applies DWA weights or the epoch ramp.
    """
    images = batch['images']
    if self.diff_augment is not None and self.training:
      images = self.diff_augment(images)
    logits = self(images, batch['meta'], batch['series_mask'])

    weights = batch['sample_weight'].clamp_min(1e-6)
    if self.curriculum is not None:
      floor = self.curriculum.weight_floor(int(self.current_epoch))
      weights = weights * (weights >= floor).float()
    sup_per_sample = self.criterion(logits, batch['y_hard'], reduction='none')
    sup_term = (sup_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)

    weak_rows = (~batch['is_gold'].bool()) & (weights > 1e-5)
    distill_term = logits.new_zeros(())
    if weak_rows.any():
      d_per_sample = self.distill_criterion(
        logits, batch['y_soft'], reduction='none'
      )
      w_weak = weights[weak_rows]
      distill_term = (d_per_sample[weak_rows] * w_weak).sum() / w_weak.sum()

    if self.dynamic_weighter is not None:
      w_sup, w_dis = self._term_weights
      total = w_sup * sup_term + w_dis * distill_term
    else:
      total = sup_term + self._distill_lambda() * distill_term
    self._term_sums[0] += float(sup_term.detach())
    self._term_sums[1] += float(distill_term.detach())
    self._term_batches += 1
    return sup_term, distill_term, total

  # ------------------------------------------------------------------ #
  def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
    """Compute the combined objective under auto or manual optimization.

    Args:
        batch: Collated batch dict.
        batch_idx: Index within the epoch (drives manual accumulation).

    Returns:
        Scalar loss (consumed by Lightning in automatic mode).
    """
    del batch_idx
    _, _, loss = self._compute_terms(batch)
    self.log(
      'train_loss',
      loss,
      on_step=True,
      on_epoch=True,
      prog_bar=True,
      sync_dist=True,
    )
    if self.automatic_optimization:
      return loss
    # --- manual optimization (dual optimizers) ------------------------ #
    opts = self.optimizers()
    opts = opts if isinstance(opts, (list, tuple)) else [opts]
    accum = max(1, int(self.trainer.accumulate_grad_batches))
    self.manual_backward(loss / accum)
    if not self.trainer.should_accumulate:
      self._apply_gradient_control()
      for opt in opts:
        opt.step()
      for opt in opts:
        opt.zero_grad(set_to_none=True)
    return loss

  def _apply_gradient_control(self) -> None:
    """Clip gradients via the configured strategy and inject noise.

    Shared by automatic (via hooks) and manual optimization paths; when
    no strategy is configured the Trainer-level ``gradient_clip_val``
    remains authoritative and nothing runs here.
    """
    params = [p for p in self.parameters() if p.requires_grad]
    if self.grad_clip is not None:
      metrics = self.grad_clip.clip(params)
      self.log_dict(
        {f'grad/{k}': v for k, v in metrics.items()},
        on_step=True,
        on_epoch=False,
        prog_bar=False,
      )
    if self.grad_noise is not None:
      self.grad_noise.inject(params)

  def configure_gradient_clipping(
    self, optimizer, gradient_clip_val=None, gradient_clip_threshold=None
  ) -> None:
    """Route automatic-mode clipping through the configured strategy.

    Args:
        optimizer: Optimizer about to step.
        gradient_clip_val: Trainer-level norm ceiling (fallback).
        gradient_clip_threshold: Unused PL hook parameter.
    """
    if self.grad_clip is None:
      super().configure_gradient_clipping(
        optimizer, gradient_clip_val, gradient_clip_threshold
      )
      return
    self._apply_gradient_control()

  def on_before_optimizer_step(self, optimizer) -> None:
    """Inject decaying gradient noise in automatic mode only.

    Args:
        optimizer: Optimizer about to step.
    """
    del optimizer
    if self.automatic_optimization and self.grad_noise is not None:
      self.grad_noise.inject([p for p in self.parameters() if p.requires_grad])

  def on_train_epoch_start(self) -> None:
    """Advance the warmup-cosine schedule and reset DWA accumulators."""
    if self._scheduler is not None:
      self._scheduler.step(self.current_epoch + 0.5)
    self._term_sums = [0.0, 0.0]
    self._term_batches = 0

  def on_train_epoch_end(self) -> None:
    """Update DWA weights from this epoch's mean term losses."""
    if self.dynamic_weighter is None or self._term_batches == 0:
      return
    means = (
      self._term_sums[0] / self._term_batches,
      self._term_sums[1] / self._term_batches,
    )
    self._term_weights = self.dynamic_weighter.update(means)
    self.log('loss_weight_sup', self._term_weights[0], sync_dist=True)
    self.log('loss_weight_distill', self._term_weights[1], sync_dist=True)

  # ------------------------------------------------------------------ #
  def validation_step(self, batch: dict, batch_idx: int) -> None:
    """Accumulate predictions for pooled epoch-level AUC.

    Args:
        batch: Same structure as the training batch.
        batch_idx: Index within the validation loop (unused).
    """
    del batch_idx
    logits = self(batch['images'], batch['meta'], batch['series_mask'])
    self._val_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    self._val_targets.append(batch['y_soft'].detach().cpu().numpy())
    self._val_gold.append(batch['is_gold'].detach().cpu().numpy())

  def on_validation_epoch_end(self) -> None:
    """Log pooled macro-AUC plus the gold-only lens when available."""
    if not self._val_probs:
      return
    probs = np.concatenate(self._val_probs)
    targets = np.concatenate(self._val_targets)
    gold = np.concatenate(self._val_gold).astype(bool)
    auc_all, _, skipped = macro_auc(targets > 0.5, probs)
    self.log('val_macro_auc', auc_all, prog_bar=True, sync_dist=True)
    self.log('val_auc_skipped_classes', float(skipped), sync_dist=True)
    if gold.any() and not gold.all():
      auc_gold, _, _ = macro_auc(targets[gold] > 0.5, probs[gold])
      self.log('val_macro_auc_gold', auc_gold, sync_dist=True)
    self._val_probs.clear()
    self._val_targets.clear()
    self._val_gold.clear()

  # ------------------------------------------------------------------ #
  def _wrap(self, optimizer):
    """Apply Lookahead wrapping when configured.

    Args:
        optimizer: Built optimizer instance.

    Returns:
        Possibly Lookahead-wrapped optimizer.
    """
    spec = self.optimizer_cfg.lookahead
    return (
      Lookahead(optimizer, **spec.params) if spec is not None else optimizer
    )

  def _build_single_optimizer(self):
    """Construct the classic differential-LR AdamW over all params.

    Returns:
        Configured optimizer (possibly Lookahead-wrapped).
    """
    spec = self.optimizer_cfg.optimizer
    params = dict(spec.params)
    lr = float(params.pop('lr'))
    weight_decay = float(params.pop('weight_decay', 1e-2))
    groups = build_param_groups(
      self.model,
      base_lr=lr,
      weight_decay=weight_decay,
      backbone_lr_scale=self.optimizer_cfg.backbone_lr_scale,
    )
    return self._wrap(resolve_target(spec.target)(groups, **params))

  def _build_dual_optimizers(self):
    """Construct separate backbone/head optimizers from their specs.

    Returns:
        List [backbone_optimizer, head_optimizer].
    """
    enc = [
      p
      for n, p in self.model.named_parameters()
      if n.startswith('encoder.') and p.requires_grad
    ]
    head = [
      p
      for n, p in self.model.named_parameters()
      if not n.startswith('encoder.') and p.requires_grad
    ]
    built = []
    for spec, group_params in (
      (self.optimizer_cfg.backbone_optimizer, enc),
      (self.optimizer_cfg.optimizer, head),
    ):
      params = dict(spec.params)
      lr = float(params.pop('lr'))
      wd = float(params.pop('weight_decay', 1e-2))
      opt = resolve_target(spec.target)(
        [{'params': group_params, 'lr': lr, 'weight_decay': wd}], **params
      )
      built.append(self._wrap(opt))
    return built

  def configure_optimizers(self):
    """Construct optimizer(s) and attach the warmup-cosine scheduler.

    Returns:
        ``{'optimizer': opt}`` in automatic mode or
        ``{'optimizer': [backbone, head]}`` in manual (dual) mode.
    """
    if self.optimizer_cfg.backbone_optimizer is not None:
      optimizer = self._build_dual_optimizers()
    else:
      optimizer = self._build_single_optimizer()
    sched_spec = self.optimizer_cfg.scheduler
    if sched_spec is not None:
      self._scheduler = resolve_target(sched_spec.target)(
        optimizer[0] if isinstance(optimizer, list) else optimizer,
        **dict(sched_spec.params),
      )
    return {'optimizer': optimizer}

  # ------------------------------------------------------------------ #
  @torch.no_grad()
  def predict_logits(
    self, images: torch.Tensor, meta: torch.Tensor, series_mask: torch.Tensor
  ) -> torch.Tensor:
    """Inference helper returning sigmoid probabilities.

    Args:
        images: Batched series stacks.
        meta: Series metadata tensor.
        series_mask: Validity mask.

    Returns:
        Probabilities of shape ``(B, n_targets)``.
    """
    return torch.sigmoid(self(images, meta, series_mask))
