#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-target losses with unknown-label masking.

Targets carry the sentinel ``-1`` for uncertain/unmentioned pseudo-labels;
every loss below masks those elements so they contribute neither positive
nor negative gradient signal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

IGNORE_INDEX = -1.0


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with configurable per-class positive weighting."""

    def __init__(
        self,
        pos_weight_mode: str = 'frequency',
        pos_weight_fixed: list[float] | None = None,
        label_smoothing: float = 0.0,
        eps: float = 1e-4,
    ) -> None:
        """Configure the criterion.

        Args:
            pos_weight_mode: ``frequency`` derives weights per batch,
                ``fixed`` uses ``pos_weight_fixed``, ``none`` disables weighting.
            pos_weight_fixed: Explicit weights used by the ``fixed`` mode.
            label_smoothing: Fraction clamping hard {0, 1} targets inward.
            eps: Clamping epsilon applied symmetrically.
        """
        super().__init__()
        self.pos_weight_mode = pos_weight_mode
        self.pos_weight_fixed = pos_weight_fixed
        self.label_smoothing = label_smoothing
        self.eps = eps

    def _smooth(self, targets: Tensor) -> Tensor:
        """Clamp binary targets according to the smoothing fraction.

        Args:
            targets: Binary target tensor (already masked of sentinels).

        Returns:
            Smoothed copy of the targets.
        """
        if self.label_smoothing <= 0.0:
            return targets
        return targets.clamp(self.label_smoothing, 1.0 - self.label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the masked, weighted BCE over valid elements only.

        Args:
            logits: ``(batch, n_targets)`` raw network outputs.
            targets: ``(batch, n_targets)`` values in {-1, 0, 1}.

        Returns:
            Scalar loss averaged over unmasked elements.
        """
        mask = targets != IGNORE_INDEX
        safe_targets = targets[mask].float()
        smoothed = self._smooth(safe_targets)
        weight = None
        if self.pos_weight_mode == 'fixed' and self.pos_weight_fixed is not None:
            fixed = torch.as_tensor(self.pos_weight_fixed, device=logits.device)
            weight = fixed.expand_as(targets)[mask]
        elif self.pos_weight_mode == 'frequency':
            positives = smoothed.sum()
            negatives = smoothed.numel() - positives
            if positives > 0 and negatives > 0:
                weight = (negatives / positives).expand_as(smoothed)
        return nn.functional.binary_cross_entropy_with_logits(
            logits[mask], smoothed, pos_weight=weight, reduction='mean'
        )


class AsymmetricFocalLoss(nn.Module):
    """Focal loss with separate focusing for positive/negative samples.

    Reference formulation (multi-label variant):

    .. math::
        L = -(1-p)^{gamma_pos} log(p) * y
          - (1-pt)^{gamma_neg} log(pt) * clip(1-y, t, 1) * (1-y)

    where ``pt = clip(p, clip_value, 1)`` suppresses easy-negative gradients.
    """

    def __init__(
        self,
        gamma_pos: float = 2.0,
        gamma_neg: float = 1.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
    ) -> None:
        """Configure focusing exponents.

        Args:
            gamma_pos: Focusing exponent applied to positives.
            gamma_neg: Focusing exponent applied to negatives.
            clip: Probability floor for easy-negative suppression (0 disables).
            label_smoothing: Fraction clamping positive targets inward.
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute masked asymmetric focal loss.

        Args:
            logits: ``(batch, n_targets)`` raw outputs.
            targets: ``(batch, n_targets)`` values in {-1, 0, 1}.

        Returns:
            Scalar loss averaged over unmasked elements.
        """
        mask = targets != IGNORE_INDEX
        y = targets[mask].float().clamp(self.label_smoothing, 1.0)
        x = logits[mask]
        x_pos = x * y
        x_neg = x * (1.0 - y)
        if self.clip > 0:
            x_neg = x_neg.clamp(min=self.clip, max=1.0e4)
        loss_pos = -torch.pow(1.0 - torch.sigmoid(x_pos), self.gamma_pos) \
            * torch.log(torch.sigmoid(x_pos) + 1e-8) * y
        loss_neg = -torch.pow(1.0 - torch.sigmoid(x_neg), self.gamma_neg) \
            * torch.log(1.0 - torch.sigmoid(x_neg) + 1e-8) * (1.0 - y)
        losses = loss_pos + loss_neg
        count = mask.sum().clamp(min=1)
        return losses.sum() / count
