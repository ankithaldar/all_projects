#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mid-training model-component swapping (dynamic architecture).

Declarative schedule of ``(epoch, attribute, ComponentSpec)`` swaps
applied by a Lightning callback. Swapped components attempt to inherit
old weights when shapes permit (``strict=False`` load), enabling
capacity upgrades (e.g. AttentionMIL -> TransformerAggregator) without
restarting the run.
"""

from __future__ import annotations

import lightning.pytorch as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from knee.config_params.loader import instantiate
from knee.helpers.logging_utils import get_logger


class ComponentSwapSpec(BaseModel):
  """One scheduled swap event.

  Attributes:
      epoch: Training epoch at which the swap fires.
      attribute: Attribute path on the LightningModule's ``model`` to
          replace (e.g. ``aggregator``).
      spec: New component specification.
  """

  model_config = ConfigDict(extra='forbid')
  epoch: int = Field(ge=0)
  attribute: str = Field(min_length=1)
  spec: dict

  @field_validator('attribute')
  @classmethod
  def _no_dunder(cls, v: str) -> str:
    """Reject attribute paths that are not plain identifiers.

    Args:
        v: Candidate attribute name.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is not a safe identifier.
    """
    if not v.isidentifier():
      raise ValueError('attribute must be a single identifier')
    return v


class ComponentSwapCallback(pl.Callback):
  """Swap model components at configured epochs.

  Args:
      schedule: List of raw dicts matching :class:`ComponentSwapSpec`.
  """

  def __init__(self, schedule: list[dict]) -> None:
    """Validate and store the swap schedule.

    Args:
        schedule: Raw spec dicts; validated eagerly so typos fail at
            config-load time rather than mid-training.
    """
    super().__init__()
    self.schedule = [ComponentSwapSpec.model_validate(s) for s in schedule]
    self._log = get_logger('knee.swap')

  def on_train_epoch_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Perform any swaps scheduled for the starting epoch.

    Args:
        trainer: Active trainer (device source for new modules).
        pl_module: Module whose ``model`` attributes may be replaced.

    Raises:
        AttributeError: If the target attribute does not exist.
    """
    epoch = trainer.current_epoch
    for event in [s for s in self.schedule if s.epoch == epoch]:
      root = pl_module.model
      old = getattr(root, event.attribute)
      new = instantiate(event.spec).to(pl_module.device)
      try:
        new.load_state_dict(old.state_dict(), strict=False)
        self._log.info(
          'epoch %d: %s swapped; inherited compatible weights',
          epoch,
          event.attribute,
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        self._log.warning(
          'epoch %d: %s swapped fresh (%s)', epoch, event.attribute, exc
        )
      setattr(root, event.attribute, new)
