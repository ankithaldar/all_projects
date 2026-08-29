#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sampler construction from YAML specifications.

Mirrors ``augmentations.factory``: experiment YAMLs declare WHAT to
sample with (class_path + init_params); runtime state (study ids,
labels frame, target order) is injected here at loader-build time,
because weights cannot exist before the fold split is known.
"""

from __future__ import annotations

import importlib

from knee.datasets.study_dataset import StudyDataset
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_CLASS_PATH = 'knee.samplers.weighted.StudyWeightedRandomSampler'


def _resolve_class(class_path: str) -> type:
  """Import a sampler class from its dotted path.

  Args:
      class_path: Fully qualified path such as
          ``knee.samplers.weighted.StudyWeightedRandomSampler``.

  Returns:
      The imported class object.

  Raises:
      ImportError: If the module or attribute cannot be resolved.
  """
  module_path, _, attr = class_path.rpartition('.')
  if not module_path:
    raise ImportError(f'Invalid class_path: {class_path!r}')
  try:
    module = importlib.import_module(module_path)
    return getattr(module, attr)
  except (ImportError, AttributeError) as exc:
    raise ImportError(f'Cannot resolve {class_path!r}: {exc}') from exc


def build_train_sampler(spec: dict | None, dataset: StudyDataset):
  """Build the training sampler for one fold, or None for uniform.

  Args:
      spec: ``train_sampler`` section (class_path/init_params) or
          None/false to keep uniform shuffling.
      dataset: Attached training dataset supplying study ids, labels
          frame, and the canonical target order.

  Returns:
      A torch Sampler instance, or None when the spec is disabled or
      the dataset carries no labels (inference-style splits).
  """
  if not spec or dataset.labels_df is None or not dataset.study_ids:
    _LOGGER.info(
      'train sampler disabled (spec=%s, labels=%s)',
      bool(spec),
      dataset.labels_df is not None,
    )
    return None
  class_path = str(spec.get('class_path') or DEFAULT_CLASS_PATH)
  sampler_cls = _resolve_class(class_path)
  if not isinstance(sampler_cls, type):
    raise ValueError(
      f'train_sampler class_path must name a class: {class_path!r}'
    )
  params = dict(spec.get('init_params') or {})
  _LOGGER.info(
    'train sampler: %s over %d studies', class_path, len(dataset.study_ids)
  )
  return sampler_cls(
    study_ids=list(dataset.study_ids),
    label_frame=dataset.labels_df,
    target_columns=list(dataset.target_columns),
    **params,
  )
