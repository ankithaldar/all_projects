#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The registry of the three validation metrics, for the Keras backend.

The engine selects its optional metrics by name, so this module is the
dispatch table for them: ``lraps``, ``f1score_clip`` and
``f1score_frame`` are the three names the shipped configs use.

The metrics are epoch-wise accumulators: ``reset_state()`` at the start
of the epoch, ``update_state(...)`` per batch, ``result()`` at the end.
'''

from birdcall.metrics.tf_metric_f1score_clip import SedF1ScoreClip
from birdcall.metrics.tf_metric_f1score_frame import SedF1ScoreFrame
from birdcall.metrics.tf_metric_lraps import SedLRAPS

METRIC_REGISTRY: dict[str, type] = {
    'lraps': SedLRAPS,
    'f1score_clip': SedF1ScoreClip,
    'f1score_frame': SedF1ScoreFrame,
}


def build_metric(name: str, **kwargs):
  '''Builds the metric of the given name.

  Args:
    name: One of the keys of ``METRIC_REGISTRY``.
    **kwargs: The arguments of the metric, for example ``threshold``.

  Returns:
    The metric.

  Raises:
    ValueError: If the name is not one of the registered metrics.
  '''
  if name not in METRIC_REGISTRY:
    raise ValueError(
        f'Unknown metric {name!r}. '
        f'Valid names: {sorted(METRIC_REGISTRY)}')
  return METRIC_REGISTRY[name](**kwargs)
