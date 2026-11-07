#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline for retraining triggers (Phase 3).

Wiring: ``should_retrain`` -> ``build_retrain_plan``.

Canonical catalog datasets (configured in ``conf/``, not
here):
- ``monitoring_report`` (in): monitoring / drift report.
- ``last_train_info`` (in): last training timestamp and
  row count.
- ``retrain_decision`` (intermediate): output of
  :func:`should_retrain`.
- ``retrain_plan`` (out): plan-only dict from
  :func:`build_retrain_plan`; it never executes Kedro.

Params (``params:retraining_params``, a dict). Expected
keys for conf alignment:
  drift_alert_threshold: drift fraction/value at/above this
    fires (default 0.2, fraction/value semantics; explicit
    thresholds >= 1 keep legacy count semantics; ``null``
    disables).
  performance_decay_threshold: decay above this fires
    (default 0.05; ``null`` disables).
  schedule_days: days since ``last_train_at`` at/above
    this fire (default 7; ``null`` disables).
  data_volume_growth_threshold: row growth at/above this
    fires (default 0.20; ``null`` disables).
  now: injected current time (datetime or ISO string) for
    deterministic schedule checks; omit in production.
  env: target env name recorded in the plan (default
    ``'local'``).
  pipeline_name: pipeline named in the plan (default
    ``'modeling'``).
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.retraining.nodes import (
  build_retrain_plan,
  should_retrain,
)


def create_pipeline(**kwargs):
  '''Create the retraining trigger Kedro pipeline.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. tags, namespace).

  Returns:
    Kedro Pipeline: should_retrain -> build_retrain_plan.
  '''
  return Pipeline([
    node(func=should_retrain,
         inputs=['monitoring_report', 'last_train_info',
                 'params:retraining_params'],
         outputs='retrain_decision',
         name='should_retrain'),
    node(func=build_retrain_plan,
         inputs=['retrain_decision',
                 'params:retraining_params'],
         outputs='retrain_plan',
         name='build_retrain_plan'),
  ], **kwargs)
