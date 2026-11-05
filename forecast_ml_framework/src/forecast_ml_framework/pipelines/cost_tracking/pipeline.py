#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline wiring for cost / performance tracking.

Wiring: ``estimate_cost`` -> ``track_run_cost`` ->
``summarize_costs``.

Canonical catalog datasets (configured in ``conf/``, not here):

- ``run_info`` (in): one run's usage/metadata dict (cluster
  runtime hours, BigQuery bytes scanned, artifact GB, model,
  env, month). No cloud API is called; those numbers arrive
  here as plain dict data.
- ``params:cost_tracking_params`` (in): config-driven unit
  prices (see ``nodes`` module docstring for per-key docs).
- ``cost_estimate``: output of :func:`estimate_cost`.
- ``cost_record``: output of :func:`track_run_cost`, one
  record dict per run. Append ``cost_record`` values to a log
  (e.g. the ``cost_log`` catalog dataset) across runs.
- ``cost_log`` (in): list of cost record dicts accumulated
  across runs (empty list when nothing has been tracked yet).
- ``cost_summary`` (out): totals by model/env/month from
  :func:`summarize_costs`.
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.cost_tracking.nodes import (
  estimate_cost,
  summarize_costs,
  track_run_cost,
)


def create_pipeline(**kwargs):
  '''Create the cost tracking Kedro pipeline.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. namespace).

  Returns:
    Kedro Pipeline with three nodes:
    estimate_cost -> track_run_cost -> summarize_costs.
  '''
  estimate = node(
    func=estimate_cost,
    inputs=['run_info', 'params:cost_tracking_params'],
    outputs='cost_estimate',
    name='estimate_cost',
  )
  track = node(
    func=track_run_cost,
    inputs=['run_info', 'cost_estimate'],
    outputs='cost_record',
    name='track_run_cost',
  )
  summarize = node(
    func=summarize_costs,
    inputs=['cost_log'],
    outputs='cost_summary',
    name='summarize_costs',
  )
  return Pipeline([estimate, track, summarize], **kwargs)
