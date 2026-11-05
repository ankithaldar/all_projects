#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline for monitoring / drift (Phase 3).

Wiring: ``compute_drift`` -> ``build_monitoring_report`` ->
``evaluate_alerts``.

Canonical catalog datasets (configured in ``conf/``, not here):
- ``reference_data`` (in): baseline window used for drift.
- ``current_data`` (in): current window compared to the
  reference.
- ``reference_predictions`` (in): baseline predictions.
- ``current_predictions`` (in): current predictions.
- ``current_metrics`` (in): current performance metrics.
- ``alerts`` (out): alert list from ``evaluate_alerts``.

Params (``params:monitoring_params``, a dict). See
``pipelines.monitoring.nodes`` for the full key list; the
ones that matter here are ``bins`` (PSI bins, default 10),
``epsilon`` (PSI smoothing), ``min_rows`` (small-input
threshold), ``baseline_metrics`` (performance baseline),
and ``thresholds`` (per-metric ``{'warning', 'critical'}``
alert thresholds by severity, e.g. psi 0.1/0.25).

Back-compat defaults: when ``create_pipeline`` is called
without ``monitoring_params``, the canonical dataset names
above are wired. When supplied, its ``reference_dataset`` /
``current_dataset`` / ``output_dataset`` / ``alerts_dataset``
entries override the first node inputs / final output.
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.monitoring.nodes import (
  build_monitoring_report,
  compute_drift,
  evaluate_alerts,
)


def create_pipeline(monitoring_params=None, **kwargs):
  '''Create the monitoring / drift Kedro pipeline.

  Args:
    monitoring_params: Optional dict; its dataset-name
      entries wire the first/last node I/O. Defaults to the
      canonical ``reference_data`` / ``current_data`` /
      ``alerts`` names when omitted.
    **kwargs: Forwarded to Pipeline (e.g. tags, namespace).

  Returns:
    Kedro Pipeline: compute_drift -> build_monitoring_report
    -> evaluate_alerts.
  '''
  params = monitoring_params or {}
  reference_dataset = params.get('reference_dataset') or 'reference_data'
  current_dataset = params.get('current_dataset') or 'current_data'
  alerts_dataset = (params.get('alerts_dataset')
                    or params.get('output_dataset') or 'alerts')
  return Pipeline([
    node(func=compute_drift,
         inputs=[reference_dataset, current_dataset,
                 'params:monitoring_params'],
         outputs='drift_report',
         name='compute_drift'),
    node(func=build_monitoring_report,
         inputs=['drift_report', 'reference_predictions',
                 'current_predictions', 'current_metrics',
                 'params:monitoring_params'],
         outputs='monitoring_report',
         name='build_monitoring_report'),
    node(func=evaluate_alerts,
         inputs=['monitoring_report', 'params:monitoring_params'],
         outputs=alerts_dataset,
         name='evaluate_alerts'),
  ], **kwargs)
