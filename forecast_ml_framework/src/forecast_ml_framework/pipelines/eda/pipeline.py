#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline wiring for exploratory data analysis.

Final parameters.yml naming (documented for catalog wiring):
- ``params:eda_params`` -> eda params dict (see
  ``pipelines.eda.nodes`` module docstring for keys)
- ``variables`` (catalog) -> variables dict with target /
  feature columns
- ``raw_data`` (catalog) -> eda_summary df input
- ``eda_summary`` output of eda_summary
- ``eda_report`` output of eda_report (summary + findings +
  recommendations); also receives ``raw_data`` and
  ``variables`` so :func:`detect_leakage` findings are
  wired into the report
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.eda.nodes import (
  eda_report,
  eda_summary,
)


def create_pipeline(**kwargs):
  '''Create Kedro pipeline with summary and report nodes.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. namespace).

  Returns:
    Kedro Pipeline with two nodes: eda_summary ->
    eda_report.
  '''
  summary_node = node(
    func=eda_summary,
    inputs=['raw_data', 'params:eda_params', 'variables'],
    outputs='eda_summary',
    name='eda_summary',
  )
  report_node = node(
    func=eda_report,
    inputs=['eda_summary', 'params:eda_params', 'raw_data',
            'variables'],
    outputs='eda_report',
    name='eda_report',
  )
  return Pipeline([summary_node, report_node], **kwargs)
