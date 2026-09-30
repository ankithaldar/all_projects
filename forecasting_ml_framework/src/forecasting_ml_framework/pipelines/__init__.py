#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Orchestration layer: the pipeline registry and the pipeline factories.

Nothing in this package imports the node layer's *implementations* directly at
module scope, so the graph's shape can be inspected without a Spark session.
"""

from forecasting_ml_framework.pipelines.registry import (
  PipelineTopology,
  describe_pipelines,
  register_pipelines,
)

__all__ = ['PipelineTopology', 'describe_pipelines', 'register_pipelines']
