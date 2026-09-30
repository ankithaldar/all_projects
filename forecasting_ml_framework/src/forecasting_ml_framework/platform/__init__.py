#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Platform layer: process-level compute resources and the Spark session lifecycle.

Nothing in this package may import a higher layer. The dependency is strictly
one-way.
'''

from forecasting_ml_framework.platform.environment import (
  RunEnvironment,
  as_flag,
  resolve_artefact_root,
  resolve_environment,
)
from forecasting_ml_framework.platform.spark import get_spark_session, refresh_access_token

__all__ = [
  'RunEnvironment',
  'as_flag',
  'get_spark_session',
  'refresh_access_token',
  'resolve_artefact_root',
  'resolve_environment',
]
