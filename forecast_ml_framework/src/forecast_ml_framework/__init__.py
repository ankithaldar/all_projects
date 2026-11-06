#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecast ML framework package.

Config-driven Kedro ML framework for forecasting, console-first,
at Phase 4: the pipeline registry, IaC, and monitoring phases
have landed. Twelve pipelines are registered (see
``pipeline_registry`` and ``pipelines``), parameters live in
the ``conf/base/parameters/*.yml`` buckets, and the sole
console entry point is ``forecast-ml``.
'''

__version__ = '0.1.0'
__all__ = ['__version__']
