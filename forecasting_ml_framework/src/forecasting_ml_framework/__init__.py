#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forecasting Framework with ML.

A configuration-driven propensity and value-segment modelling framework.

Its defining property is that the unit of reuse is not a model but a *model
program*: a declarative description of which SQL builds the training frame and
derives the label, which feature transforms run in what order over which
columns, which estimator and which search to fit, which metric and decile-lift
tables to emit, and how the fitted model is promoted to "current" in the
enterprise model registry. A new production model is stood up by authoring
configuration, not by writing Python.

The package's layers, with a strict one-way dependency discipline — upper layers
depend on lower layers, and lower layers never import upper layers:

===============  =============================================================
Layer            Contents
===============  =============================================================
L6 orchestration  Scheduler DAGs, deployment scripts, egress SQL templates.
                 Knows pipeline names, parameter names and table names. Never
                 imports the framework; all coupling is by string.
L5 configuration  Parameters, globals, data catalog, Spark tuning, logging,
                 project settings, hooks, the config loader, the context.
L4 orchestration  Pipeline registry and the four pipeline factories.
L3 nodes         Feature-engineering nodes, modelling nodes, validation nodes.
L2 core          Metadata, the routine contract, the routine library, the
                 registry, the trainers, scoring, metrics, promotion, the
                 sequence tier.
L1 platform      Spark session lifecycle, Kedro runner utilities, the bootstrap
                 rewriter, the dataset adapters.
===============  =============================================================
"""

from __future__ import annotations

__version__ = '1.0.0'

__all__ = ['__version__']
