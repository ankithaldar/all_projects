#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Centralised, framework-wide constants.

Every string that appears in more than one module lives here.

A convention spelled out separately in the node layer, the dataset adapters and
the SQL templates is three places to change when it changes, and nothing catches
a miss. Centralising them makes a convention change a one-line edit, and puts
the whole surface of a convention in view at once -- which is what makes it
reviewable.
"""

from __future__ import annotations

import enum

# --------------------------------------------------------------------------- #
# Column-role keys used by the ``parameters`` and ``globals`` documents
# --------------------------------------------------------------------------- #
ROLE_NUMERICAL = 'NUM_COL'
ROLE_CATEGORICAL = 'CAT_COL'
ROLE_INDICATOR = 'IND_COL'
ROLE_SEQUENCE = 'SEQ_COL'
ROLE_IDENTIFIER = 'ID_COL'

COLUMN_ROLE_KEYS = (
  ROLE_NUMERICAL,
  ROLE_CATEGORICAL,
  ROLE_INDICATOR,
  ROLE_SEQUENCE,
  ROLE_IDENTIFIER,
)

#: Roles that contribute scalar, estimator-consumable features. The
#: concatenation order of these three keys *is* the model's feature-vector order
#: for the life of the artefact, so it must not be reordered casually.
FEATURE_ROLE_KEYS = (ROLE_NUMERICAL, ROLE_CATEGORICAL, ROLE_INDICATOR)

# --------------------------------------------------------------------------- #
# Model identity and provenance
# --------------------------------------------------------------------------- #
MODEL_KEY = 'model_key'
RUN_DATE = 'run_date'
TRAIN_RUN_DATE = 'train_run_dt'
SCORE_RUN_DATE = 'score_run_dt'
REPORT_DATE = 'rpt_dt'
INSERT_TIMESTAMP = 'insert_ts'
LAST_UPDATE_DATE = 'last_upd_dt'

#: Column names that, if any of them is present among the identifiers, satisfy
#: the report-date convention. A bare literal in both ``fit_registry`` and
#: ``apply_registry`` would be two places to change and nothing to catch a miss.
CANONICAL_DATE_COLUMNS = ('rpt_dt', 'report_period', 'create_date')

#: Dotted module that aggregates every built-in preprocessing routine class, and
#: therefore the first candidate the reflective resolver searches. It is derived
#: from this module's own package rather than written as a literal, so the
#: resolver cannot name a prefix that disagrees with where it is actually
#: installed.
_ROOT_PACKAGE = __package__ or 'forecasting_ml_framework'
ROUTINES_MODULE = _ROOT_PACKAGE + '.preprocessing.routines'

# --------------------------------------------------------------------------- #
# Score-table contract (the enterprise downstream contract)
# --------------------------------------------------------------------------- #
SCORE_VALUE = 'score_value'
ORIG_SCORE_VALUE = 'orig_score_value'
SCORE_DECILE = 'score_decile'
SCORE_CENTILE = 'score_centile'
CALIB_DECILE = 'calib_decile'
CALIB_CENTILE = 'calib_centile'
HIGH_SCORE_IND = 'high_score_ind'
HIGH_SCORE_YES = 'Y'
HIGH_SCORE_NO = 'N'

#: Number of bands used for the decile / centile columns.
DECILE_COUNT = 10
CENTILE_COUNT = 100

#: The decile (1-indexed) that the promotion gate defaults to.
TOP_DECILE = 10

#: The metric the regression band profile is computed on. Regression has no
#: positive class to lift over, so the decile error profile is the analogue of a
#: lift table and needs an explicit, banded error metric to be well defined.
#: Declared here rather than spelled at the call site so the "which band, which
#: metric" decision is a single reviewable value instead of a literal repeated by
#: the node that builds the profile and the node that reads it.
DEFAULT_REGRESSION_BAND_METRIC = 'RMSE_DECILE_10'

#: ``high_score_ind`` requires both an above-threshold score *and* membership of
#: the top half of the score distribution.
#:
#: The band boundary is deliberately NOT a constant here. It was `5`, which is
#: wrong twice over: bands run 1..10 so `>= 5` admits deciles 5-10 (the top 60
#: percent, not the top half), and a coarse score can be banded into fewer than
#: five bands, so the flag then matched nothing at all. ``scoring._top_half_floor``
#: derives the boundary from the banding actually produced.

DEFAULT_PREDICTION_COLUMN = 'prediction'
DEFAULT_PREDICTION_PROBABILITY_COLUMN = 'prediction_probability'
CALIBRATION_PREFIX = 'calib_'

# --------------------------------------------------------------------------- #
# Data-slice tokens injected by the custom run command
# --------------------------------------------------------------------------- #
SLICE_TRAIN = 'train_data'
SLICE_VALID = 'valid_data'
SLICE_TEST = 'test_data'
SLICE_CALIB_TRAIN = 'calib_train_data'
SLICE_CALIB_VALID = 'calib_valid_data'
SLICE_CALIB_TEST = 'calib_test_data'

#: Injected into the parameter dictionary immediately before session creation so
#: that a single node function can serve both calibrated and uncalibrated
#: evaluation. The ``calib_`` prefix is the string dispatch mechanism.
SLICE_TOKEN_MAP = {
  'train': SLICE_TRAIN,
  'valid': SLICE_VALID,
  'test': SLICE_TEST,
  'calib_train': SLICE_CALIB_TRAIN,
  'calib_valid': SLICE_CALIB_VALID,
  'calib_test': SLICE_CALIB_TEST,
}

# --------------------------------------------------------------------------- #
# Pipeline registry environment variables
# --------------------------------------------------------------------------- #
ENV_SUFFIX = 'suffix'
ENV_SPLIT = 'split'
ENV_IS_REGRESSION = 'is_regression'
ENV_RUN_MODE = 'run_mode'
ENV_TRAIN_MODE = 'train_mode'

# --------------------------------------------------------------------------- #
# Environment flags
#
# Distinct from the four mode flags above. A mode flag says what a run computes;
# an environment flag says where it runs. Keeping the two categories apart is what
# stops a change of environment from altering pipeline semantics -- which is
# exactly what makes a local run a valid correctness test of a production one.
# --------------------------------------------------------------------------- #

#: The run executes inside a managed workspace rather than on the managed
#: cluster. Governs how the run obtains credentials; it does not select a
#: persistence back end.
ENV_IS_HOSTED = 'is_hosted_env'

#: The legacy spelling of :data:`ENV_IS_HOSTED`, kept for one compatibility
#: window. It is honoured when the current name is absent, and its use is logged.
#: A flag named for the product that first required it cannot be reasoned about
#: or transferred to a different workspace, so the rename is the point.
ENV_IS_HOSTED_LEGACY = 'is_hosted_platform'

#: The run executes on a developer's own machine against no cloud resources.
#: Governs which persistence back end every data transaction uses, and it is the
#: flag that binds the framework to the embedded store.
ENV_IS_LOCAL = 'is_local'

#: An explicit back-end override, for a deployment that genuinely needs one other
#: than the one ``is_local`` implies. The implicit rule remains the default, so
#: ordinary configuration needs only the flag.
ENV_DB_BACKEND = 'db_backend'

#: Back-end identifiers. ``distributed`` is the managed, serverless columnar
#: warehouse; ``embedded`` is the single-file relational store.
BACKEND_DISTRIBUTED = 'distributed'
BACKEND_EMBEDDED = 'embedded'
DB_BACKENDS = (BACKEND_DISTRIBUTED, BACKEND_EMBEDDED)

#: The back end that binds artefacts when ``is_local`` is set.
LOCAL_ARTEFACT_ROOT = '/tmp/forecasting_ml_framework'

PIPELINE_FE_TRAINING = 'fe_training'
PIPELINE_MODEL_TRAINING = 'model_training'
PIPELINE_FE_SCORING = 'fe_scoring'
PIPELINE_MODEL_SCORING = 'model_scoring'
PIPELINE_DEFAULT = '__default__'

DEFAULT_SUFFIX = 'params'

# --------------------------------------------------------------------------- #
# Label-derivation windows (govern the statistical validity of every model)
# --------------------------------------------------------------------------- #
#: How long a subscriber must remain active before they are declared retained.
OUTCOME_WINDOW_DAYS = 58
#: How far past the training date outcomes must be observed for a label to be
#: trustworthy. Outcome window plus a 30-day settlement lag.
MATURITY_WINDOW_DAYS = 88


class RunMode(str, enum.Enum):
  """Operational mode of a scoring run."""

  PROD = 'prod'
  EVAL = 'eval'


class TrainMode(str, enum.Enum):
  """Champion/challenger promotion mode of a training run."""

  TRAIN_ONLY = 'train_only'
  UPDATE_METADATA = 'train_updateMETADATA'
  COMPARE_AND_PROMOTE = 'train_compareMODEL_updateMETADATA'
  COMPARE_ONLY = 'train_compare_update'


class ModelType(str, enum.Enum):
  """Modelling tier selected by ``global_params.model_type``."""

  SKLEARN = 'sklearn'
  PYSPARK = 'pyspark'
  TRANSFORMER = 'transformer'


#: ``status_ind`` values of the bitemporal model registry.
STATUS_CURRENT = 'c'
STATUS_SUPERSEDED = 'o'

#: Human-readable help for the promotion gate metric grammar.
#:
#: The name list is filled in by :func:`render_metric_grammar_help` from
#: ``METRIC_REGISTRY`` rather than written out here. This text is what an operator
#: reads when a metric name is *rejected*, so a hand-maintained copy of the
#: vocabulary is a copy that will drift: ``KSI`` was listed here for a long time
#: while the parser refused it, which made the error message recommend the name
#: it was complaining about. Deriving the list makes that unrepresentable.
METRIC_GRAMMAR_PREAMBLE = """
Supported metric names:
{metric_names}
Banded forms select a single band of the score distribution:
  LIFT_DECILE_10      -> lift within the top decile
  LIFT_CENTILE_100    -> lift within the top centile
An unbanded name is evaluated over the whole population.
""".strip()


def render_metric_grammar_help(metric_names: list[str]) -> str:
  '''Render the promotion-metric grammar help from the live metric registry.

  Args:
    metric_names: The registry's names, in the order they should be listed.

  Returns:
    The help text an operator sees when a metric name is rejected.
  '''
  regression_prefixes = ('MAE', 'MAPE', 'MSE', 'MSLE', 'RMSE', 'RMSLE', 'R2', 'ADJ')
  classification = [name for name in metric_names if not name.startswith(regression_prefixes)]
  regression = [name for name in metric_names if name.startswith(regression_prefixes)]
  return METRIC_GRAMMAR_PREAMBLE.format(
    metric_names=(
      f"  classification: {', '.join(repr(name) for name in classification)}\n"
      f"  regression:     {', '.join(repr(name) for name in regression)}"
    )
  )
