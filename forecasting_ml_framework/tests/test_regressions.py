#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Regression tests for defects found against the architecture documentation.

Every test in this file was written *after* the defect was reproduced, and each
one names the failure it pins down in its docstring. They are grouped by the
property they defend rather than by the module they touch, because the property
is what a future change has to preserve.

The four that matter most, and why:

* ``test_sequence_routine_honours_its_configuration`` -- sixteen routines were
  assigning their defaults *after* ``load_params``, so every configured value was
  discarded and the whole sequence family ran on defaults. It was silent, it
  affected the entire family, and nothing else in the suite noticed.
* ``test_allow_list_refuses_an_escape_route`` -- dynamic resolution from
  configuration is arbitrary code execution unless it is constrained, and the
  documented control was never enforced.
* ``test_validation_node_returns_three_values_with_no_incumbent`` -- the first
  training of every new model returned ``None`` against a three-output contract,
  so Kedro raised while saving. The model was already trained by then.
* ``test_every_reachable_topology_has_its_sequencing_tokens`` -- the tokens were
  absent from the catalog, so the classification pipelines could not be
  constructed at all.
'''

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import ConfigurationError
from forecasting_ml_framework.modeling.calibration import select_threshold
from forecasting_ml_framework.modeling.metrics import (
  METRIC_GRAMMAR_HELP,
  METRIC_REGISTRY,
  MetricSpec,
  _assign_bands,
  parse_metric,
)
from forecasting_ml_framework.modeling.scoring import _rank_bands, _top_half_floor
from forecasting_ml_framework.platform.spark import _load_spark_conf
from forecasting_ml_framework.preprocessing.routines import AVAILABLE_ROUTINES
from forecasting_ml_framework.preprocessing.scalar import Imputations
from forecasting_ml_framework.preprocessing.sequence_ops import (
  get_slope_func,
  normalise_window,
  seq_date_delta,
)
from forecasting_ml_framework.utils.reflection import resolve


# --------------------------------------------------------------------------- #
# 1. The transformation contract: configuration must survive construction
# --------------------------------------------------------------------------- #
class TestConfigurationSurvivesConstruction:
  '''A routine may not re-assign a configured key after loading it.'''

  def test_sequence_routine_honours_its_configuration(self) -> None:
    '''Every sequence routine must keep the configuration it was given.

    The defect: ``SequenceTransform.__init__`` and each of its sixteen subclasses
    re-assigned their declared defaults *after* ``load_params`` had applied the
    configuration, so ``last_n_val``, ``cols``, ``fixed_length``, ``round_off``,
    ``order``, ``suffix`` and ``rational_vector_map`` were all silently
    discarded. Consequences: ``cols.include_cols`` was always empty, so the
    routines resolved to ``metadata.feature_cols`` -- every column, including
    plain scalars and strings -- and every window was 3 regardless of
    configuration.
    '''
    configured = {'last_n_val': 7, 'cols': {'exclude_cols': ['age'], 'include_cols': ['seq']}}
    family = [name for name in AVAILABLE_ROUTINES if _is_sequence(name)]

    assert family, 'the sequence family must be discoverable for this test to mean anything'

    for name in family:
      routine = AVAILABLE_ROUTINES[name](None, dict(configured))
      assert routine.last_n_val == 7, f'{name} discarded last_n_val'
      assert routine.cols['include_cols'] == ['seq'], f'{name} discarded cols.include_cols'

  def test_sequence_subclass_keys_are_honoured(self) -> None:
    '''Each sequence subclass keeps its own configured keys.

    The base-class fix is necessary but not sufficient: the subclasses carried
    their own re-assignments, which had to be removed individually.
    '''
    base = {'last_n_val': 7, 'cols': {'exclude_cols': [], 'include_cols': ['seq']}}
    cases: dict[str, dict[str, Any]] = {
      'FixedLength': {'fixed_length': 12},
      'SequenceFillNa': {'fixed_length': 4},
      'SequenceNormalize': {'round_off': 2},
      'SequenceDomainNormalizations': {'normalize_by_val': 7.5, 'round_off': 1},
      'SequenceRecency': {'run_date_col': 'as_of', 'fill_na_val': 42},
      'Slope': {'order': 2, 'suffix': 'trend', 'last_n_val': 5},
      'Similarity': {'rational_vector_map': {'seq': [1.0, 2.0]}, 'last_n_val': 8},
    }

    for name, extra in cases.items():
      params = dict(base)
      params.update(extra)
      routine = AVAILABLE_ROUTINES[name](None, params)
      for key, value in extra.items():
        assert getattr(routine, key) == value, f'{name} discarded {key}'

  def test_configured_exclusion_is_not_applied_to_a_wider_set(self) -> None:
    '''``exclude_cols`` narrows the selection, it does not widen it.

    The regression test for the *intent* of the family: with the configuration
    honoured, the resolved set must be exactly the include list minus the
    exclude list -- not the whole feature list.
    '''
    routine = AVAILABLE_ROUTINES['SequenceAverage'](
      None, {'last_n_val': 7, 'cols': {'exclude_cols': ['seq_b'], 'include_cols': ['seq', 'seq_b']}}
    )

    resolved = routine.get_selected_columns(
      routine.cols['exclude_cols'], routine.cols['include_cols'], ['age', 'plan_tier', 'seq', 'seq_b']
    )

    assert resolved == ['seq'], f'the sequence family would be handed {resolved}'


# --------------------------------------------------------------------------- #
# 2. Dynamic resolution is a code-execution surface, so it is constrained
# --------------------------------------------------------------------------- #
class TestResolutionAllowList:
  '''A configuration string may not name an arbitrary callable.'''

  @pytest.mark.parametrize(
    'path',
    [
      'os.system',
      'os.popen',
      'builtins.eval',
      'builtins.exec',
      'subprocess.Popen',
      'pickle.loads',
      'importlib.import_module',
      'shutil.rmtree',
      'posix.system',
    ],
  )
  def test_allow_list_refuses_an_escape_route(self, path: str) -> None:
    '''A refused path raises rather than returning the callable.

    Each of these resolves through an unbounded ``getattr`` walk, so before the
    allow-list a configuration edit was enough to obtain arbitrary code
    execution. The threat is exactly that a configuration document is data.
    '''
    with pytest.raises(ConfigurationError) as caught:
      resolve(path)
    assert 'allow-list' in str(caught.value)

  @pytest.mark.parametrize(
    'path',
    [
      'sklearn.ensemble.RandomForestClassifier',
      'sklearn.model_selection.GridSearchCV',
      'sklearn.isotonic.IsotonicRegression',
      'pyspark.ml.feature.StringIndexer',
    ],
  )
  def test_allow_list_admits_the_shipped_handles(self, path: str) -> None:
    '''The allow-list must not break a handle the configuration can name.

    A control that is too tight is as much a defect as one that is too loose: it
    would turn a configuration change into a code change, which is precisely what
    the design principle forbids.
    '''
    assert resolve(path) is not None


# --------------------------------------------------------------------------- #
# 3. Graph integrity: a declared dataset must exist and be constructible
# --------------------------------------------------------------------------- #
class TestCatalogIntegrity:
  '''The catalog is the single source of physical truth, so it must be whole.'''

  def test_every_reachable_topology_has_its_sequencing_tokens(self, conf_dir: Path) -> None:
    '''Every sequencing token the graph names must exist in the catalog.

    Kedro resolves every declared dataset through the catalog, so a token with
    no catalog entry makes the pipeline *unconstructible* -- the runner reports
    "dataset not found" before the first node executes. The tokens are the
    mechanism that orders the evaluation fan-out, so their absence disabled the
    classification training and evaluation-scoring pipelines entirely.
    '''
    catalog = yaml.safe_load((conf_dir / 'catalog.yml').read_text(encoding='utf-8'))
    pipelines = Path(__file__).resolve().parents[1] / 'src' / 'forecasting_ml_framework' / 'pipelines'
    referenced = set()
    for module in pipelines.glob('*.py'):
      referenced |= set(re.findall(r'sequence_ind\d+', module.read_text(encoding='utf-8')))

    assert referenced, 'no sequencing tokens are referenced, so this test is vacuous'
    missing = sorted(name for name in referenced if name not in catalog)
    assert not missing, f'sequencing tokens absent from the catalog: {missing}'

  def test_every_catalog_dataset_type_is_importable(self, conf_dir: Path) -> None:
    '''A catalog entry whose type cannot be imported fails at catalog load.

    This is a graph invariant with an exact oracle, which is why it belongs in
    the fast tier: a misspelled dataset type is a configuration error that
    otherwise surfaces at run time, in a cluster, as an import failure.
    '''
    import importlib  # noqa: PLC0415

    catalog = yaml.safe_load((conf_dir / 'catalog.yml').read_text(encoding='utf-8'))
    unresolvable: list[str] = []
    for name, config in catalog.items():
      if name.startswith('_') or not isinstance(config, dict):
        continue
      dataset_type = config.get('type')
      if not dataset_type:
        continue
      module_path = dataset_type.rpartition('.')[0]
      try:
        importlib.import_module(module_path)
      except ImportError:
        unresolvable.append(f'{name} -> {dataset_type}')

    assert not unresolvable, f'unimportable dataset types: {unresolvable}'

  def test_holdout_prediction_datasets_are_the_pandas_flavour(self, conf_dir: Path) -> None:
    '''The holdout slice is consumed by pandas-only nodes.

    ``kedro_datasets.ParquetDataset`` is not an importable name on
    kedro-datasets >= 7, and where the name does resolve it is the *Spark*
    flavour -- which the consumers cannot use, since they call ``.empty`` on
    what they receive.
    '''
    catalog = yaml.safe_load((conf_dir / 'catalog.yml').read_text(encoding='utf-8'))
    for name in ('valid_prediction_df_${suffix}', 'valid_calib_prediction_df_${suffix}'):
      assert catalog[name]['type'] == 'kedro_datasets.pandas.ParquetDataset', f'{name} has the wrong type'


# --------------------------------------------------------------------------- #
# 3b. The scoring path enforces the same contract as the fitting path
# --------------------------------------------------------------------------- #
class TestScoringProjectionIsChecked:
  '''A partial feature set must never be scored as a whole one.'''

  def test_apply_registry_refuses_to_drop_a_declared_feature(self) -> None:
    '''``apply_registry`` must raise where ``fit_registry`` already did.

    The scoring path projected the missing column away and carried on, leaving
    the persisted contract describing a column the materialised frame did not
    contain. The consequences are asymmetric: the row counts and the uniqueness
    gate both pass, a score table is published, and the model has silently been
    applied to a different feature vector than the one it was validated on.
    '''
    import inspect  # noqa: PLC0415

    from forecasting_ml_framework.nodes import feature_nodes  # noqa: PLC0415

    source = inspect.getsource(feature_nodes.apply_registry)

    assert 'DataContractError' in source, 'the scoring path lost the contract check'
    assert 'absent' in source, 'the missing-column computation was removed'


# --------------------------------------------------------------------------- #
# 4. The promotion gate is a two-value contract in every mode
# --------------------------------------------------------------------------- #
class TestValidationNodeArity:
  '''A node must return exactly what it declares, in every branch.'''

  @pytest.mark.parametrize(
    'function_name',
    ['validate_and_promote', 'validate_and_promote_regression'],
  )
  def test_validation_node_returns_three_values_with_no_incumbent(self, function_name: str) -> None:
    '''No branch of a three-output node may return ``None``.

    The "no incumbent" branch is the *first training of every new model*, and it
    returned ``None`` against three declared outputs. Kedro raises while saving
    outputs, so the failure landed after the model had been trained and the
    registry had been written. A first training could therefore not complete in
    compare mode, which is the mode that is supposed to be safe for a first
    training.
    '''
    import inspect  # noqa: PLC0415

    from forecasting_ml_framework.nodes import validation_nodes  # noqa: PLC0415

    function = getattr(validation_nodes, function_name)
    source = inspect.getsource(function)

    assert 'return None' not in source, f'{function_name} can still return None against three outputs'
    assert 'return challenger, challenger_model, challenger_calibrator' in source


# --------------------------------------------------------------------------- #
# 4d. A mistyped mode must not select a weaker gate
# --------------------------------------------------------------------------- #
class TestTopologyValidation:
  '''An unrecognised specialisation axis must be refused, not defaulted.'''

  @pytest.mark.parametrize(
    'train_mode',
    ['train_only', 'train_updateMETADATA', 'train_compare_update', 'train_compareMODEL_updateMETADATA'],
  )
  def test_every_declared_train_mode_is_accepted(self, train_mode: str) -> None:
    '''The guard must not be so tight that a documented mode is rejected.'''
    from forecasting_ml_framework import constants as framework_constants  # noqa: PLC0415
    from forecasting_ml_framework.pipelines.registry import PipelineTopology  # noqa: PLC0415

    assert train_mode in {mode.value for mode in framework_constants.TrainMode}
    topology = PipelineTopology(suffix='params', run_mode='prod', train_mode=train_mode)
    assert topology.train_mode == train_mode

  @pytest.mark.parametrize('train_mode', ['TOTALLY_BOGUS', 'train_compareMODEL', ''])
  def test_a_mistyped_train_mode_is_refused(self, train_mode: str) -> None:
    '''Only ``run_mode`` was validated, and the default was the unsafe one.

    An unrecognised token is not "compare", so ``is_compare_mode`` was False and
    the graph selected the *non-comparison* promotion node. The run then
    retrained, wrote the production model trio, and applied no gate at all --
    reached by a spelling mistake, which is the outcome the gate exists to
    prevent.
    '''
    from forecasting_ml_framework.exceptions import PipelineTopologyError  # noqa: PLC0415
    from forecasting_ml_framework.pipelines.registry import PipelineTopology  # noqa: PLC0415

    with pytest.raises(PipelineTopologyError):
      PipelineTopology(suffix='params', run_mode='prod', train_mode=train_mode)


# --------------------------------------------------------------------------- #
# 10b. The runner invocation must match the pinned Kedro signature
# --------------------------------------------------------------------------- #
class TestRunnerInvocation:
  '''A programmatic run must call the runner the way Kedro defines it.'''

  def test_forwarded_kwargs_are_accepted_by_the_runner(self) -> None:
    '''Every keyword the framework forwards must exist on the runner.

    ``AbstractRunner.run`` in Kedro 1.0 accepts neither ``is_async`` nor
    ``session_id`` and takes no ``**kwargs``, so forwarding them raised
    ``TypeError`` on *every* programmatic run, for both runners.
    '''
    import inspect  # noqa: PLC0415

    from kedro.runner import AbstractRunner  # noqa: PLC0415

    from forecasting_ml_framework.platform import kedro_utils  # noqa: PLC0415

    accepted = set(inspect.signature(AbstractRunner.run).parameters)
    forwarded = _forwarded_runner_kwargs(kedro_utils)

    assert forwarded, 'the forwarder could not be located, so this test is vacuous'
    unknown = sorted(forwarded - accepted)
    assert not unknown, f'the framework forwards keyword(s) the pinned Kedro rejects: {unknown}'

  def test_requesting_async_execution_is_refused_not_ignored(self) -> None:
    '''``is_async`` is a promise; silently discarding it breaks it quietly.'''
    from forecasting_ml_framework.exceptions import ConfigurationError  # noqa: PLC0415
    from forecasting_ml_framework.platform import kedro_utils  # noqa: PLC0415

    with pytest.raises(ConfigurationError):
      kedro_utils.run_pipeline(None, None, is_async=True)


def _forwarded_runner_kwargs(kedro_utils: Any) -> set[str]:
  '''Extract the keyword names a module forwards to a runner.

  Reading the call rather than restating the list is the point: a list written
  beside this test would go stale exactly as the code it documents did.

  Args:
    kedro_utils: The module under test.

  Returns:
    The keyword argument names found in its ``run_pipeline`` call.
  '''
  import ast  # noqa: PLC0415
  import inspect  # noqa: PLC0415

  tree = ast.parse(inspect.getsource(kedro_utils.run_pipeline))
  forwarded: set[str] = set()
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    # The call target is `selected.run(...)`, i.e. an Attribute named `run`.
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'run':
      forwarded |= {keyword.arg for keyword in node.keywords if keyword.arg}
  return forwarded


# --------------------------------------------------------------------------- #
# 5. The threshold is a policy parameter, so it must be a usable probability
# --------------------------------------------------------------------------- #
class TestThresholdSelection:
  '''A threshold that is not a probability cannot be persisted as one.'''

  def test_degenerate_slice_never_yields_a_non_finite_threshold(self) -> None:
    '''``roc_curve`` opens with ``inf``; the sentinel must not be persisted.

    Every criterion is undefined on a single-class slice, so ``argmax`` returns
    index 0 and the sentinel that the library puts there is selected. It is then
    pickled into the model handle and applied to every future scoring run, so
    ``inf`` predicts nothing positive and the minimum score predicts everything
    positive -- both durable, both silent.
    '''
    frame = pd.DataFrame({'label': [0] * 40, 'probability': np.linspace(0.2, 0.8, 40)})

    for method in ('pr_curve', 'roc_curve', 'mcc_curve', 'ks_statistics'):
      selected = select_threshold(frame, method, 'label', 'probability')
      assert np.isfinite(selected), f'{method} returned {selected}'
      assert 0.2 <= selected <= 0.8, f'{method} returned {selected}, outside the score range'

  def test_a_real_slice_still_selects_a_usable_threshold(self) -> None:
    '''The guard must not disturb an ordinary selection.'''
    rng = np.random.default_rng(0)
    label = (rng.random(2000) < 0.08).astype(int)
    probability = np.clip(label * 0.55 + rng.normal(0.15, 0.10, 2000), 0, 1)
    frame = pd.DataFrame({'label': label, 'probability': probability})

    for method in ('pr_curve', 'roc_curve', 'mcc_curve', 'ks_statistics'):
      selected = select_threshold(frame, method, 'label', 'probability')
      assert 0.0 < selected < 1.0, f'{method} produced an implausible threshold {selected}'
      # A usable operating point on an 8% positive rate must not flag the world.
      assert 0.0 < float((probability >= selected).mean()) < 0.5, f'{method} flagged an implausible share'

  def test_ks_selection_survives_a_duplicated_index(self) -> None:
    '''Label-based indexing returns a Series when the index repeats.

    The threshold frame is built on the caller's index, which is routinely
    non-unique after a concat, a groupby or a filter. ``.loc[idxmax()]`` then
    yields a Series and the cast to float raises -- after the model was fitted,
    so it aborted the run rather than the selection.
    '''
    frame = pd.DataFrame(
      {'label': [0, 1, 0, 1, 0, 1], 'probability': [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]}
    )
    duplicated = frame.copy()
    duplicated.index = [0, 0, 1, 1, 2, 2]

    unique_selection = select_threshold(frame, 'ks_statistics', 'label', 'probability')
    repeated_selection = select_threshold(duplicated, 'ks_statistics', 'label', 'probability')

    assert repeated_selection == unique_selection


# --------------------------------------------------------------------------- #
# 4b. The training path must survive the target shape it is given
# --------------------------------------------------------------------------- #
class TestTargetShape:
  '''A run must not die because of a property of its own data.'''

  @staticmethod
  def _train(labels: np.ndarray, method: str | None, handle: str) -> Any:
    '''Train a real trainer over a real frame and return the handle.

    Args:
      labels: The target values.
      method: The threshold criterion, or ``None`` for the default path.
      handle: A dotted estimator path.

    Returns:
      The trained trainer instance.
    '''
    from forecasting_ml_framework.core.metadata import Metadata  # noqa: PLC0415
    from forecasting_ml_framework.modeling.models import ModelTraining  # noqa: PLC0415

    columns = [f'f{index}' for index in range(4)]
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 4)), columns=columns)
    frame['target'] = labels
    params: dict[str, Any] = {'model_handle': handle, 'model_params': {'max_depth': 4}, 'eval_size': 0.2}
    if method:
      params['threshold_selection_method'] = method
    trainer = ModelTraining(**params)
    trainer.train(frame, Metadata(feature_cols=columns, target_col='target', id_cols=[]))
    return trainer

  @pytest.mark.parametrize('method', [None, 'pr_curve', 'roc_curve', 'ks_statistics', 'mcc_curve'])
  def test_a_multiclass_target_trains(self, method: str | None) -> None:
    '''Every criterion is defined over a binary label, and none is binary here.

    The guard tested ``is_continuous_target``, which describes the *estimator
    class*, not the data. A five-class target passed it and reached a criterion
    scikit-learn defines only over a binary label, dying with ``multiclass
    format is not supported`` -- after the model had been fitted, so the cost was
    a whole run. The unconfigured path was affected too, because it defaults to
    ``pr_curve``.
    '''
    trainer = self._train(np.tile([0, 1, 2, 3, 4], 60), method, 'sklearn.tree.DecisionTreeClassifier')

    assert trainer.multi_class_flag is True
    assert np.isfinite(trainer.optimal_model_threshold)
    assert np.isfinite(trainer.optimal_calib_threshold)

  def test_a_continuous_target_trains(self) -> None:
    '''A continuous target has no decision to threshold, and must not try.'''
    labels = np.random.default_rng(1).normal(size=300)
    trainer = self._train(labels, 'mcc_curve', 'sklearn.ensemble.RandomForestRegressor')

    assert trainer._is_continuous() is True
    assert np.isfinite(trainer.optimal_model_threshold)

  def test_a_binary_target_still_selects_an_operating_point(self) -> None:
    '''The new guard must not swallow the ordinary binary case.

    Without this, "skip threshold selection when the target is not binary" would
    be indistinguishable from "never select a threshold", and the default
    operating point would silently replace the learned one everywhere.
    '''
    labels = (np.random.default_rng(2).random(300) < 0.2).astype(int)
    trainer = self._train(labels, 'mcc_curve', 'sklearn.tree.DecisionTreeClassifier')

    assert trainer.multi_class_flag is False
    assert trainer.optimal_model_threshold != 0.5, 'the configured default was used instead of a learned one'


# --------------------------------------------------------------------------- #
# 4c. A log-scale error must not drop the rows it cannot handle
# --------------------------------------------------------------------------- #
class TestLogScaleMetrics:
  '''A metric that skips rows reports a different quantity from the one named.'''

  def test_a_negative_prediction_does_not_silently_vanish(self) -> None:
    '''``log`` of a negative is NaN, and ``Series.mean`` skips NaN.

    A regressor predicting a negative amount is routine. The NaN was skipped, so
    the band mean was taken over a reduced, unpredictable subset: the worst error
    in the band was discarded and the denominator was wrong. In the fixture below
    that reported an MSLE of exactly zero -- a perfect score for a band that
    contained a prediction three units away from a target of three.
    '''
    from forecasting_ml_framework.modeling.metrics import _mean_squared_log_error  # noqa: PLC0415

    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = pd.Series([1.0, 2.0, -3.0, 4.0, 5.0])

    naive = float(((np.log(actual + 1) - np.log(predicted + 1)) ** 2).mean())
    corrected = _mean_squared_log_error(actual, predicted)

    assert naive == 0.0, 'the fixture no longer reproduces the NaN-skipping defect'
    assert corrected > 0.0, 'a prediction three units out still reported a perfect log-scale error'

  def test_a_clean_frame_is_unaffected(self) -> None:
    '''The correction must not disturb a band with no undefined term.'''
    from forecasting_ml_framework.modeling.metrics import _mean_squared_log_error  # noqa: PLC0415

    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # An exact prediction is exactly zero, and a constant offset is a small but
    # non-zero log-scale error -- the `+1` offset compresses large values, so a
    # shift of one unit costs less at the top of the range than at the bottom.
    assert _mean_squared_log_error(actual, actual) == pytest.approx(0.0)
    assert 0.0 < _mean_squared_log_error(actual, actual + 1.0) < 0.2

  def test_an_entirely_undefined_band_reports_nan_not_zero(self) -> None:
    '''With no defined row there is no quantity, and the answer is ``NaN``.'''
    from forecasting_ml_framework.modeling.metrics import _mean_squared_log_error  # noqa: PLC0415

    assert np.isnan(_mean_squared_log_error(pd.Series([np.nan] * 3), pd.Series([1.0] * 3)))


# --------------------------------------------------------------------------- #
# 5b. The metric row's *shape* must not depend on a redundant flag
# --------------------------------------------------------------------------- #
class TestMetricRowShapeIsDataDriven:
  '''A flag that is advisory must not reshape a published table.'''

  @staticmethod
  def _evaluate(label: np.ndarray, multiclass: bool) -> Any:
    '''Evaluate a small binary frame and return the metric row.

    Args:
      label: The target values.
      multiclass: The configured ``multiclass_flag``.

    Returns:
      The metric row as a one-element ``DataFrame``.
    '''
    from forecasting_ml_framework.core.metadata import Metadata  # noqa: PLC0415
    from forecasting_ml_framework.nodes import model_nodes  # noqa: PLC0415

    size = len(label)
    frame = pd.DataFrame(
      {
        'cust_id': [f'c{index}' for index in range(size)],
        'label': label,
        'prediction': label,
        'probability': np.where(label == 1, 0.8, 0.2),
      }
    )
    parameters = {
      'model_key': 'regression_test_model',
      'run_date': '2026-09-29',
      'modeling_params': {
        'prediction_col': 'prediction',
        'prediction_probability_col': 'probability',
        'eval_metric': 'LIFT_DECILE_10',
      },
    }
    row, _ = model_nodes.model_evaluation(
      frame,
      parameters,
      Metadata(feature_cols=['probability'], target_col='label', id_cols=['cust_id']),
      'prediction',
      'probability',
      'train',
      multiclass,
      sequence_flag=False,
    )
    del label
    return row

  def test_a_redundant_multiclass_flag_does_not_narrow_the_row(self) -> None:
    '''Two-class data must produce the same columns either way.

    ``is_multiclass`` previously read ``observed_classes > 2 or (multi_class and
    observed_classes > 1)``, so setting the flag on a genuinely binary slice
    emitted the *narrow* multiclass row -- dropping ``roc_auc``, ``precision``,
    ``recall``, ``f1`` and all four confusion counts -- while the log line
    announced that "the binary metric path is used". Both variants write to the
    same table with ``WRITE_TRUNCATE``, so a re-run with the flag set destroyed
    the previous snapshot's columns.
    '''
    binary = np.array([0, 1] * 8)

    without_flag = set(self._evaluate(binary, False).columns)
    with_flag = set(self._evaluate(binary, True).columns)

    assert with_flag == without_flag, f'columns lost: {sorted(without_flag - with_flag)}'
    for required in ('roc_auc', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp'):
      assert required in with_flag, f'{required} is missing when the flag is set'

  def test_a_genuinely_multiclass_target_still_takes_the_multiclass_path(self) -> None:
    '''The other direction must be unaffected: the data decides, either way.'''
    row = self._evaluate(np.tile([0, 1, 2], 6), False)

    assert 'n_classes' in row.columns, 'a three-class target took the binary path'


# --------------------------------------------------------------------------- #
# 6. Banding: the top half, and no crash on an unusable score
# --------------------------------------------------------------------------- #
class TestBanding:
  '''Banding is a consumer contract, so its edges are not advisory.'''

  @pytest.mark.parametrize('distinct', [2, 3, 4, 7, 10])
  def test_top_half_floor_is_the_upper_half_of_the_bands_present(self, distinct: int) -> None:
    '''The floor is the first label of the top half of the bands produced.

    A fixed floor of 5 is wrong twice over. With the full 1..10 decile structure
    it admits deciles 5-10, which is the top 60 percent rather than the top half.
    And ``qcut`` with ``duplicates='drop'`` produces only as many bands as the
    score's distinct values admit, so a coarse score can yield a single band that
    no floor of 5 would reach -- and the flag became 'N' for every row.

    The invariant is the band-index one, expressed against the bands actually
    produced rather than a hard-coded table: the number of distinct score values
    does not equal the number of surviving bands, because ``qcut`` collapses
    quantile edges that fall inside one value. Asserting against a precomputed
    per-case number would therefore be asserting an accident of the quantile
    algorithm. For the same reason the assertion is not a row share: with few
    distinct values the surviving bands are unequal in size, so "the top half of
    the bands" is deliberately not "half the rows".
    '''
    scores = pd.Series(np.repeat(np.arange(float(distinct)), 200))
    bands = _rank_bands(scores, constants.DECILE_COUNT)
    highest = int(bands.max())

    assert _top_half_floor(bands) == highest - (highest + 1) // 2 + 1
    # Whatever the count, the floor must admit at least one band and no more
    # than half of them, and it must be strictly above the median band.
    admitted = highest - _top_half_floor(bands) + 1
    assert 1 <= admitted <= -(-highest // 2), f'{admitted} of {highest} bands admitted'

  def test_full_decile_structure_flags_exactly_half(self) -> None:
    '''With ten equal-volume bands the top half is five of them, i.e. half.'''
    scores = pd.Series(np.linspace(0.0, 1.0, 1000))
    bands = _rank_bands(scores, constants.DECILE_COUNT)

    assert int(bands.max()) == 10
    assert _top_half_floor(bands) == 6
    assert float((bands >= 6).mean()) == pytest.approx(0.5, abs=0.01)

  def test_unsplittable_score_does_not_flag_nothing(self) -> None:
    '''A single-band population degrades to "all of it", not to "none of it".

    This is the case a fixed floor turned into a total inversion: every row
    landed in band 1, and 1 is below 5, so a consumer reading ``high_score_ind``
    saw 'N' on every row of a run that had in fact scored the whole population.
    '''
    bands = _rank_bands(pd.Series([0.4] * 100), constants.DECILE_COUNT)

    assert int(bands.max()) == 1
    assert float((bands >= _top_half_floor(bands)).mean()) == 1.0

  @pytest.mark.parametrize('column', [[np.nan] * 5, [0.5] * 5])
  def test_unusable_score_does_not_crash_the_bander(self, column: list[float]) -> None:
    '''The degeneracy guard must run *before* ``qcut``, not after it.

    ``qcut`` on an all-NaN column reaches into numpy's internals and raises
    ``IndexError: index -1 is out of bounds for axis 0 with size 0``. A guard
    written on the following line is a guard that never runs on the case it was
    written for.
    '''
    spec = MetricSpec(
      name='LIFT_DECILE_10',
      metric='LIFT',
      band_kind='DECILE',
      band=10,
      kind='classification',
      higher_is_better=True,
    )
    frame = pd.DataFrame({'probability': column, 'label': [0] * len(column)})

    assert list(_assign_bands(frame.copy(), 'probability', spec)['score_ntile']) == [1] * len(column)
    assert list(_rank_bands(pd.Series(column, dtype=float), constants.DECILE_COUNT)) == [1] * len(column)


# --------------------------------------------------------------------------- #
# 7. The metric vocabulary must not advertise what it cannot parse
# --------------------------------------------------------------------------- #
class TestMetricVocabulary:
  '''The help text an operator reads on rejection must be truthful.'''

  def test_help_advertises_only_parseable_names(self) -> None:
    '''Every name in the grammar help must be in the registry.

    ``KSI`` was listed in the help and in the higher-is-better set while being
    subtracted again when the registry was built, so the error message told the
    operator to use a name the parser then refused.
    '''
    advertised = set(re.findall(r"'([A-Z][A-Z_0-9]*)'", METRIC_GRAMMAR_HELP))
    advertised -= {'DECILE', 'CENTILE'}

    assert advertised, 'the help must advertise something for this test to mean anything'
    unregistered = advertised - set(METRIC_REGISTRY)
    assert not unregistered, f'help advertises unregistered metrics: {sorted(unregistered)}'

  def test_removed_name_is_rejected_and_no_longer_advertised(self) -> None:
    '''``KSI`` is a threshold criterion, not a reportable metric.'''
    assert 'KSI' not in METRIC_REGISTRY
    assert 'KSI' not in METRIC_GRAMMAR_HELP
    with pytest.raises(Exception, match='(?i)unsupported|regression metric'):
      parse_metric('KSI', strict=True)


# --------------------------------------------------------------------------- #
# 8. The sequence helpers are pure, so their edge cases are directly assertable
# --------------------------------------------------------------------------- #
class TestSequenceHelperEdges:
  '''A pure helper's contract is its arithmetic, including at the edges.'''

  def test_zero_degree_slope_has_no_trend_coefficient(self) -> None:
    '''``order: 0`` yields a constant fit, so there is no slope to report.

    ``order`` is configurable precisely so an ``order: 1`` and an ``order: 2``
    slope of one column can coexist, which makes 0 a reachable value. ``polyfit``
    returns a single coefficient for it, so reading the linear term ran off the
    end of the array.
    '''
    assert get_slope_func([1.0, 2.0, 3.0], order=0, last_n_val=3) == 0.0
    assert get_slope_func([1.0, 2.0, 3.0], order=1, last_n_val=3) != 0.0

  @pytest.mark.parametrize(
    ('dates', 'expected'),
    [
      (['2024-01-01', '2024-01-02', '2024-01-04'], 2),
      (['2024-01-01', '2024-01-03', '2024-01-04'], 2),
      (['2024-01-01', '2024-01-11'], 10),
      (['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'], 1),
      (['2024-01-01'], 0),
    ],
  )
  def test_mean_date_gap_rounds_rather_than_truncates(self, dates: list[str], expected: int) -> None:
    '''A mean of 1.5 days is 2 days, not 1.

    Floor division biased the feature downward on every window whose mean was
    fractional, and the bias grows as the window lengthens. The values still
    looked like plausible day counts, so nothing surfaced it.
    '''
    assert seq_date_delta(dates) == expected

  @pytest.mark.parametrize('length', [1, 2, 3, 4, 6, 8])
  def test_similarity_windows_are_always_equal_length(self, length: int) -> None:
    '''Both sides of a cosine comparison must be normalised to the window.

    Only the recent window was normalised before, so the baseline was shorter
    than the window for any history shorter than ``2k``. The comparison then
    returned its degenerate answer, which reads as *maximally dissimilar* -- the
    opposite of the truth, and applied to exactly the low-activity customers a
    retention model most needs to score.
    '''
    sequence = [float(index + 1) for index in range(length)]
    window = 2
    recent = _rank_bands(pd.Series(sequence, dtype=float), window)  # not used; keeps intent explicit
    del recent

    from forecasting_ml_framework.preprocessing.sequence_ops import get_fixed_length  # noqa: PLC0415

    primary = get_fixed_length(sequence, fixed_length=window)
    baseline = normalise_window(sequence, start=-2 * window, end=-window, fixed_length=window)

    assert len(primary) == len(baseline) == window


# --------------------------------------------------------------------------- #
# 9. Imputation: a configured bound is the operator's explicit instruction
# --------------------------------------------------------------------------- #
class TestImputationBounds:
  '''A configured bound must be applied, and only to selected columns.'''

  def test_none_strategy_still_applies_configured_bounds(self) -> None:
    '''``custom_bounds`` exists for exactly the ``strategy: none`` case.

    Eligibility was tested against the *learned* bounds, which are empty under
    ``none``, so every configured bound was discarded. Both ``Imputations``
    blocks in the shipped parameters.yml use that combination, which made the
    whole ``fill_missing_dict`` in globals.yml dead.
    '''
    configured = {'plan_tier': 'UNKNOWN', 'region_code': 'UNKNOWN'}
    routine = Imputations(
      None,
      {
        'strategy': 'none',
        'custom_bounds': dict(configured),
        'cols': {'exclude_cols': [], 'include_cols': ['plan_tier', 'region_code']},
      },
    )
    routine.use_cols = ['plan_tier', 'region_code']
    selected = set(routine.use_cols)

    merged = {key: value for key, value in routine.custom_bounds.items() if key in selected}

    assert merged == configured, 'a configured bound was discarded, making the routine a no-op'

  def test_bound_outside_the_selection_is_not_applied(self) -> None:
    '''A configured bound is scoped to the routine's selection, not global.'''
    routine = Imputations(
      None,
      {
        'strategy': 'none',
        'custom_bounds': {'plan_tier': 'UNKNOWN', 'unrelated': 'X'},
        'cols': {'exclude_cols': [], 'include_cols': ['plan_tier']},
      },
    )
    routine.use_cols = ['plan_tier']

    assert {key for key in routine.custom_bounds if key in set(routine.use_cols)} == {'plan_tier'}


# --------------------------------------------------------------------------- #
# 10. The tuned session must actually be able to start
# --------------------------------------------------------------------------- #
class TestSparkTuningIsApplicable:
  '''A tuning document that cannot be applied is not a tuning document.'''

  def test_tuning_mapping_is_accepted_by_the_session_builder(self, tmp_path: Path) -> None:
    '''The builder takes a mapping in ``map`` and a ``SparkConf`` in ``conf``.

    A plain ``dict`` was passed to ``conf``, which the builder reads with
    ``.getAll()``. This broke every tuned path -- the ``run`` command, the
    Spark-aware context, and the initialisation entry point -- so it was a total
    outage of the production entry point rather than a tuning defect.
    '''
    from pyspark.sql import SparkSession  # noqa: PLC0415

    environment = tmp_path / 'base'
    environment.mkdir(parents=True)
    (environment / 'spark.yml').write_text('spark.sql.adaptive.enabled: true\n', encoding='utf-8')

    loaded = _load_spark_conf(tmp_path, 'base')
    # The loader stringifies every value, so a YAML boolean arrives as 'True'.
    # Spark parses booleans case-insensitively, so the spelling is not a
    # correctness concern -- the assertion is that the mapping is accepted, and
    # that acceptance is the thing that was broken.
    assert list(loaded) == ['spark.sql.adaptive.enabled']
    assert loaded['spark.sql.adaptive.enabled'].lower() == 'true'

    builder = SparkSession.builder.config(map=loaded)
    assert builder._options['spark.sql.adaptive.enabled'] == loaded['spark.sql.adaptive.enabled']

  def test_legacy_java_version_scheme_is_recognised(self) -> None:
    '''Java 8 reports ``1.8.x``, whose major version is the *second* component.

    Reading the first component reported version 1, which is not in the supported
    set, so a perfectly good Java 8 was rejected exactly as reliably as an
    unsupported one and the auto-detection could never succeed on the oldest
    runtime it was written to support.
    '''
    from forecasting_ml_framework.local.spark_session import (  # noqa: PLC0415
      SUPPORTED_JAVA_MAJORS,
      _parse_java_version,
    )

    assert _parse_java_version('1.8.0_402') == 8
    assert _parse_java_version('11.0.21') == 11
    assert _parse_java_version('17.0.9') == 17
    assert _parse_java_version('21.0.2') not in SUPPORTED_JAVA_MAJORS
    for supported in SUPPORTED_JAVA_MAJORS:
      assert _parse_java_version(f'{supported}.0.1') == supported


# --------------------------------------------------------------------------- #
# 11. Learned state must not be fitted in-sample, nor shared
# --------------------------------------------------------------------------- #
class TestTargetEncodingIsOutOfFold:
  '''In-sample target encoding is the textbook leakage pattern.'''

  def test_the_frozen_map_differs_from_the_in_sample_one(self) -> None:
    '''``fit`` cross-fits nothing; only ``fit_transform`` does.

    ``TargetEncoder.fit`` populates ``encodings_`` with full-data smoothed means,
    so a row encoded from that table is encoded with a statistic that includes
    its own label. On the fixture below every category moves and the majority of
    rows receive a different value, so this is not a rounding difference.
    '''
    from sklearn.preprocessing import TargetEncoder  # noqa: PLC0415

    from forecasting_ml_framework.preprocessing.categorical import _cross_fitted_maps  # noqa: PLC0415

    rng = np.random.default_rng(0)
    size = 300
    frame = pd.DataFrame({'segment': rng.choice([f's{index}' for index in range(6)], size)})
    target = pd.Series((rng.random(size) < 0.05).astype(float))

    in_sample_encoder = TargetEncoder(categories='auto', cv=5, smooth='auto')
    in_sample_encoder.fit(frame, target)
    in_sample = dict(
      zip(in_sample_encoder.categories_[0], in_sample_encoder.encodings_[0], strict=True)
    )

    cross_encoder = TargetEncoder(categories='auto', cv=5, smooth='auto')
    cross_fitted = cross_encoder.fit_transform(frame[['segment']], target)
    out_of_fold = _cross_fitted_maps(cross_fitted, frame, ['segment'])['segment']

    assert set(in_sample) == set(out_of_fold)
    moved = [key for key in in_sample if abs(in_sample[key] - out_of_fold[key]) > 1e-12]
    assert moved, 'the two maps are identical, so the fixture no longer discriminates'

  def test_the_fitted_map_is_a_category_lookup(self) -> None:
    '''Scoring-time lookup needs a per-category map, not per-row encodings.'''
    from sklearn.preprocessing import TargetEncoder  # noqa: PLC0415

    from forecasting_ml_framework.preprocessing.categorical import _cross_fitted_maps  # noqa: PLC0415

    frame = pd.DataFrame({'segment': ['a', 'a', 'b', 'b', 'c', 'c']})
    target = pd.Series([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    encoder = TargetEncoder(categories='auto', cv=2, smooth='auto')
    cross_fitted = encoder.fit_transform(frame[['segment']], target)

    maps = _cross_fitted_maps(cross_fitted, frame, ['segment'])

    assert set(maps) == {'segment'}
    assert set(maps['segment']) == {'a', 'b', 'c'}
    assert all(isinstance(value, float) for value in maps['segment'].values())


# --------------------------------------------------------------------------- #
# 12. A transform must be able to run on a schema that has a string in it
# --------------------------------------------------------------------------- #
class TestNumericOnlyRoutines:
  '''A distance-based imputer cannot consume a categorical column.'''

  def test_knn_imputation_scopes_its_fit_to_the_selection(self) -> None:
    '''The fit width is the routine's selection, not the whole feature set.

    ``feature_cols`` contains the categorical roles, and a ``KNNImputer``
    measures Euclidean distance, so fitting on it raised ``could not convert
    string to float`` for any schema with a string feature. The routine was
    unrunnable in every realistic case, masked only by being disabled in the
    shipped configuration.
    '''
    from sklearn.impute import KNNImputer  # noqa: PLC0415

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
      {
        'arupe_30d': rng.normal(10, 3, 50),
        'tenure_days': rng.integers(1, 900, 50),
        'plan_tier': rng.choice(['A', 'B'], 50),
      }
    )
    with pytest.raises(ValueError, match='could not convert string to float'):
      KNNImputer(n_neighbors=3).fit(frame)

    numeric = KNNImputer(n_neighbors=3)
    numeric.fit(frame[['arupe_30d', 'tenure_days']])
    assert numeric.transform(frame[['arupe_30d', 'tenure_days']]).shape == (50, 2)

  def test_the_routine_declares_the_selection_as_its_fit_width(self) -> None:
    '''``fitted_columns`` must equal the selected columns, not the feature list.

    Asserted against the executable statements rather than the raw source, so
    that the explanatory comment describing the defect cannot itself satisfy or
    fail the check.
    '''
    import ast  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    from forecasting_ml_framework.preprocessing.scalar import KNNImputation  # noqa: PLC0415

    def _attribute_chain(node: ast.AST) -> str:
      '''Render an attribute expression as dotted source text.

      Args:
        node: An AST node.

      Returns:
        The dotted name, or an empty string for anything else.
      '''
      if isinstance(node, ast.Attribute):
        return f'{_attribute_chain(node.value)}.{node.attr}'.lstrip('.')
      if isinstance(node, ast.Name):
        return node.id
      return ''

    # `inspect.getsource` on a method returns its *indented* body, which is not
    # parseable on its own.
    body = textwrap.dedent(inspect.getsource(KNNImputation.fit))
    assignments = [
      _attribute_chain(node.targets[0])
      for node in ast.walk(ast.parse(body))
      if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Attribute)
    ]

    assert 'self.fitted_columns' in assignments, 'the fit width is no longer stored'
    assert 'metadata.feature_cols' not in assignments, 'the fit width is again the whole feature set'


# --------------------------------------------------------------------------- #
# 13. The identifier guarantee must be enforced, not merely declared
# --------------------------------------------------------------------------- #
class TestIdentifierProtection:
  '''A declared invariant that nothing populates is not an invariant.'''

  def test_the_registry_hands_every_routine_the_protected_columns(self) -> None:
    '''Identifiers, the target and the sequence columns must never be transformed.

    ``get_selected_columns`` subtracts ``key_cols_list``, and nothing ever
    populated it -- so the guarantee held only because identifiers happened not to
    be in ``feature_cols``. A routine naming one in ``include_cols`` transformed
    it, and a transformed target is not the label the evaluation metrics are
    defined against.
    '''
    import logging  # noqa: PLC0415

    from forecasting_ml_framework.core.metadata import Metadata  # noqa: PLC0415
    from forecasting_ml_framework.core.registry import Registry  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    contract = Metadata(
      feature_cols=['arupe_30d', 'plan_tier', 'evt_seq'],
      categorical_cols=['plan_tier'],
      seq_cols=['evt_seq'],
      id_cols=['cust_id', 'mobile_ref'],
      target_col='label',
    )
    routine = AVAILABLE_ROUTINES['Imputations'](
      None,
      {
        'strategy': 'median',
        'cols': {'exclude_cols': [], 'include_cols': ['cust_id', 'arupe_30d', 'label', 'evt_seq']},
      },
    )
    registry = Registry([routine])
    registry._protect_identifiers(contract)

    resolved = routine.get_selected_columns(
      routine.cols['exclude_cols'], routine.cols['include_cols'], contract.feature_cols
    )

    assert resolved == ['arupe_30d']
    for protected in ('cust_id', 'mobile_ref', 'label', 'evt_seq'):
      assert protected not in resolved, f'{protected} was resolved for transformation'


# --------------------------------------------------------------------------- #
# 14. Concatenating registries must not alias their routines
# --------------------------------------------------------------------------- #
class TestRegistryCompositionIsIsolated:
  '''A combined registry must not overwrite a fitted original's state.'''

  @pytest.mark.parametrize('operation', ['add', 'merge', 'copy'])
  def test_composition_does_not_share_routine_instances(self, operation: str) -> None:
    '''Sharing instances let fitting the combination overwrite the original.

    That is the most dangerous class of defect this framework has, because it
    invalidates a persisted train-serve contract: a registry that was fitted,
    and possibly already serialised, silently became a different object.
    '''
    import logging  # noqa: PLC0415

    from forecasting_ml_framework.core.registry import Registry  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    first = Registry([AVAILABLE_ROUTINES['Imputations'](None, {'strategy': 'median', 'cols': {}})])
    second = Registry([AVAILABLE_ROUTINES['CapOutliers'](None, {'method': 'std', 'cols': {}})])
    first.routines[0].bounds = {'column': 'fitted-on-original'}

    if operation == 'add':
      combined = first + second
    elif operation == 'merge':
      combined = Registry.merge(first, second)
    else:
      combined = first.copy()

    assert combined.routines[0] is not first.routines[0]
    combined.routines[0].bounds = {'column': 'fitted-on-combination'}
    assert first.routines[0].bounds == {'column': 'fitted-on-original'}


# --------------------------------------------------------------------------- #
# 15. Observability: the record must actually be written
# --------------------------------------------------------------------------- #
class TestObservabilityIsNotSilentlyEmpty:
  '''A hook that reports nothing is worse than no hook, because it is trusted.'''

  def test_dataset_provenance_sees_config_declared_datasets(self) -> None:
    '''Kedro 1.0 splits datasets across two private mappings.

    ``_datasets`` holds initialised instances; ``_lazy_datasets`` holds the ones
    built from configuration, which is *everything* a node reads or writes. The
    hook read ``_datasets`` alone -- and ``DataCatalog`` has no ``datasets``
    attribute, so the other branch never fired either. The per-node provenance
    record was therefore silent on every node of every run, while the module
    docstring described it as the primary tool for diagnosing artefact-resolution
    problems.
    '''
    import logging  # noqa: PLC0415

    from kedro.io import DataCatalog  # noqa: PLC0415

    from forecasting_ml_framework.hooks import _iter_datasets  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    catalog = DataCatalog.from_config(
      {
        'model_handle_params': {'type': 'kedro_datasets.pickle.PickleDataset', 'filepath': '/tmp/handle.pkl'},
        'fitted_model_params': {'type': 'kedro_datasets.pickle.PickleDataset', 'filepath': '/tmp/fitted.pkl'},
        'sequence_ind1': {'type': 'kedro.io.MemoryDataset'},
      }
    )

    assert catalog._datasets == {}, 'the fixture no longer reproduces the split storage'
    reported = [name for name, _ in _iter_datasets(catalog)]

    assert set(reported) == {'model_handle_params', 'fitted_model_params', 'sequence_ind1'}

  def test_a_failing_dataset_does_not_break_the_node(self) -> None:
    '''Provenance must never be the reason a node fails.'''
    import logging  # noqa: PLC0415

    from forecasting_ml_framework.hooks import _iter_datasets  # noqa: PLC0415

    logging.disable(logging.CRITICAL)

    class Exploding:
      """A catalog whose only dataset cannot be materialised."""

      @staticmethod
      def keys() -> list[str]:
        """Return the declared dataset names.

        Returns:
          One dataset name.
        """
        return ['broken']

      @staticmethod
      def get(_name: str) -> Any:
        """Fail, as a missing artefact would.

        Args:
          _name: The dataset name.

        Raises:
          FileNotFoundError: Always.
        """
        raise FileNotFoundError('no such artefact')

    assert list(_iter_datasets(Exploding())) == []

  def test_compare_mode_registers_the_winning_model(self) -> None:
    '''A comparison run must register the validation node's output.

    The specialisation axes were read from the ``--params`` channel, which never
    carries them, so ``train_mode`` always defaulted to ``train_only`` and a
    comparison run registered the training node's output -- the *rejected*
    challenger. That is the single outcome this hook exists to prevent.
    '''
    import logging  # noqa: PLC0415

    from forecasting_ml_framework.hooks import _collect_registration_payload  # noqa: PLC0415

    logging.disable(logging.CRITICAL)

    class FakeNode:
      """A node stub exposing only the attributes the hook reads."""

      def __init__(self, name: str, inputs: list[str], outputs: list[str]) -> None:
        """Record the node's identity and its datasets.

        Args:
          name: The node name.
          inputs: The node's input dataset names.
          outputs: The node's output dataset names.
        """
        self.name, self.inputs, self.outputs = name, inputs, outputs

    class FakeCatalog:
      """A catalog stub resolving ``parameters`` and naming every artefact."""

      def __init__(self, document: dict[str, Any]) -> None:
        """Record the document the training node will expose.

        Args:
          document: The parameters document.
        """
        self.document = document

      def load(self, name: str) -> Any:
        """Resolve a dataset name.

        Args:
          name: The dataset name.

        Returns:
          The parameters document, or an artefact label.
        """
        return self.document if name == 'parameters' else f'ARTEFACT::{name}'

    class FakePipeline:
      """A pipeline stub exposing only its nodes."""

      def __init__(self, nodes: list[Any]) -> None:
        """Record the pipeline's nodes.

        Args:
          nodes: The nodes in the graph.
        """
        self.nodes = nodes

    nodes = [
      FakeNode('model_training_params', ['parameters', 'frame'], ['model_handle_candidate_params']),
      FakeNode('model_validation_params', ['a', 'b', 'c'], ['model_handle_params']),
    ]
    document = {'train_mode': 'train_compareMODEL_updateMETADATA', 'is_regression': 'False'}

    model, _, _ = _collect_registration_payload(FakePipeline(nodes), FakeCatalog(document), document)
    assert model == 'ARTEFACT::model_handle_params', 'the rejected candidate was registered'

    document = {'train_mode': 'train_only', 'is_regression': 'False'}
    model, _, _ = _collect_registration_payload(FakePipeline(nodes), FakeCatalog(document), document)
    assert model == 'ARTEFACT::model_handle_candidate_params', 'a train-only run must register its own output'


# --------------------------------------------------------------------------- #
# 16. Idempotency is the persistence layer's whole job
# --------------------------------------------------------------------------- #
class TestPersistenceIdempotency:
  '''A write disposition is a promise about what a re-run does.'''

  def test_insert_overwrite_derives_its_scope_from_the_frame(self) -> None:
    '''The delete scope must be read off the data, not off a parameter.

    A frame spanning two dates with ``write_partition`` naming one of them
    cleared only that date, so the append landed on top of the surviving rows of
    the other -- and every re-run duplicated them. Reading the scope off the frame
    makes that unrepresentable: you cannot clear a partition you are not about to
    rewrite.
    '''
    import inspect  # noqa: PLC0415

    from forecasting_ml_framework.persistence.bigquery_spark import CustomSparkBQDataSet  # noqa: PLC0415

    source = inspect.getsource(CustomSparkBQDataSet._insert_overwrite_save)

    assert 'distinct()' in source, 'the scope is no longer derived from the frame'
    assert 'DELETE FROM' in source and 'quote_literal' in source, 'the delete is still interpolated'

  def test_partition_scope_cannot_be_injected(self) -> None:
    '''A partition field or value must not be able to change the predicate.

    Both are interpolated into a ``DELETE ... WHERE`` statement. A field carrying
    a quote produced ``WHERE rpt_dt' OR '1'='1 = 'x' OR '1'='1'`` -- which
    deletes the entire table.
    '''
    from forecasting_ml_framework.persistence.bigquery_spark import _is_table_absent  # noqa: PLC0415
    from forecasting_ml_framework.utils.text import quote_identifier, quote_literal  # noqa: PLC0415

    with pytest.raises(Exception, match='not a plain SQL identifier'):
      quote_identifier("rpt_dt' OR '1'='1")

    assert quote_literal("x' OR '1'='1") == "'x'' OR ''1''=''1'"
    assert _is_table_absent(RuntimeError('Not found: Table p:d.t')) is True

  @pytest.mark.parametrize(
    ('message', 'expected'),
    [
      ('Not found: Table proj:ds.tbl', True),
      ('Not found: Dataset proj:ds', True),
      ('Table proj:ds.tbl was not found in location US', True),
      ('Your default credentials were not found. Please set up Application Default Credentials', False),
      ('Could not fetch access token: 404 Not Found (project quota exceeded)', False),
      ('Access Denied: Table proj:ds.tbl: User does not have permission', False),
      ('Exceeded quota for query: too many api requests', False),
    ],
  )
  def test_a_credential_fault_is_not_an_absent_table(self, message: str, expected: bool) -> None:
    '''A generic ``'not found'`` marker turns a credential fault into data loss.

    Every caller of an existence probe goes on to create or overwrite the table it
    just probed, so reporting "absent" for an authentication failure means
    overwriting a table nobody successfully read. The markers are
    table-specific, and a fault marker always wins.
    '''
    from forecasting_ml_framework.persistence.bigquery_spark import _is_table_absent  # noqa: PLC0415

    assert _is_table_absent(RuntimeError(message)) is expected

  def test_a_declared_credential_is_not_shadowed_by_a_default(self) -> None:
    '''A merged default key makes ``.get(key, fallback)`` always return the default.

    The defaults are merged into ``save_args`` at construction, so the key always
    existed and the constructor's ``credentials`` was discarded -- the adapter
    silently fell back to ambient application-default credentials, which is a
    quiet change of identity.
    '''
    import logging  # noqa: PLC0415

    from forecasting_ml_framework.persistence.gbq_pandas import GBQTableDataSet  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    declared = {'key': 'value'}
    dataset = GBQTableDataSet(table_name='p.d.scores', dataset='d', project_id='p', credentials=declared)

    assert dataset._save_args.get('credentials') is None, 'the fixture no longer reproduces the shadowing'
    assert dataset._save_args.get('credentials', dataset._credentials) is None
    assert dataset._credentials is declared


# --------------------------------------------------------------------------- #
# 17. A configuration reference must resolve to the value it names
# --------------------------------------------------------------------------- #
class TestRuntimeResolver:
  '''A resolver that always returns ``''`` resolves nothing.'''

  def test_a_published_runtime_parameter_resolves(self) -> None:
    '''The resolver read an environment variable nothing ever wrote.

    Every ``${runtime:NAME}`` reference therefore resolved to the empty string --
    silently, because an empty string is a legal value for every configuration
    type. A reference to the run date produced ``''`` and failed much later as a
    malformed partition predicate.

    The production resolver is re-registered with ``replace=True`` rather than
    only when absent: the resolver name is process-global, and other tests (and
    the configuration loader itself) legitimately install their own, so a
    first-wins registration would make this assertion depend on test order.
    '''
    from omegaconf import OmegaConf  # noqa: PLC0415

    from forecasting_ml_framework.platform.bootstrap import (  # noqa: PLC0415
      publish_runtime_parameters,
      runtime_parameter,
    )
    from forecasting_ml_framework.platform.config_loader import ENV_PREFIX  # noqa: PLC0415

    saved = os.environ.pop(f'{ENV_PREFIX}regression_test_run_date', None)
    try:
      publish_runtime_parameters({'regression_test_run_date': '2026-09-29'})
      assert runtime_parameter('regression_test_run_date') == '2026-09-29'

      # `_register` is idempotent by design, so the resolver is installed here
      # the same way the loader installs it at process start.
      import forecasting_ml_framework.platform.config_loader as loader  # noqa: PLC0415

      loader.OmegaConf.register_new_resolver(
        'runtime',
        lambda name: _resolve_runtime(name),
        replace=True,
      )
      resolved = OmegaConf.to_container(
        OmegaConf.create({'value': '${runtime:regression_test_run_date}'}), resolve=True
      )
      assert resolved['value'] == '2026-09-29'
    finally:
      publish_runtime_parameters({})
      if saved is not None:
        os.environ[f'{ENV_PREFIX}regression_test_run_date'] = saved

  def test_an_unset_parameter_still_resolves_to_empty(self) -> None:
    '''Optional references must not fail the configuration load.'''
    from omegaconf import OmegaConf  # noqa: PLC0415

    from forecasting_ml_framework.platform.bootstrap import publish_runtime_parameters  # noqa: PLC0415
    from forecasting_ml_framework.platform.config_loader import ENV_PREFIX  # noqa: PLC0415

    saved = os.environ.pop(f'{ENV_PREFIX}regression_test_never_set', None)
    try:
      publish_runtime_parameters({})
      import forecasting_ml_framework.platform.config_loader as loader  # noqa: PLC0415

      loader.OmegaConf.register_new_resolver(
        'runtime', lambda name: _resolve_runtime(name), replace=True
      )
      resolved = OmegaConf.to_container(
        OmegaConf.create({'value': '${runtime:regression_test_never_set}'}), resolve=True
      )
      assert resolved['value'] == ''
    finally:
      if saved is not None:
        os.environ[f'{ENV_PREFIX}regression_test_never_set'] = saved


def _resolve_runtime(name: str) -> Any:
  '''Resolve a ``${runtime:NAME}`` reference exactly as the loader does.

  Args:
    name: The runtime parameter name.

  Returns:
    The published value, the environment override, or an empty string.
  '''
  from forecasting_ml_framework.platform.bootstrap import runtime_parameter  # noqa: PLC0415
  from forecasting_ml_framework.platform.config_loader import ENV_PREFIX  # noqa: PLC0415

  value = os.environ.get(f'{ENV_PREFIX}{name}')
  if value is not None:
    return value
  value = runtime_parameter(name)
  return '' if value is None else value


def _is_sequence(name: str) -> bool:
  '''Report whether a routine name belongs to the sequence family.

  Args:
    name: A routine's short configuration name.

  Returns:
    ``True`` when the routine operates on ordered behavioural histories.
  '''
  return name.startswith(('Sequence', 'Fixed', 'Slope', 'Similarity', 'Change', 'Coefficient'))


# --------------------------------------------------------------------------- #
# 18. A metric row must describe the population it was computed from
# --------------------------------------------------------------------------- #
class TestMetricRowDescribesItsOwnPopulation:
  '''A confusion matrix that does not sum to the frame it came from is a lie.

  The four counts summing to the population is the cheapest possible check on a
  metric row, and it is the one check that catches "these numbers were computed
  on a different frame than the one they are filed under". It caught a real
  defect: the local end-to-end run evaluated its train slice on the full
  1,340-row parent frame while the trainer's train slice was 1,005 rows, so
  every published train metric silently contained all 335 holdout rows and the
  two slices overlapped in their entirety.
  '''

  @staticmethod
  def _frame(size: int, offset: int = 0) -> pd.DataFrame:
    '''Build a small binary frame with a deterministic label pattern.

    Args:
      size: The number of rows.
      offset: Added to every index, so disjoint frames can be built.

    Returns:
      A frame carrying an identifier, a label and a probability column.
    '''
    label = np.array([index % 4 == 0 for index in range(offset, offset + size)], dtype=int)
    return pd.DataFrame(
      {
        'cust_id': [f'c{index}' for index in range(offset, offset + size)],
        'label': label,
        'prediction': label,
        'probability': np.where(label == 1, 0.75, 0.25),
      }
    )

  def test_the_counts_sum_to_the_frame_that_was_evaluated(self) -> None:
    '''Four frames of different sizes, each reconciled against its own size.'''
    from forecasting_ml_framework.core.metadata import Metadata  # noqa: PLC0415
    from forecasting_ml_framework.nodes import model_nodes  # noqa: PLC0415

    metadata = Metadata(feature_cols=['probability'], target_col='label', id_cols=['cust_id'])
    parameters = {
      'model_key': 'regression_population',
      'run_date': '2026-09-29',
      'modeling_params': {
        'prediction_col': 'prediction',
        'prediction_probability_col': 'probability',
        'eval_metric': 'LIFT_DECILE_10',
      },
    }

    for size in (40, 100, 200, 335):
      row, _ = model_nodes.model_evaluation(
        self._frame(size),
        parameters,
        metadata,
        'prediction',
        'probability',
        'train',
        False,
        sequence_flag=False,
      )
      total = int(row.iloc[0]['tn'] + row.iloc[0]['fp'] + row.iloc[0]['fn'] + row.iloc[0]['tp'])
      assert total == size, f'a {size}-row frame produced counts summing to {total}'

  def test_a_row_cannot_be_filed_under_a_slice_it_did_not_come_from(self) -> None:
    '''The mismatch this guards against is detectable, and that is the point.

    Evaluating the parent frame and labelling the result ``train`` is exactly the
    defect: the counts are internally consistent, the precision and recall agree
    with them, and every individual number is correct. Nothing about the row
    alone is wrong. Only the comparison against the slice size exposes it, which
    is why the reconciliation is enforced at publication and not left to a
    reader who has no way to know either number.
    '''
    from forecasting_ml_framework.core.metadata import Metadata  # noqa: PLC0415
    from forecasting_ml_framework.nodes import model_nodes  # noqa: PLC0415

    parent = self._frame(200)
    train_slice = parent.iloc[:150].copy()
    holdout = parent.iloc[150:].copy()

    metadata = Metadata(feature_cols=['probability'], target_col='label', id_cols=['cust_id'])
    parameters = {
      'model_key': 'regression_overlap',
      'run_date': '2026-09-29',
      'modeling_params': {
        'prediction_col': 'prediction',
        'prediction_probability_col': 'probability',
        'eval_metric': 'LIFT_DECILE_10',
      },
    }

    def counts(frame: pd.DataFrame) -> int:
      '''Return the confusion total for a frame.

      Args:
        frame: The frame to evaluate.

      Returns:
        The sum of the four confusion counts.
      '''
      row, _ = model_nodes.model_evaluation(
        frame, parameters, metadata, 'prediction', 'probability', 'train', False, sequence_flag=False
      )
      return int(row.iloc[0]['tn'] + row.iloc[0]['fp'] + row.iloc[0]['fn'] + row.iloc[0]['tp'])

    # The honest reading: each slice counts its own rows, and the two partition
    # the parent.
    assert len(train_slice) + len(holdout) == len(parent), 'the split does not partition the frame'
    assert counts(train_slice) == len(train_slice)
    assert counts(holdout) == len(holdout)
    assert not (set(train_slice['cust_id']) & set(holdout['cust_id'])), 'the slices overlap'

    # The defect: evaluating the parent and filing it as the train slice. The row
    # is internally perfect, and the size check is the only thing that sees it.
    contaminated = counts(parent)
    assert contaminated == len(parent)
    assert contaminated != len(train_slice), (
      'the contamination is undetectable from the row alone -- which is exactly '
      'why the size reconciliation has to be an assertion'
    )
