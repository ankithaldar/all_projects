#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Artifact storage backends: local, GCS, and S3.

Trust boundary: artifacts are trusted only when produced by this
framework. Loading uses a restricted unpickler (see
:func:`restricted_loads`) with an EXACT ``(module, name)``
allowlist of classes/reconstruction helpers for sklearn, numpy,
pandas, and verified zoo models (xgboost/lightgbm/catboost).
Prefix/module-wide allowlisting is NOT used:
loader functions such as ``pandas.read_pickle``/``read_csv``,
``numpy.load``, ``eval``/``exec``/``open``, and any
``os``/``subprocess``/``sys`` global are rejected, so a chained
``__reduce__`` payload cannot reach a file loader or shell.
A ``pickletools`` pre-scan additionally rejects ``REDUCE`` /
``NEWOBJ_EX`` construction of allowlisted ``Booster`` classes:
framework pickles rebuild a ``Booster`` only via ``NEWOBJ``
with empty args plus ``BUILD`` state, so a file-backed
constructor call (valid model file, ``/etc/passwd``, a FIFO
that would block forever) is pure attack surface and is
rejected before any file is opened. Pickle payloads larger
than ``MAX_ARTIFACT_BYTES`` (default 512 MB, env-overridable)
are rejected before unpickling.
Path jail checks stop keys escaping a store root.
'''
# pylint: disable=import-outside-toplevel, unused-variable
# pylint: disable=broad-exception-caught

import abc
import io
import os
import pickle
import pickletools
from pathlib import Path

# EXACT (module, name) allowlist. No prefix/module-wide entries:
# any global not listed here (including every read_*/load/loader,
# eval/exec/open, and os/subprocess/sys callable) is rejected.
# Entries are classes/constructors and the narrow reconstruction
# helpers sklearn/numpy/pandas pickles actually reference.
_SAFE_GLOBALS = {
  ('builtins', 'set'),
  ('builtins', 'frozenset'),
  ('builtins', 'slice'),
  ('builtins', 'range'),
  ('builtins', 'bytearray'),
  ('builtins', 'complex'),
  ('builtins', 'object'),
  ('copyreg', '__newobj__'),
  ('copyreg', '__newobj_ex__'),
  ('copyreg', '_reconstructor'),
  ('__builtin__', 'set'),
  ('__builtin__', 'frozenset'),
  ('collections', 'OrderedDict'),
  ('collections', 'defaultdict'),
  ('collections', 'Counter'),
  ('collections', 'deque'),
  ('datetime', 'datetime'),
  ('datetime', 'date'),
  ('datetime', 'time'),
  ('datetime', 'timedelta'),
  ('datetime', 'timezone'),
  ('decimal', 'Decimal'),
  ('fractions', 'Fraction'),
  ('uuid', 'UUID'),
  ('pathlib', 'PosixPath'),
  ('pathlib', 'WindowsPath'),
  ('pathlib', 'PurePosixPath'),
  ('pathlib', 'PureWindowsPath'),
  ('pathlib', 'PurePath'),
  ('re', 'Pattern'),
  # numpy (both numpy<2 `numpy.core` and numpy>=2 `numpy._core`)
  ('numpy', 'dtype'),
  ('numpy', 'ndarray'),
  ('numpy', 'int64'),
  ('numpy', 'int32'),
  ('numpy', 'float64'),
  ('numpy', 'float32'),
  ('numpy', 'bool_'),
  ('numpy', 'str_'),
  ('numpy.core.multiarray', '_reconstruct'),
  ('numpy._core.multiarray', '_reconstruct'),
  ('numpy.core.multiarray', 'scalar'),
  ('numpy._core.multiarray', 'scalar'),
  ('numpy.core.numeric', '_frombuffer'),
  ('numpy._core.numeric', '_frombuffer'),
  ('numpy.core._multiarray_umath', '_reconstruct'),
  ('numpy._core._multiarray_umath', '_reconstruct'),
  ('numpy.dtypes', 'Float64DType'),
  ('numpy.dtypes', 'Float32DType'),
  ('numpy.dtypes', 'Int64DType'),
  ('numpy.dtypes', 'Int32DType'),
  ('numpy.dtypes', 'BoolDType'),
  ('numpy.dtypes', 'StrDType'),
  ('numpy.dtypes', 'ObjectDType'),
  ('numpy.dtypes', 'DateTime64DType'),
  ('numpy.dtypes', 'Timedelta64DType'),
  ('numpy.dtypes', 'VoidDType'),
  # pandas classes / reconstruction helpers (no read_* / load)
  ('pandas', 'DataFrame'),
  ('pandas', 'Series'),
  ('pandas', 'Index'),
  ('pandas', 'RangeIndex'),
  ('pandas', 'MultiIndex'),
  ('pandas', 'Categorical'),
  ('pandas', 'CategoricalDtype'),
  ('pandas', 'StringDtype'),
  ('pandas', 'Interval'),
  ('pandas', 'IntervalDtype'),
  ('pandas', 'Period'),
  ('pandas', 'PeriodDtype'),
  ('pandas.core.frame', 'DataFrame'),
  ('pandas.core.series', 'Series'),
  ('pandas.core.indexes.base', 'Index'),
  ('pandas.core.indexes.base', '_new_Index'),
  ('pandas.core.indexes.range', 'RangeIndex'),
  ('pandas.core.indexes.multi', 'MultiIndex'),
  ('pandas.core.indexes.category', 'CategoricalIndex'),
  ('pandas.core.indexes.datetimes', 'DatetimeIndex'),
  ('pandas.core.indexes.timedeltas', 'TimedeltaIndex'),
  ('pandas.core.indexes.period', 'PeriodIndex'),
  ('pandas.core.indexes.interval', 'IntervalIndex'),
  ('pandas.core.indexes.extension', 'ExtensionIndex'),
  ('pandas.core.internals.managers', 'BlockManager'),
  ('pandas.core.internals.managers', 'SingleBlockManager'),
  ('pandas.core.internals.managers', 'SingleArrayManager'),
  ('pandas.core.internals.managers', 'ArrayManager'),
  ('pandas._libs.internals', '_unpickle_block'),
  ('pandas._libs.internals', 'Block'),
  ('pandas._libs.internals', 'NumericBlock'),
  ('pandas._libs.internals', 'DatetimeLikeBlock'),
  ('pandas._libs.internals', 'ExtensionBlock'),
  ('pandas._libs.internals', 'ObjectBlock'),
  ('pandas._libs.internals', 'DatetimeBlock'),
  ('pandas._libs.internals', 'TimedeltaBlock'),
  ('pandas._libs.internals', 'CategoricalBlock'),
  ('pandas._libs.arrays', '__pyx_unpickle_NDArrayBacked'),
  ('pandas._libs.missing', 'NAType'),
  ('pandas._libs.tslibs.timestamps', 'Timestamp'),
  ('pandas._libs.tslibs.timedeltas', 'Timedelta'),
  ('pandas._libs.tslibs.nattype', 'NaTType'),
  ('pandas._libs.tslibs.nattype', 'NaT'),
  ('pandas._libs.tslibs.period', 'Period'),
  ('pandas._libs.tslibs.offsets', 'DateOffset'),
  ('pandas.core.arrays', 'DatetimeArray'),
  ('pandas.core.arrays', 'TimedeltaArray'),
  ('pandas.core.arrays', 'PeriodArray'),
  ('pandas.core.arrays', 'IntervalArray'),
  ('pandas.core.arrays', 'Categorical'),
  ('pandas.core.arrays', 'IntegerArray'),
  ('pandas.core.arrays', 'FloatingArray'),
  ('pandas.core.arrays', 'BooleanArray'),
  ('pandas.core.arrays', 'StringArray'),
  ('pandas.core.arrays', 'ArrowExtensionArray'),
  ('pandas.core.arrays', 'NumpyExtensionArray'),
  ('pandas.arrays', 'DatetimeArray'),
  ('pandas.arrays', 'TimedeltaArray'),
  ('pandas.arrays', 'PeriodArray'),
  ('pandas.arrays', 'IntervalArray'),
  ('pandas.arrays', 'ArrowStringArray'),
  ('pandas.arrays', 'ArrowExtensionArray'),
  ('pandas.arrays', 'NumpyExtensionArray'),
  ('pandas.core.dtypes.dtypes', 'CategoricalDtype'),
  ('pandas.core.dtypes.dtypes', 'DatetimeTZDtype'),
  ('pandas.core.dtypes.dtypes', 'PeriodDtype'),
  ('pandas.core.dtypes.dtypes', 'IntervalDtype'),
  ('pandas.core.dtypes.dtypes', 'SparseDtype'),
  ('pandas.core.dtypes.dtypes', 'ArrowDtype'),
  ('pandas.core.dtypes.common', 'CoerceMixin'),
  # pyarrow array reconstruction (data only, no file access)
  ('pyarrow.lib', '_restore_array'),
  ('pyarrow.lib', 'type_for_alias'),
  ('pyarrow.lib', 'py_buffer'),
  ('pyarrow.lib', 'ChunkedArray'),
  ('pyarrow.lib', 'Array'),
  ('pyarrow', 'ChunkedArray'),
  ('pyarrow', 'Array'),
  ('pyarrow', 'Table'),
  # scipy sparse containers used inside sklearn estimators
  ('scipy.sparse._csr', 'csr_matrix'),
  ('scipy.sparse._csc', 'csc_matrix'),
  ('scipy.sparse._coo', 'coo_matrix'),
  ('scipy.sparse._bsr', 'bsr_matrix'),
  ('scipy.sparse._dia', 'dia_matrix'),
  ('scipy.sparse._dok', 'dok_matrix'),
  ('scipy.sparse._lil', 'lil_matrix'),
  ('scipy.sparse.csr', 'csr_matrix'),
  ('scipy.sparse.csc', 'csc_matrix'),
  ('scipy.sparse.coo', 'coo_matrix'),
  ('scipy.sparse.bsr', 'bsr_matrix'),
  ('scipy.sparse.dia', 'dia_matrix'),
  ('scipy.sparse.dok', 'dok_matrix'),
  ('scipy.sparse.lil', 'lil_matrix'),
  # sklearn estimators / pipelines / transformers (classes only)
  ('sklearn.linear_model._base', 'LinearRegression'),
  ('sklearn.linear_model._base', 'ElasticNet'),
  ('sklearn.linear_model._base', 'Lasso'),
  ('sklearn.linear_model._base', 'Ridge'),
  ('sklearn.linear_model._ridge', 'Ridge'),
  ('sklearn.linear_model._ridge', 'RidgeCV'),
  ('sklearn.linear_model._ridge', 'RidgeClassifier'),
  ('sklearn.linear_model._logistic', 'LogisticRegression'),
  ('sklearn.linear_model._logistic', 'LogisticRegressionCV'),
  ('sklearn.linear_model._logistic', 'SGDClassifier'),
  ('sklearn.linear_model._coordinate_descent', 'Lasso'),
  ('sklearn.linear_model._coordinate_descent', 'ElasticNet'),
  ('sklearn.linear_model._coordinate_descent', 'Lars'),
  ('sklearn.linear_model._huber', 'HuberRegressor'),
  ('sklearn.linear_model._theil_sen', 'TheilSenRegressor'),
  ('sklearn.linear_model._ransac', 'RANSACRegressor'),
  ('sklearn.linear_model._passive_aggressive', 'PassiveAggressiveRegressor'),
  ('sklearn.linear_model._passive_aggressive', 'PassiveAggressiveClassifier'),
  ('sklearn.linear_model._perceptron', 'Perceptron'),
  ('sklearn.linear_model._stochastic_gradient', 'SGDRegressor'),
  ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier'),
  ('sklearn.linear_model._least_angle', 'Lars'),
  ('sklearn.linear_model._least_angle', 'LassoLars'),
  ('sklearn.linear_model._least_angle', 'LassoLarsIC'),
  ('sklearn.linear_model._quantile_regression', 'QuantileRegressor'),
  ('sklearn.linear_model._bayes', 'BayesianRidge'),
  ('sklearn.linear_model._bayes', 'ARDRegression'),
  ('sklearn.ensemble._forest', 'RandomForestRegressor'),
  ('sklearn.ensemble._forest', 'RandomForestClassifier'),
  ('sklearn.ensemble._forest', 'ExtraTreesRegressor'),
  ('sklearn.ensemble._forest', 'ExtraTreesClassifier'),
  ('sklearn.ensemble._gb', 'GradientBoostingRegressor'),
  ('sklearn.ensemble._gb', 'GradientBoostingClassifier'),
  ('sklearn.ensemble._hist_gradient_boosting',
   'HistGradientBoostingRegressor'),
  ('sklearn.ensemble._hist_gradient_boosting',
   'HistGradientBoostingClassifier'),
  ('sklearn.ensemble._iforest', 'IsolationForest'),
  ('sklearn.ensemble._bagging', 'BaggingRegressor'),
  ('sklearn.ensemble._bagging', 'BaggingClassifier'),
  ('sklearn.ensemble._adaboosting', 'AdaBoostRegressor'),
  ('sklearn.ensemble._adaboosting', 'AdaBoostClassifier'),
  ('sklearn.ensemble._stacking', 'StackingRegressor'),
  ('sklearn.ensemble._stacking', 'StackingClassifier'),
  ('sklearn.ensemble._voting', 'VotingRegressor'),
  ('sklearn.ensemble._voting', 'VotingClassifier'),
  ('sklearn.tree._classes', 'DecisionTreeRegressor'),
  ('sklearn.tree._classes', 'DecisionTreeClassifier'),
  ('sklearn.tree._classes', 'ExtraTreeRegressor'),
  ('sklearn.tree._classes', 'ExtraTreeClassifier'),
  ('sklearn.tree._tree', 'Tree'),
  ('sklearn.pipeline', 'Pipeline'),
  ('sklearn.pipeline', 'FeatureUnion'),
  ('sklearn.compose._column_transformer', 'ColumnTransformer'),
  ('sklearn.preprocessing._data', 'StandardScaler'),
  ('sklearn.preprocessing._data', 'MinMaxScaler'),
  ('sklearn.preprocessing._data', 'RobustScaler'),
  ('sklearn.preprocessing._data', 'MaxAbsScaler'),
  ('sklearn.preprocessing._data', 'Normalizer'),
  ('sklearn.preprocessing._data', 'QuantileTransformer'),
  ('sklearn.preprocessing._data', 'PowerTransformer'),
  ('sklearn.preprocessing._data', 'FunctionTransformer'),
  ('sklearn.preprocessing._label', 'LabelEncoder'),
  ('sklearn.preprocessing._label', 'OneHotEncoder'),
  ('sklearn.preprocessing._label', 'OrdinalEncoder'),
  ('sklearn.preprocessing._label', 'MultiLabelBinarizer'),
  ('sklearn.preprocessing._discretization', 'KBinsDiscretizer'),
  ('sklearn.preprocessing._polynomial', 'PolynomialFeatures'),
  ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
  ('sklearn.feature_extraction.text', 'TfidfVectorizer'),
  ('sklearn.feature_extraction.text', 'CountVectorizer'),
  ('sklearn.feature_extraction.text', 'TfidfTransformer'),
  ('sklearn.feature_extraction.text', 'CountVectorizer'),
  ('sklearn.feature_selection._univariate_selection', 'SelectKBest'),
  ('sklearn.feature_selection._univariate_selection', 'SelectPercentile'),
  ('sklearn.feature_selection._rfe', 'RFE'),
  ('sklearn.feature_selection._rfe', 'RFECV'),
  ('sklearn.decomposition._pca', 'PCA'),
  ('sklearn.decomposition._pca', 'TruncatedSVD'),
  ('sklearn.decomposition._pca', 'IncrementalPCA'),
  ('sklearn.decomposition._factor_analysis', 'FactorAnalysis'),
  ('sklearn.decomposition._fastica', 'FastICA'),
  ('sklearn.decomposition._nmf', 'NMF'),
  ('sklearn.decomposition._lda', 'LinearDiscriminantAnalysis'),
  ('sklearn.svm._base', 'BaseLibSVM'),
  ('sklearn.svm._classes', 'SVR'),
  ('sklearn.svm._classes', 'SVC'),
  ('sklearn.svm._classes', 'NuSVR'),
  ('sklearn.svm._classes', 'NuSVC'),
  ('sklearn.svm._classes', 'LinearSVR'),
  ('sklearn.svm._classes', 'LinearSVC'),
  ('sklearn.neighbors._regression', 'KNeighborsRegressor'),
  ('sklearn.neighbors._classification', 'KNeighborsClassifier'),
  ('sklearn.neighbors._uniform_weight', 'KNeighborsRegressor'),
  ('sklearn.neighbors._uniform_weight', 'KNeighborsClassifier'),
  ('sklearn.neighbors._kd_tree', 'KDTree'),
  ('sklearn.neighbors._ball_tree', 'BallTree'),
  ('sklearn.naive_bayes', 'GaussianNB'),
  ('sklearn.naive_bayes', 'MultinomialNB'),
  ('sklearn.naive_bayes', 'BernoulliNB'),
  ('sklearn.naive_bayes', 'ComplementNB'),
  ('sklearn.cluster._kmeans', 'KMeans'),
  ('sklearn.cluster._kmeans', 'MiniBatchKMeans'),
  ('sklearn.cluster._agglomerative', 'AgglomerativeClustering'),
  ('sklearn.cluster._dbscan', 'DBSCAN'),
  ('sklearn.cluster._birch', 'Birch'),
  ('sklearn.neural_network._multilayer_perceptron', 'MLPRegressor'),
  ('sklearn.neural_network._multilayer_perceptron', 'MLPClassifier'),
  ('sklearn.neural_network._base', 'BaseMultilayerPerceptron'),
  ('sklearn.neural_network._stochastic_gradient', 'SGDRegressor'),
  ('sklearn.utils._array_api', 'ArrayAPIWrapper'),
  ('sklearn.utils._tags', 'Tags'),
  # date/time helpers embedded in pandas/sklearn pickles.
  # NOTE: dateutil tzfile is deliberately NOT allowlisted: its
  # constructor opens the path it is given, so allowing it would
  # be a file-existence oracle (and its real pickle global is
  # dateutil.tz._tzfile.tzfile / _unpickle, neither of which is
  # needed -- modern pandas uses zoneinfo/pytz, both covered).
  ('dateutil.tz.tz', 'tzutc'),
  ('dateutil.tz.tz', 'tzoffset'),
  ('pytz.tzinfo', 'DstTzInfo'),
  ('pytz.tzinfo', 'StaticTzInfo'),
  ('pytz', 'UTC'),
  # zoo models verified by inspecting pickle globals of tiny
  # trained models in the uv venv (see tests/test_core.py):
  # xgboost Booster + sklearn wrappers, lightgbm Booster +
  # sklearn wrappers, catboost sklearn wrappers. No loaders,
  # no os/subprocess/sys, no DMatrix/Pool (not picklable /
  # not needed to unpickle a fitted model).
  ('xgboost.core', 'Booster'),
  ('xgboost.sklearn', 'XGBRegressor'),
  ('xgboost.sklearn', 'XGBClassifier'),
  ('lightgbm.basic', 'Booster'),
  ('lightgbm.sklearn', 'LGBMRegressor'),
  ('lightgbm.sklearn', 'LGBMClassifier'),
  ('catboost.core', 'CatBoostRegressor'),
  ('catboost.core', 'CatBoostClassifier'),
}

# Booster classes whose constructor accepts a filesystem path
# (xgboost/lightgbm ``model_file``, 3rd positional parameter).
# Their framework pickles use NEWOBJ with empty args + BUILD,
# never REDUCE/NEWOBJ_EX, so those call shapes are rejected
# by the pre-scan below.
_BOOSTER_GLOBALS = frozenset({
  ('xgboost.core', 'Booster'),
  ('lightgbm.basic', 'Booster'),
})

_DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024

_MARK = ('mark',)
_UNKNOWN = ('unknown',)

_VALUE_OPCODES = frozenset({
  'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
  'STRING', 'BINSTRING', 'SHORT_BINSTRING',
  'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8',
  'UNICODE', 'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8',
  'FLOAT', 'BINFLOAT', 'NONE', 'NEWTRUE', 'NEWFALSE',
})


def _max_artifact_bytes():
  '''Return the artifact size cap in bytes.

  Read from the ``MAX_ARTIFACT_BYTES`` environment variable on
  every call so operators and tests can override it without
  reimporting. Invalid or non-positive values fall back to
  the default.

  Returns:
    Size cap in bytes.
  '''
  text = os.environ.get('MAX_ARTIFACT_BYTES')
  if text:
    try:
      cap = int(text)
    except ValueError:
      cap = 0
    if cap > 0:
      return cap
  return _DEFAULT_MAX_ARTIFACT_BYTES


#: Chunk size for capped streaming reads (1 MiB).
_READ_CHUNK_BYTES = 1024 * 1024


def _read_capped(handle, cap=None):
  '''Read a binary stream in chunks, rejecting once over ``cap``.

  Reads at most ``cap + 1 MiB`` bytes: each 1 MiB chunk is
  checked against the running total before it is retained, and
  the first chunk that pushes the total over ``cap`` raises
  without reading further, so an oversized payload is never
  fully materialised in memory (C2-F2/R-03). This single helper
  backs local, GCS, S3, and pickle-dataset loads.

  Args:
    handle: Binary file-like object with a ``read(size)`` method.
    cap: Maximum bytes to accept; None uses
      :func:`_max_artifact_bytes`.

  Returns:
    The stream bytes (at most ``cap`` bytes).

  Raises:
    ValueError: When the running total exceeds ``cap``.
  '''
  limit = _max_artifact_bytes() if cap is None else cap
  chunks = []
  total = 0
  while True:
    chunk = handle.read(_READ_CHUNK_BYTES)
    if not chunk:
      break
    total += len(chunk)
    if total > limit:
      raise ValueError(
        'Artifact exceeds MAX_ARTIFACT_BYTES size limit')
    chunks.append(chunk)
  return b''.join(chunks)


def _is_booster_token(token):
  '''Return True when ``token`` names an allowlisted Booster.

  Args:
    token: Symbolic pickle stack token.

  Returns:
    True for xgboost/lightgbm Booster global tokens.
  '''
  return (
      isinstance(token, tuple) and len(token) == 3
      and token[0] == 'global'
      and (token[1], token[2]) in _BOOSTER_GLOBALS)


def _token_string(token):
  '''Return the string carried by a value token, if any.

  Args:
    token: Symbolic pickle stack token.

  Returns:
    The string value, or an empty string.
  '''
  if (isinstance(token, tuple) and len(token) == 2
      and token[0] == 'value' and isinstance(token[1], str)):
    return token[1]
  return ''


def _pop_token(stack):
  '''Pop one symbolic token, or return unknown when empty.

  Args:
    stack: Mutable symbolic pickle stack.

  Returns:
    The popped token.
  '''
  if stack:
    return stack.pop()
  return _UNKNOWN


def _pop_until_mark(stack):
  '''Pop tokens up to and including the nearest mark.

  Args:
    stack: Mutable symbolic pickle stack.

  Returns:
    Tokens popped above the mark, in stack order.
  '''
  items = []
  while stack and stack[-1] is not _MARK:
    items.append(stack.pop())
  if stack:
    stack.pop()
  items.reverse()
  return items


def _dict_with_item(token, key, value):
  '''Return a dict token with one item added, when possible.

  Args:
    token: Base symbolic token.
    key: Key symbolic token.
    value: Value symbolic token.

  Returns:
    The updated dict token, or unknown for non-dict bases.
  '''
  if (isinstance(token, tuple) and token and token[0] == 'dict'):
    return ('dict', token[1] + ((key, value),))
  return _UNKNOWN


def _global_token_from_arg(arg):
  '''Build a global token from a GLOBAL/INST opcode argument.

  Args:
    arg: ``'<module> <name>'`` string from ``pickletools``.

  Returns:
    A global token, or unknown when malformed.
  '''
  if isinstance(arg, str):
    parts = arg.split()
    if len(parts) == 2:
      return ('global', parts[0], parts[1])
  return _UNKNOWN


def _prescan_pickle(raw):
  '''Reject Booster construction before any unpickling happens.

  A symbolic ``pickletools`` stack walk identifies the target
  of every ``REDUCE`` / ``NEWOBJ_EX`` / ``NEWOBJ`` / ``OBJ`` /
  ``INST`` opcode, following memo references, so indirection
  through ``MEMOIZE``/``BINGET`` cannot hide the target. Any
  such call targeting an allowlisted ``Booster`` class is
  rejected: framework pickles only rebuild a ``Booster`` via
  ``NEWOBJ`` with empty args plus ``BUILD`` state, so a
  constructor call carrying a model file (including a FIFO,
  which would block the unpickler indefinitely) never occurs
  in a legitimate artifact. The rejection message is static
  and never echoes a probed path.

  Args:
    raw: Pickle byte string to scan.

  Raises:
    ValueError: When the pickle is malformed or constructs
      a ``Booster`` via a rejected call shape.
  '''
  try:
    ops = list(pickletools.genops(raw))
  except Exception:
    raise ValueError('Invalid or forbidden pickle payload') from None
  stack = []
  memo = {}
  for opcode, arg, pos in ops:
    name = opcode.name
    if name == 'MARK':
      stack.append(_MARK)
    elif name in _VALUE_OPCODES:
      stack.append(('value', arg))
    elif name == 'GLOBAL':
      stack.append(_global_token_from_arg(arg))
    elif name == 'STACK_GLOBAL':
      name_token = _pop_token(stack)
      module_token = _pop_token(stack)
      stack.append(
          ('global', _token_string(module_token),
           _token_string(name_token)))
    elif name == 'MEMOIZE':
      if stack:
        memo[len(memo)] = stack[-1]
    elif name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
      if stack:
        memo[arg] = stack[-1]
    elif name in ('GET', 'BINGET', 'LONG_BINGET'):
      stack.append(memo.get(arg, _UNKNOWN))
    elif name == 'EMPTY_TUPLE':
      stack.append(('tuple', ()))
    elif name == 'EMPTY_DICT':
      stack.append(('dict', ()))
    elif name == 'TUPLE':
      stack.append(('tuple', tuple(_pop_until_mark(stack))))
    elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
      items = [_pop_token(stack) for _ in range(int(name[-1]))]
      items.reverse()
      stack.append(('tuple', tuple(items)))
    elif name == 'DICT':
      items = _pop_until_mark(stack)
      if len(items) % 2 == 0:
        pairs = tuple(
            (items[i], items[i + 1])
            for i in range(0, len(items), 2))
        stack.append(('dict', pairs))
      else:
        stack.append(_UNKNOWN)
    elif name in ('LIST', 'FROZENSET'):
      _pop_until_mark(stack)
      stack.append(_UNKNOWN)
    elif name == 'SETITEM':
      value = _pop_token(stack)
      key = _pop_token(stack)
      base = _pop_token(stack)
      stack.append(_dict_with_item(base, key, value))
    elif name == 'SETITEMS':
      items = _pop_until_mark(stack)
      base = _pop_token(stack)
      for i in range(0, len(items) - 1, 2):
        base = _dict_with_item(base, items[i], items[i + 1])
      stack.append(base)
    elif name == 'APPEND':
      _pop_token(stack)
      stack.append(_pop_token(stack))
    elif name in ('APPENDS', 'ADDITEMS'):
      _pop_until_mark(stack)
      stack.append(_pop_token(stack))
    elif name == 'BUILD':
      _pop_token(stack)
      stack.append(_pop_token(stack))
    elif name == 'REDUCE':
      _pop_token(stack)
      func = _pop_token(stack)
      if _is_booster_token(func):
        raise ValueError(
            'Forbidden pickle payload: Booster construction '
            'is not allowed')
      stack.append(_UNKNOWN)
    elif name == 'NEWOBJ':
      args = _pop_token(stack)
      cls = _pop_token(stack)
      empty = (
          isinstance(args, tuple) and len(args) == 2
          and args[0] == 'tuple' and args[1] == ())
      if _is_booster_token(cls) and not empty:
        raise ValueError(
            'Forbidden pickle payload: Booster construction '
            'is not allowed')
      stack.append(_UNKNOWN)
    elif name == 'NEWOBJ_EX':
      _pop_token(stack)
      _pop_token(stack)
      cls = _pop_token(stack)
      if _is_booster_token(cls):
        raise ValueError(
            'Forbidden pickle payload: Booster construction '
            'is not allowed')
      stack.append(_UNKNOWN)
    elif name == 'OBJ':
      items = _pop_until_mark(stack)
      if items and _is_booster_token(items[0]):
        raise ValueError(
            'Forbidden pickle payload: Booster construction '
            'is not allowed')
      stack.append(_UNKNOWN)
    elif name == 'INST':
      _pop_until_mark(stack)
      if _is_booster_token(_global_token_from_arg(arg)):
        raise ValueError(
            'Forbidden pickle payload: Booster construction '
            'is not allowed')
      stack.append(_UNKNOWN)
    elif name == 'POP':
      _pop_token(stack)
    elif name == 'POP_MARK':
      _pop_until_mark(stack)
    elif name == 'DUP':
      if stack:
        stack.append(stack[-1])
    elif name in ('PERSID', 'EXT1', 'EXT2', 'EXT4', 'NEXT_BUFFER'):
      stack.append(_UNKNOWN)
    elif name == 'BINPERSID':
      _pop_token(stack)
      stack.append(_UNKNOWN)
    elif name in ('PROTO', 'FRAME', 'STOP', 'READONLY_BUFFER'):
      pass
    else:
      stack.append(_UNKNOWN)


class _RestrictedUnpickler(pickle.Unpickler):
  '''Unpickler allowing only allowlisted sklearn/numpy/pandas/zoo
  model classes plus safe types.'''

  def find_class(self, module, name):
    '''Return allowlisted global or raise ValueError.

    Only EXACT ``(module, name)`` pairs in ``_SAFE_GLOBALS``
    pass; there is no prefix fallback.

    Args:
      module: Module name from the pickle stream.
      name: Attribute name from the pickle stream.

    Returns:
      The requested class or function.

    Raises:
      ValueError: When the global is not allowlisted.
    '''
    if (module, name) in _SAFE_GLOBALS:
      return super().find_class(module, name)
    raise ValueError(
        f'Forbidden pickle global: {module}.{name}')


def restricted_loads(raw):
  '''Unpickle ``raw`` with an allowlist, rejecting other globals.

  Trusted-artifact boundary: only artifacts written by this
  framework (sklearn/numpy/pandas/zoo objects and safe builtin
  types) should be loaded. Untrusted pickles must be rejected
  before this call; this function is defence in depth. The
  size cap and the Booster pre-scan run before unpickling, so
  an oversized payload or a file-backed ``Booster`` call is
  rejected without reading any probed path.

  Args:
    raw: Pickle byte string.

  Returns:
    The unpickled object.

  Raises:
    ValueError: When the pickle exceeds the size cap,
      references a forbidden global, constructs a ``Booster``
      via a rejected call shape, or is malformed.
  '''
  if not isinstance(raw, (bytes, bytearray)):
    raise ValueError('restricted_loads requires bytes')
  if len(raw) > _max_artifact_bytes():
    raise ValueError(
        'Artifact exceeds MAX_ARTIFACT_BYTES size limit')
  _prescan_pickle(bytes(raw))
  # Preserve only our own forbidden-global error (its message
  # names the attacker-supplied module/attribute, not a probed
  # path). Every other error -- including ValueError subclasses
  # from allowlisted constructors such as XGBoostError (a
  # ValueError) from Booster(model_file=...) and LightGBMError
  # from Booster(model_file=...) -- may embed a probed filesystem
  # path, so it is genericized. The generic error is raised
  # after the except blocks so neither __cause__ nor __context__
  # chains the original (path-bearing) exception.
  generic = None
  try:
    return _RestrictedUnpickler(io.BytesIO(bytes(raw))).load()
  except ValueError as exc:
    if str(exc).startswith('Forbidden pickle global:'):
      raise
    generic = 'Forbidden pickle payload'
  except Exception:  # pylint: disable=broad-exception-caught
    generic = 'Invalid or forbidden pickle payload'
  if generic is not None:
    raise ValueError(generic)


def _decode_payload(raw):
  '''Decode bytes as pickle (restricted), utf-8 text, or raw.

  Args:
    raw: Stored byte payload.

  Returns:
    The decoded payload.
  '''
  if raw[:1] == b'\x80':
    return restricted_loads(raw)
  try:
    return raw.decode('utf-8')
  except UnicodeDecodeError:
    return raw


def _validate_cloud_key(path):
  '''Validate an object key for GCS/S3 stores.

  Args:
    path: Object key or prefix to validate.

  Returns:
    The key as a string.

  Raises:
    ValueError: When the key is absolute or escapes via ``..``.
  '''
  key = str(path)
  if key.startswith('/') or key.startswith('\\'):
    raise ValueError('Absolute object keys are not allowed')
  parts = key.replace('\\', '/').split('/')
  if '..' in parts:
    raise ValueError('Object keys escaping the store are not allowed')
  return key


class ArtifactStore(abc.ABC):
  '''Abstract artifact store interface.

  Implementations must provide byte-level persistence behind a
  small, backend-neutral surface used by pipelines.
  '''

  @abc.abstractmethod
  def save(self, path, data):
    '''Save ``data`` at ``path``.

    Args:
      path: Destination object key or relative file path.
      data: Payload to persist (bytes, str, or picklable object).
    '''

  @abc.abstractmethod
  def load(self, path):
    '''Load and return the payload stored at ``path``.

    Args:
      path: Source object key or relative file path.

    Returns:
      The previously saved payload.
    '''

  @abc.abstractmethod
  def exists(self, path):
    '''Check whether ``path`` exists.

    Args:
      path: Object key or relative file path.

    Returns:
      True when an object exists at ``path``.
    '''

  @abc.abstractmethod
  def list(self, prefix=''):
    '''List object keys under ``prefix``.

    Args:
      prefix: Key prefix filter; empty lists everything.

    Returns:
      A sorted list of object keys.
    '''


class LocalArtifactStore(ArtifactStore):
  '''Store artifacts as files under a local directory (jailed).

  Args:
    base_path: Root directory for stored artifacts.
  '''

  def __init__(self, base_path='.'):
    '''Initialize with a filesystem root directory.

    Args:
      base_path: Root directory; created lazily on save.
    '''
    self.base_path = str(base_path)

  def _full_path(self, path):
    '''Resolve ``path`` inside the base directory (path jail).

    Args:
      path: Relative object key; empty means the base itself.

    Returns:
      The resolved absolute filesystem path.

    Raises:
      ValueError: When ``path`` is absolute or escapes base.
    '''
    text = str(path)
    if not text:
      return str(Path(self.base_path).resolve())
    if os.path.isabs(text) or Path(text).is_absolute():
      raise ValueError('Absolute paths are not allowed')
    base = Path(self.base_path).resolve()
    target = (base / text).resolve()
    if target != base and base not in target.parents:
      raise ValueError('Path escapes the artifact store base')
    return str(target)

  def save(self, path, data):
    '''Save ``data`` under the base directory.

    Args:
      path: Relative destination path.
      data: Bytes, string, or picklable object to persist.
    '''
    full = self._full_path(path)
    parent = os.path.dirname(full)
    if parent:
      os.makedirs(parent, exist_ok=True)
    if isinstance(data, bytes):
      with open(full, 'wb') as handle:
        handle.write(data)
    elif isinstance(data, str):
      with open(full, 'w', encoding='utf-8') as handle:
        handle.write(data)
    else:
      with open(full, 'wb') as handle:
        pickle.dump(data, handle)

  def load(self, path):
    '''Load the payload stored at ``path``.

    Pickle payloads go through :func:`restricted_loads`.
    The file is read via :func:`_read_capped`, so an oversized
    file is rejected without being fully read into memory.

    Args:
      path: Relative source path.

    Returns:
      The stored string, bytes, or picklable object.
    '''
    with open(self._full_path(path), 'rb') as handle:
      raw = _read_capped(handle)
    return _decode_payload(raw)

  def exists(self, path):
    '''Check whether ``path`` exists on disk.

    Args:
      path: Relative path to check.

    Returns:
      True when the file exists.
    '''
    return os.path.isfile(self._full_path(path))

  def list(self, prefix=''):
    '''List stored file paths under ``prefix``.

    Args:
      prefix: Relative directory prefix filter.

    Returns:
      Sorted relative paths of all files under ``prefix``.
    '''
    root = self._full_path(prefix) if prefix else self.base_path
    results = []
    if not os.path.isdir(root):
      return results
    for dirpath, dirnames, filenames in os.walk(root):
      for name in filenames:
        full = os.path.join(dirpath, name)
        results.append(os.path.relpath(full, self.base_path))
    return sorted(results)


class GCSArtifactStore(ArtifactStore):
  '''Store artifacts in a Google Cloud Storage bucket.

  The ``google-cloud-storage`` SDK is imported lazily, only when
  an operation runs, so importing this module never breaks local
  pandas-mode execution.

  Args:
    bucket: Name of the GCS bucket.
  '''

  def __init__(self, bucket):
    '''Initialize with only a bucket name (no SDK import).

    Args:
      bucket: Name of the GCS bucket.
    '''
    self.bucket = bucket

  def _client(self):
    '''Create a GCS client, importing the SDK lazily.

    Returns:
      A configured ``google.cloud.storage.Client``.

    Raises:
      ImportError: When the GCS SDK is not installed.
    '''
    try:
      from google.cloud import storage
    except ImportError as exc:
      raise ImportError(
          'google-cloud-storage is required for GCSArtifactStore'
      ) from exc
    return storage.Client()

  def save(self, path, data):
    '''Upload ``data`` to ``path`` in the bucket.

    Args:
      path: Object key in the bucket.
      data: Bytes, string, or picklable object to persist.
    '''
    key = _validate_cloud_key(path)
    blob = self._client().bucket(self.bucket).blob(key)
    if isinstance(data, bytes):
      blob.upload_from_string(data)
    elif isinstance(data, str):
      blob.upload_from_string(data.encode('utf-8'))
    else:
      blob.upload_from_string(pickle.dumps(data))

  def load(self, path):
    '''Download and decode the payload at ``path``.

    The blob is streamed via :func:`_read_capped` (blob
    ``open()``), so an oversized blob is rejected without
    being fully downloaded into memory. There is no
    ``download_as_bytes`` fallback: that call would
    materialise the whole blob before the size cap could
    be enforced.

    Args:
      path: Object key in the bucket.

    Returns:
      The stored string, bytes, or picklable object.

    Raises:
      RuntimeError: When the blob has no streaming
        ``open()``, so a capped read is impossible.
    '''
    key = _validate_cloud_key(path)
    blob = self._client().bucket(self.bucket).blob(key)
    open_method = getattr(blob, 'open', None)
    if not callable(open_method):
      raise RuntimeError('GCS blob has no streaming open()')
    with open_method('rb') as handle:
      raw = _read_capped(handle)
    return _decode_payload(raw)

  def exists(self, path):
    '''Check whether ``path`` exists in the bucket.

    Args:
      path: Object key in the bucket.

    Returns:
      True when the object exists.
    '''
    key = _validate_cloud_key(path)
    return self._client().bucket(self.bucket).blob(key).exists()

  def list(self, prefix=''):
    '''List object keys in the bucket under ``prefix``.

    Args:
      prefix: Object key prefix filter.

    Returns:
      Sorted object keys.
    '''
    if prefix:
      _validate_cloud_key(prefix)
    bucket = self._client().bucket(self.bucket)
    return sorted(blob.name for blob in bucket.list_blobs(
        prefix=str(prefix)))


class S3ArtifactStore(ArtifactStore):
  '''Store artifacts in an Amazon S3 bucket.

  The ``boto3`` SDK is imported lazily, only when an operation
  runs, so importing this module never breaks local execution.

  Args:
    bucket: Name of the S3 bucket.
  '''

  def __init__(self, bucket):
    '''Initialize with only a bucket name (no SDK import).

    Args:
      bucket: Name of the S3 bucket.
    '''
    self.bucket = bucket

  def _client(self):
    '''Create an S3 client, importing boto3 lazily.

    Returns:
      A configured boto3 S3 client.

    Raises:
      ImportError: When boto3 is not installed.
    '''
    try:
      import boto3
    except ImportError as exc:
      raise ImportError(
          'boto3 is required for S3ArtifactStore') from exc
    return boto3.client('s3')

  def save(self, path, data):
    '''Upload ``data`` to ``path`` in the bucket.

    Args:
      path: Object key in the bucket.
      data: Bytes, string, or picklable object to persist.
    '''
    key = _validate_cloud_key(path)
    if isinstance(data, bytes):
      body = data
    elif isinstance(data, str):
      body = data.encode('utf-8')
    else:
      body = pickle.dumps(data)
    self._client().put_object(Bucket=self.bucket, Key=key, Body=body)

  def load(self, path):
    '''Download and decode the payload at ``path``.

    The body is read via :func:`_read_capped`, so an oversized
    body is rejected without being fully read into memory.

    Args:
      path: Object key in the bucket.

    Returns:
      The stored string, bytes, or picklable object.
    '''
    key = _validate_cloud_key(path)
    obj = self._client().get_object(Bucket=self.bucket, Key=key)
    return _decode_payload(_read_capped(obj['Body']))

  def exists(self, path):
    '''Check whether ``path`` exists in the bucket.

    Args:
      path: Object key in the bucket.

    Returns:
      True when the object exists.
    '''
    key = _validate_cloud_key(path)
    try:
      self._client().head_object(Bucket=self.bucket, Key=key)
      return True
    except Exception:
      return False

  def list(self, prefix=''):
    '''List object keys in the bucket under ``prefix``.

    Args:
      prefix: Object key prefix filter.

    Returns:
      Sorted object keys.
    '''
    if prefix:
      _validate_cloud_key(prefix)
    paginator = self._client().get_paginator('list_objects_v2')
    keys = []
    for page in paginator.paginate(Bucket=self.bucket, Prefix=str(prefix)):
      for obj in page.get('Contents', []):
        keys.append(obj['Key'])
    return sorted(keys)


def create_artifact_store(config=None):
  '''Create an artifact store from configuration.

  Args:
    config: Mapping with a ``type`` of ``'local'``, ``'gcs'``, or
      ``'s3'``. A nested ``artifact_store`` (or ``store``) mapping
      is unwrapped automatically. Local stores use ``base_path``;
      cloud stores use ``bucket``. Missing config defaults to a
      local store rooted at ``./artifacts``.

  Returns:
    The configured :class:`ArtifactStore` instance.

  Raises:
    ValueError: When ``type`` names an unknown backend.
  '''
  config = dict(config) if config else {}
  nested = config.get('artifact_store') or config.get('store')
  if isinstance(nested, dict):
    config = dict(nested)
  store_type = str(
      config.get('type') or config.get('store_type')
      or config.get('backend') or 'local').lower()
  if store_type == 'local':
    base = config.get('base_path') or config.get('path') \
        or config.get('local_path') or './artifacts'
    return LocalArtifactStore(base)
  if store_type in ('gcs', 'gs', 'google'):
    return GCSArtifactStore(config.get('bucket') or config.get('bucket_name'))
  if store_type in ('s3', 'aws'):
    return S3ArtifactStore(config.get('bucket') or config.get('bucket_name'))
  raise ValueError(f'Unknown artifact store type: {store_type!r}')
