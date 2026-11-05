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
Path jail checks stop keys escaping a store root.
'''
# pylint: disable=import-outside-toplevel, unused-variable
# pylint: disable=broad-exception-caught

import abc
import io
import os
import pickle
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
  # date/time helpers embedded in pandas/sklearn pickles
  ('dateutil.tz.tz', 'tzfile'),
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
  before this call; this function is defence in depth.

  Args:
    raw: Pickle byte string.

  Returns:
    The unpickled object.

  Raises:
    ValueError: When the pickle references a forbidden global
      or is malformed.
  '''
  if not isinstance(raw, (bytes, bytearray)):
    raise ValueError('restricted_loads requires bytes')
  try:
    return _RestrictedUnpickler(io.BytesIO(bytes(raw))).load()
  except ValueError:
    raise
  except Exception as exc:
    raise ValueError('Invalid or forbidden pickle payload') from exc


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

    Args:
      path: Relative source path.

    Returns:
      The stored string, bytes, or picklable object.
    '''
    with open(self._full_path(path), 'rb') as handle:
      raw = handle.read()
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

    Args:
      path: Object key in the bucket.

    Returns:
      The stored string, bytes, or picklable object.
    '''
    key = _validate_cloud_key(path)
    raw = self._client().bucket(self.bucket).blob(key).download_as_bytes()
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

    Args:
      path: Object key in the bucket.

    Returns:
      The stored string, bytes, or picklable object.
    '''
    key = _validate_cloud_key(path)
    obj = self._client().get_object(Bucket=self.bucket, Key=key)
    return _decode_payload(obj['Body'].read())

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
