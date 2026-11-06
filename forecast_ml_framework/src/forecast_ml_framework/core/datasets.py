#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro datasets for SQL sources and Spark SQL sources.'''
# pylint: disable=import-outside-toplevel, super-init-not-called
# pylint: disable=broad-exception-caught

import os
import pickle
import re

try:
  from kedro.io import AbstractDataset as _KedroDataset
except ImportError:  # pragma: no cover - older/newer Kedro names
  try:
    from kedro.io.core import AbstractDataSet as _KedroDataset
  except ImportError:
    _KedroDataset = object


_FORBIDDEN_SQL_KEYWORDS = (
    'INTO', 'OUTFILE', 'DUMPFILE', 'COPY', 'DELETE', 'UPDATE',
    'INSERT', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'MERGE',
    'GRANT', 'REVOKE', 'CALL', 'EXECUTE', 'REPLACE', 'VACUUM',
    'ATTACH', 'DETACH', 'PRAGMA', 'REINDEX',
    'PG_SLEEP', 'PG_TERMINATE_BACKEND', 'PG_CANCEL_BACKEND',
    'PG_RELOAD_CONF', 'PG_ROTATE_LOGFILE', 'SLEEP', 'BENCHMARK',
    'LOAD_FILE', 'WAITFOR', 'XP_CMDSHELL',
)
_FORBIDDEN_SQL_RE = re.compile(
    r'\b(?:' + '|'.join(_FORBIDDEN_SQL_KEYWORDS) + r')\b',
    re.IGNORECASE)


def _strip_sql_literals(text):
  '''Remove single/double-quoted literals so keywords inside
  string values are not mistaken for SQL keywords.

  Args:
    text: SQL text to strip.

  Returns:
    SQL text with quoted literal contents blanked out.
  '''
  out = []
  quote = None
  i = 0
  while i < len(text):
    ch = text[i]
    if quote is not None:
      if ch == quote:
        if i + 1 < len(text) and text[i + 1] == quote:
          i += 2
          out.append(' ')
          continue
        quote = None
        out.append(' ')
      else:
        out.append(' ')
      i += 1
      continue
    if ch in ("'", '"'):
      quote = ch
      out.append(' ')
    else:
      out.append(ch)
    i += 1
  return ''.join(out)


def _validate_read_only_query(query):
  '''Enforce single-statement, read-only intent for a SQL query.

  Enforced rules: exactly one statement (a trailing ``;`` is
  allowed, no other ``;``), the statement must start with
  ``SELECT`` or ``WITH``, and the denylisted keywords
  (``INTO``, ``OUTFILE``, ``COPY``, ``DELETE``, ``UPDATE``,
  ``INSERT``, ``DROP``, ``ALTER``, ``CREATE``, ``TRUNCATE``,
  ``MERGE``, ``GRANT``, ``CALL``, ``EXECUTE``, and the
  ``pg_sleep``/sleep family, matched as whole words
  case-insensitively anywhere in the text, including inside
  CTE bodies) are rejected. ``UNION`` is deliberately allowed:
  it is a legitimate read, and any ``SELECT`` can already
  exfiltrate readable rows, so UNION exfiltration is inherent
  to read access rather than a write primitive.

  Trust boundary: ``query`` is trusted configuration (catalog /
  params), never untrusted user input. This is keyword
  enforcement, not a SQL sandbox: it cannot make arbitrary SQL
  from an untrusted source safe (e.g. a ``SELECT`` against a
  sensitive table still reads it). Only trusted queries must
  be used, and the DB account should be read-only.

  Args:
    query: SQL query string to validate.

  Returns:
    The query unchanged.

  Raises:
    ValueError: When the query is not a single read-only
      SELECT/WITH or contains a forbidden keyword.
  '''
  if not isinstance(query, str) or not query.strip():
    raise ValueError('SQLQueryDataSet requires a query')
  text = query.strip()
  if text.endswith(';'):
    text = text[:-1].rstrip()
  stripped = _strip_sql_literals(text)
  if ';' in stripped:
    raise ValueError('Only a single SQL statement is allowed')
  first = stripped.split(None, 1)[0].lower() if stripped.split() else ''
  if first not in ('select', 'with'):
    raise ValueError('Only SELECT/WITH read-only queries are allowed')
  match = _FORBIDDEN_SQL_RE.search(stripped)
  if match:
    raise ValueError(
        f'Forbidden SQL keyword in read-only query: {match.group(0)!r}')
  return query


class SQLQueryDataSet(_KedroDataset):
  '''Load a SQL query into a pandas DataFrame.

  Trust boundary: connection and query come from trusted
  configuration (Kedro catalog/params), never from untrusted
  user input. :func:`_validate_read_only_query` enforces a
  single SELECT/WITH statement with no DML/DDL/INTO/COPY or
  sleep keywords (see its docstring for the exact rules); it
  is keyword enforcement, not a SQL sandbox, so only trusted
  queries against a read-only DB account must be used.

  SQLAlchemy and pandas are imported lazily inside :meth:`load`
  so importing this module never requires them locally.

  Args:
    connection: SQLAlchemy URL string, engine, or config dict
      with a ``connection``/``url`` entry.
    query: The SQL query string to execute.
  '''

  def __init__(self, connection=None, query=None, **kwargs):
    '''Initialize with a connection and a query.

    Args:
      connection: SQLAlchemy URL, engine, or config dict.
      query: SQL query text; may also arrive via ``kwargs``.
      **kwargs: Aliases ``url``, ``dsn``, ``engine``, ``sql``.
    '''
    if isinstance(connection, dict):
      merged = dict(connection)
      merged.update(kwargs)
      connection = (merged.get('connection') or merged.get('url')
                    or merged.get('dsn') or merged.get('engine'))
      query = query or merged.get('query') or merged.get('sql')
    connection = connection or kwargs.get('url') or kwargs.get('dsn') \
        or kwargs.get('engine')
    query = query or kwargs.get('sql') or kwargs.get('query')
    self.connection = connection
    self.query = query

  def _describe(self):
    '''Describe the dataset for Kedro catalog introspection.

    Returns:
      A dict with the dataset type and query presence flag.
    '''
    return {'type': type(self).__name__, 'has_query': bool(self.query)}

  def _engine(self):
    '''Build a SQLAlchemy engine lazily.

    Returns:
      A SQLAlchemy engine or the configured engine object.

    Raises:
      ImportError: When SQLAlchemy is not installed.
      ValueError: When no connection is configured.
    '''
    if hasattr(self.connection, 'connect'):
      return self.connection
    if not self.connection:
      raise ValueError('SQLQueryDataSet requires a connection')
    try:
      from sqlalchemy import create_engine
    except ImportError as exc:
      raise ImportError(
          'sqlalchemy is required for SQLQueryDataSet') from exc
    return create_engine(self.connection)

  def _load(self):
    '''Execute the query and return a pandas DataFrame.

    Returns:
      The query result as a pandas DataFrame.

    Raises:
      ImportError: When pandas/SQLAlchemy are not installed.
      ValueError: When no query is configured.
    '''
    _validate_read_only_query(self.query)
    try:
      import pandas as pd
    except ImportError as exc:
      raise ImportError(
          'pandas is required for SQLQueryDataSet') from exc
    return pd.read_sql(self.query, self._engine())

  def load(self):
    '''Load the query result.

    Returns:
      The query result as a pandas DataFrame.
    '''
    return self._load()

  def _save(self, data):
    '''Saving is not supported for a read-only query dataset.

    Args:
      data: Payload that cannot be saved.

    Raises:
      NotImplementedError: Always, by design.
    '''
    raise NotImplementedError('SQLQueryDataSet is read-only')

  def save(self, data):
    '''Saving is not supported for a read-only query dataset.

    Args:
      data: Payload that cannot be saved.

    Raises:
      NotImplementedError: Always, by design.
    '''
    return self._save(data)

  def _exists(self):
    '''Check connectivity by running a trivial query.

    Returns:
      True when the connection answers, else False.
    '''
    try:
      engine = self._engine()
      with engine.connect() as conn:
        conn.exec_driver_sql('SELECT 1')
      return True
    except Exception:
      return False

  def exists(self):
    '''Check whether the configured database is reachable.

    Returns:
      True when a trivial query succeeds.
    '''
    return self._exists()


class SparkSQLDataSet(_KedroDataset):
  '''Stub dataset for Spark SQL (BigQuery/Athena via Spark).

  Intended for JDBC/connector-backed queries such as BigQuery
  or Athena through Spark. Without a live Spark session the
  dataset raises a clear error instead of failing obscurely.
  The same :func:`_validate_read_only_query` enforcement as
  :class:`SQLQueryDataSet` is applied to ``query`` before any
  session is resolved; it is keyword enforcement, not a SQL
  sandbox, so only trusted queries must be used.

  Args:
    spark: An existing Spark session, or None.
    query: The Spark SQL query text.
    config: Optional JDBC/connector configuration mapping.
  '''

  def __init__(self, spark=None, query=None, config=None, **kwargs):
    '''Initialize with a session, query, and connector config.

    Args:
      spark: Existing Spark session, or a config dict.
      query: Spark SQL query text.
      config: JDBC/connector configuration mapping.
      **kwargs: Aliases ``session``, ``sql``, ``jdbc_config``.
    '''
    if isinstance(spark, dict) and config is None:
      config, spark = spark, None
    spark = spark or kwargs.get('session')
    query = query or kwargs.get('sql') or kwargs.get('query')
    config = config or kwargs.get('jdbc_config') or {}
    self.spark = spark
    self.query = query
    self.config = dict(config)

  def _describe(self):
    '''Describe the dataset for Kedro catalog introspection.

    Returns:
      A dict with the dataset type and query presence flag.
    '''
    return {'type': type(self).__name__, 'has_query': bool(self.query)}

  def _resolve_spark(self):
    '''Return the session or try to create one from config.

    Returns:
      A Spark session.

    Raises:
      RuntimeError: When no session exists and none can be made.
    '''
    if self.spark is not None:
      return self.spark
    from forecast_ml_framework.core.spark_factory import (
        SparkSessionFactory)
    spark = SparkSessionFactory.get_spark(
        {'spark': dict(self.config, enabled=True)})
    if spark is None:
      raise RuntimeError(
          'SparkSQLDataSet requires a Spark session; provide one '
          'or enable Spark (pyspark available)')
    return spark

  def _load(self):
    '''Run the Spark SQL query.

    Returns:
      The Spark DataFrame produced by the query.

    Raises:
      RuntimeError: When no Spark session is available.
      ValueError: When no query is configured or the query is
        not a single read-only SELECT/WITH.
    '''
    if not self.query:
      raise ValueError('SparkSQLDataSet requires a query')
    _validate_read_only_query(self.query)
    return self._resolve_spark().sql(self.query)

  def load(self):
    '''Run the Spark SQL query.

    Returns:
      The Spark DataFrame produced by the query.
    '''
    return self._load()

  def _save(self, data):
    '''Saving is not supported for this stub dataset.

    Args:
      data: Payload that cannot be saved.

    Raises:
      NotImplementedError: Always, by design.
    '''
    raise NotImplementedError('SparkSQLDataSet is read-only')

  def save(self, data):
    '''Saving is not supported for this stub dataset.

    Args:
      data: Payload that cannot be saved.

    Raises:
      NotImplementedError: Always, by design.
    '''
    return self._save(data)

  def _exists(self):
    '''Check whether a Spark session is available.

    Returns:
      True when a session exists or can be created.
    '''
    try:
      self._resolve_spark()
      return True
    except Exception:
      return False

  def exists(self):
    '''Check whether Spark is available for this dataset.

    Returns:
      True when a session exists or can be created.
    '''
    return self._exists()


class RestrictedPickleDataset(_KedroDataset):
  '''Pickle dataset loading only via the restricted unpickler.

  Drop-in replacement for
  ``kedro_datasets.pickle.PickleDataset`` for model,
  calibration, and filtered pickle entries in the catalog.
  Saving uses ``pickle.dumps`` (dumping never executes code);
  loading reads raw bytes and unpickles via
  ``core.artifact_store.restricted_loads`` with its exact
  ``(module, name)`` allowlist, never raw ``pickle.load``.
  Only sklearn/numpy/pandas objects whose globals are
  allowlisted roundtrip; zoo models (xgboost/lightgbm/
  catboost/h2o) referencing non-allowlisted globals are
  rejected with ``ValueError`` by design (sklearn-only).

  Args:
    filepath: Local filesystem path of the pickle file.
      May also arrive via ``kwargs`` as ``path``.
    **kwargs: Accepted for Kedro catalog compatibility
      (``backend``, ``load_args``, ``save_args``,
      ``version``, ``credentials``, ``metadata``,
      ``protocol``) and stored, not otherwise used.
  '''

  def __init__(self, filepath=None, *args, **kwargs):
    '''Initialize with a pickle filepath.

    Args:
      filepath: Local filesystem path of the pickle file;
        may also be the first positional arg or ``path``.
      *args: Positional compatibility (filepath).
      **kwargs: Kedro catalog keys (see class docstring).
    '''
    if args and filepath is None:
      filepath = args[0]
    filepath = filepath or kwargs.pop('path', None)
    self.filepath = filepath
    self.load_args = kwargs.pop('load_args', None) or {}
    self.save_args = kwargs.pop('save_args', None) or {}
    self.version = kwargs.pop('version', None)
    self.credentials = kwargs.pop('credentials', None)
    self.metadata = kwargs.pop('metadata', None)
    self.protocol = kwargs.pop('protocol', None)
    self.backend = kwargs.pop('backend', None)
    self._kwargs = kwargs

  def _describe(self):
    '''Describe the dataset for Kedro catalog introspection.

    Returns:
      A dict with the dataset type and filepath.
    '''
    return {'type': type(self).__name__,
            'filepath': str(self.filepath) if self.filepath else None}

  def _path(self):
    '''Return the local filepath as a string.

    Returns:
      The filepath as a string.

    Raises:
      ValueError: When no filepath is configured.
    '''
    if not self.filepath:
      raise ValueError('RestrictedPickleDataset requires a filepath')
    return str(self.filepath).removeprefix('file://')

  def _load(self):
    '''Load via the restricted unpickler.

    Returns:
      The unpickled object.

    Raises:
      ValueError: When no filepath is configured, the file
        is missing, or the pickle is malformed/forbidden.
    '''
    from forecast_ml_framework.core.artifact_store import (
      restricted_loads,
    )
    path = self._path()
    with open(path, 'rb') as handle:
      raw = handle.read()
    return restricted_loads(raw)

  def load(self):
    '''Load the unpickled object.

    Returns:
      The unpickled object.
    '''
    return self._load()

  def _save(self, data):
    '''Serialize ``data`` as pickle bytes to the filepath.

    Args:
      data: Picklable payload to persist.

    Raises:
      ValueError: When no filepath is configured.
    '''
    path = self._path()
    parent = os.path.dirname(path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    protocol = self.protocol if self.protocol is not None else -1
    with open(path, 'wb') as handle:
      handle.write(pickle.dumps(data, protocol=protocol))

  def save(self, data):
    '''Serialize ``data`` as pickle bytes to the filepath.

    Args:
      data: Picklable payload to persist.
    '''
    return self._save(data)

  def _exists(self):
    '''Check whether the pickle file exists.

    Returns:
      True when the file exists on disk.
    '''
    try:
      return os.path.isfile(self._path())
    except Exception:
      return False

  def exists(self):
    '''Check whether the pickle file exists.

    Returns:
      True when the file exists on disk.
    '''
    return self._exists()
