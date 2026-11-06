#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pydantic schemas and OmegaConf loaders for shipped config.

Lazy-dependency note: importing this module requires pydantic
and omegaconf. ``core.config_loader`` only imports it lazily,
so plain-dict use never needs those packages.
'''

import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Serializes _blocked_oc_resolvers across threads: the lock is
# held for the whole blocked section, so one thread exiting can
# never restore the real oc.* resolvers while another thread is
# still resolving inside its blocked section, and two threads can
# never interleave clear/register and crash with "already
# registered". RLock (not Lock) so nested/re-entrant entry on the
# same thread works; the refcount ensures only the outermost
# entry registers blockers and only the outermost exit restores.
_OC_BLOCKED_LOCK = threading.RLock()
_oc_blocked_count = 0
_oc_blocked_saved = None
_oc_blocked_registry = None


class _Allow(BaseModel):
  '''Base model allowing unknown keys.

  ``extra='allow'`` is deliberate forward-compatibility: shipped
  conf gains new keys (e.g. a new bucket in
  ``conf/base/parameters/``) without breaking older schemas.
  Known sections are typed fields on ``ParametersConfig`` (not
  extras), so a typo in a known top-level section name is
  caught by the round-trip test asserting every shipped key
  lands in a typed field; unknown keys inside a section are
  preserved rather than silently dropped.
  '''

  model_config = ConfigDict(extra='allow')


class VariablesConfig(_Allow):
  '''Variables (variables.yml) contract.

  Attributes:
    target_col: Target column name.
    id_cols: Identifier columns.
    cat_cols: Categorical columns.
    num_cols: Numeric columns.
    date_cols: Date columns.
    seq_cols: Sequence columns.
    ind_cols: Indicator columns.
    drop_cols: Columns to drop.
    bucket_segment_col: Segment column, if any.
    business_relevant_features: Features always kept.
    override_features: Features forced in.
    exclude_features: Features always excluded.
  '''

  target_col: str = 'target'
  id_cols: List[str] = Field(default_factory=list)
  cat_cols: List[str] = Field(default_factory=list)
  num_cols: List[str] = Field(default_factory=list)
  date_cols: List[str] = Field(default_factory=list)
  seq_cols: List[str] = Field(default_factory=list)
  ind_cols: List[str] = Field(default_factory=list)
  drop_cols: List[str] = Field(default_factory=list)
  bucket_segment_col: Optional[str] = None
  business_relevant_features: List[str] = Field(default_factory=list)
  override_features: List[str] = Field(default_factory=list)
  exclude_features: List[str] = Field(default_factory=list)


class GlobalParams(_Allow):
  '''Global run parameters.

  ``task_type`` defaults to ``'classification'``: this is the
  single schema default, aligned with ``ModelingParams`` and
  with ``is_regression=False``. (``ConfigLoader.task_type``
  keeps its legacy ``'regression'`` fallback only when no
  explicit task type is present anywhere in the config.)
  '''

  task_type: str = 'classification'
  is_regression: bool = False
  is_domino: bool = False
  score_model_id: str = 'demo_model_v1'
  model_type: str = 'sklearn'
  random_state: int = 42
  target_col: str = 'target'
  key_cols_list: List[str] = Field(default_factory=list)


class DataParams(_Allow):
  '''Data source parameters.'''

  train_data: str = 'raw_data'
  test_data: str = 'raw_data'


class SamplingParams(_Allow):
  '''Sampling parameters (code contract).'''

  method: str = 'stratified'
  test_size: float = 0.2
  stratify_col: str = 'target'


class PandasConversionParams(_Allow):
  '''Spark to pandas conversion parameters.'''

  convert_to_pandas: bool = True
  category_cols: List[str] = Field(default_factory=list)
  # Guard threshold matching conf/base/parameters/data.yml and
  # core.spark_factory.check_pandas_limits.
  max_rows_for_pandas: int = 5000000


class EvaluationParams(_Allow):
  '''Evaluation parameters.'''

  multiclass_flag: bool = False
  prediction_col: str = 'xgboost_pred'
  prediction_probability_col: str = 'score_value'
  metrics: Dict[str, Any] = Field(default_factory=dict)


class TrainTestSplitParams(_Allow):
  '''Train/test split parameters.'''

  training_split_ratio: float = 0.8
  test_split_ratio: float = 0.2
  stratify_col: str = 'target'
  shuffle: bool = True
  random_state: int = 42


class FeatureSelectionParams(_Allow):
  '''Feature selection parameters (code contract).'''

  enabled: bool = True
  random_state: int = 42
  is_regression: bool = False
  variance_threshold: Optional[float] = None
  missing_threshold: Optional[float] = None
  correlation: Dict[str, Any] = Field(default_factory=dict)
  kbest: Dict[str, Any] = Field(default_factory=dict)
  l1: Dict[str, Any] = Field(default_factory=dict)
  tree: Dict[str, Any] = Field(default_factory=dict)
  business_features: List[str] = Field(default_factory=list)
  override_features: List[str] = Field(default_factory=list)
  exclude_features: List[str] = Field(default_factory=list)
  output: str = 'selected_features'


class ModelingParams(_Allow):
  '''Modelling parameters (code contract).'''

  task_type: str = 'classification'
  target_col: str = 'target'
  random_state: int = 42
  test_size: float = 0.2
  models_to_use: List[str] = Field(default_factory=list)
  model_params: Dict[str, Any] = Field(default_factory=dict)
  calibration: Dict[str, Any] = Field(default_factory=dict)
  sampling_params: Dict[str, Any] = Field(default_factory=dict)
  shap_params: Dict[str, Any] = Field(default_factory=dict)


class ModelingRegParams(ModelingParams):
  '''Modelling (regression) parameters.'''

  task_type: str = 'regression'


class CalibrationParams(_Allow):
  '''Calibration parameters.'''

  enabled: bool = True
  method: str = 'betacal'
  cv: int = 5


class CvParams(_Allow):
  '''Cross-validation parameters.'''

  n_splits: int = 5
  shuffle: bool = True
  random_state: int = 42
  scoring: str = 'roc_auc'
  strategy: str = 'stratified_kfold'


class SparkParams(_Allow):
  '''Spark parameters.'''

  master: str = 'local[*]'
  app_name: str = 'forecast-ml-framework'
  spark: Dict[str, Any] = Field(default_factory=dict)
  jar_packages: List[str] = Field(default_factory=list)


class ArtifactStoreParams(_Allow):
  '''Artifact store parameters.'''

  type: str = 'local'
  base_path: str = 'artifacts'
  bucket: str = ''
  versioning: bool = True


class MlflowParams(_Allow):
  '''MLflow parameters.'''

  tracking_uri: str = 'mlruns'
  experiment_name: str = 'forecast-ml-framework'
  registry_uri: str = ''
  log_params: bool = True
  log_metrics: bool = True
  log_artifacts: bool = True


class DvcParams(_Allow):
  '''DVC parameters.'''

  enabled: bool = True
  remote: str = 'local'
  remote_url: str = ''
  push_on_run: bool = False


class ShapParams(_Allow):
  '''SHAP parameters.'''

  enabled: bool = True
  explainer: str = 'TreeExplainer'
  background_size: int = 100
  top_k: int = 20
  output_dir: str = 'artifacts/shap'


class ChartParams(_Allow):
  '''Chart parameters.'''

  enabled: bool = True
  output_dir: str = 'artifacts/charts'
  dpi: int = 150
  charts: List[str] = Field(default_factory=list)


class ValidationParams(_Allow):
  '''Validation parameters (code contract).'''

  fail_fast: bool = False
  required_cols: List[str] = Field(default_factory=list)
  max_null_ratio: Optional[float] = None


class RegistryParams(_Allow):
  '''Model registry parameters (Phase 3).'''

  tracking_uri: str = 'mlruns'
  registry_uri: str = ''
  model_name: str = 'forecast_model'
  alias: str = 'champion'
  promote_on_metric: str = 'f1_weighted'


class ScoringParams(_Allow):
  '''Batch scoring parameters (Phase 3).'''

  input_dataset: str = 'scoring_input'
  output_dataset: str = 'predictions'
  score_model_id: str = 'demo_model_v1'
  batch_size: int = 1000
  max_rows: int = 100000
  partition_batch_size: int = 10000
  spark_scoring: Dict[str, Any] = Field(default_factory=dict)
  max_cols: int = 1000
  model_path: str = ''
  model_name: str = 'forecast_model'
  alias: str = 'champion'
  id_cols: List[str] = Field(default_factory=lambda: ['id'])
  feature_cols: Optional[List[str]] = None
  task_type: str = 'classification'


class GcpInfraParams(_Allow):
  '''GCP infra locations (env placeholders in shipped conf).'''

  project: str = ''
  bucket: str = ''
  dataset: str = ''
  cluster: str = ''


class AwsInfraParams(_Allow):
  '''AWS infra locations (env placeholders in shipped conf).'''

  bucket: str = ''
  database: str = ''


class InfraParams(_Allow):
  '''Infra parameters (Phase 3).'''

  gcp: GcpInfraParams = Field(default_factory=GcpInfraParams)
  aws: AwsInfraParams = Field(default_factory=AwsInfraParams)


class MonitoringParams(_Allow):
  '''Monitoring / drift parameters (Phase 3).'''

  psi_threshold: float = 0.1
  ks_threshold: float = 0.05
  bins: int = 10
  alert_severities: List[str] = Field(
    default_factory=lambda: ['warn', 'critical'])


class RetrainingParams(_Allow):
  '''Retraining parameters (Phase 3).'''

  schedule_days: int = 7
  drift_alert_threshold: float = 0.2
  decay_threshold: float = 0.05


class FeatureStoreParams(_Allow):
  '''Feature store parameters (Phase 3).'''

  offline_path: str = 'artifacts/feature_store/offline'
  online_backend: str = 'redis'
  entity_key: str = 'id'


class AbParams(_Allow):
  '''A/B testing parameters (Phase 3).'''

  traffic_split: float = 0.1
  mode: str = 'shadow'


class CostParams(_Allow):
  '''Cost accounting parameters (Phase 3).

  Canonical keys are ``data_scan_price_per_gb`` (USD per GB
  scanned, 0.005 == 5.0 USD/TB) and
  ``storage_price_per_gb_month``. Deprecated aliases
  ``storage_price_per_gb`` and ``scan_price_per_tb_scanned``
  are accepted on input and translated to the canonical
  keys (scan alias is converted from USD/TB to USD/GB).
  '''

  compute_price_per_hour: float = 0.5
  storage_price_per_gb_month: float = 0.02
  data_scan_price_per_gb: float = 0.005

  @model_validator(mode='before')
  @classmethod
  def _translate_deprecated_aliases(cls, data):
    '''Translate deprecated cost keys to canonical keys.'''
    if isinstance(data, dict):
      data = dict(data)
      if 'storage_price_per_gb' in data and (
          'storage_price_per_gb_month' not in data):
        data['storage_price_per_gb_month'] = data.pop(
            'storage_price_per_gb')
      else:
        data.pop('storage_price_per_gb', None)
      if 'scan_price_per_tb_scanned' in data and (
          'data_scan_price_per_gb' not in data):
        try:
          data['data_scan_price_per_gb'] = float(
              data.pop('scan_price_per_tb_scanned')) / 1000.0
        except (TypeError, ValueError):
          data.pop('scan_price_per_tb_scanned', None)
      else:
        data.pop('scan_price_per_tb_scanned', None)
    return data


class GovernanceParams(_Allow):
  '''Governance / audit parameters (Phase 3).'''

  retention_days: int = 365
  audit_actor_required: bool = True


class ParametersConfig(_Allow):
  '''Parameters (parameters.yml) contract.

  Every shipped top-level section is a nested model with
  sensible defaults; unknown sections/keys pass through
  via ``extra='allow'`` (see ``_Allow``) so shipped conf
  always validates and round-trips.
  '''

  global_params: GlobalParams = Field(
    default_factory=GlobalParams)
  data_params: DataParams = Field(default_factory=DataParams)
  sampling_params: SamplingParams = Field(
    default_factory=SamplingParams)
  sampling_reg_params: SamplingParams = Field(
    default_factory=SamplingParams)
  pandas_conversion_params: PandasConversionParams = Field(
    default_factory=PandasConversionParams)
  evaluation_params: EvaluationParams = Field(
    default_factory=EvaluationParams)
  train_test_split_params: TrainTestSplitParams = Field(
    default_factory=TrainTestSplitParams)
  data_prep_params: List[Any] = Field(default_factory=list)
  feature_selection_params: FeatureSelectionParams = Field(
    default_factory=FeatureSelectionParams)
  models_to_use: List[str] = Field(default_factory=list)
  h2o_automl_params: Dict[str, Any] = Field(default_factory=dict)
  modeling_params: ModelingParams = Field(
    default_factory=ModelingParams)
  modeling_reg_params: ModelingRegParams = Field(
    default_factory=ModelingRegParams)
  model_params_by_name: Dict[str, Any] = Field(
    default_factory=dict)
  calibration_params: CalibrationParams = Field(
    default_factory=CalibrationParams)
  cv_params: CvParams = Field(default_factory=CvParams)
  spark_params: SparkParams = Field(default_factory=SparkParams)
  artifact_store_params: ArtifactStoreParams = Field(
    default_factory=ArtifactStoreParams)
  mlflow_params: MlflowParams = Field(
    default_factory=MlflowParams)
  dvc_params: DvcParams = Field(default_factory=DvcParams)
  shap_params: ShapParams = Field(default_factory=ShapParams)
  chart_params: ChartParams = Field(
    default_factory=ChartParams)
  validation_params: ValidationParams = Field(
    default_factory=ValidationParams)
  registry_params: RegistryParams = Field(
    default_factory=RegistryParams)
  scoring_params: ScoringParams = Field(
    default_factory=ScoringParams)
  infra_params: InfraParams = Field(default_factory=InfraParams)
  monitoring_params: MonitoringParams = Field(
    default_factory=MonitoringParams)
  retraining_params: RetrainingParams = Field(
    default_factory=RetrainingParams)
  feature_store_params: FeatureStoreParams = Field(
    default_factory=FeatureStoreParams)
  ab_params: AbParams = Field(default_factory=AbParams)
  cost_params: CostParams = Field(default_factory=CostParams)
  governance_params: GovernanceParams = Field(
    default_factory=GovernanceParams)


def validate_variables(data):
  '''Validate a variables mapping.

  Args:
    data: Mapping or VariablesConfig.

  Returns:
    Validated VariablesConfig.
  '''
  if isinstance(data, VariablesConfig):
    return data
  return VariablesConfig.model_validate(dict(data or {}))


def validate_parameters(data):
  '''Validate a parameters mapping.

  Args:
    data: Mapping or ParametersConfig.

  Returns:
    Validated ParametersConfig.
  '''
  if isinstance(data, ParametersConfig):
    return data
  return ParametersConfig.model_validate(dict(data or {}))


def _load_merged(base_dir, env_dir, name):
  '''Load one conf file merging env overlay over base.

  Args:
    base_dir: Directory holding the base file.
    env_dir: Overlay directory, or None.
    name: File name, e.g. 'parameters.yml'.

  Returns:
    OmegaConf config, or empty config when absent.
  '''
  base_path = Path(base_dir) / name
  cfg = (_safe_load(base_path) if base_path.exists() else (
    OmegaConf.create({})))
  if env_dir is not None:
    env_path = Path(env_dir) / name
    if env_path.exists():
      cfg = OmegaConf.merge(cfg, _safe_load(env_path))
  return cfg


def _safe_load(path):
  '''Load a YAML file, rejecting non-canonical ``oc.*`` first.

  OmegaConf parses interpolation grammar at load time
  and would raise ``GrammarParseError`` for variants
  such as ``${oc .env:...}`` before the sanitizer runs.
  Scanning the raw text first ensures every spaced /
  case variant fails closed with a ``ValueError``
  naming ``oc.`` instead.

  Args:
    path: Path to the YAML file.

  Returns:
    OmegaConf config for the file.
  '''
  text = Path(path).read_text()
  _reject_noncanonical_oc(text)
  return OmegaConf.load(path)


def _load_parameters_dir(directory):
  '''Load parameters from one conf directory, or None.

  Single-file layout wins: ``parameters.yml`` is used when it
  exists. Otherwise ``parameters/`` is merged in sorted filename
  order with OmegaConf.merge, later files winning. Non-yml
  entries in the folder are ignored.

  Args:
    directory: Conf directory (base or env overlay).

  Returns:
    OmegaConf config, or None when neither layout is present.
  '''
  directory = Path(directory)
  single = directory / 'parameters.yml'
  if single.exists():
    return _safe_load(single)
  folder = directory / 'parameters'
  if folder.is_dir():
    cfg = OmegaConf.create({})
    for path in sorted(folder.glob('*.yml')):
      cfg = OmegaConf.merge(cfg, _safe_load(path))
    return cfg
  return None


def _load_merged_parameters(base_dir, env_dir):
  '''Load parameters merging env overlay over base.

  Each side may use the single-file or per-bucket folder layout
  (see _load_parameters_dir); env wins over base. Variables and
  catalog loading are unchanged.

  Args:
    base_dir: Directory holding the base parameters.
    env_dir: Overlay directory, or None.

  Returns:
    OmegaConf config, or empty config when absent on both sides.
  '''
  cfg = _load_parameters_dir(base_dir)
  if cfg is None:
    cfg = OmegaConf.create({})
  if env_dir is not None:
    overlay = _load_parameters_dir(env_dir)
    if overlay is not None:
      cfg = OmegaConf.merge(cfg, overlay)
  return cfg


_OC_INTERPOLATION = re.compile(r'\$\{oc\.([a-z_]+)(?::([^}]*))?\}')

_OC_ANY = re.compile(r'\$\{\s*oc\s*\.', re.IGNORECASE)


def _reject_noncanonical_oc(text):
  '''Reject spaced/case-variant ``oc.*`` interpolations.

  OmegaConf's own resolver parsing tolerates whitespace
  and case variation, so ``${ oc.env:...}``,
  ``${ OC.env:...}`` and ``${oc .env:...}`` would bypass
  the canonical regex above. Any ``oc.`` interpolation
  that is not exactly the canonical form is rejected
  outright (fail-closed), never resolved against the
  process environment.

  Args:
    text: String to inspect.

  Raises:
    ValueError: If ``text`` contains a non-canonical
      ``oc.*`` interpolation.
  '''
  for match in _OC_ANY.finditer(text):
    if _OC_INTERPOLATION.match(text, match.start()) is None:
      raise ValueError(
        'oc.* resolvers with whitespace/case variation are '
        'blocked: ' + text)


def _oc_replacement(match):
  '''Resolve one ``oc.*`` match without reading the environment.

  ``oc.env:VAR,default`` becomes the literal ``default`` (the
  process environment is never read, so a secret placed in an
  env var cannot leak into the returned dict). ``oc.env:VAR``
  with no default, ``oc.create`` and every other ``oc.*``
  resolver raise ``ValueError``.
  '''
  kind = match.group(1)
  rest = match.group(2) or ''
  if kind == 'env':
    if ',' not in rest:
      raise ValueError(
        'oc.env without a default is blocked: '
        + match.group(0))
    default = rest.split(',', 1)[1].strip()
    if len(default) >= 2 and default[0] == default[-1] and (
        default[0] in ('"', "'")):
      default = default[1:-1]
    return default
  raise ValueError(
    'oc.%s resolver is blocked: %s' % (kind, match.group(0)))


def _sanitize_oc_interpolations(cfg):
  '''Return a copy of ``cfg`` with ``oc.*`` handled safely.

  Args:
    cfg: OmegaConf config loaded with ``resolve=False``.

  Returns:
    New OmegaConf config where ``oc.env`` defaults are literal
    strings and any other ``oc.*`` interpolation has raised.
    Non-``oc`` interpolations (``${variables...}``,
    ``${params...}``, relative ``${.global_params...}``) are
    preserved for OmegaConf to resolve.
  '''
  plain = OmegaConf.to_container(cfg, resolve=False)

  def walk(node):
    if isinstance(node, dict):
      return {key: walk(value) for key, value in node.items()}
    if isinstance(node, list):
      return [walk(value) for value in node]
    if isinstance(node, str):
      _reject_noncanonical_oc(node)
      result = _OC_INTERPOLATION.sub(_oc_replacement, node)
      # Substitution can recombine fragments into a fresh
      # ``${oc.*}`` (e.g. a ``$`` default adjacent to a literal
      # ``{oc.env:...}``), which re.sub never re-scans and which
      # OmegaConf.resolve would then resolve against the process
      # environment. Reject any such result fail-closed.
      if _OC_ANY.search(result):
        raise ValueError(
          'oc.* resolvers blocked after substitution: '
          + result)
      return result
    return node

  return OmegaConf.create(walk(plain))


_INTERPOLATION_INNER = re.compile(r'\$\{([^{}]+)\}')


def _flatten_strings(node, path=(), out=None):
  '''Map tuple paths to string leaf values in ``node``.'''
  if out is None:
    out = {}
  if isinstance(node, dict):
    for key, value in node.items():
      _flatten_strings(value, path + (str(key),), out)
  elif isinstance(node, (list, tuple)):
    for index, value in enumerate(node):
      _flatten_strings(value, path + (str(index),), out)
  elif isinstance(node, str):
    out[path] = node
  return out


def _walk_strings(node):
  '''Yield every string in ``node``, including dict keys.'''
  if isinstance(node, dict):
    for key, value in node.items():
      if isinstance(key, str):
        yield key
      yield from _walk_strings(value)
  elif isinstance(node, (list, tuple)):
    for value in node:
      yield from _walk_strings(value)
  elif isinstance(node, str):
    yield node


def _parse_ref_segments(text):
  '''Parse a reference path into flat-key segments, or None.

  Supports dot segments (``a.b``), bracket indexing
  (``lst[0]``, ``a.b[2].c``, ``lst[-1]``), unquoted
  bracket keys (``d[k1]``, ``d[k-1]``, ``d[a][b]``)
  and quoted segments (``a["b.c"]``, ``a.'b'`` /
  ``"a".b``) as OmegaConf interpolations allow.
  Malformed paths return None.
  '''
  parts = []
  index = 0
  length = len(text)
  while index < length:
    if text[index] in ('"', "'"):
      quote = text[index]
      end = text.find(quote, index + 1)
      if end == -1:
        return None
      parts.append(text[index + 1:end])
      index = end + 1
    elif text[index] == '[':
      end = text.find(']', index)
      if end == -1:
        return None
      inside = text[index + 1:end].strip()
      if re.fullmatch(r'-?[0-9]+', inside):
        parts.append(inside)
      elif len(inside) >= 2 and inside[0] in ('"', "'") and (
          inside[-1] == inside[0]):
        parts.append(inside[1:-1])
      elif re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_\-]*', inside):
        parts.append(inside)
      else:
        return None
      index = end + 1
    elif text[index] == '.':
      if not parts:
        return None
      index += 1
      if index >= length:
        return None
      continue
    else:
      match = re.match(r'[A-Za-z0-9_][A-Za-z0-9_\-]*', text[index:])
      if match is None:
        return None
      parts.append(match.group(0))
      index += match.end()
    # After a segment, only '.', '[' or end may follow.
    if index < length and text[index] not in ('.', '['):
      return None
  if not parts:
    return None
  return parts


def _resolve_negative_indices(parts, flat):
  '''Map negative list indices in ``parts`` to flat keys.'''
  resolved = []
  for part in parts:
    if re.fullmatch(r'-[0-9]+', part):
      prefix = tuple(resolved)
      indices = sorted({
        int(key[len(prefix)]) for key in flat
        if len(key) > len(prefix) and key[:len(prefix)] == prefix
        and key[len(prefix)].isdigit()})
      if not indices:
        return None
      offset = int(part)
      if -offset > len(indices):
        return None
      resolved.append(str(indices[offset]))
    else:
      resolved.append(part)
  return tuple(resolved)


def _referenced_string(inner, current_path, flat):
  '''Return the string value an interpolation references, or None.

  Only plain key references (absolute, or relative via leading
  dots) to string leaves are expanded, including bracket
  indexing (``lst[0]``), unquoted bracket keys
  (``d[k1]``, ``d[a][b]``) and quoted segments. Resolver calls
  (any ``:`` in the reference outside brackets/quotes, e.g.
  ``oc.env:...`` or Kedro-style ``params:...``) and references
  to missing or non-string nodes return None and are left
  unexpanded; they cannot be shown to recombine into
  ``${oc...}`` here.
  '''
  candidate = inner.strip()
  if not candidate:
    return None
  # Main reference ends at the first comma outside
  # brackets/quotes; a colon there marks a resolver call.
  depth = 0
  quote = None
  main = candidate
  for pos, char in enumerate(candidate):
    if quote is not None:
      if char == quote:
        quote = None
    elif char in ('"', "'"):
      quote = char
    elif char == '[':
      depth += 1
    elif char == ']':
      depth = max(0, depth - 1)
    elif depth == 0 and char == ',':
      main = candidate[:pos]
      break
    elif depth == 0 and char == ':':
      return None
  main = main.strip()
  if not main:
    return None
  if main.startswith('.'):
    dots = len(main) - len(main.lstrip('.'))
    rest = main[dots:]
    if not rest:
      return None
    parent = current_path[:-1] if current_path else ()
    ups = dots - 1
    if ups > 0:
      parent = parent[:-ups] if ups <= len(parent) else ()
    segments = _parse_ref_segments(rest)
    if segments is None:
      return None
    ref_path = _resolve_negative_indices(
      list(parent) + segments, flat)
    if ref_path is None:
      return None
  else:
    segments = _parse_ref_segments(main)
    if segments is None:
      return None
    ref_path = _resolve_negative_indices(segments, flat)
    if ref_path is None:
      return None
  return flat.get(ref_path)


def _check_one_string(text, current_path, flat):
  '''Expand references in ``text``; raise if it forms ``${oc...}``.

  OmegaConf resolves interpolations recursively, so fragments
  split across keys (``a: '$'``, ``b: '{oc.env:SECRET}'``,
  ``x: '${a}${b}'``) recombine into a fresh ``${oc.env:...}``
  during ``resolve()`` and would leak the environment. This
  simulates that expansion on the unresolved container (without
  invoking any resolver) and fails closed on any recombined
  ``oc.*`` pattern.
  '''
  current = text
  for _pass in range(50):
    if _OC_ANY.search(current):
      raise ValueError(
        'oc.* resolvers are blocked after cross-key '
        'interpolation: ' + current)
    matches = list(_INTERPOLATION_INNER.finditer(current))
    if not matches:
      break
    parts = []
    last = 0
    changed = False
    for match in matches:
      parts.append(current[last:match.start()])
      referenced = _referenced_string(
        match.group(1), current_path, flat)
      if referenced is None:
        parts.append(match.group(0))
      else:
        parts.append(referenced)
        changed = True
      last = match.end()
    parts.append(current[last:])
    expanded = ''.join(parts)
    if not changed or expanded == current:
      break
    current = expanded
  if _OC_ANY.search(current):
    raise ValueError(
      'oc.* resolvers are blocked after cross-key '
      'interpolation: ' + current)


def _check_strings_with_paths(node, path=(), flat=None):
  '''Run the cross-key check on every string in ``node``.'''
  if flat is None:
    flat = _flatten_strings(node)
  if isinstance(node, dict):
    for key, value in node.items():
      if isinstance(key, str):
        _check_one_string(key, path, flat)
      _check_strings_with_paths(value, path + (str(key),), flat)
  elif isinstance(node, (list, tuple)):
    for index, value in enumerate(node):
      _check_strings_with_paths(
        value, path + (str(index),), flat)
  elif isinstance(node, str):
    _check_one_string(node, path, flat)


def assert_no_cross_key_oc(plain):
  '''Reject containers where interpolation forms ``${oc...}``.

  Args:
    plain: Unresolved plain container (dict/list/str leaves).

  Raises:
    ValueError: If expanding any string's interpolations with
      the referenced string values yields a ``${oc...}``
      pattern, however the fragments were split across keys,
      lists, or nested mappings.
  '''
  _check_strings_with_paths(plain)


def reject_oc_in_resolved(plain):
  '''Reject a resolved container still containing ``${oc...}``.

  Args:
    plain: Fully resolved plain container.

  Raises:
    ValueError: If any dict key or value string matches
      ``${oc...}`` (including whitespace/case variants).
  '''
  for text in _walk_strings(plain):
    if _OC_ANY.search(text):
      raise ValueError(
        'oc.* resolvers are blocked in resolved config: ' + text)


@contextmanager
def _blocked_oc_resolvers():
  '''Temporarily replace every ``oc.*`` resolver with a blocker.

  Defence in depth around ``OmegaConf.resolve``: a recombined
  ``${oc...}`` that the static expansion check did not
  anticipate is then refused (as a ``ValueError`` subclass)
  instead of reading the process environment. Original
  resolvers are restored on exit. Canonical
  ``oc.env:VAR,default`` values are substituted to literals
  before this is entered, so legitimate configs are unaffected.

  Thread-safety: a module-level ``RLock`` is held for the whole
  blocked section and a refcount tracks nesting, so concurrent
  threads are serialized -- a thread exiting never restores the
  real resolvers while another thread is still inside (which
  would let it resolve ``oc.env`` against the environment),
  and concurrent clear/register can never crash with
  "already registered". Only the outermost entry registers
  blockers; only the outermost exit restores the originals.
  '''
  global _oc_blocked_count, _oc_blocked_saved, _oc_blocked_registry
  with _OC_BLOCKED_LOCK:
    if _oc_blocked_count == 0:
      saved = {}
      names = []
      registry = None
      try:
        from omegaconf.basecontainer import (  # pylint: disable=import-outside-toplevel
          BaseContainer,
        )
        registry = BaseContainer._resolvers
        names = [
          name for name in list(registry.keys())
          if name.startswith('oc.')]
        saved = {name: registry[name] for name in names}
      except (ImportError, AttributeError):
        names = [
          name for name in (
            'oc.coerce', 'oc.create', 'oc.decode', 'oc.deprecated',
            'oc.env', 'oc.select', 'oc.dict.keys', 'oc.dict.values')
          if OmegaConf.has_resolver(name)]
        saved = {}

      def _make_blocker(name):
        def _blocked(*args, **kwargs):
          raise ValueError('oc.* resolvers are blocked: ' + name)
        return _blocked

      try:
        for name in names:
          OmegaConf.clear_resolver(name)
          OmegaConf.register_resolver(name, _make_blocker(name))
      except Exception:
        # Never leave half-registered blockers behind.
        for name in names:
          OmegaConf.clear_resolver(name)
        if registry is not None:
          registry.update(saved)
        else:
          for name, fn in saved.items():
            OmegaConf.register_resolver(name, fn, replace=True)
        raise
      _oc_blocked_saved = (names, saved)
      _oc_blocked_registry = registry
    _oc_blocked_count += 1
    try:
      yield
    finally:
      _oc_blocked_count -= 1
      if _oc_blocked_count == 0:
        names, saved = _oc_blocked_saved or ([], {})
        registry = _oc_blocked_registry
        _oc_blocked_saved = None
        _oc_blocked_registry = None
        for name in names:
          OmegaConf.clear_resolver(name)
        if registry is not None:
          registry.update(saved)
        else:
          for name, fn in saved.items():
            OmegaConf.register_resolver(name, fn, replace=True)


def load_config(env_dir='base'):
  '''Load shipped config with env overlay, resolved.

  Args:
    env_dir: Env name ('base'|'local'|'gcp'|'aws') or a
      path to an env directory / the conf root / base dir.

  Returns:
    Plain dict with 'variables', 'parameters' and
    'catalog' keys (OmegaConf.to_container). Variables and
    parameters interpolations are resolved together (they
    reference each other, e.g. ``${variables.target_col}``);
    catalog entries use Kedro-style ``${params:...}``
    interpolations that OmegaConf cannot resolve, so the
    catalog is converted with ``resolve=False`` and left
    for Kedro to resolve at runtime. ``${oc.*}`` resolvers
    are never resolved against the process environment:
    ``oc.env`` with a default is replaced by that literal
    default, while ``oc.env`` without a default, ``oc.create``
    and other ``oc.*`` resolvers raise ``ValueError``.
  '''
  raw = str(env_dir)
  env_path = Path(raw)
  if env_path.is_dir():
    if (env_path / 'base').is_dir():
      conf_root, env_name = env_path, 'base'
    elif env_path.name == 'base':
      conf_root, env_name = env_path.parent, 'base'
    else:
      # Env dir such as conf/local: base is its sibling.
      conf_root, env_name = env_path.parent, env_path.name
  else:
    # Plain env name: conf/ is two levels above this file's
    # package root (src/forecast_ml_framework/core/).
    root = Path(__file__).resolve().parents[3]
    conf_root, env_name = root / 'conf', raw
  base_dir = conf_root / 'base'
  overlay = None if env_name == 'base' else conf_root / env_name
  merged = OmegaConf.create({
    'variables': _load_merged(base_dir, overlay, 'variables.yml'),
    'parameters': _load_merged_parameters(base_dir, overlay),
  })
  # Never resolve oc.* against the process environment: strip
  # them (default-literal substitution, see above) before the
  # only resolve() call, which then handles ${variables/...}
  # and relative interpolations. Env values cannot leak.
  merged = _sanitize_oc_interpolations(merged)
  # Fragments split across keys recombine only during
  # resolve(); reject them on the unresolved container first,
  # resolve with oc.* resolvers blocked, then re-scan the
  # fully resolved container (every key and value) so no
  # ${oc...} survives into the returned plain dict.
  assert_no_cross_key_oc(OmegaConf.to_container(merged, resolve=False))
  # to_container(resolve=True) also resolves interpolations, so
  # it must stay inside the blocked-oc context; doing it
  # outside would resolve a recombined ${oc...} against the
  # process environment and leak it.
  with _blocked_oc_resolvers():
    OmegaConf.resolve(merged)
    plain = OmegaConf.to_container(merged, resolve=True)
  reject_oc_in_resolved(plain)
  plain['catalog'] = OmegaConf.to_container(
    _sanitize_oc_interpolations(
      _load_merged(base_dir, overlay, 'catalog.yml')),
    resolve=False)
  assert_no_cross_key_oc(plain['catalog'])
  return plain
