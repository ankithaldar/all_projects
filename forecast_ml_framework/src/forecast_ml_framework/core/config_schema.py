#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pydantic schemas and OmegaConf loaders for shipped config.

Lazy-dependency note: importing this module requires pydantic
and omegaconf. ``core.config_loader`` only imports it lazily,
so plain-dict use never needs those packages.
'''

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class _Allow(BaseModel):
  '''Base model allowing unknown keys.'''

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
  '''Global run parameters.'''

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


class ParametersConfig(_Allow):
  '''Parameters (parameters.yml) contract.

  Every shipped top-level section is a nested model with
  sensible defaults; unknown sections/keys pass through
  via ``extra='allow'`` so shipped conf always validates.
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
  cfg = OmegaConf.load(base_path) if base_path.exists() else (
    OmegaConf.create({}))
  if env_dir is not None:
    env_path = Path(env_dir) / name
    if env_path.exists():
      cfg = OmegaConf.merge(cfg, OmegaConf.load(env_path))
  return cfg


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
    for Kedro to resolve at runtime.
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
    'parameters': _load_merged(base_dir, overlay, 'parameters.yml'),
  })
  OmegaConf.resolve(merged)
  plain = OmegaConf.to_container(merged, resolve=True)
  plain['catalog'] = OmegaConf.to_container(
    _load_merged(base_dir, overlay, 'catalog.yml'), resolve=False)
  return plain
