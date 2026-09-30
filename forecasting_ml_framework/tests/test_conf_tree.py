#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests that the shipped configuration tree is loadable and coherent.

A configuration tree that does not parse is invisible to the unit suite -- every
test supplies its own parameters -- and it is fatal on the first real run. The
defect these pin was nineteen interpolations written unquoted inside YAML flow
mappings, which is a parse error rather than an interpolation: ``{key: ${x}}`` is
not valid YAML at all, so ``conf/base/parameters.yml`` could not be loaded by any
loader, in any environment.

The second half of the module checks coherence rather than syntax: the environment
flags an environment declares, the back end each environment resolves to, and the
column-role contract the routines are configured against.
'''

import pathlib
import re

import pytest
import yaml
from omegaconf import OmegaConf

from forecasting_ml_framework import constants
from forecasting_ml_framework.platform.environment import resolve_environment

CONF_ROOT = pathlib.Path(__file__).resolve().parents[1] / 'conf'

DOCUMENTS = sorted(CONF_ROOT.rglob('*.yml'))

#: Stands in for the values a scheduler supplies per run. Only the keys the routed
#: document interpolates are needed, and the dates are fixed so the assertions are
#: deterministic.
_RUNTIME = {
  'model_key': 'forecast_demo_retention',
  'target_col': 'label',
  'run_date': '2026-06-30',
  'train_run_dt': '2026-06-30',
  'score_run_dt': '2026-06-30',
  'external_project': 'demo-project',
  'data_project': 'demo-project',
  'db_staging_data': 'staging',
  'db_score_data': 'scored',
  'db_metrics_data': 'metrics',
  'db_internal_data': 'internal',
  'db_source_data': 'source',
  'output_gcs_location': 'gs://demo-bucket/out',
  'tmp_bucket': 'gs://demo-bucket/tmp',
  'is_hosted_env': False,
  'is_local': False,
  'is_regression': False,
  'split': False,
  'run_mode': 'prod',
  'train_mode': 'train_only',
  'suffix': 'params',
}

#: Every key the bind-and-rewrite routes. The set is the union of the runtime
#: channel and whatever each environment's globals document declares, so adding a
#: key to either document makes it routable without touching the test.
def _known_keys() -> list:
  """Collect every routable interpolation key.

  Returns:
    The key names.
  """
  keys = set(_RUNTIME)
  for environment in ('base', 'local'):
    keys.update(OmegaConf.load(CONF_ROOT / environment / 'globals.yml').keys())
  return sorted(keys)


_KNOWN_KEYS = _known_keys()


class TestEveryDocumentParses:
  '''A document that does not parse is fatal on the first real run.'''

  @pytest.mark.parametrize('path', DOCUMENTS, ids=lambda item: item.name)
  def test_document_is_valid_yaml(self, path: pathlib.Path) -> None:
    try:
      document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:  # pragma: no cover - the failure being pinned
      pytest.fail(f'{path} does not parse: {error}')
    assert isinstance(document, dict), f'{path} must be a mapping at the top level'

  @pytest.mark.parametrize('path', DOCUMENTS, ids=lambda item: item.name)
  def test_document_declares_at_least_one_key(self, path: pathlib.Path) -> None:
    assert yaml.safe_load(path.read_text(encoding='utf-8'))


#: The exact shape that made the original parse error: an interpolation written
#: bare as the value of a key inside a *flow* mapping -- ``{key: ${x}}``. YAML
#: reads the brace as the start of a nested mapping and fails, rather than
#: treating the interpolation as a scalar. The lookahead for ``,`` or ``}`` is
#: what keeps ordinary block-mapping values (``key: ${x}`` on its own line) and
#: quoted values out of the match.
_UNQUOTED_FLOW_VALUE = re.compile(r':\s+\$\{[^}]*\}\s*[,}]')


def _unquoted_flow_interpolations(path: pathlib.Path) -> list:
  """Report lines whose flow mapping holds an unquoted interpolation.

  Args:
    path: The document to scan.

  Returns:
    A list of ``(line_number, text)`` pairs, one per offending line.
  """
  found = []
  for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
      continue
    if '{' not in stripped:
      continue
    if _UNQUOTED_FLOW_VALUE.search(stripped):
      found.append((number, line))
  return found


class TestInterpolationsAreQuoted:
  '''An interpolation inside a flow mapping must be quoted to be valid YAML.'''

  @pytest.mark.parametrize('path', DOCUMENTS, ids=lambda item: item.name)
  def test_no_unquoted_interpolation_sits_inside_a_flow_mapping(self, path: pathlib.Path) -> None:
    """A bare ``${...}`` as a flow-mapping value is a parse error, not an interpolation.

    Nineteen such sites made ``conf/base/parameters.yml`` unloadable by any loader
    in any environment, which no unit test would have caught because every test
    supplies its own parameters.
    """
    offenders = _unquoted_flow_interpolations(path)
    assert not offenders, f'{path} has unquoted interpolations in flow mappings: {offenders}'


class TestEnvironmentFlags:
  '''Each environment declares the flags the resolver reads.'''

  def test_base_declares_both_environment_flags(self) -> None:
    document = yaml.safe_load((CONF_ROOT / 'base' / 'parameters.yml').read_text())
    assert constants.ENV_IS_HOSTED in document
    assert constants.ENV_IS_LOCAL in document

  def test_base_does_not_declare_the_legacy_flag(self) -> None:
    """The legacy spelling must not be reintroduced as the declared value."""
    document = yaml.safe_load((CONF_ROOT / 'base' / 'parameters.yml').read_text())
    assert constants.ENV_IS_HOSTED_LEGACY not in document

  def test_global_params_carries_the_current_flag_names(self) -> None:
    document = yaml.safe_load((CONF_ROOT / 'base' / 'parameters.yml').read_text())
    globals_block = document['global_params']
    assert constants.ENV_IS_HOSTED in globals_block
    assert constants.ENV_IS_LOCAL in globals_block
    assert constants.ENV_IS_HOSTED_LEGACY not in globals_block

  def test_catalog_binds_the_current_flag_name(self) -> None:
    document = yaml.safe_load((CONF_ROOT / 'base' / 'catalog.yml').read_text())
    anchor = document['_spark_bq_data']
    assert constants.ENV_IS_HOSTED in anchor
    assert constants.ENV_IS_HOSTED_LEGACY not in anchor

  def test_local_environment_is_local_and_unhosted(self) -> None:
    """The local environment is the embedded store with no credential path."""
    environment = resolve_environment(yaml.safe_load((CONF_ROOT / 'local' / 'parameters.yml').read_text()))
    assert environment.is_local is True
    assert environment.db_backend == constants.BACKEND_EMBEDDED
    assert environment.uses_local_artefacts is True

  def test_the_local_output_root_is_a_bare_path(self) -> None:
    """A local address carries no scheme."""
    document = yaml.safe_load((CONF_ROOT / 'local' / 'parameters.yml').read_text())
    assert '://' not in str(document['output_gcs_location'])


class TestColumnRoleContract:
  '''The role lists the routines are configured against must be coherent.'''

  @pytest.mark.parametrize('environment', ['base', 'local'])
  def test_a_sequence_column_is_not_also_a_scalar_one(self, environment: str) -> None:
    document = yaml.safe_load((CONF_ROOT / environment / 'globals.yml').read_text())
    scalar = set(document['NUM_COL']) | set(document['CAT_COL']) | set(document['IND_COL'])
    overlap = scalar & set(document['SEQ_COL'])
    # A sequence column receives a dense tensor, not a raw array, so the trainer
    # cannot fit it if it is also declared scalar.
    assert not overlap, f'{environment}: sequence columns also declared scalar: {sorted(overlap)}'

  @pytest.mark.parametrize('environment', ['base', 'local'])
  def test_identifier_columns_do_not_overlap_the_feature_roles(self, environment: str) -> None:
    document = yaml.safe_load((CONF_ROOT / environment / 'globals.yml').read_text())
    features = set(document['NUM_COL']) | set(document['CAT_COL']) | set(document['IND_COL'])
    overlap = features & set(document['ID_COL'])
    assert not overlap, f'{environment}: identifier columns also declared as features'

  @pytest.mark.parametrize('environment', ['base', 'local'])
  def test_no_column_occupies_two_role_lists(self, environment: str) -> None:
    document = yaml.safe_load((CONF_ROOT / environment / 'globals.yml').read_text())
    seen: set = set()
    for role in ('NUM_COL', 'CAT_COL', 'IND_COL', 'SEQ_COL'):
      columns = set(document[role])
      assert not (columns & seen), f'{environment}: a column appears in more than one role list'
      seen |= columns


class TestRoutineProgramIsResolvable:
  '''Every active routine block must name a routine that exists.'''

  def test_every_named_routine_is_registered(self) -> None:
    from forecasting_ml_framework.preprocessing.routines import AVAILABLE_ROUTINES  # noqa: PLC0415

    document = yaml.safe_load((CONF_ROOT / 'base' / 'parameters.yml').read_text())
    for block in document['data_prep_params']:
      name = next(iter(block))
      assert name in AVAILABLE_ROUTINES, f'{name} is configured but not registered'

  def test_every_block_declares_the_activation_key(self) -> None:
    """A block that omits ``skip`` is silently excluded, so the key is load-bearing."""
    document = yaml.safe_load((CONF_ROOT / 'base' / 'parameters.yml').read_text())
    for block in document['data_prep_params']:
      body = next(iter(block.values()))
      assert 'skip' in body, f'{next(iter(block))} omits skip and will be silently excluded'


class TestResolvedParameterDocument:
  '''The merged document must resolve and expose what the nodes read.'''

  @staticmethod
  def _route(environment: str) -> str:
    """Apply the bind-and-rewrite the bootstrap performs before OmegaConf loads.

    The production loader prefixes every unbound key with its resolution scope, so
    ``${model_key}`` becomes ``${globals:model_key}`` when the globals document
    declares it and ``${runtime:model_key}`` otherwise. Reproducing that routing is
    what makes the result the document a run actually receives: without it,
    ``model_key: ${model_key}`` is a self-reference and OmegaConf reports a
    recursive interpolation rather than a missing value.

    Args:
      environment: The environment directory name.

    Returns:
      The parameters document with every interpolation routed to a scope.
    """
    globals_ = OmegaConf.load(CONF_ROOT / environment / 'globals.yml')
    OmegaConf.register_new_resolver('globals', lambda name: globals_.get(name), replace=True)
    OmegaConf.register_new_resolver('runtime', lambda name: _RUNTIME.get(name, ''), replace=True)

    raw = (CONF_ROOT / environment / 'parameters.yml').read_text()
    # Longest key first, so a key that is a prefix of another cannot shadow it.
    for key in sorted(_KNOWN_KEYS, key=len, reverse=True):
      scope = 'globals' if key in globals_.keys() else 'runtime'
      raw = raw.replace('${' + key, '${' + scope + ':' + key)
    return raw

  @classmethod
  def _resolve(cls, environment: str, *, resolve: bool) -> dict:
    """Load a routed parameters document.

    Args:
      environment: The environment directory name.
      resolve: Whether to resolve interpolations, or return them as written.

    Returns:
      The merged document.
    """
    globals_ = OmegaConf.load(CONF_ROOT / environment / 'globals.yml')
    routed = cls._route(environment)
    return OmegaConf.to_container(OmegaConf.merge(globals_, OmegaConf.create(routed)), resolve=resolve)

  @pytest.mark.parametrize('environment', ['base', 'local'])
  def test_the_document_parses_after_routing(self, environment: str) -> None:
    document = self._resolve(environment, resolve=False)
    assert document['model_key'] is not None

  @pytest.mark.parametrize('environment', ['base', 'local'])
  def test_the_document_declares_a_routine_program(self, environment: str) -> None:
    document = self._resolve(environment, resolve=False)
    if 'data_prep_params' in document:
      assert isinstance(document['data_prep_params'], list)

  def test_a_categorical_imputation_resolves_its_column_list(self) -> None:
    """The interpolation must yield a list of column names, not the interpolation text."""
    document = OmegaConf.to_container(OmegaConf.create(self._route('base')), resolve=True)
    block = document['data_prep_params'][0]['Imputations']['args']
    assert isinstance(block['cols']['include_cols'], list)
    assert block['cols']['include_cols'], 'the categorical role list resolved to nothing'

  def test_the_fill_dictionary_is_resolved(self) -> None:
    document = OmegaConf.to_container(OmegaConf.create(self._route('base')), resolve=True)
    block = document['data_prep_params'][0]['Imputations']['args']
    assert isinstance(block['custom_bounds'], dict)
    assert block['custom_bounds']

  def test_no_self_referential_interpolation_survives_routing(self) -> None:
    """``model_key: ${model_key}`` must be rewritten before OmegaConf sees it.

    Left unrewritten it is a self-reference, and OmegaConf reports a recursive
    interpolation -- an error that names neither the missing value nor the file.
    """
    document = OmegaConf.to_container(OmegaConf.create(self._route('base')), resolve=True)
    assert document['model_key'] == _RUNTIME['model_key']
