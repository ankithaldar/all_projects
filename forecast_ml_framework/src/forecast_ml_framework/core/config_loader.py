#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Configuration access and feature-list merging helpers.'''


class ConfigLoader:
  '''Read variables and parameters with documented fallbacks.

  Args:
    variables: Global variables mapping (Kedro ``inputs`` style).
    parameters: Model parameters mapping (Kedro ``params`` style).

  Attributes:
    variables: The variables mapping supplied at construction.
    parameters: The parameters mapping supplied at construction.
  '''

  def __init__(self, variables=None, parameters=None):
    '''Initialize with optional mappings.

    Accepts plain dicts, OmegaConf ``DictConfig`` objects, or
    pydantic models (e.g. ``VariablesConfig`` /
    ``ParametersConfig`` from ``core.config_schema``). Inputs
    are normalized to plain dicts; pydantic inputs are
    validated (re-validated from their dump) on construction.
    Behaviour of ``get``/``task_type``/``merged_feature_list``
    is unchanged for dict inputs.

    Args:
      variables: Variables mapping/config; default empty.
      parameters: Parameters mapping/config; default empty.
    '''
    self.variables = self._to_plain(variables)
    self.parameters = self._to_plain(parameters)

  @staticmethod
  def _to_plain(value):
    '''Normalize dict/DictConfig/pydantic input to dict.

    Args:
      value: Input in any supported form, or None.

    Returns:
      Equivalent plain dict ({} for None).
    '''
    if value is None:
      return {}
    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
      # Pydantic model: validate on construction by
      # re-validating its dump through its own class. Validation
      # errors propagate: swallowing them would silently accept
      # a config that does not match its schema.
      dumped = model_dump()
      value.__class__.model_validate(dumped)
      return dict(dumped)
    if isinstance(value, dict):
      return dict(value)
    # OmegaConf DictConfig/ListConfig (lazy import).
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
    if OmegaConf.is_config(value):
      from forecast_ml_framework.core.config_schema import (  # pylint: disable=import-outside-toplevel
        _blocked_oc_resolvers,
        assert_no_cross_key_oc,
        reject_oc_in_resolved,
      )
      unresolved = OmegaConf.to_container(value, resolve=False)
      ConfigLoader._reject_oc_interpolations(unresolved)
      # Fragments split across keys recombine only during
      # resolve(); reject them before resolving, resolve with
      # oc.* resolvers blocked, then re-scan the fully
      # resolved container (every key and value).
      assert_no_cross_key_oc(unresolved)
      with _blocked_oc_resolvers():
        plain = OmegaConf.to_container(value, resolve=True)
      reject_oc_in_resolved(plain)
      return dict(plain) if isinstance(plain, dict) else {}
    return dict(value)

  @staticmethod
  def _reject_oc_interpolations(node):
    '''Reject any ``${oc.*}`` interpolation in ``node``.

    ``oc.env`` would read process environment (a secret-leak
    vector when the variable holds a credential) and
    ``oc.create`` builds arbitrary objects; neither is ever
    resolved here. Use ``core.config_schema.load_config``,
    which replaces ``oc.env`` defaults statically without
    reading the environment, instead of passing a raw
    DictConfig to this loader. Whitespace/case variants
    (``${ oc.env:...}``, ``${ OC.env:...}``,
    ``${oc .env:...}``) are rejected too.

    Args:
      node: Plain container from ``OmegaConf.to_container``.

    Raises:
      ValueError: When any string contains ``${oc...}``,
        including whitespace/case variants.
    '''
    import re as _re
    _pattern = _re.compile(r'\$\{\s*oc\s*\.', _re.IGNORECASE)
    if isinstance(node, dict):
      for item in node.values():
        ConfigLoader._reject_oc_interpolations(item)
    elif isinstance(node, (list, tuple)):
      for item in node:
        ConfigLoader._reject_oc_interpolations(item)
    elif isinstance(node, str) and _pattern.search(node):
      raise ValueError(
        'oc.* resolvers are blocked in ConfigLoader: ' + node)

  def _sources(self):
    '''Return sources in search order (parameters first).'''
    return (self.parameters, self.variables)

  def _lookup(self, keys, sections=()):
    '''Find the first present key across sources and sections.

    Args:
      keys: Candidate key names, in priority order.
      sections: Optional nested section names to search.

    Returns:
      The found value, or None when no key is present.
    '''
    for source in self._sources():
      for key in keys:
        if key in source:
          return source[key]
      for section in sections:
        nested = source.get(section)
        if isinstance(nested, dict):
          for key in keys:
            if key in nested:
              return nested[key]
    return None

  def get(self, section=None, key=None, default=None):
    '''Get ``key`` under ``section`` from parameters, then variables.

    Args:
      section: Nested section name, or None for a flat lookup.
      key: Key to look up, or None when only a section is given.
      default: Value returned when the key is absent.

    Returns:
      The configured value, or ``default`` when missing.
    '''
    if key is None:
      key, section = section, None
    for source in self._sources():
      if section is None:
        if key in source:
          return source[key]
        continue
      nested = source.get(section)
      if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default

  def task_type(self):
    '''Return ``'classification'`` or ``'regression'``.

    An explicit ``task_type`` parameter wins. Otherwise the type
    attached to ``target_col`` (``target_type`` / ``target_col``
    mapping with a ``type`` entry) is auto-detected. Unknown or
    missing types default to ``'classification'`` (the single
    default, aligned with the schema defaults).

    Returns:
      ``'classification'`` or ``'regression'``.
    '''
    explicit = self._lookup(
        ('task_type', 'task', 'model_type', 'problem_type'),
        sections=('model', 'task', 'target', 'global_params'))
    if explicit is not None:
      return self._normalize(explicit)
    target_type = self._lookup(
        ('target_type', 'target_col_type', 'target_kind', 'label_type'),
        sections=('target', 'target_col'))
    if target_type is not None:
      return self._normalize(target_type)
    target_col = self._lookup(('target_col',), sections=('target',))
    if isinstance(target_col, dict):
      for key in ('type', 'task_type', 'kind'):
        if key in target_col:
          return self._normalize(target_col[key])
    return 'classification'

  @staticmethod
  def _normalize(value):
    '''Normalize a task-type hint to the canonical strings.

    Args:
      value: Free-form task-type hint from configuration.

    Returns:
      ``'classification'`` or ``'regression'``.
    '''
    text = str(value).strip().lower()
    if 'regress' in text:
      return 'regression'
    if any(token in text for token in (
        'class', 'binary', 'categor', 'label', 'clf')):
      return 'classification'
    return 'classification'

  def merged_feature_list(self, selected=None):
    '''Merge selected features with business rules.

    When a non-empty feature override is configured it wins and is
    returned deduplicated. A string override is a single feature
    name, never a sequence of characters. Override length is
    capped at 10000 entries. Existence of named columns is the
    caller's responsibility: non-existent names pass through
    unchanged (documented behaviour, not validated here).

    Args:
      selected: Feature names chosen by selection logic.

    Returns:
      The merged, deduplicated feature list.

    Raises:
      ValueError: When the override is not a list/tuple or
        exceeds the 10000-entry cap.
    '''
    selected = list(selected) if selected else []
    override = self._lookup(
        ('feature_override', 'features_override', 'override_features',
         'feature_list_override'),
        sections=('features', 'feature_selection'))
    if override:
      if isinstance(override, str):
        override = [override]
      if not isinstance(override, (list, tuple)):
        raise ValueError('feature override must be a list of names')
      if len(override) > 10000:
        raise ValueError('feature override exceeds 10000 entries')
      return self._dedup(list(override))
    business = self._lookup(
        ('business_features', 'business_feature_list', 'business_cols',
         'business_columns', 'additional_features'),
        sections=('features', 'feature_selection')) or []
    if isinstance(business, str):
      business = [business]
    exclude = set(self._lookup(
        ('exclude_features', 'excluded_features', 'exclude',
         'exclude_columns', 'dropped_features'),
        sections=('features', 'feature_selection')) or [])
    if isinstance(exclude, str):
      exclude = {exclude}
    merged = self._dedup(selected + list(business))
    return [name for name in merged if name not in exclude]

  @staticmethod
  def _dedup(items):
    '''Deduplicate ``items`` preserving first-seen order.

    Args:
      items: Iterable of hashable items.

    Returns:
      A new list without duplicates.
    '''
    seen = set()
    result = []
    for item in items:
      if item not in seen:
        seen.add(item)
        result.append(item)
    return result
