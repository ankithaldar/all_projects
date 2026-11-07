#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''MLflow model registry with alias promotion.

Local registries use a sqlite (or file) tracking URI, so no
tracking server is required. The ``mlflow`` package is imported
lazily, only when a registry operation runs, so importing this
module never breaks environments without mlflow installed.
'''
# pylint: disable=broad-exception-caught

import importlib
import ipaddress
import logging
import math
import re
import socket
from collections.abc import Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: Well-known aliases (custom aliases are also allowed).
KNOWN_ALIASES = (
  'production', 'staging', 'challenger', 'champion')

#: Metrics where a higher value is better.
_HIGHER_BETTER = {
  'f1', 'f1_weighted', 'accuracy', 'balanced_accuracy',
  'roc_auc', 'auc', 'r2', 'precision', 'recall',
}

_ALIAS_RE = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')

#: URI schemes allowed for MLflow tracking/registry URIs.
ALLOWED_URI_SCHEMES = frozenset({'http', 'https', 'sqlite'})

#: Hostnames that serve cloud instance metadata.
_BLOCKED_HOSTS = frozenset({
  'metadata.google.internal', 'metadata', 'metadata.azure',
  'metadata.azure.internal', '100.100.100.200',
})

_SCHEME_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*):')

#: One numeric host label: decimal/octal digits or 0x hex.
_NUMERIC_PART_RE = re.compile(r'(?:0[xX][0-9a-fA-F]+|[0-9]+)')

#: Max characters kept for URIs in error messages.
_SAFE_URI_MAX_LEN = 120

#: Redact ``user:pass@`` userinfo on raw URI text before parsing,
#: so opaque URIs (``scheme:user:pass@host``) also lose credentials
#: (C2-F4). The scheme form preserves the ``scheme:``/``scheme://``
#: prefix; the netloc form preserves a leading ``//``.
_USERINFO_SCHEME_RE = re.compile(
  r'^([a-zA-Z][a-zA-Z0-9+.-]*:)(//)?[^/@\s?#]+@')
_USERINFO_NETLOC_RE = re.compile(r'^(//)[^/@\s?#]+@')
_USERINFO_BARE_RE = re.compile(r'^[^/@\s?#:]+(?::[^/@\s?#]*)?@')


def _safe_uri(uri):
  '''Return a redacted, truncated form of ``uri`` for errors.

  Userinfo (username/password), query string and fragment
  are stripped so passwords and tokens never appear in
  ``ValueError`` messages (Sec-F3). Userinfo is redacted on
  the raw text before ``urlparse`` runs, so opaque URIs such
  as ``https:user:pass@host`` (which ``urlparse`` treats as
  scheme + path, not an authority) are covered too (C2-F4);
  scheme-less values have any ``?query``/``#fragment`` cut.
  The result is cut to ``_SAFE_URI_MAX_LEN`` characters with
  an ellipsis when longer.

  Args:
    uri: Registry/tracking URI as given by the caller.

  Returns:
    Redacted URI string safe to include in errors.
  '''
  if not isinstance(uri, str):
    text = repr(uri)
    if len(text) > _SAFE_URI_MAX_LEN:
      return text[:_SAFE_URI_MAX_LEN] + '...'
    return text
  redacted = _USERINFO_SCHEME_RE.sub(r'\1\2', uri, count=1)
  redacted = _USERINFO_NETLOC_RE.sub(r'\1', redacted, count=1)
  redacted = _USERINFO_BARE_RE.sub('', redacted, count=1)
  parsed = urlparse(redacted)
  if parsed.scheme or parsed.netloc:
    host = parsed.hostname or ''
    if ':' in host and not host.startswith('['):
      host = f'[{host}]'
    try:
      port = parsed.port
    except ValueError:
      port = None
    authority = host
    if port is not None:
      authority = f'{host}:{port}'
    path = parsed.path or ''
    if parsed.scheme:
      text = f'{parsed.scheme}://{authority}{path}'
    else:
      text = f'//{authority}{path}'
  else:
    text = re.split(r'[?#]', redacted, maxsplit=1)[0]
  if len(text) > _SAFE_URI_MAX_LEN:
    return text[:_SAFE_URI_MAX_LEN] + '...'
  return text


def _is_numeric_host(host):
  '''Return True when every label of ``host`` is numeric.

  Covers pure decimal integers, hex (``0x...``), octal
  (leading-zero multi-digit) and dotted combinations of those,
  e.g. ``0xa9.0xfe.0xa9.0xfe``. These all resolve via
  ``getaddrinfo`` to an IP, but ``ipaddress`` only parses the
  canonical dotted-decimal form.
  '''
  parts = host.split('.')
  if not parts or any(part == '' for part in parts):
    return False
  return all(_NUMERIC_PART_RE.fullmatch(part) for part in parts)


#: Canonical loopback forms treated as explicit localhost.
_EXPLICIT_LOCALHOST_IPS = frozenset({'127.0.0.1', '::1'})


def _embedded_ip_blocked(ip):
  '''Return True when ``ip`` must be rejected as non-public.

  Unwraps IPv4-mapped IPv6 (``ipv4_mapped``) and checks the
  embedded IPv4; rejects 6to4/teredo transitions outright
  (their embedded IPv4 could otherwise smuggle a private /
  metadata address past IPv6 checks); and rejects any
  address that is not global (e.g. CGNAT 100.64.0.0/10,
  where is_private is False but is_global is False),
  private, loopback, link-local, reserved, multicast,
  or unspecified. Explicit canonical localhost
  (``127.0.0.1``/``::1``) is handled by the caller, not here,
  so mapped/encoded loopback forms are still blocked.
  '''
  if isinstance(ip, ipaddress.IPv6Address):
    mapped = ip.ipv4_mapped
    if mapped is not None:
      return _embedded_ip_blocked(mapped)
    # 6to4 (2002::/16) and Teredo (2001::/32) embed an IPv4
    # address that may be private/metadata; reject them all.
    if ip.sixtofour is not None or ip.teredo is not None:
      return True
  return bool(
      not ip.is_global or ip.is_private or ip.is_loopback
      or ip.is_link_local or ip.is_reserved or ip.is_multicast
      or ip.is_unspecified or str(ip) == '100.100.100.200')


def _validate_dns_host(uri, host):
  '''Reject DNS names resolving to any non-public IP.

  Args:
    uri: Registry/tracking URI for error messages.
    host: DNS hostname to resolve.

  Raises:
    ValueError: When DNS lookup fails or any resolved
      address is non-public (SSRF guard, fail closed).
  '''
  try:
    infos = socket.getaddrinfo(
      host, None, type=socket.SOCK_STREAM)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'DNS resolution failed.') from exc
  if not infos:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'DNS resolution failed.')
  for info in infos:
    try:
      raw = info[4][0]
      resolved = ipaddress.ip_address(raw)
    except (IndexError, TypeError, ValueError) as exc:
      raise ValueError(
        f'Invalid registry URI {_safe_uri(uri)!r}: '
        'DNS resolution failed.') from exc
    if _embedded_ip_blocked(resolved):
      raise ValueError(
        f'Invalid registry URI {_safe_uri(uri)!r}: '
        'metadata host.')


def _validate_uri_host(uri, host):
  '''Reject metadata/encoded/non-public hosts for ``uri``.

  Canonical 127.0.0.1/::1 (and the ``localhost`` hostname)
  remain allowed by design as explicit localhost. Encoded
  forms of any IP are rejected unless the host IS the
  canonical string of a parsed IP (``str(ip) == host``).
  Subdomains of localhost (``*.localhost``, e.g.
  ``foo.localhost``), ``localhost.localdomain`` and its
  subdomains (``*.localhost.localdomain``, e.g.
  ``foo.localhost.localdomain`` -- the RFC6761 alias for
  localhost) are rejected: they also
  resolve to ::1 via getaddrinfo, so allowing them would
  let a non-canonical spelling bypass the explicit-
  localhost allowance. IPv4-mapped IPv6 is unwrapped and
  its embedded IPv4 is judged; 6to4/Teredo forms are
  rejected; any other private/loopback/link-local/
  reserved/multicast/unspecified IP is rejected (SSRF
  guard).

  DNS names are resolved via ``socket.getaddrinfo`` and
  every resolved address is judged by ``_embedded_ip_blocked``;
  one non-public address rejects the URI and any DNS
  failure fails closed (Sec-F4/Antisec-F04).

  Raises:
    ValueError: When the host contains non-ASCII characters,
      is %-encoded, blocked, non-public, or numeric in a
      non-canonical form, or resolves to a non-public IP.
  '''
  if host and not host.isascii():
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  if '%' in host:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  if host in _BLOCKED_HOSTS:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  # *.localhost, localhost.localdomain and
  # *.localhost.localdomain resolve to ::1 just like
  # localhost, but only the exact canonical 'localhost'
  # hostname is allowed by design; every other localhost
  # spelling is rejected.
  if host == 'localhost.localdomain' or host.endswith(
          ('.localhost', '.localhost.localdomain')):
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  if not host or host == 'localhost':
    return
  try:
    ip = ipaddress.ip_address(host)
  except ValueError:
    ip = None
  if _is_numeric_host(host) and (ip is None or str(ip) != host):
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  if ip is not None and str(ip) in _EXPLICIT_LOCALHOST_IPS:
    # Explicit canonical localhost only; mapped/encoded
    # loopback (e.g. ::ffff:127.0.0.1) does not reach here
    # because str(ip) is the mapped form, and is blocked by
    # the unwrapping check below.
    return
  if ip is not None and _embedded_ip_blocked(ip):
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'metadata host.')
  if ip is None:
    _validate_dns_host(uri, host)


def _is_mlflow_exception(exc):
  '''Return True when ``exc`` is an MlflowException.

  Detection is by isinstance when the real type is importable,
  and by class name/module as a fallback so fakes/doubles and
  the version-merged ``mlflow.exceptions`` layout in newer
  MLflow releases are both handled. Only errors classified
  here may be swallowed/logged by callers.
  '''
  for klass in type(exc).__mro__:
    if klass.__name__ == 'MlflowException':
      return True
    if klass.__name__ in (
        'ResourceDoesNotExistException', 'RestException') \
        and (klass.__module__ or '').startswith('mlflow'):
      return True
  return False


def _is_not_found(exc):
  '''Return True only when ``exc`` means the alias is absent.

  A missing alias (``RESOURCE_DOES_NOT_EXIST``) is the only
  lookup failure that may promote as a first version. Any
  other MLflow failure (e.g. ``INTERNAL_ERROR``) is not a
  not-found signal and must fail closed, so a transient
  registry outage can never promote blindly. When the
  exception carries an ``error_code`` it decides; otherwise
  the ``ResourceDoesNotExistException`` class name is the
  fallback.
  '''
  code = getattr(exc, 'error_code', None)
  if code is not None:
    return 'RESOURCE_DOES_NOT_EXIST' in str(code).upper()
  return any(
    klass.__name__ == 'ResourceDoesNotExistException'
    for klass in type(exc).__mro__)


def validate_alias(alias):
  '''Validate a registry alias.

  Args:
    alias: Alias such as 'production', 'staging',
      'challenger', or a custom lowercase alias.

  Returns:
    The alias unchanged.

  Raises:
    ValueError: When the alias is not lowercase
      alphanumeric (with '-'/'_' after the first
      character), or is empty/not a string.
  '''
  if not isinstance(alias, str) or not _ALIAS_RE.fullmatch(alias):
    raise ValueError(
      f'Invalid alias {alias!r}: expected lowercase '
      'alphanumeric, starting with a letter/digit '
      '(e.g. production, staging, challenger, or a '
      'custom alias).')
  return alias


def validate_model_name(name):
  '''Validate a registered model name.

  Uses the same documented allowlist charset as aliases:
  lowercase alphanumeric, starting with a letter/digit,
  then letters/digits/'-'/'_' (max 64 chars). This rejects
  path traversal ('../../'), separators, whitespace,
  newlines and model-URI injection ('name@alias').

  Args:
    name: Registered model name.

  Returns:
    The name unchanged.

  Raises:
    ValueError: When the name is empty, not a string, or
      contains characters outside the allowlist.
  '''
  if not isinstance(name, str) or not _ALIAS_RE.fullmatch(name):
    raise ValueError(
      f'Invalid model name {name!r}: expected lowercase '
      'alphanumeric, starting with a letter/digit '
      '(e.g. my-model).')
  return name


def validate_registry_uri(uri):
  '''Validate an MLflow tracking/registry URI.

  Allowlist: ``http``, ``https`` and ``sqlite`` schemes,
  plus scheme-less local paths such as 'mlruns'/'./mlruns'
  (the framework defaults). Anything else is rejected,
  including 'file://', 'javascript:' and other schemes.

  Rejected regardless of scheme: embedded userinfo
  ('user:pass@host'), whitespace/control characters, and
  cloud metadata hosts (169.254.0.0/16 link-local IPs,
  'metadata.google.internal', 'metadata', 100.100.100.200).
  Also rejected: any non-public IP literal -- private
  (10/8, 172.16/12, 192.168/16, ...), loopback other than
  explicit localhost/127.0.0.1/::1, link-local, reserved,
  multicast or unspecified -- including IPv4-mapped IPv6
  (embedded IPv4 unwrapped and checked), 6to4 and Teredo
  transition forms.
  Also rejected: %-encoded hostnames, non-ASCII hostnames
  (e.g. unicode dots U+3002/U+FF0E/U+FF61/U+2024 that
  ``getaddrinfo`` normalises to '.'), and IPs encoded as a
  decimal integer, hex or octal (single or dotted), e.g.
  'http://0xA9FEA9FE/' or '//169.254.169.254', which
  ``getaddrinfo`` resolves to the metadata IP. Canonical
  127.0.0.1/localhost stay allowed; encoded forms do not.
  ``*.localhost`` hosts, ``localhost.localdomain`` and
  ``*.localhost.localdomain`` hosts (which also resolve
  to ::1) are rejected; only the exact ``localhost``
  hostname is allowed.
  DNS hostnames are resolved via ``socket.getaddrinfo``;
  any non-public resolved address rejects the URI and
  DNS failures fail closed. Error messages redact
  userinfo, query and fragment, truncated to 120 chars.

  Args:
    uri: Tracking/registry URI, or ''/None for ambient.

  Returns:
    The URI unchanged.

  Raises:
    ValueError: When the URI is not a string, uses a
      disallowed scheme, embeds credentials, or targets
      a metadata host.
  '''
  if uri is None or uri == '':
    return uri
  if not isinstance(uri, str) or not uri.strip() \
      or uri != uri.strip() \
      or any(ord(ch) < 32 or ord(ch) == 127 for ch in uri) \
      or ' ' in uri:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}.')
  match = _SCHEME_RE.match(uri)
  if match is None:
    # Scheme-less local path (e.g. 'mlruns', './mlruns').
    # A '//host/...' network-path reference carries a host,
    # so it gets the same host validation as http URIs.
    if uri.startswith('//'):
      parsed = urlparse(uri)
      if parsed.username is not None or parsed.password is not None:
        raise ValueError(
          f'Invalid registry URI {_safe_uri(uri)!r}: '
          'userinfo embedded.')
      host = (parsed.hostname or '').lower().rstrip('.')
      _validate_uri_host(uri, host)
    return uri
  scheme = match.group(1).lower()
  if scheme not in ALLOWED_URI_SCHEMES:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      f'scheme {scheme!r} '
      'is not allowed (http, https, sqlite only).')
  parsed = urlparse(uri)
  if scheme in ('http', 'https') and not uri.lower().startswith(
      scheme + '://'):
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}.')
  if scheme == 'sqlite' and not uri.lower().startswith(
      'sqlite://'):
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}.')
  # Reject any userinfo in the authority component.
  if parsed.username is not None or parsed.password is not None:
    raise ValueError(
      f'Invalid registry URI {_safe_uri(uri)!r}: '
      'userinfo embedded.')
  host = (parsed.hostname or '').lower().rstrip('.')
  _validate_uri_host(uri, host)
  return uri


def _higher_better(metric):
  '''Return True when higher values of ``metric`` win.

  Args:
    metric: Metric name.

  Returns:
    True for known higher-is-better metrics.
  '''
  key = str(metric).lower()
  if key in _HIGHER_BETTER:
    return True
  return any(key.startswith(p) for p in
             ('f1_', 'accuracy', 'roc_auc'))


class ModelRegistry:
  '''MLflow-backed model registry with aliases.

  Library API by design: no pipeline or serving code in this
  repository calls :class:`ModelRegistry`/`register_model`;
  callers wire it explicitly. The promotion gate therefore
  only protects callers that pass shipped config through:
  when ``config`` carries ``promote_on_metric`` (top-level,
  or nested under ``registry_params``/``mlflow_params`` as
  shipped in ``parameters/platform.yml``, where it is
  ``'f1_weighted'``) that metric becomes the default gate
  for :meth:`register_model`. Constructing without such
  config leaves the gate disabled, by design for bare
  library use. The gate compares via ``compare_runs``, so
  it also needs an ``experiment_id``/``experiment_name`` in
  config to function (other than first-version promotion).

  Args:
    config: Mapping with 'tracking_uri', 'registry_uri'
      and optionally 'experiment_name'/'experiment_id'
      and 'promote_on_metric'. Nested 'registry_params',
      'mlflow_params' (or 'mlflow') mappings are unwrapped
      automatically. Local sqlite URIs such as
      'sqlite:////tmp/mlflow.db' need no server.
    tracking_uri: Tracking URI, overriding ``config``.
    registry_uri: Registry URI, overriding ``config``.
      Defaults to the tracking URI when unset.
    experiment_id: Experiment ID for compare_runs.
    experiment_name: Experiment name for compare_runs.
    promote_on_metric: Registry default promotion gate
      metric (e.g. 'f1_weighted'); ''/None disables
      the gate. Per-call value overrides.
  '''

  def __init__(self, config=None, tracking_uri=None,
               registry_uri=None, experiment_id=None,
               experiment_name=None, promote_on_metric=None):
    '''Store config; do not import mlflow yet.

    Args:
      config: Optional config mapping (see class docs).
      tracking_uri: Tracking URI override.
      registry_uri: Registry URI override.
      experiment_id: Experiment ID override.
      experiment_name: Experiment name override.
      promote_on_metric: Promotion gate metric override;
        None falls back to ``config``.
    '''
    cfg = dict(config) if config else {}
    for nested_key in ('registry_params', 'mlflow_params', 'mlflow'):
      nested = cfg.get(nested_key)
      if isinstance(nested, Mapping):
        merged = dict(cfg)
        merged.update(dict(nested))
        cfg = merged
    self.tracking_uri = (
      tracking_uri or cfg.get('tracking_uri') or 'mlruns')
    self.registry_uri = (
      registry_uri or cfg.get('registry_uri') or '')
    validate_registry_uri(self.tracking_uri)
    validate_registry_uri(self.registry_uri)
    self.experiment_id = experiment_id or cfg.get(
      'experiment_id')
    self.experiment_name = experiment_name or cfg.get(
      'experiment_name') or cfg.get('experiment')
    if promote_on_metric is not None:
      self.promote_on_metric = promote_on_metric
    else:
      self.promote_on_metric = cfg.get('promote_on_metric') or None
    self._mlflow = None
    self._sklearn = None
    self._client = None

  def _load_mlflow(self):
    '''Import mlflow lazily, with a clear error if absent.

    Returns:
      The imported mlflow module.

    Raises:
      ImportError: When mlflow is not installed.
    '''
    if self._mlflow is not None:
      return self._mlflow
    try:
      self._mlflow = importlib.import_module('mlflow')
    except ImportError as exc:
      raise ImportError(
        'mlflow is required for ModelRegistry; install '
        'it with `pip install mlflow` (or `uv sync`).'
      ) from exc
    return self._mlflow

  def _load_sklearn(self):
    '''Import mlflow.sklearn lazily.

    Returns:
      The mlflow.sklearn module.

    Raises:
      ImportError: When mlflow is not installed.
    '''
    if self._sklearn is not None:
      return self._sklearn
    self._load_mlflow()
    try:
      self._sklearn = importlib.import_module(
        'mlflow.sklearn')
    except ImportError as exc:
      raise ImportError(
        'mlflow is required for ModelRegistry; install '
        'it with `pip install mlflow` (or `uv sync`).'
      ) from exc
    return self._sklearn

  def _get_client(self):
    '''Create (once) an MlflowClient for our URIs.

    Returns:
      A configured MlflowClient.
    '''
    if self._client is not None:
      return self._client
    self._load_mlflow()
    try:
      tracking = importlib.import_module('mlflow.tracking')
    except ImportError as exc:
      raise ImportError(
        'mlflow is required for ModelRegistry; install '
        'it with `pip install mlflow` (or `uv sync`).'
      ) from exc
    self._client = tracking.MlflowClient(
      tracking_uri=self.tracking_uri or None,
      registry_uri=self.registry_uri or None)
    return self._client

  def _ensure_tracking(self):
    '''Point the global mlflow client at our URIs.'''
    mlflow = self._load_mlflow()
    if self.tracking_uri:
      mlflow.set_tracking_uri(self.tracking_uri)
    if self.registry_uri and hasattr(
        mlflow, 'set_registry_uri'):
      mlflow.set_registry_uri(self.registry_uri)

  def _gate_promotion(self, name, alias, candidate_run_id,
                      gate_metric):
    '''Decide whether the gate allows alias promotion.

    The candidate run is compared with the incumbent behind
    ``alias`` (via ``get_model_by_alias``) using
    ``compare_runs`` rows for ``gate_metric``. Direction
    follows ``_higher_better`` (e.g. f1/_weighted higher
    wins; other metrics lower wins, matching the
    validation default) and only strict improvement
    promotes. No incumbent promotes; any missing data or
    comparison failure fails closed (no promotion). Only a
    genuinely missing alias (``KeyError`` or an MLflow
    ``RESOURCE_DOES_NOT_EXIST`` error) counts as "no
    incumbent"; any other incumbent-lookup failure (e.g.
    ``INTERNAL_ERROR``) also fails closed rather than
    promoting blindly.

    Args:
      name: Registered model name.
      alias: Alias to gate.
      candidate_run_id: New run ID for the candidate.
      gate_metric: Metric name to compare on.

    Returns:
      True when the candidate strictly improves and the
      alias may move, False otherwise.
    '''
    try:
      incumbent = self.get_model_by_alias(name, alias)
    except KeyError:
      return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if _is_not_found(exc):
        return True
      logger.warning(
        'promotion gate failed closed for %r %r: %s',
        name, alias, type(exc).__name__)
      return False
    incumbent_run_id = getattr(incumbent, 'run_id', None)
    if not candidate_run_id or not incumbent_run_id:
      logger.warning(
        'promotion gate failed closed for %r %r: '
        'missing run id', name, alias)
      return False
    try:
      rows = self.compare_runs(metric=str(gate_metric))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'promotion gate failed closed for %r %r: %s',
        name, alias, type(exc).__name__)
      return False
    candidate = next(
      (row for row in rows
       if row.get('run_id') == candidate_run_id), None)
    current = next(
      (row for row in rows
       if row.get('run_id') == incumbent_run_id), None)
    if candidate is None or current is None:
      logger.warning(
        'promotion gate failed closed for %r %r: '
        'candidate/incumbent row missing', name, alias)
      return False
    try:
      candidate_value = float(candidate['metric_value'])
      current_value = float(current['metric_value'])
    except (TypeError, ValueError):
      logger.warning(
        'promotion gate failed closed for %r %r: '
        'metric is not numeric', name, alias)
      return False
    if not math.isfinite(candidate_value) or not math.isfinite(
        current_value):
      logger.warning(
        'promotion gate failed closed for %r %r: '
        'metric is not finite', name, alias)
      return False
    if _higher_better(str(gate_metric)):
      return candidate_value > current_value
    return candidate_value < current_value

  def _find_version(self, name, run_id):
    '''Find a model version for ``run_id`` under ``name``.

    Only an exact ``run_id`` match wins; there is no
    wrong-version fallback, so typos return None and the
    caller raises instead of promoting an unrelated
    version (MLops-2).

    Args:
      name: Registered model name.
      run_id: Run ID to look up.

    Returns:
      The version string, or None when not found.
    '''
    client = self._get_client()
    try:
      versions = client.search_model_versions(
        filter_string=f'name={name!r} and run_id={run_id!r}')
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not _is_mlflow_exception(exc):
        raise
      logger.warning(
        'search_model_versions failed for %r: %s', name,
        type(exc).__name__)
      versions = []
    for item in list(versions or []):
      if getattr(item, 'run_id', None) == run_id:
        return str(getattr(item, 'version', '') or '')
    return None

  def register_model(self, model, name, alias=None,
                     metrics=None, params=None, tags=None,
                     promote_on_metric=None):
    '''Register ``model`` and optionally point ``alias`` at it.

    Promotion is explicit: ``alias`` defaults to None
    (no alias move) so routine registrations never hit
    production silently (MLops-4). Pass an alias to opt
    in. When a gate metric is set, the candidate must
    strictly improve on the incumbent behind the alias
    (via ``get_model_by_alias`` + ``compare_runs``);
    worse metrics leave the alias unchanged. Direction
    follows ``_higher_better`` (higher-better for f1 etc,
    lower-better otherwise); equal scores do not promote.
    A missing alias (first version) promotes.

    Args:
      model: Fitted sklearn model to log/register.
      name: Registered model name.
      alias: Alias to set; None (default) skips aliasing.
      metrics: Metric dict to log on the run.
      params: Param dict to log on the run.
      tags: Tag dict to log on the run.
      promote_on_metric: Per-call gate metric; None uses
        the registry default, '' disables the gate.

    Returns:
      Dict with 'run_id', 'uri', 'model_uri', 'version',
      'name', 'alias' and 'promoted' (True only when the
      alias was actually set).

    Raises:
      ImportError: When mlflow is not installed.
      ValueError: When name/alias is invalid.
    '''
    validate_model_name(name)
    if alias is not None:
      validate_alias(alias)
    mlflow = self._load_mlflow()
    sklearn = self._load_sklearn()
    self._ensure_tracking()
    if self.experiment_name and hasattr(
        mlflow, 'set_experiment'):
      mlflow.set_experiment(self.experiment_name)
    metrics = dict(metrics or {})
    params = dict(params or {})
    tags = dict(tags or {})
    run_id = None
    model_uri = None
    version = None
    with mlflow.start_run() as run:
      if params:
        mlflow.log_params(params)
      if metrics:
        mlflow.log_metrics(metrics)
      if tags:
        mlflow.set_tags(tags)
      logged = sklearn.log_model(
        sk_model=model, name='model',
        registered_model_name=name, tags=tags or None)
      model_uri = (getattr(logged, 'model_uri', None)
                   or getattr(logged, 'uri', None))
      version = getattr(logged, 'version', None)
      run_id = getattr(logged, 'run_id', None)
      if run is not None and hasattr(run, 'info'):
        run_id = run_id or getattr(run.info, 'run_id', None)
      if model_uri and not version:
        try:
          created = mlflow.register_model(model_uri, name)
          version = getattr(created, 'version', None)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          if not _is_mlflow_exception(exc):
            raise
          logger.warning(
            'register_model fallback failed for %r: %s', name,
            type(exc).__name__)
          version = None
    if run_id and not version:
      version = self._find_version(name, run_id)
    if not model_uri and run_id:
      model_uri = f'runs:/{run_id}/model'
    if promote_on_metric is not None:
      gate_metric = promote_on_metric
    else:
      gate_metric = self.promote_on_metric
    promoted = False
    if alias and version is not None:
      if not gate_metric:
        self._get_client().set_registered_model_alias(
          name, alias, str(version))
        promoted = True
      else:
        candidate_run_id = str(run_id) if run_id else None
        promoted = self._gate_promotion(
          name, alias, candidate_run_id,
          str(gate_metric))
        if promoted:
          self._get_client().set_registered_model_alias(
            name, alias, str(version))
        else:
          logger.warning(
            'promotion gate blocked %r %r (candidate %s)',
            name, alias, candidate_run_id)
    return {
      'run_id': str(run_id) if run_id else None,
      'uri': model_uri,
      'model_uri': model_uri,
      'version': str(version) if version is not None else None,
      'name': name,
      'alias': alias,
      'promoted': promoted,
    }

  def get_model_by_alias(self, name, alias):
    '''Return the model version behind ``alias``.

    Args:
      name: Registered model name.
      alias: Alias such as 'production'.

    Returns:
      The model version object from MLflow.

    Raises:
      ValueError: When name/alias is invalid.
    '''
    validate_model_name(name)
    validate_alias(alias)
    return self._get_client().get_model_version_by_alias(
      name, alias)

  def promote_alias(self, name, alias, run_id=None,
                    version=None):
    '''Point ``alias`` at the version for ``run_id``.

    Args:
      name: Registered model name.
      alias: Alias to promote (validated).
      run_id: Run ID whose version should win; lookup
        is exact with no fallback, so digit-only run IDs
        are never treated as version strings (MLops-3).
        Pass an explicit ``version`` for version-based
        promotion.
      version: Explicit version, overriding run lookup.

    Returns:
      The promoted version as a string.

    Raises:
      ValueError: When alias is invalid or no version
        can be resolved for ``run_id``.
    '''
    validate_model_name(name)
    validate_alias(alias)
    resolved = str(version) if version is not None else None
    if resolved is None:
      if run_id is None:
        raise ValueError(
          'promote_alias requires a run_id or version.')
      resolved = self._find_version(name, str(run_id))
      if not resolved:
        raise ValueError(
          f'No model version found for run_id {run_id!r} '
          f'under model {name!r}.')
    self._get_client().set_registered_model_alias(
      name, alias, str(resolved))
    return str(resolved)

  def list_models(self, name=None):
    '''List registered model versions for ``name``.

    Args:
      name: Registered model name. When None, list
        registered models instead of versions.

    Returns:
      A list of model version (or registered model)
      objects.
    '''
    client = self._get_client()
    if name is not None:
      validate_model_name(name)
      versions = client.search_model_versions(
        filter_string=f'name={name!r}')
      return list(versions or [])
    models = client.search_registered_models()
    return list(models or [])

  def compare_runs(self, metric='rmse', experiment_id=None,
                   experiment_name=None, higher_better=None,
                   top_n=None):
    '''Rank runs in an experiment by ``metric``.

    Used for champion/challenger comparison: the
    first row is the champion, the rest are
    challengers.

    Args:
      metric: Metric name to rank by.
      experiment_id: Experiment ID; defaults to the
        registry's configured experiment.
      experiment_name: Experiment name to resolve.
      higher_better: Override the direction; when
        None it is inferred from the metric name.
      top_n: Keep only the top N rows when set.

    Returns:
      A list of dicts sorted best-first, each with
      'run_id', 'metric', 'metric_value', 'params',
      'tags', 'status', 'champion' and 'challenger'.

    Raises:
      ValueError: When no experiment is configured
        or the metric is empty.
    '''
    if not isinstance(metric, str) or not metric.strip():
      raise ValueError(
        'compare_runs requires a non-empty metric name.')
    self._load_mlflow()
    exp_id = experiment_id or self.experiment_id
    if not exp_id:
      name = experiment_name or self.experiment_name
      if name:
        exp = self._get_client().get_experiment_by_name(name)
        exp_id = getattr(exp, 'experiment_id', None)
    if not exp_id:
      raise ValueError(
        'compare_runs requires an experiment_id or '
        'experiment_name.')
    if higher_better is None:
      higher_better = _higher_better(metric)
    rows = []
    for run in self._get_client().search_runs([str(exp_id)]):
      data = getattr(run, 'data', None)
      metrics = dict(getattr(data, 'metrics', {}) or {})
      if metric not in metrics:
        continue
      info = getattr(run, 'info', None)
      rows.append({
        'run_id': getattr(info, 'run_id', None),
        'metric': metric,
        'metric_value': metrics[metric],
        'params': dict(getattr(data, 'params', {}) or {}),
        'tags': dict(getattr(data, 'tags', {}) or {}),
        'status': getattr(info, 'status', None),
        'champion': False,
        'challenger': False,
      })
    rows.sort(key=lambda row: row['metric_value'],
              reverse=bool(higher_better))
    if top_n is not None:
      rows = rows[:top_n]
    for index, row in enumerate(rows):
      row['champion'] = index == 0
      row['challenger'] = index > 0
    return rows
