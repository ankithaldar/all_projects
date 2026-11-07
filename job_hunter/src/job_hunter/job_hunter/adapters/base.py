#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Abstract source adapter contract.'''

from __future__ import annotations

import abc
import ipaddress
import re
from typing import TYPE_CHECKING, List, Tuple
from urllib.parse import quote

from job_hunter.core.errors import AdapterError
from job_hunter.core.models import CompanyTarget, RawJobRecord

if TYPE_CHECKING:
  from job_hunter.adapters.http_client import HttpClient

BOARD_REF_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
'''Board refs are single safe path segments: no separators, no dots.'''

DOMAIN_RE = re.compile(
  r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?'
  r'(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$',
)
'''Bare registrable-domain shape; mirrors the API company validator.

Single-source truth: the API wraps valid_domain; import from here
instead of duplicating another implementation.
'''


def valid_domain(domain: str) -> str:
  '''Validate a bare company domain, returning its normalized form.

  Args:
    domain: Candidate domain (empty means unknown and passes through).

  Returns:
    Lowercased domain, or empty string when unknown.

  Raises:
    ValueError: On overlong domains, userinfo/URL characters, IP
      literals, or non-domain shapes.
  '''
  candidate = (domain or '').strip().lower()
  if not candidate:
    return ''
  if len(candidate) > 253 or any(
    char in candidate for char in '@:/?#'
  ):
    raise ValueError(f'invalid domain: {domain!r}')
  try:
    ipaddress.ip_address(candidate)
  except ValueError:
    pass
  else:
    raise ValueError(f'invalid domain: {domain!r}')
  if not DOMAIN_RE.match(candidate):
    raise ValueError(f'invalid domain: {domain!r}')
  return candidate

WORKDAY_HOST_RE = re.compile(r'^[a-z0-9-]+\.(?:wd\d+\.)?myworkdayjobs\.com$')
'''Workday hosts are tenant[.wdN].myworkdayjobs.com only.'''


def validate_board_ref(board_ref: str, source: str = '') -> str:
  '''Validate a board_ref before it is interpolated into a URL.

  Args:
    board_ref: Raw board reference from a target.
    source: Source key for error attribution.

  Returns:
    URL-safe rendering of the ref (quote is a no-op for valid refs).

  Raises:
    AdapterError: When the ref is not a single safe segment.
  '''
  if not isinstance(board_ref, str) or not BOARD_REF_RE.match(board_ref):
    raise AdapterError(
      f'invalid board_ref for {source}: {board_ref!r}', source=source
    )
  return quote(board_ref, safe='')


def validate_workday_board_ref(board_ref: str) -> Tuple[str, str]:
  '''Validate a ``host/site`` workday board_ref.

  Args:
    board_ref: Raw board reference from a target.

  Returns:
    (host, site), both URL-safe.

  Raises:
    AdapterError: On malformed host or site.
  '''
  if not isinstance(board_ref, str) or '/' not in board_ref:
    raise AdapterError(
      f'bad workday board_ref: {board_ref!r}', source='workday'
    )
  host, site = board_ref.split('/', 1)
  if not WORKDAY_HOST_RE.match(host.lower()):
    raise AdapterError(
      f'bad workday board_ref: {board_ref!r}', source='workday'
    )
  if not BOARD_REF_RE.match(site):
    raise AdapterError(
      f'bad workday board_ref: {board_ref!r}', source='workday'
    )
  return quote(host.lower(), safe=''), quote(site, safe='')


class SourceAdapter(abc.ABC):
  '''One job-source integration: fetch postings for a target.'''

  source_key: str = ''

  def __init__(self, http: 'HttpClient') -> None:
    '''Initialize the adapter.

    Args:
      http: Shared polite HTTP client.
    '''
    self._http = http

  @abc.abstractmethod
  async def fetch(
    self, target: CompanyTarget, limit: int = 200
  ) -> List[RawJobRecord]:
    '''Fetch current postings for one company/board.

    Args:
      target: Company plus board reference.
      limit: Safety cap on returned records.

    Returns:
      Raw posting records.
    '''
