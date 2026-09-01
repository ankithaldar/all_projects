#!/usr/bin/env python
# -- coding: utf-8 --

'''Exception types used across the LLM gateway.'''


from __future__ import annotations

from typing import Optional


class GatewayError(Exception):
  '''Base exception for all gateway errors.'''

  def __init__(
    self,
    message: str,
    provider: str = '',
    status_code: Optional[int] = None,
  ) -> None:
    '''Initialize the error.

    Args:
      message: Human readable error message.
      provider: Provider name associated with the error.
      status_code: Optional HTTP status code.
    '''
    super().__init__(message)
    self.message = message
    self.provider = provider
    self.status_code = status_code


class ConfigError(GatewayError):
  '''Raised when gateway configuration is invalid.'''


class ProviderError(GatewayError):
  '''Raised when a provider returns a non-recoverable error.'''


class TransientProviderError(ProviderError):
  '''Raised for temporary provider or network failures.'''


class RateLimitedProviderError(ProviderError):
  '''Raised when a provider is rate limited.'''


class AuthenticationError(ProviderError):
  '''Raised when provider authentication fails.'''


class AllProvidersFailedError(GatewayError):
  '''Raised when all configured providers fail.'''
