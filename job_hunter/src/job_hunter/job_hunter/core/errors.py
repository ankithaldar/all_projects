#!/usr/bin/env python
# -- coding: utf-8 --

'''Exception hierarchy for the Job Hunter application.'''


from __future__ import annotations


class AppError(Exception):
  '''Base exception for all application errors.'''


class ConfigError(AppError):
  '''Raised when configuration is missing or invalid.'''


class DatabaseError(AppError):
  '''Raised on SQLite failures.'''


class AdapterError(AppError):
  '''Raised when a source adapter fails.'''

  def __init__(self, message: str, source: str = '') -> None:
    '''Initialize the error.

    Args:
      message: Human readable description.
      source: Source key associated with the failure.
    '''
    super().__init__(message)
    self.source = source


class NotFoundError(AppError):
  '''Raised when a requested entity does not exist.'''


class ConflictError(AppError):
  '''Raised on conflicting state, e.g. a run already in progress.'''


class StructuredOutputError(AppError):
  '''Raised when an LLM cannot produce schema-valid output after repairs.'''


class GatewayUnavailableError(AppError):
  '''Raised when every LLM provider failed.'''
