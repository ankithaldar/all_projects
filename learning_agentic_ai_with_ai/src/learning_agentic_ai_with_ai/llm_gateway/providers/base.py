#!/usr/bin/env python
# -- coding: utf-8 --

'''Abstract provider interface.'''


from __future__ import annotations

import abc
from typing import AsyncIterator, Iterator

from llm_gateway.schemas import (ProviderChunk, ProviderRequest,
                                 ProviderResponse)


class LLMProvider(abc.ABC):
  '''Abstract base class for all LLM providers.'''

  def __init__(self, name: str) -> None:
    '''Initialize provider.

    Args:
      name: Provider name.
    '''
    self.name = name

  @abc.abstractmethod
  def chat(self, request: ProviderRequest) -> ProviderResponse:
    '''Execute a synchronous chat request.

    Args:
      request: Provider request.

    Returns:
      Provider response.
    '''

  @abc.abstractmethod
  async def achat(self, request: ProviderRequest) -> ProviderResponse:
    '''Execute an asynchronous chat request.

    Args:
      request: Provider request.

    Returns:
      Provider response.
    '''

  @abc.abstractmethod
  def stream(self, request: ProviderRequest) -> Iterator[ProviderChunk]:
    '''Execute a synchronous streaming request.

    Args:
      request: Provider request.

    Yields:
      Streaming chunks.
    '''

  @abc.abstractmethod
  async def astream(
    self,
    request: ProviderRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Execute an asynchronous streaming request.

    Args:
      request: Provider request.

    Yields:
      Streaming chunks.
    '''
