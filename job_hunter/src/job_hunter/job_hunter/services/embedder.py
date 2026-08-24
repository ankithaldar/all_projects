#!/usr/bin/env python
# -- coding: utf-8 --

'''Embedding providers with graceful degradation to keyword scoring.'''


from __future__ import annotations

import abc
import logging
import math
from typing import List, Optional

from job_hunter.core.config import AppSettings

logger = logging.getLogger(__name__)


class EmbeddingProvider(abc.ABC):
  '''Interface for semantic embedding backends.'''

  model_id: str = 'none'

  @abc.abstractmethod
  def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
    '''Embed a batch of texts.

    Args:
      texts: Input strings.

    Returns:
      Equal-length list of vectors, or None when unavailable.
    '''


class FastembedProvider(EmbeddingProvider):
  '''ONNX MiniLM-class embeddings via the fastembed package.'''

  def __init__(self, model: str) -> None:
    '''Load the model once.

    Args:
      model: fastembed model id.

    Raises:
      ImportError: When fastembed is not installed.
    '''
    from fastembed import TextEmbedding
    self.model_id = model
    self._model = TextEmbedding(model_name=model)

  def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
    '''Embed using fastembed.

    Args:
      texts: Input strings.

    Returns:
      Vectors, or None for an empty batch.
    '''
    if not texts:
      return None
    vectors = [
      [float(x) for x in vector]
      for vector in self._model.embed(texts)
    ]
    return vectors if len(vectors) == len(texts) else None


class NullProvider(EmbeddingProvider):
  '''Fallback provider signaling that semantics are unavailable.'''

  model_id = 'keyword-fallback'

  def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
    '''Always reports unavailability.

    Args:
      texts: Input strings (ignored).

    Returns:
      None.
    '''
    return None


def get_embedder(settings: AppSettings) -> EmbeddingProvider:
  '''Build the configured embedder with a safe fallback.

  Args:
    settings: Application settings.

  Returns:
    Provider instance.
  '''
  model = str(settings.embeddings.get('model', 'BAAI/bge-small-en-v1.5'))
  try:
    return FastembedProvider(model)
  except Exception as exc:
    logger.warning('fastembed unavailable (%s); using keyword fallback', exc)
    return NullProvider()


def cosine(a: List[float], b: List[float]) -> float:
  '''Compute cosine similarity between two equal-length vectors.

  Args:
    a: First vector.
    b: Second vector.

  Returns:
    Similarity in [-1, 1]; 0.0 on degenerate input.
  '''
  if len(a) != len(b) or not a:
    return 0.0
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(y * y for y in b))
  if norm_a == 0 or norm_b == 0:
    return 0.0
  return dot / (norm_a * norm_b)
