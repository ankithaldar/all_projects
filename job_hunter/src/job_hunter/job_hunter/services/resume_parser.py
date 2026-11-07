#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Resume text extraction for LaTeX-generated PDFs.'''


from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Dict

import pypdf

LIGATURES = {
  'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
  '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-',
}

MAX_PAGES = 50
MAX_TEXT_CHARS = 100_000
MAX_CHARS_PER_PAGE = 20_000
'''Per-page extraction cap; each page contributes at most this many chars.'''
MAX_RESUME_BYTES = 10_000_000
'''Files larger than this are rejected before any bytes are read.'''


class ResumeParser:
  '''Extract clean text from resume PDFs via pypdf.'''

  def __init__(self) -> None:
    '''Create the parser.'''
    self._ligature_table = str.maketrans(LIGATURES)

  def parse_file(self, path: str | Path) -> Dict[str, Any]:
    '''Parse a PDF resume file.

    Args:
      path: File location.

    Returns:
      Mapping with text, pages, sha256, ok flag.

    Raises:
      ValueError: When the file is over the byte cap or does not start
        with a PDF header.
    '''
    if Path(path).stat().st_size > MAX_RESUME_BYTES:
      raise ValueError('resume too large')
    data = Path(path).read_bytes()
    if not data.startswith(b'%PDF'):
      raise ValueError('not a PDF')
    reader = pypdf.PdfReader(str(path))
    capped = reader.pages[:MAX_PAGES + 1]
    truncated_pages = len(capped) > MAX_PAGES
    total_pages = min(len(capped), MAX_PAGES)
    texts: list = []
    total_chars = 0
    truncated_text = False
    for page in capped[:MAX_PAGES]:
      try:
        page_text = page.extract_text() or ''
      except Exception:  # pylint: disable=broad-exception-caught
        page_text = ''
      if len(page_text) > MAX_CHARS_PER_PAGE:
        page_text = page_text[:MAX_CHARS_PER_PAGE]
        truncated_text = True
      texts.append(page_text)
      total_chars += len(page_text)
      if total_chars >= MAX_TEXT_CHARS:
        # Stop accumulating (and extracting) once the global cap is
        # reached, so a pathological PDF is bounded in memory too.
        truncated_text = True
        break
    raw = '\n'.join(texts)
    text = self.clean(raw)
    if len(text) > MAX_TEXT_CHARS:
      truncated_text = True
      text = text[:MAX_TEXT_CHARS]
    return {
      'text': text,
      'pages': total_pages,
      'sha256': self.sha256_of(data),
      'ok': bool(raw.strip()),
      'truncated': bool(truncated_pages or truncated_text),
    }

  def clean(self, text: str) -> str:
    '''Normalize unicode, fix ligatures, and collapse whitespace.

    Args:
      text: Raw extracted text.

    Returns:
      Cleaned single-line-spaced text.
    '''
    normalized = unicodedata.normalize('NFKC', text or '')
    translated = normalized.translate(self._ligature_table)
    lines = [' '.join(line.split()) for line in translated.splitlines()]
    return '\n'.join(line for line in lines if line)

  @staticmethod
  def sha256_of(data: bytes) -> str:
    '''Hash file bytes.

    Args:
      data: Raw bytes.

    Returns:
      Hex digest.
    '''
    return hashlib.sha256(data).hexdigest()

  @staticmethod
  def looks_like_cv(text: str) -> bool:
    '''Heuristic sanity check that parsed text resembles a resume.

    Args:
      text: Cleaned text.

    Returns:
      True when plausible.
    '''
    lowered = text.lower()
    hits = sum(
      token in lowered
      for token in ('experience', 'education', 'skills', 'project', 'work')
    )
    return hits >= 2 and len(text) > 400
