#!/usr/bin/env python
# -- coding: utf-8 --

'''Resume text extraction for LaTeX-generated PDFs.'''


from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Dict

LIGATURES = {
  'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
  '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-',
}


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
    '''
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [page.extract_text() or '' for page in reader.pages]
    raw = '\n'.join(pages)
    return {
      'text': self.clean(raw),
      'pages': len(pages),
      'sha256': self.sha256_of(Path(path).read_bytes()),
      'ok': bool(raw.strip()),
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
