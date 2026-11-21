#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared fixtures and fakes for the cat-game-bot test suite."""

# Pytest fixtures intentionally share their name with their own arguments.
# pylint: disable=redefined-outer-name

import json
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
  """Return the absolute project root directory."""
  return Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest(project_root):
  """Return the committed template manifest mapping."""
  path = project_root / 'templates' / 'manifest.json'
  return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def manifest_names(manifest):
  """Return the canonical template names from the committed manifest."""
  return frozenset(manifest['templates'])


class FakeOcr:
  """OCR stub keyed by ``(x, y, width, height)`` rectangle coordinates."""

  def __init__(self, texts=None, default=''):
    """Store the rectangle-to-text mapping and fallback text.

    Args:
      texts: Mapping of rectangle key to recognised text.
      default: Text returned for unconfigured rectangles.
    """
    self.texts = dict(texts or {})
    self.default = default

  def read(self, unused_screen, rect):
    """Return the text configured for ``rect``.

    Args:
      unused_screen: Ignored screen argument.
      rect: Rectangle whose text is requested.

    Returns:
      The recognised text.
    """
    key = (rect.x, rect.y, rect.width, rect.height)
    return self.texts.get(key, self.default)


class FakeMatcher:
  """Template matcher stub returning configured matches per name."""

  def __init__(self, matches=None):
    """Store the name-to-match-list mapping.

    Args:
      matches: Mapping of template name to a list of matches.
    """
    self.matches = dict(matches or {})

  def match(self, unused_screen, name):
    """Return matches for ``name``.

    Args:
      unused_screen: Ignored screen argument.
      name: Template name to look up.

    Returns:
      The configured matches, or an empty list.
    """
    return self.matches.get(name, [])


class FakeLibrary:
  """Template library stub exposing a fixed set of template names."""

  def __init__(self, names):
    """Store the available template names.

    Args:
      names: Iterable of template names.
    """
    self._names = list(names)

  def names(self):
    """Return the available template names."""
    return self._names


@pytest.fixture
def fake_ocr():
  """Return the OCR stub class for test collaboration graphs."""
  return FakeOcr


@pytest.fixture
def fake_matcher():
  """Return the matcher stub class for test collaboration graphs."""
  return FakeMatcher


@pytest.fixture
def fake_library():
  """Return the library stub class for test collaboration graphs."""
  return FakeLibrary
