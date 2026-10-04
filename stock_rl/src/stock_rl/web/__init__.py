#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Dashboard assets, loaded from the standard library's own resource API.

The dashboard is three files -- a document, one stylesheet and one script
-- with no build step, no bundler and no CDN. That is a deliberate
consequence of the project's research, not asceticism: the design doc
proposed React for a view whose entire job is to draw one line and print
two tables, which costs a package.json, a node_modules tree and a build
artefact to produce roughly four thousand lines against a curve this
module renders as a single SVG ``path``. The standard library already
ships everything needed to read a sibling file inside an installed
package, so the dependency that would have been added is not added.

Assets are read through :func:`importlib.resources.files` rather than by
joining ``__file__`` with a directory. The difference matters once the
package is zipped: ``Path(__file__).parent / 'index.html'`` stops
resolving the moment the wheel is installed as a zipimport, and an API
that serves its own dashboard should not have a filesystem layout
assumption buried in it.

``asset_text`` therefore takes a *name*, not a path. To be precise about
what that closes: this is a hardening gap in a public function, not a
reachable vulnerability. No HTTP route passes a caller-supplied name --
:func:`stock_rl.api.dispatch` names the three assets with module-level
constants -- so ``'../api.py'`` or ``'/etc/passwd'`` is not something a
request can express today. It is fixed anyway because ``asset_text`` is
in ``__all__``, a caller of this package can pass it anything, and a
loader that reads whatever it is handed is one refactor away from being
a file-disclosure primitive.

PONYTAIL: one asset per endpoint shape, all static. Ceiling: no build
step means no bundling, minification or content hashing, so a large
dashboard would ship unminified and without cache busting. Upgrade path:
add a build target in the Makefile and keep this module reading the
built output, so callers do not learn where the bytes came from.
'''

from __future__ import annotations

from importlib.resources import files
from pathlib import PurePosixPath, PureWindowsPath

__all__ = ['asset_text', 'index_html', 'script', 'stylesheet']

#: File name of the dashboard document.
index_document = 'index.html'

#: File name of the dashboard stylesheet.
stylesheet_document = 'style.css'

#: File name of the dashboard script.
script_document = 'app.js'


def _refuse_traversal(name: str) -> None:
  '''Refuse an asset name that could leave this package.

  The checks are lexical and both path flavours are consulted: the
  process may be POSIX, and a name shaped for Windows is still a name.
  This holds for a zipimport too, where there is no filesystem to resolve
  against and the only possible defence is lexical.

  Args:
    name: Candidate asset name.

  Raises:
    ValueError: If the name is empty, is not a string, is absolute in
      either path flavour, or contains a ``..`` component. A refused name
      is a programming error rather than a missing file, so it is not
      reported as :class:`FileNotFoundError`: a caller that catches that
      and falls back to a default would be silently handed the wrong
      document.
  '''
  if not isinstance(name, str) or not name.strip():
    raise ValueError(f'asset name must be a non-empty string, got {name!r}')
  if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
    raise ValueError(
      f'asset name must be relative to the package, got {name!r}')
  parts = PurePosixPath(name).parts
  if '..' in parts or '..' in PureWindowsPath(name).parts:
    raise ValueError(
      f'asset name must not walk out of the package, got {name!r}')


def asset_text(name: str) -> str:
  '''Return the text of one dashboard asset.

  Args:
    name: File name relative to this package, e.g. ``'index.html'``. Must
      be a relative name inside the package; see :func:`_refuse_traversal`.

  Returns:
    The file's contents decoded as UTF-8.

  Raises:
    ValueError: If the name is absolute or walks out of the package.
    FileNotFoundError: If no such asset exists. Raised rather than
      returning an empty string, because an empty dashboard body is
      indistinguishable from a working one with no data.
  '''
  _refuse_traversal(name)
  return files(__name__).joinpath(name).read_text(encoding='utf-8')


def index_html() -> str:
  '''Return the dashboard document.

  Returns:
    The full HTML of ``index.html``.
  '''
  return asset_text(index_document)


def stylesheet() -> str:
  '''Return the dashboard stylesheet.

  Returns:
    The full CSS of ``style.css``.
  '''
  return asset_text(stylesheet_document)


def script() -> str:
  '''Return the dashboard script.

  Returns:
    The full JavaScript of ``app.js``.
  '''
  return asset_text(script_document)
