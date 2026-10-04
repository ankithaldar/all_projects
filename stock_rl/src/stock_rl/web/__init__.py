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

PONYTAIL: one asset per endpoint shape, all static. Ceiling: no build
step means no bundling, minification or content hashing, so a large
dashboard would ship unminified and without cache busting. Upgrade path:
add a build target in the Makefile and keep this module reading the
built output, so callers do not learn where the bytes came from.
'''

from __future__ import annotations

from importlib.resources import files

__all__ = ['asset_text', 'index_html', 'script', 'stylesheet']

#: File name of the dashboard document.
index_document = 'index.html'

#: File name of the dashboard stylesheet.
stylesheet_document = 'style.css'

#: File name of the dashboard script.
script_document = 'app.js'


def asset_text(name: str) -> str:
  '''Return the text of one dashboard asset.

  Args:
    name: File name relative to this package, e.g. ``'index.html'``.

  Returns:
    The file's contents decoded as UTF-8.

  Raises:
    FileNotFoundError: If no such asset exists. Raised rather than
      returning an empty string, because an empty dashboard body is
      indistinguishable from a working one with no data.
  '''
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
