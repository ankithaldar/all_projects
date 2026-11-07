#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-6 JS harness: run web/js/api.js under node with a stub DOM.'''


from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_DRIVER = r'''
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeNode(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    className: '',
    textContent: '',
    nodeType: 1,
    attributes: {},
    listeners: {},
    setAttribute(k, v) { this.attributes[k] = String(v); },
    getAttribute(k) {
      return Object.prototype.hasOwnProperty.call(this.attributes, k)
        ? this.attributes[k] : null;
    },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    append(...children) {
      this.children = (this.children || []).concat(children);
    },
  };
  return node;
}

const documentStub = {
  createElement: (tag) => makeNode(tag),
  createTextNode: (text) => ({ nodeType: 3, textContent: String(text) }),
  getElementById: () => null,
  body: makeNode('body'),
};

const context = {
  console,
  document: documentStub,
  window: {},
  sessionStorage: {
    getItem: () => '',
    setItem: () => {},
  },
  fetch: async () => { throw new Error('no fetch in harness'); },
  prompt: () => '',
  AbortController,
  TextDecoder,
  URLSearchParams,
  setTimeout,
  clearTimeout,
  JSON,
  Object,
  String,
  Array,
};
context.window = context;
context.globalThis = context;
vm.createContext(context);

const source = fs.readFileSync(process.argv[2], 'utf8');
vm.runInContext(source, context, { filename: 'api.js' });
const jh = context.jh || context.window.jh;
assert(jh && typeof jh.el === 'function', 'jh.el missing');

function hrefOf(node) { return node.getAttribute('href'); }

// Rejected schemes / protocol-relative URLs: no href attribute set.
for (const bad of [
  'javascript:x',
  'javascript:alert(1)',
  'JaVaScRiPt:x',
  'data:text/html,x',
  'vbscript:x',
  '//evil.com/x',
  '/\\evil.com/x',
]) {
  const node = jh.el('a', { href: bad });
  assert.strictEqual(hrefOf(node), null, 'href must be omitted: ' + bad);
}

// Control-character vectors: leading / embedded control chars in the
// scheme must not smuggle a href through the anchored allowlist.
for (const bad of [
  '\u0000javascript:alert(1)',
  'java\u0000script:alert(1)',
  '\u001fhttps://example.com/x',
  '\u000djavascript:x',
]) {
  const node = jh.el('a', { href: bad });
  assert.strictEqual(hrefOf(node), null, 'ctrl href must be omitted');
}

// Allowed: absolute http(s) gets href + rel, site-relative and
// fragment hrefs pass through.
const ext = jh.el('a', { href: 'https://example.com/x' });
assert.strictEqual(hrefOf(ext), 'https://example.com/x');
assert.strictEqual(ext.getAttribute('rel'), 'noopener noreferrer');

const rel = jh.el('a', { href: '/safe/path' });
assert.strictEqual(hrefOf(rel), '/safe/path');

const frag = jh.el('a', { href: '#section' });
assert.strictEqual(hrefOf(frag), '#section');

console.log('js-harness-ok');
'''

pytestmark = pytest.mark.skipif(
  shutil.which('node') is None,
  reason='node unavailable; JS harness skipped',
)


def test_js_harness_el_href_vectors(tmp_path: Path) -> None:
  '''jh.el href allowlist holds under node with a stub DOM.

  Args:
    tmp_path: Pytest temp dir.
  '''
  driver = tmp_path / 'harness.js'
  driver.write_text(_DRIVER, encoding='utf-8')
  api_js = ROOT / 'web' / 'js' / 'api.js'
  proc = subprocess.run(
    ['node', str(driver), str(api_js)],
    capture_output=True, text=True, timeout=30, check=False,
  )
  assert proc.returncode == 0, proc.stderr or proc.stdout
  assert 'js-harness-ok' in proc.stdout
