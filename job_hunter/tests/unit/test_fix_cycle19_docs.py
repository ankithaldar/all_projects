#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-19 pins: docs match cited code truth (Cycle-18 findings).'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
ARCH = (DOCS / 'architecture.md').read_text(encoding='utf-8')
OVERVIEW = (DOCS / 'app_overview.md').read_text(encoding='utf-8')
ROADMAP = (DOCS / 'roadmap.md').read_text(encoding='utf-8')
SRC = ROOT / 'src/job_hunter/job_hunter'
WEB = ROOT / 'web'


def _section(text: str, start: str, end: str) -> str:
  '''Extract a markdown section between two headings.

  Args:
    text: Full document text.
    start: Heading marker for the section start.
    end: Heading marker for the section end.

  Returns:
    Text between the markers.
  '''
  begin = text.index(start)
  finish = text.index(end, begin)
  return text[begin:finish]


def _row(text: str, page: str) -> str:
  '''Return the table row for a web page.

  Args:
    text: Full document text.
    page: Page filename such as `index.html`.

  Returns:
    The markdown row containing the page name.
  '''
  return next(line for line in text.splitlines() if f'`{page}`' in line)


def test_arch_s18_no_keyboard_claims_excluding_native_tab() -> None:
  '''§18: no keyboard claims once the native Tab note is excluded.'''
  s18 = _section(ARCH, '## 18.', '## 19.')
  without_tab = s18.replace('native Tab focus only', '')
  assert 'keyboard' not in without_tab.lower()


def test_arch_s18_companies_no_filters_pause_resume_blacklist() -> None:
  '''§18 companies: no filters, Pause/Resume + +prio + Blacklist only.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'companies.html')
  assert '(no filters)' in row
  assert 'Pause/Resume' in row
  assert '+prio' in row
  assert 'Blacklist' in row
  assert 'import textarea' in row
  assert 'add-manually form' in row
  assert 'no refresh/label UI' in row
  assert 'refresh endpoint planned §17' in row
  assert 'Table w/ filters' not in row
  assert 'pin/pause/refresh/label' not in row
  assert 'import dialog' not in row
  # Code truth: companies.html has no filter inputs and no refresh call;
  # actions are Pause/Resume, +prio, Blacklist only.
  src = (WEB / 'companies.html').read_text(encoding='utf-8')
  assert 'Pause' in src and 'Resume' in src
  assert '+prio' in src and 'Blacklist' in src
  assert 'importYaml' in src and 'cName' in src
  assert '/api/companies/' in src
  assert '/refresh' not in src


def test_arch_s18_recommendations_mouse_only_buttons_no_bulk() -> None:
  '''§18 recommendations: mouse-only per-card buttons, no bulk claims.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'recommendations.html')
  assert 'mouse-only Save/Dismiss per-card buttons' in row
  assert 'native Tab focus only' in row
  assert 'keyboard nav' not in row
  assert 'bulk dismiss' not in row
  # Code truth: recCard emits Save/Dismiss buttons; no key handlers.
  comp = (WEB / 'js/components.js').read_text(encoding='utf-8')
  assert 'Save' in comp and 'Dismiss' in comp
  assert 'keydown' not in comp and 'keyup' not in comp


def test_arch_s18_dashboard_no_recent_runs() -> None:
  '''§18 dashboard: KPIs + per_page=8 recs only, no recent runs.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'index.html')
  assert 'KPI cards (New today, Saved, Active jobs, Companies)' in row
  assert 'per_page=8' in row
  assert 'recent runs' not in row.lower()
  # Code truth: index.html fetches KPIs + per_page=8 recs, never runs.
  src = (WEB / 'index.html').read_text(encoding='utf-8')
  assert '/api/stats/dashboard' in src
  assert 'per_page=8' in src
  assert '/api/runs' not in src


def test_arch_s18_profile_comma_status_api_only() -> None:
  '''§18 profile: comma input, status line only, skills API-only.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'profile.html')
  assert 'roles plain comma-separated input' in row
  assert 'resume upload + status line only' in row
  assert 'no parsed-preview editor' in row
  assert 'skills manager API-only (no page UI)' in row
  assert 'roles chips' not in row
  # Code truth: roles is a comma-separated input; resume writes a
  # status line; there is no skills page UI (only the status line
  # echoes parsed skill names).
  src = (WEB / 'profile.html').read_text(encoding='utf-8')
  assert 'comma-separated' in src
  assert 'resumeOut' in src
  assert 'skills manager' not in src.lower()
  assert '/api/profile/skills' not in src
  assert 'id="skills' not in src


def test_arch_s18_jobs_top_row_source_badge_only() -> None:
  '''§18 jobs: top filter row, source badge only.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'jobs.html')
  assert 'top filter row' in row
  assert 'source badge only' in row
  assert 'no duplicate/stale badges' in row
  assert 'filter sidebar' not in row
  unnegated = row.replace('no duplicate/stale badges', '')
  assert 'duplicate/stale' not in unnegated
  # Code truth: jobs.html renders a top row and a single source badge.
  src = (WEB / 'jobs.html').read_text(encoding='utf-8')
  assert 'source_key' in src
  assert 'sidebar' not in src.lower()


def test_arch_s18_runs_console_trigger_no_drawer() -> None:
  '''§18 runs: history + Run-discovery trigger + live SSE console.'''
  row = _row(_section(ARCH, '## 18.', '## 19.'), 'runs.html')
  assert 'Run history table' in row
  assert 'Run-discovery trigger' in row
  assert 'live SSE console' in row
  assert 'level-coloured' in row
  assert 'no detail drawer/stats counters/errors panel' in row
  unnegated = row.replace('no detail drawer/stats counters/errors panel', '')
  assert 'detail drawer' not in unnegated
  assert 'stats counters' not in unnegated
  assert 'errors panel' not in unnegated
  # Code truth: runs.html has a history table, a trigger button and a
  # level-classed SSE console; no drawer/stats panel markup.
  src = (WEB / 'runs.html').read_text(encoding='utf-8')
  assert 'Run discovery now' in src
  assert '/api/runs/discovery' in src
  assert 'console' in src.lower()
  assert 'drawer' not in src.lower()


def test_app_overview_button_review_no_keypress() -> None:
  '''Overview: button-based review, no one-keypress claim.'''
  assert 'button-based review' in OVERVIEW
  assert 'one keypress' not in OVERVIEW.lower()
  assert 'keyboard' not in OVERVIEW.lower()


def test_roadmap_feature12_button_review_downgraded_planned() -> None:
  '''Roadmap feature_12: button-based review, downgraded to planned.'''
  row = _row(ROADMAP, 'feature_12_web_ui')
  assert 'button-based review flow' in row
  assert 'keyboard' not in ROADMAP.lower()
  assert 'keyboard-only' not in row
  assert 'keyboard review' not in row
  assert 'mouse-only via per-card buttons' in row
  assert 'planned' in row.lower()
  assert 'partial' in row.lower()
  assert 'done' not in row.lower().replace('not done', '')


def test_ddl_country_not_null() -> None:
  '''DDL: country is NOT NULL in architecture §16.'''
  assert "country TEXT NOT NULL DEFAULT 'IN'" in ARCH
  # Code truth: 0001_init.sql declares country NOT NULL.
  sql = (SRC / 'db/migrations/0001_init.sql').read_text(encoding='utf-8')
  assert "country TEXT NOT NULL DEFAULT 'IN'" in sql
