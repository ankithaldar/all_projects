#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for gates, scoring math, and feedback reweighting.'''


from __future__ import annotations

from job_hunter.core.models import CandidateProfile
from job_hunter.services.matcher import (
  company_fit,
  gate_failures,
  recency_score,
  salary_fit,
  seniority_fit,
  skill_coverage,
  title_fit,
)
from job_hunter.services.scorer import aggregate, feedback_adjust, normalize_weights


def _profile(**overrides) -> CandidateProfile:
  '''Build a profile with overrides.

  Args:
    **overrides: Field replacements.

  Returns:
    CandidateProfile instance.
  '''
  fields = {
    'cities': ['Bengaluru'],
    'remote_pref': 'any',
    'salary_floor_lpa': 45.0,
    'experience_years': 9.0,
    'employment_types': ['full_time'],
    'target_roles': ['Staff Data Scientist'],
    'target_verticals': ['fintech'],
    'skills': [],
  }
  fields.update(overrides)
  return CandidateProfile(**fields)


def test_gates_salary_floor() -> None:
  '''Postings below the floor fail with a named reason.'''
  candidate = _profile()
  assert gate_failures(candidate, {'salary_max_lpa': 40.0}, None)
  assert not gate_failures(candidate, {'salary_max_lpa': 50.0}, None)
  assert not gate_failures(candidate, {}, None)


def test_gates_location_and_mode() -> None:
  '''Location, work-mode, and employment gates behave per policy.'''
  base = {'salary_max_lpa': 60.0}
  assert any(f.startswith('location') for f in gate_failures(_profile(), {**base, 'city': 'Pune'}, None))
  remote_profile = _profile(remote_pref='remote')
  failures = gate_failures(remote_profile, {**base, 'work_mode': 'onsite', 'city': ''}, None)
  assert any(f.startswith('work_mode') for f in failures)
  contract_failures = gate_failures(
    _profile(), {**base, 'city': '', 'employment_type': 'internship'}, None,
  )
  assert any(f.startswith('employment_type') for f in contract_failures)


def test_component_scores_bounds() -> None:
  '''All components return values within [0, 1].'''
  assert salary_fit(None, None, 45.0) == 0.5
  assert salary_fit(50.0, 70.0, 45.0) == 1.0
  assert 0.0 <= seniority_fit(9.0, 8.0, 12.0) <= 1.0
  must, nice = skill_coverage([1, 2], [{'skill_id': 1, 'kind': 'must_have'}])
  assert must == 1.0 and nice == 0.5
  must2, _ = skill_coverage([1], [{'skill_id': 1, 'kind': 'must_have'}, {'skill_id': 2, 'kind': 'must_have'}])
  assert abs(must2 - 0.5) < 1e-9
  assert title_fit(['Data Scientist'], 'Staff Data Scientist - ML') > 0.3
  assert recency_score(None) == 0.5
  assert company_fit('fintech', ['fintech'], 3) == 1.0
  assert company_fit('gaming_media', ['fintech'], 5) >= 0.5


def test_aggregate_and_weights() -> None:
  '''Aggregation produces 0..100 totals and weights renormalize.'''
  weights = normalize_weights({'skills_must': 0.6})
  assert abs(sum(weights.values()) - 1.0) < 1e-6
  total, breakdown = aggregate({k: 1.0 for k in weights}, weights)
  assert total == 100.0 and len(breakdown) == len(weights)


def test_feedback_adjust_bounded() -> None:
  '''Dismissal signal shifts weight to skills within bounds.'''
  adjusted = feedback_adjust(DEFAULT_WEIGHTS := __import__(
    'job_hunter.services.scorer', fromlist=['DEFAULT_WEIGHTS'],
  ).DEFAULT_WEIGHTS, dismiss_ratio=1.0)
  assert adjusted['skills_must'] <= DEFAULT_WEIGHTS['skills_must'] + 0.05 + 1e-9
  assert adjusted['semantic'] >= 0.01
