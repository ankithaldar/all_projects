#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for adapter payload parsing (no network).'''


from __future__ import annotations

from job_hunter.adapters.aggregators import parse_remotive
from job_hunter.adapters.greenhouse import parse_payload as parse_greenhouse
from job_hunter.adapters.lever import parse_payload as parse_lever
from job_hunter.adapters.ashby import parse_payload as parse_ashby
from job_hunter.adapters.workable import parse_payload as parse_workable
from job_hunter.adapters.smartrecruiters import parse_list_payload as parse_sr
from job_hunter.adapters.personio import parse_xml


def test_greenhouse_parse() -> None:
  '''Greenhouse payload maps to records with html content.'''
  payload = {
    'jobs': [
      {
        'id': 123,
        'title': 'Staff Data Scientist',
        'absolute_url': 'https://boards.example.com/jobs/123',
        'location': {'name': 'Bengaluru, India'},
        'content': '<p>Build models</p>',
        'updated_at': '2026-08-01T00:00:00Z',
      },
      {'id': 124, 'title': '', 'absolute_url': ''},
    ],
  }
  records = parse_greenhouse(payload)
  assert len(records) == 1
  assert records[0].title == 'Staff Data Scientist'
  assert records[0].location_text == 'Bengaluru, India'


def test_lever_parse() -> None:
  '''Lever payload maps categories into fields.'''
  payload = [{
    'id': 'abc',
    'text': 'Data Scientist',
    'hostedUrl': 'https://jobs.lever.co/co/abc',
    'createdAt': 1755000000000,
    'categories': {
      'location': 'Remote, India',
      'commitment': 'Full Time',
      'workplaceType': 'remote',
    },
    'descriptionPlain': 'Do DS work',
  }]
  records = parse_lever(payload)
  assert records[0].work_mode_hint == 'remote'
  assert records[0].employment_type_raw == 'Full Time'
  assert records[0].posted_at == '1755000000000'


def test_ashby_parse_with_compensation() -> None:
  '''Ashby compensation tranches render a salary string.'''
  payload = {
    'jobs': [{
      'id': 'j1',
      'title': 'Staff DS',
      'jobUrl': 'https://jobs.ashbyhq.com/co/j1',
      'location': 'Bengaluru',
      'isRemote': True,
      'publishedAt': '2026-08-10T00:00:00Z',
      'compensation': {
        'period': 'year',
        'tranches': [{
          'currency': 'INR',
          'compensationTierSummary': {'minimumAmount': 4500000, 'maximumAmount': 6500000},
        }],
      },
    }],
  }
  records = parse_ashby(payload)
  assert records[0].salary_raw.startswith('INR')
  assert '4500000' in records[0].salary_raw


def test_workable_and_smartrecruiters_parse() -> None:
  '''Workable widget and SR list payloads map cleanly.'''
  workable = {
    'jobs': [{
      'id': 'w1',
      'title': 'Senior DS',
      'shortlink': 'https://apply.workable.com/co/j/w1/',
      'location': {'city': 'Pune', 'country': 'India'},
      'remote': True,
      'created_at': '2026-08-05',
    }],
  }
  sr = {
    'content': [{
      'id': 's1',
      'name': 'Lead DS',
      'location': {'city': 'Gurugram', 'country': 'India'},
      'releasedDate': '2026-08-11T00:00:00Z',
    }],
  }
  assert parse_workable(workable)[0].url.endswith('w1/')
  sr_records = parse_sr(sr, 'co')
  assert sr_records[0].url == 'https://jobs.smartrecruiters.com/co/s1'


def test_personio_parse() -> None:
  '''Personio XML feed parses positions.'''
  xml = '''<?xml version="1.0"?>
  <positions>
    <position><id>7</id><name>Staff Data Scientist</name>
    <detailUrl>https://co.jobs.personio.de/job/7</detailUrl>
    <office>Bengaluru</office></position>
  </positions>'''
  records = parse_xml(xml)
  assert records[0].external_id == '7'


def test_remotive_parse() -> None:
  '''Remotive API payload maps with remote hint.'''
  payload = {
    'jobs': [{
      'id': 9,
      'url': 'https://remotive.com/remote-jobs/data/9',
      'company_name': 'Acme',
      'title': 'DS II',
      'candidate_required_location': 'India',
      'description': '<p>Analyze</p>',
      'publication_date': '2026-08-09T00:00:00',
    }],
  }
  records = parse_remotive(payload)
  assert records[0].work_mode_hint == 'remote'
