#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for adapter payload parsing (no network).'''


from __future__ import annotations

from job_hunter.adapters.aggregators import (
  parse_arbeitnow,
  parse_himalayas,
  parse_jobicy,
  parse_remotive,
)
from job_hunter.adapters.cutshort import (
  extract_category_jobs,
  parse_cutshort,
)
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


def test_himalayas_camelcase_parse() -> None:
  '''The live Himalayas camelCase payload maps to records.'''
  payload = {
    'jobs': [{
      'title': 'Data Scientist',
      'companyName': 'Teya',
      'companySlug': 'teya',
      'applicationLink': 'https://himalayas.app/companies/teya/jobs/ds',
      'description': '<p>Model things</p>',
      'employmentType': 'Full Time',
      'locationRestrictions': ['United Kingdom'],
      'minSalary': '50000',
      'maxSalary': '60000',
      'salaryPeriod': 'annual',
      'currency': 'GBP',
      'pubDate': '1790655361',
    }],
  }
  records = parse_himalayas(payload)
  assert len(records) == 1
  assert records[0].company_name == 'Teya'
  assert records[0].location_text == 'United Kingdom'
  assert records[0].salary_raw == 'GBP 50000-60000 annual'
  assert records[0].posted_at == '2026-09-29'


def test_arbeitnow_and_jobicy_parse() -> None:
  '''Arbeitnow and Jobicy payloads map to records with company names.'''
  arbeitnow = {
    'data': [{
      'slug': 'ds-at-acme-1',
      'company_name': 'Acme',
      'title': 'Data Scientist',
      'description': '<p>Build models</p>',
      'remote': True,
      'url': 'https://acme.com/careers/ds',
      'tags': ['Data Science'],
      'job_types': ['Full-time'],
      'location': 'Berlin',
      'created_at': 1786516800,
    }],
  }
  records = parse_arbeitnow(arbeitnow)
  assert records[0].company_name == 'Acme'
  assert records[0].work_mode_hint == 'remote'
  assert records[0].posted_at == '2026-08-12'

  jobicy = {
    'jobs': [{
      'id': 143120,
      'url': 'https://jobicy.com/jobs/143120-ds',
      'jobTitle': 'Senior Data Scientist',
      'companyName': 'RevenueCat',
      'jobGeo': 'LATAM, USA',
      'jobType': ['Full-Time'],
      'jobDescription': '<p>Analyze</p>',
      'pubDate': '2026-09-29T03:35:14+00:00',
      'salaryMin': 208000,
      'salaryMax': 240000,
      'salaryCurrency': 'USD',
      'salaryPeriod': 'yearly',
    }],
  }
  jobicy_records = parse_jobicy(jobicy)
  assert jobicy_records[0].company_name == 'RevenueCat'
  assert '208000-240000' in jobicy_records[0].salary_raw
  assert 'USD' in jobicy_records[0].salary_raw


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


def _cutshort_page(jobs: list) -> str:
  '''Wrap job mappings in a minimal rendered Cutshort category page.

  Args:
    jobs: Job mappings to embed in the Next.js island.

  Returns:
    Page HTML containing the island.
  '''
  import json as _json
  island = {
    'props': {
      'pageProps': {
        'dehydratedState': {
          'queries': [
            {
              'queryKey': ['jobListData', 'datascience-jobs'],
              'state': {
                'data': {'data': {'pageData': {'jobs': jobs}}},
              },
            },
          ],
        },
      },
    },
  }
  return (
    '<html><body><script id="__NEXT_DATA__" type="application/json">'
    f'{_json.dumps(island)}</script></body></html>'
  )


def test_cutshort_salary_uses_display_text_not_structured_bounds() -> None:
  '''Cutshort salary comes from the rendered string, not salaryRange min/max.

  A posting shown as Rs 15L - Rs 20L reports min=750000, so trusting the
  structured bounds would halve the floor. The display text is authoritative.
  '''
  from job_hunter.services.normalizer import parse_salary_lpa

  page = _cutshort_page([{
    '_id': 'abc123',
    'headline': 'Sr. Data Scientist',
    'publicUrl': 'https://cutshort.io/job/Sr-Data-Scientist-abc123',
    'companyDetails': {'name': 'Acme Analytics'},
    'locations': ['Pune'],
    'locationsText': 'Pune',
    'remoteType': 'remote_not_okay',
    'roleTypes': ['full_time'],
    'salaryRangeText': '₹15L - ₹20L / yr',
    'salaryRange': {'min': 750000, 'max': 2000000, 'duration': 'YEAR'},
    'expRange': {'min': 8, 'max': 12},
    'allSkills': ['Python', 'Data Science'],
    'sanitizedComment': '<p>Build models</p>',
  }])
  records = parse_cutshort(page)
  assert len(records) == 1
  assert records[0].salary_raw == '15-20 lakhs per annum'
  assert parse_salary_lpa(records[0].salary_raw) == (15.0, 20.0)
  assert records[0].company_name == 'Acme Analytics'
  assert records[0].work_mode_hint == 'onsite'
  assert records[0].employment_type_raw == 'full_time'
  assert records[0].description_text == 'Build models'
  assert records[0].extra['experience_min_years'] == 8


def test_cutshort_monthly_salary_is_annualized() -> None:
  '''A monthly posting is annualized from the structured rupee bounds.'''
  from job_hunter.services.normalizer import parse_salary_lpa

  page = _cutshort_page([{
    '_id': 'mon1',
    'headline': 'Data Analyst',
    'publicUrl': 'https://cutshort.io/job/Data-Analyst-mon1',
    'companyDetails': {'name': 'Beta'},
    'locations': ['Bengaluru'],
    'remoteType': 'remote_only',
    'salaryRangeText': '₹15000 - ₹18000 / mo',
    'salaryRange': {
      'min': 180000, 'max': 216000, 'duration': 'MONTH', 'currency': 'INR',
    },
  }])
  record = parse_cutshort(page)[0]
  assert record.salary_raw == '1.8-2.2 lakhs per annum'
  assert parse_salary_lpa(record.salary_raw) == (1.8, 2.2)
  assert record.work_mode_hint == 'remote'


def test_cutshort_skips_incomplete_and_client_only_pages() -> None:
  '''Pages without embedded jobs or with unusable entries yield nothing.'''
  assert extract_category_jobs('<html><body>nothing</body></html>') == []
  assert extract_category_jobs(_cutshort_page([])) == []

  # A posting missing both title and URL cannot be stored, so it is dropped.
  page = _cutshort_page([
    {'_id': 'x', 'headline': '', 'publicUrl': ''},
    {'_id': 'y', 'headline': 'ML Engineer', 'publicUrl': 'https://cutshort.io/job/y'},
  ])
  records = parse_cutshort(page)
  assert len(records) == 1
  assert records[0].title == 'ML Engineer'


def test_cutshort_reversed_salary_bracket_is_reordered() -> None:
  '''A reversed lakhs bracket is normalized low-to-high rather than dropped.'''
  page = _cutshort_page([{
    '_id': 'rev',
    'headline': 'DS Lead',
    'publicUrl': 'https://cutshort.io/job/rev',
    'salaryRangeText': '₹30L - ₹20L / yr',
  }])
  assert parse_cutshort(page)[0].salary_raw == '20-30 lakhs per annum'
