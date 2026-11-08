#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P2-O2 alert routing tests: Slack/PagerDuty file refs + stub rules.'''

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ALERTMANAGER = (
  ROOT / 'deploy' / 'observability' / 'alertmanager' / 'alertmanager.yaml'
)
RULES = ROOT / 'deploy' / 'observability' / 'prometheus' / 'rules.yaml'
SCRIPT = (
  ROOT / 'deploy' / 'observability' / 'scripts' / 'synthetic-drift.sh'
)
SLACK_FILE = '/etc/alertmanager/secrets/slack-webhook'
PAGERDUTY_FILE = '/etc/alertmanager/secrets/pagerduty-routing-key'


def _load(path):
  '''Load a YAML file as a dict.

  Args:
    path: Path to the YAML file.

  Returns:
    Parsed YAML mapping.

  Raises:
    FileNotFoundError: If the file does not exist.
  '''
  with Path(path).open(encoding='utf-8') as handle:
    data = yaml.safe_load(handle)
  assert isinstance(data, dict)
  return data


def _receivers(config):
  '''Index Alertmanager receivers by name.

  Args:
    config: Parsed Alertmanager config.

  Returns:
    Mapping of receiver name to receiver mapping.
  '''
  return {item['name']: item for item in config['receivers']}


def _all_rules(config):
  '''Flatten Prometheus rule groups into a list of rules.

  Args:
    config: Parsed Prometheus rules config.

  Returns:
    List of every rule mapping across all groups.
  '''
  rules = []
  for group in config['groups']:
    rules.extend(group['rules'])
  return rules


def test_alertmanager_receivers_use_file_refs():
  config = _load(ALERTMANAGER)
  receivers = _receivers(config)
  slack = receivers['slack-ops']['slack_configs'][0]
  assert slack['api_url_file'] == SLACK_FILE
  assert 'api_url' not in slack
  pager = receivers['pagerduty-critical']['pagerduty_configs'][0]
  assert pager['routing_key_file'] == PAGERDUTY_FILE
  assert 'routing_key' not in pager
  for ref in (slack['api_url_file'], pager['routing_key_file']):
    assert ref.startswith('/etc/alertmanager/secrets/')


def test_alertmanager_route_groups_by_alertname():
  config = _load(ALERTMANAGER)
  route = config['route']
  assert route['group_by'] == ['alertname']
  assert route['receiver'] == 'slack-ops'
  by_severity = {}
  for child in route['routes']:
    text = ' '.join(str(item) for item in child['matchers'])
    found = re.search(r'severity\s*=\s*"([^"]+)"', text)
    assert found is not None
    by_severity[found.group(1)] = child['receiver']
  assert by_severity['critical'] == 'pagerduty-critical'
  assert by_severity['warning'] == 'slack-ops'


def test_no_literal_webhook_host_or_key_in_yaml():
  text = ALERTMANAGER.read_text(encoding='utf-8')
  slack_host = 'hooks' + '.slack' + '.com'
  pager_host = 'events' + '.pagerduty' + '.com'
  assert slack_host not in text
  assert pager_host not in text
  assert re.search(r'https?' + r'://', text) is None
  assert 'api_url:' not in text
  assert 'routing_key:' not in text


def test_rules_yaml_has_recording_rules():
  rules = _all_rules(_load(RULES))
  recorded = {rule['record'] for rule in rules if 'record' in rule}
  assert 'job:predict_rate5m' in recorded
  assert 'job:http_p95_5m' in recorded
  for rule in rules:
    if 'record' in rule:
      assert rule['expr']


def test_drift_psi_high_stub_threshold_and_for():
  rules = _all_rules(_load(RULES))
  alerts = {rule['alert']: rule for rule in rules if 'alert' in rule}
  assert 'HighPredictErrorRate' in alerts
  assert 'ApiLatencyHigh' in alerts
  assert 'DriftPsiHigh' in alerts
  drift = alerts['DriftPsiHigh']
  assert 'drift_psi' in drift['expr']
  assert '> 0.2' in drift['expr']
  assert str(drift['for']) == '5m'
  assert drift['labels']['severity'] == 'critical'
  latency = alerts['ApiLatencyHigh']
  assert 'job:http_p95_5m' in latency['expr']
  assert '> 0.5' in latency['expr']
  assert str(latency['for']) == '5m'


def test_drift_psi_high_fires_on_synthetic_stub_value():
  rules = _all_rules(_load(RULES))
  drift = next(
    rule for rule in rules if rule.get('alert') == 'DriftPsiHigh'
  )
  threshold = float(re.search(r'>\s*([0-9.]+)', drift['expr']).group(1))
  assert 0.25 > threshold
  assert 0.1 <= threshold


def test_synthetic_drift_script_prints_fired_alert():
  assert SCRIPT.exists()
  assert SCRIPT.stat().st_mode & 0o111
  text = SCRIPT.read_text(encoding='utf-8')
  assert 'DriftPsiHigh' in text
  result = subprocess.run(
    ['bash', str(SCRIPT)],
    capture_output=True,
    text=True,
    timeout=60,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  assert 'DriftPsiHigh' in result.stdout
