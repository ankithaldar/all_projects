#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P2-C1 container and Helm deploy tests.

RED: Dockerfile and Helm chart do not exist yet, so the
first read raises FileNotFoundError. GREEN: multi-stage
uv Dockerfile plus Helm chart render the frozen serving
entrypoint, securityContext, probes, and NetworkPolicy.
'''

import os
import re
from pathlib import Path

import yaml

_ENTRYPOINT = (
    'uvicorn forecast_ml_framework.serving.app:app '
    '--host 0.0.0.0 --port 8000')


def _root():
  '''Return the forecast_ml_framework project root.'''
  return Path(__file__).resolve().parents[1]


def _read(relative):
  '''Read a project-relative file as text.

  Args:
    relative: Path relative to the project root.

  Returns:
    File contents as text.

  Raises:
    FileNotFoundError: When the file does not exist.
  '''
  return (_root() / relative).read_text(encoding='utf-8')


def _chart():
  '''Return the Helm chart directory path.'''
  return _root() / 'deploy' / 'helm' / 'forecast-ml'


def _values():
  '''Load and return the Helm values mapping.'''
  return yaml.safe_load(_read(
      'deploy/helm/forecast-ml/values.yaml'))


def test_dockerfile_frozen_entrypoint_and_user():
  text = _read('Dockerfile')
  assert 'FROM python:3.14-slim' in text
  assert 'uv sync --frozen --no-dev' in text
  assert 'USER 10001' in text
  assert 'EXPOSE 8000' in text
  lines = [line for line in text.splitlines()
           if line.strip().startswith('ENTRYPOINT')]
  assert lines, 'missing ENTRYPOINT line'
  assert any(_ENTRYPOINT in line for line in lines)
  assert 'pyproject.toml' in text
  assert 'uv.lock' in text
  assert 'src' in text
  assert 'conf' in text
  assert 'onnx' not in text.lower()


def test_values_yaml_frozen_keys():
  values = _values()
  assert values['image']['repository'] == 'forecast-ml'
  assert str(values['image']['tag']) == '0.1.0'
  assert values['replicaCount'] == 2
  assert isinstance(values['env'], dict)
  assert isinstance(values['envFrom'], list)
  assert isinstance(values['secrets'], dict)
  assert values['serviceAccount']['create'] is True
  assert values['serviceAccount']['name'] == 'forecast-ml'
  assert values['autoscaling']['enabled'] is False
  assert values['autoscaling']['minReplicas'] == 1
  assert values['autoscaling']['maxReplicas'] == 5
  assert values['ingress']['enabled'] is False
  assert values['service']['type'] == 'ClusterIP'
  assert values['service']['port'] == 8000


def test_values_yaml_has_no_secret_values():
  values = _values()
  assert not values['secrets']
  assert not values['envFrom']
  text = _read('deploy/helm/forecast-ml/values.yaml')
  assert not re.search(r'sk-[A-Za-z0-9]', text)
  assert not re.search(r'AKIA[0-9A-Z]{16}', text)
  for name, spec in values['secrets'].items():
    assert set(spec) <= {'secretName', 'key'}, name


def test_chart_yaml_identity():
  chart = yaml.safe_load(_read(
      'deploy/helm/forecast-ml/Chart.yaml'))
  assert chart['apiVersion'] == 'v2'
  assert chart['name'] == 'forecast-ml'
  assert str(chart['version']) == '0.1.0'
  assert str(chart['appVersion']) == '0.1.0'


def test_deployment_probes_security_and_env():
  text = _read(
      'deploy/helm/forecast-ml/templates/deployment.yaml')
  assert text.count('/health') >= 2
  assert 'readinessProbe:' in text
  assert 'livenessProbe:' in text
  assert 'serviceAccountName:' in text
  assert 'containerPort: 8000' in text
  assert 'runAsNonRoot: true' in text
  assert 'runAsUser: 10001' in text
  assert 'allowPrivilegeEscalation: false' in text
  assert 'readOnlyRootFilesystem: true' in text
  assert 'drop:' in text
  assert 'ALL' in text
  assert 'seccompProfile:' in text
  assert 'RuntimeDefault' in text
  assert 'envFrom' in text
  assert 'secretKeyRef' in text
  assert '.Values.env' in text
  assert '.Values.resources' in text
  assert '.Values.image.repository' in text


def test_service_ingress_serviceaccount_hpa():
  service = _read(
      'deploy/helm/forecast-ml/templates/service.yaml')
  assert 'ClusterIP' in service
  assert 'port: 8000' in service
  ingress = _read(
      'deploy/helm/forecast-ml/templates/ingress.yaml')
  assert '.Values.ingress.enabled' in ingress
  account = _read(
      'deploy/helm/forecast-ml/templates/serviceaccount.yaml')
  assert '.Values.serviceAccount.create' in account
  assert 'annotations' in account
  hpa = _read('deploy/helm/forecast-ml/templates/hpa.yaml')
  assert '.Values.autoscaling.enabled' in hpa
  assert 'minReplicas' in hpa
  assert 'maxReplicas' in hpa


def test_networkpolicy_default_deny_scrape_only():
  path = _chart() / 'templates' / 'networkpolicy.yaml'
  assert path.exists()
  policy = yaml.safe_load(path.read_text(encoding='utf-8'))
  assert policy['kind'] == 'NetworkPolicy'
  assert policy['spec']['policyTypes'] == ['Ingress']
  text = path.read_text(encoding='utf-8')
  assert '8000' in text
  assert 'namespaceSelector' in text
  assert 'prometheus' in text
  assert 'otel' in text
  assert 'ingress' in text
  assert '0.0.0.0/0' not in text


def test_rollback_drill_executable_and_mentions():
  path = _root() / 'deploy' / 'scripts' / 'rollback-drill.sh'
  assert path.exists()
  assert os.access(str(path), os.X_OK)
  text = path.read_text(encoding='utf-8')
  assert 'helm rollback' in text
  assert 'rollout status' in text
  assert 'helm upgrade' in text


def test_dockerignore_keeps_required_inputs():
  text = _read('.dockerignore')
  lines = [line.strip() for line in text.splitlines()
           if line.strip() and not line.strip().startswith('#')]
  assert 'src' not in lines
  assert 'conf' not in lines
  assert 'pyproject.toml' not in lines
  assert 'uv.lock' not in lines
  assert 'README.md' not in lines
