# Deploy — forecast-ml

Index for the chart, observability stack, and secrets wiring.
Start here, then follow [`../docs/runbook-p2.md`](../docs/runbook-p2.md) for the operator steps.

## Chart (Helm + K8s)

- Chart: [`helm/forecast-ml/`](helm/forecast-ml/Chart.yaml)
  (`Chart.yaml` + `values.yaml` + `templates/{deployment,service,ingress,serviceaccount,hpa,networkpolicy}.yaml`)
- Deploy: `helm upgrade --install forecast-ml <chart> --namespace default --set image.tag=<tag>` —
  see [Deploy](../docs/runbook-p2.md#deploy-helm-upgrade) in the runbook.
- Rollback drill: [`scripts/rollback-drill.sh`](scripts/rollback-drill.sh)
  (`helm rollback` + `kubectl rollout status`; CI job `k8s-drill`)
  — see [Rollback](../docs/runbook-p2.md#rollback-helm-rollback).
- Retention CronJob (P2-S1): [`k8s/retention-cronjob.yaml`](k8s/retention-cronjob.yaml)
  (`python -m forecast_ml_framework.governance.retention_job --retention --governance-params ...`)
  — see [Retention](../docs/runbook-p2.md#retentionerasure-ops).

## Observability (OTel / Prometheus / Alertmanager / Grafana)

- Collector: [`observability/otel-collector.yaml`](observability/otel-collector.yaml)
  (OTLP receiver, Prometheus exporter)
- Scrape: [`observability/prometheus.yaml`](observability/prometheus.yaml)
- Dashboards: [`observability/dashboards/`](observability/dashboards/model-drift.json)
  (`model-drift.json`, `api-latency.json` — query the live series emitted by
  the `/metrics` producer in
  [`../src/forecast_ml_framework/serving/metrics.py`](../src/forecast_ml_framework/serving/metrics.py)
  (`drift_psi`, `drift_ks`, `http_request_duration_seconds`,
  `predict_requests_total`, `predict_errors_total`), served by
  `serving/app.py` `GET /metrics`)
- Alertmanager: [`observability/alertmanager/alertmanager.yaml`](observability/alertmanager/alertmanager.yaml)
  (Slack via `url_file`, PagerDuty via `routing_key_file` from K8s Secrets)
- Rules: [`observability/prometheus/rules.yaml`](observability/prometheus/rules.yaml)
  (`drift_psi > 0.2` alert `DriftPsiHigh` on the live `drift_psi` gauge)
- Firing proof: [`observability/scripts/synthetic-drift.sh`](observability/scripts/synthetic-drift.sh)
  (feeds synthetic `drift_psi = 0.25` values against the live-named rule to
  prove firing; the values are synthetic, the metric name is live)
  — see [Alert firing](../docs/runbook-p2.md#alert-firing-synthetic-drift).

## Secrets wiring (out-of-band, references only)

Secrets stay out-of-band (env-only, no rotation code, no residency
pin — D-SECRETS-PII-GDPR-1, section 5.2(e)):

- Terraform references (names only, never values):
  [`../infra/gcp/secrets.tf`](../infra/gcp/secrets.tf),
  [`../infra/aws/secrets.tf`](../infra/aws/secrets.tf),
  [`../infra/azure/secrets.tf`](../infra/azure/secrets.tf) (P2-S1, references only).
- Helm keys (names/keys only, never values):
  [`helm/forecast-ml/values.yaml`](helm/forecast-ml/values.yaml)
  `envFrom:` + `secrets:` map (`secretKeyRef` entries), owned by P2-C1.
- OIDC env: `FORECAST_OIDC_ISSUER`, `FORECAST_OIDC_AUDIENCE`,
  `FORECAST_OIDC_JWKS_URL` (see [OIDC rotation](../docs/runbook-p2.md#oidc-rotation-out-of-band)
  in the runbook).
- Retention params: [`../conf/base/platform.yml`](../conf/base/platform.yml)
  (`governance_params.retention_days`, P2-S1).

## Readiness

CI gates: `terraform-validate` (matrix gcp/aws/azure), `helm-lint`,
`observability-check`, `k8s-drill` (see
[`../../.github/workflows/forecast-ml-ci.yml`](../../.github/workflows/forecast-ml-ci.yml)).
Local (from the project root): `bash ci/run_checks.sh`
([`../ci/run_checks.sh`](../ci/run_checks.sh);
terraform fmt + init/validate for all three stacks). Readiness is
not production promotion — see [Readiness ≠ promotion](../docs/runbook-p2.md#readiness--promotion).
