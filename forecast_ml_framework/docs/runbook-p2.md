# P2 Runbook — forecast-ml platform readiness

> Read this before you deploy, roll back, rotate OIDC, fire alerts, or
> run retention/erasure. All commands run from the repo root unless
> stated otherwise. No secret value may appear in any command, file, or
> log below — use environment variables and K8s Secret mounts only.

## Deploy (helm upgrade)

The app is deployed from the Helm chart only — never hand-applied YAML.

1. Lint and render the chart:
   `helm lint forecast_ml_framework/deploy/helm/forecast-ml`
   `helm template forecast-ml forecast_ml_framework/deploy/helm/forecast-ml`
2. Upgrade the release:
   `helm upgrade --install forecast-ml forecast_ml_framework/deploy/helm/forecast-ml --namespace default --set image.tag=<tag>`
   (`image.tag` is the CI-built artifact tag; `values.yaml` holds
   `image.repository`, `replicaCount`, `env`, `envFrom`, `secrets` —
   names/keys only, never values — `serviceAccount`, `autoscaling`,
   `ingress`.)
3. Verify the rollout is green:
   `kubectl rollout status deployment/forecast-ml --namespace default --timeout=120s`
   `curl -fsS http://localhost:8000/health` returns **200**
   (probes: `readinessProbe` + `livenessProbe` on `/health`;
   frozen `securityContext`: `runAsNonRoot`, `runAsUser 10001`,
   `allowPrivilegeEscalation false`, `readOnlyRootFilesystem`,
   `capabilities.drop [ALL]`, `seccompProfile RuntimeDefault`).
4. Record the version, commit, and `helm history` output in the
   deploy handoff note.

Secrets wiring owner: P2-C1 owns the frozen Helm secret keys in
`values.yaml` (`envFrom` + `secretKeyRef`); P2-S1 owns the Secret
names via `infra/{gcp,aws,azure}/secrets.tf` (references only).
OIDC wiring: `serving/auth.py`, RS256 only, issuer/audience from env
(`FORECAST_OIDC_ISSUER`, `FORECAST_OIDC_AUDIENCE`,
`FORECAST_OIDC_JWKS_URL`).

## Rollback (helm rollback)

Every release ships with a tested rollback path — `helm rollback`
plus the drill script. A green pipeline with a red health check is
not a deploy.

1. Roll back to the previous revision:
   `helm rollback forecast-ml --namespace default`
   Verify: `kubectl rollout status deployment/forecast-ml --namespace default --timeout=120s`
   and `helm history forecast-ml --namespace default`.
2. Rehearse with the drill script (CI job `k8s-drill` runs the same):
   `bash forecast_ml_framework/deploy/scripts/rollback-drill.sh`
   This upgrades the chart twice (v1 to v2 image tag) then
   `helm rollback` and asserts `kubectl rollout status` green.
   On CI without a cluster it runs the template-only fallback
   (`helm lint` + `helm template`) and logs which path it took.
3. If a rollback fails, stop — do not retry forward. Escalate to the
   on-call with the `helm history` + `kubectl rollout status` output.

## OIDC rotation out-of-band

OIDC rotation is **out-of-band via cloud** — there is no in-repo
rotation code, no rotation schedule, and no `rotate`/`refresh`
helper (D-SECRETS-PII-GDPR-1, section 5.2(e)). Rotation steps:

1. GCP: create the new workload-identity / secret version in Secret
   Manager via console or `gcloud` (out-of-band), then re-point the
   Helm `secrets:` map value (secret name only) and re-deploy.
2. AWS: `aws secretsmanager put-secret-value` (out-of-band CLI),
   then re-point the Helm `secrets:` map value (name only) and
   re-deploy.
3. Azure: `az keyvault secret set` (out-of-band CLI) to fill the
   Key Vault secret, then re-point the Helm `secrets:` map value
   (name only) and re-deploy.
4. Entra OIDC issuer keys rotate out-of-band in the Entra console;
   the app picks up new JWKS keys on the next `verify()` after the
   cache TTL (`cache_ttl_s=300`, default in `OIDCVerifier`) — no
   restart needed, but roll the pods if you need the old `kid` dead
   immediately (note the Istio/JWKS cache TTL: gateways in front of
   the app may cache JWKS documents separately, so allow up to the
   gateway TTL plus 300 s, or restart the gateway, before declaring
   the old key dead).
5. Revocation: `FORECAST_OIDC_REVOKED_HASHES` (comma-separated hashes)
   and/or `FORECAST_OIDC_REVOKED_FILE` (one hash per line) hold
   `subject_hash` values only — never clear `jti`/`sub`. Reload at
   startup.
6. Never log the token, `jti`, or `sub` in clear (hashed via
   `subject_hash`); unknown OIDC issuer/audience fails closed (401,
   not 500, not allow). OIDC roundtrip proof:
   `pytest forecast_ml_framework/tests/test_p2_oidc_auth.py -q`
   (valid RS256 token 200; expired/revoked/wrong-audience 401).

## Alert firing (synthetic drift)

Alert syntax fires end-to-end on the live `/metrics` producer
(`serving/metrics.py`, served by `serving/app.py` `GET /metrics`).
Dashboards and rules reference the live metric names emitted by
that producer (`http_request_duration_seconds`, `drift_psi`,
`drift_ks`, `predict_requests_total`, `predict_errors_total`).
Only the firing-proof values are synthetic — the metric names
are live.

1. Check the Alertmanager + Prometheus configs:
   `amtool check-config forecast_ml_framework/deploy/observability/alertmanager/alertmanager.yaml`
   `promtool check config forecast_ml_framework/deploy/observability/prometheus.yaml`
   `promtool check rules forecast_ml_framework/deploy/observability/prometheus/rules.yaml`
2. Prove firing on synthetic drift values:
   `bash forecast_ml_framework/deploy/observability/scripts/synthetic-drift.sh`
   This feeds synthetic drift values (`drift_psi = 0.25`, above the
   live `DriftPsiHigh` threshold of `0.2`) against the live-named
   rule, runs `promtool test rules` against the rules fixture, and
   prints the fired alert name (`DriftPsiHigh`). Exit 0 is the
   exit-gate evidence. CI job `observability-check` runs the same steps.
3. Routing: `severity=critical` goes to PagerDuty
   (`routing_key_file` from the `alertmanager-pagerduty` Secret),
   `severity=warning` goes to Slack (`api_url_file` from the
   `alertmanager-slack` Secret). No webhook URL or routing key in
   the repo — `url_file` / `routing_key_file` refs only.

## Retention/erasure ops

Retention sweep runs daily via the K8s CronJob; erasure is
operator-invoked through the secure channel only.

1. Retention (automatic): the `CronJob`
   (`forecast_ml_framework/deploy/k8s/retention-cronjob.yaml`,
   schedule `0 3 * * *`, owned by P2-S1) runs
   `python -m forecast_ml_framework.governance.retention_job --retention --governance-params /etc/forecast-ml/platform.yml`.
   Retention params come only from
   `forecast_ml_framework/conf/base/platform.yml`
   (`governance_params.retention_days` via
   `--governance-params`); never from catalog metadata.
2. Dry-run (plan only, deletes nothing):
   `python -m forecast_ml_framework.governance.retention_job --retention --dry-run --governance-params forecast_ml_framework/conf/base/platform.yml`
   (`--dry-run` calls `check_retention` only; missing/invalid
   `platform.yml` exits 2 with no store mutation.)
3. Erasure (secure channel only): clear subject ID from env
   `FORECAST_ERASE_SUBJECT_ID` only — never argv, never logs —
   via `run_erasure` delegating to frozen `erase_subject`.
   Operator CLI erasure is gated on new ADR `D-ERASURE-CLI-1`
   (owner `solution-architect` + anti-team gate). Until that ADR
   is frozen, erasure acceptance exercises the library path with
   real fake stores; no production CLI erasure is enabled.
   The hash-only `--erase-subject-hash` CLI is removed (frozen
   `erase_subject(subject_id, ...)` requires the clear ID).
4. Audit: actor hashed via `subject_hash` (never clear actor);
   reports carry hashes/counts only, never clear subject ID.

## Residency note (non-binding)

Terraform `region`/`location` variables are **non-binding**: they
make no residency or data-boundary claim (D-SECRETS-PII-GDPR-1,
section 5.2(e)). Artifacts created before the P11 decision carry
no residency guarantee. The residency decision is taken at P2 IaC
variables (non-binding) before P11; no region literal in this repo
is a commitment.

## Readiness ≠ promotion

Completing P2 proves the platform is **ready** (build/deploy/
observability criteria met). It does **not** authorize production
traffic, model promotion, or release (D-MVP-READINESS-1).
Production promotion is P11 only — no `production=true` apply, no
alias flip, no serving traffic beyond readiness checks.
**Readiness ≠ promotion.**
