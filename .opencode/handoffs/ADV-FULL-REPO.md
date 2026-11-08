# ADV-FULL-REPO — Full-repo adversarial approval (Lead report)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work; attack
  scripts only under /tmp/opencode/adv/, no src/ modified by this review).
- DEGRADED MODE DISCLOSURE: nested subagent dispatch is blocked in this
  session (subagent depth limit 1), so the tester could NOT be tasked.
  All attacks below were executed directly against real code paths by the
  adversarial tester (scripts /tmp/opencode/adv/*.py). No independent
  third-party verification is claimed.
- Verdict: **REJECTED** (see findings F1-F3; blocked list is extensive).

## Successful attacks (REJECTED basis)
- F1 [HIGH auth/deploy] Default Docker/Helm deployment is unauthenticated:
  Dockerfile ENTRYPOINT uvicorn --host 0.0.0.0 (Dockerfile:32) + module-level
  app=create_app() with host=None (serving/app.py:1281) skips validate_bind_host
  (serving/app.py:1095, only when host is not None); _validate_auth_config only
  raises under prod profile/REQUIRE_AUTH (serving/app.py:260-295). Helm defaults
  env:{}/secrets:{} (deploy/helm/forecast-ml/values.yaml:9,13) set neither.
  Proven: no-env create_app -> POST /predict = 200. The stated claim
  "non-loopback bind without a key is refused" does not hold for the shipped
  image path. Runbook deploy steps set no auth env (docs/runbook-p2.md:8-24).
- F2 [HIGH/MED injection] woe_iv_optimizer SQLite UNION exfiltration proven:
  table='t" UNION SELECT s FROM secret --' returned TOPSECRET (sqlite_loader.py:75
  unescaped f-string). Same pattern: BigQuery backtick breakout
  (bigquery_loader.py:36, string proof) and GBQ sql.format(**ctx) raw
  interpolation (run_etl_in_levels.py:520, string proof). No validator, no
  documented trusted-config boundary (unlike forecast datasets.py:165-170).
  Write/stack blocked only by sqlite3 single-statement rule.
- F3 [MED consent] Consent is opt-in, fail-open by omission:
  guard_write returns when consent_ledger is None (governance/pii.py:216-217);
  proven OfflineFeatureStore.materialize wrote a declared-email frame with no
  ledger. PII detection false negatives proven: phone, full_name, SSN value in
  neutral column, IBAN in notes all PASS with pii_classes=['none']
  (governance/pii.py detect_pii). Unsalted subject_hash dictionary-reversible
  (governance/scrub.py:70; documented accepted risk in pii.py header).
- F4 [MED extraction, amplifies F1] /predict 600/min/process token bucket
  (serving/app.py:483-516,952-961), no per-principal limit, 2 replicas by
  default; proven 626/650 burst successes. Full-precision predictions.
- F5 [LOW] OIDC token without sub VERIFIED (serving/auth.py:394-401 no sub
  requirement; authenticate principal_hash=None -> 'oidc-anonymous').
- F6 [LOW supply-chain] Dockerfile pip install uv unpinned + python:3.14-slim
  no digest (Dockerfile:4,9,16); gbq_parallel_runner/requirements.txt fully
  unpinned; woe requirements lower-bound only. (Forecast mainline is well
  pinned: bounded deps + uv.lock `uv lock --check` clean + SHA-pinned CI.)
- F7 [LOW] JWKS default_fetcher has no scheme allowlist
  (serving/auth.py:161-191): file:///etc/hostname was read (rejected only at
  JSON parse); redirects followed. URL is env-operator-only.

## Blocked (proven, executed)
- Hostile pickle: os.system/eval/pd.read_pickle-chain/STACK_GLOBAL/EXT1,
  Booster NEWOBJ/NEWOBJ_EX model_file (incl. MEMOIZE indirection),
  FunctionTransformer-os.system, CatBoost file kwargs (not accepted by API),
  oversized cap, .onnx - all rejected, no marker files; legit LinearRegression
  roundtrips (core/artifact_store.py:764-837, serving/model_loader.py).
- Forecast SQL validator: 20/23 payloads blocked incl. stacked, CTE-DELETE,
  pg_read_file quoted, READFILE, advisory locks; sqlite table intact
  (core/datasets.py:82-197). UNION-SELECT allowed by documented design
  (trusted read access, datasets.py:160-163).
- SSRF registry URI: metadata/private/hex/decimal/encoded/userinfo/file/
  javascript all blocked; localhost allowed by design
  (registry/model_registry.py:369-452). JWKS URL not request-controllable.
- oc.env smuggling: direct/spaced/case/oc.create/cross-key (dict+list)/
  dict-key/substitution-recombination all rejected, canary never leaked;
  shipped load_config canary-free (core/config_schema.py, core/config_loader.py:81-112).
- Auth (when configured): no/wrong/empty key 401, duplicate headers 401,
  non-ASCII key no-500, OIDC alg=none/HS256/expired/wrong-iss-aud/unknown-kid/
  missing-kid-exp/nbf/tampered-sig/no-scope/revoked all rejected, shared-key
  ignored in OIDC mode (serving/auth.py, serving/app.py:898-944). Existing
  suites: test_core+test_model_loader+test_app+test_p2_oidc*+test_governance+
  test_reject_oc_direct+test_settings = 1142 passed.
- DoS/caps: 11MB body 413, 10001 records 413, 1001 keys 413, bool/string/NaN/
  10**400 features 422, scrub 64KB/2000-filler 0.02s no leak, token bucket
  exact 1000/1000 concurrent, loader single-object under 20 threads,
  artifact path jail incl. symlink escape (artifact_store.py:982-1003),
  concurrent oc-resolver threads no canary leak.
- XSS/command/prompt: web esc()/safeChartFile proven via node (script escaped,
  ../../ and javascript: rejected); CSP/nosniff/DENY headers present;
  CLI subprocess argv-list only (cli.py:238), handle allowlist rejects
  os.system/subprocess/eval (core/handle.py); prompt injection N/A - no LLM
  surface in repo (grep: only fullmatch false positives).

## Race (not reproduced)
- Online store 50 concurrent distinct-ts writes kept 50/50; same-ts dedup is
  the documented merge rule. Offline delete+verify proven; single-slot
  erasure marker overwrite + predictions/monitoring legal-hold retention are
  documented (governance/erasure.py:22-32).

## Minimal remediation for re-review
1. Make the shipped image fail closed: Dockerfile ENTRYPOINT via a wrapper
   that calls validate_bind_host('0.0.0.0') / require REQUIRE_AUTH or OIDC in
   the chart values, or document that ingress must never be enabled without
   auth env + gateway. Add a CI test asserting create_app() module-level app
   with 0.0.0.0 uvicorn host refuses without key.
2. Parameterize/validate woe sqlite table + bigquery identifiers (allowlist
   regex like forecast validate_model_name) and GBQ interpolation (quote
   literals / use query parameters), or document+enforce trusted-config-only
   with the forecast-style validator.
3. Require consent_ledger (or explicit consent_opt_out) at feature-store
   writes; extend detect_pii value heuristics (phone/SSN/IBAN patterns) or
   require explicit column inventory for ['none'].
