# ADV-CYCLE4 — Cycle 4 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work; attack scripts only under /tmp/opencode/cycle4_adv/, no src/ modified by this review)
- Scope: forecast_ml_framework only (woe/GBQ F2, supply-chain F6 out of scope per Lead)
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this subagent session); attacks executed directly against real code paths by adversarial tester (scripts /tmp/opencode/cycle4_adv/cycle4.py, pickle_extra*.py). No independent third-party verification claimed.
- Verdict: **REJECTED** — 1 blocking (C-1 consent detection false-negative, F3 residual). All other 6 vectors blocked in shipped forecast paths. Residuals below are non-blocking/documented unless Lead/anti-team reclassifies C-1.

## Attack plan
1 auth bypass default | 2 consent bypass | 3 JWKS file/redirect | 4 OIDC no-sub | 5 pickle hostile | 6 extraction burst | 7 envelope invalid via file loader
Expected blocked: RuntimeError/401/429/ValueError/ModelCorruptError, no marker, no PII write for declared/detected PII.

## Blocked (proven, executed)
- 1 Auth default: module import with FORECAST_BIND_HOST=0.0.0.0 no key rc=1 RuntimeError (serving/app.py:1374-1381,257-284); Helm env (deploy/helm/forecast-ml/values.yaml:14-16) rc=1; create_app host=0.0.0.0/::/'0.0.0.0 '/'0', env BIND, short key, partial OIDC all RuntimeError (app.py:1175-1179); Helm+key POST no/wrong=401 good=200; Dockerfile:35-36,43 validate_bind_host before uvicorn; cli.py:215-222 serve validates host.
- 3 JWKS: validate+fetcher blocked file/http/private/metadata/localhost/encoded/userinfo/::ffff:169... (serving/auth.py:266-305,462-528); 302 to file/private/http/metadata//169... blocked, seen=[initial only] (auth.py:485-506,367-380); autofollow final-URL revalidated (auth.py:511-514).
- 4 OIDC no-sub: real signed RS256 token missing/empty/ws/int sub, +jti, +no-scope all BLOCKED ValueError: missing sub (auth.py:731-733); with-sub control ALLOWED.
- 5 Pickle: os.system/eval REDUCE, pd.read_pickle chain, FunctionTransformer, Booster REDUCE/NEWOBJ_EX model_file, oversized cap100 all BLOCKED ValueError, marker absent (core/artifact_store.py:764-837,790-837); legit LinearRegression/XGBRegressor roundtrip ok; FileLoader hostile/.onnx ModelCorruptError (serving/model_loader.py:505-515).
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 641x200/9x429 (token bucket 600/min app.py:99,511-540, refill 10/s explains >600); limiter works, unauth blocked.
- 7 Envelope via file: FileModelLoader/load_model/load_best bad label_set, missing/extra keys, None model, classifier-None, regressor+label all ModelCorruptError (model_loader.py:63-107,430-515; pipelines/modeling/envelope.py:146-252); valid classification/regression controls unwrap correctly via FileLoader.
- Shipped suites: test_p2_default_auth+test_consent_failclosed+test_jwks_ssrf+test_jwks_redirect+test_oidc_sub_required+test_envelope_enforcement+test_file_loader_validate+test_core+test_model_loader = 239 passed; test_app+test_p2_oidc*+test_governance = 1066 passed.

## Successful (blocking)
- C-1 [MED consent/PII, F3 residual] Detection false-negatives write PII without consent via default materialize:
  Payloads: phone/full_name columns, SSN/IBAN/phone values in neutral notes column, pii_classes=['none'] or default None, consent_ledger=None.
  Result: PASS-WRITE, persisted+retrievable. Examples (cycle4_output.txt §2, materialize probe): notes='555-123-4567', full_name='Alice Smith', notes='123-45-6789', notes='DE89370400440532013000' all materialize() OK.
  Evidence: governance/pii.py:90-131 detect_pii only email/user_/device_ value heuristics, column keywords 38-48 (no phone/name, SSN/IBAN column-only); guard_write 205-223 trusts ['none']/None when findings empty; feature_store/offline.py:56-91,88-91 defaults pii_classes=None,consent_ledger=None.
  Fix (from ADV-FULL-REPO remediation 3, not done): extend detect_pii value heuristics for phone/SSN/IBAN/name, or require explicit column inventory for ['none']/None, or require consent_ledger/explicit opt-out at materialize/set_features defaults. Add regression tests for above 4 payloads via OfflineFeatureStore.materialize + guard_write.

## Residuals (non-blocking, documented — Lead/anti-team to adjudicate)
- R1 Auth dev: bare create_app() no host/no env => 200 dev mode, WARNING app.py:318-319; loopback allowed by design (test_p2_default_auth.py:195-201). Manual `uvicorn --host 0.0.0.0` flag invisible at import unless FORECAST_BIND_HOST mirrors it — shipped paths (Dockerfile/Helm/CLI) cover it.
- R2 JWKS DNS TOCTOU: validate-then-connect re-resolves (auth.py:308-364,319-321 documented, egress allowlist recommended). No reproducible bypass without attacker DNS+race; hypothesis only.
- R3 Extraction at-limit: authenticated 600/min/process (1200 with Helm replicaCount=2, values.yaml:7), full-precision, shared key, single-process bucket (app.py:51-56 documented). Limiter/unauth controls work; at-limit authorized extraction is intended use.
- R4 load_model(path) valid envelope returns dict not unwrapped (model_loader.py:151-153 vs FileLoader _validated_unwrap 515). Invalid still blocked; production serving/scoring use FileModelLoader (app.py:719-722, scoring/nodes.py:368-371). API inconsistency, not a security bypass — 🟡.

## Gate
Verdict REJECTED (1 blocking C-1). Not accepted until anti-team-security APPROVED. Lead to adjudicate C-1 severity vs documented accepted-risk header pii.py:11-17; if reclassified non-blocking, remaining vectors are 0-blocking.

Scripts/output: /tmp/opencode/cycle4_adv/cycle4.py, cycle4_output.txt, pickle_extra*.py, attack_plan.md
