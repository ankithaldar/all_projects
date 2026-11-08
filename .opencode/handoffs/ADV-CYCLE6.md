# ADV-CYCLE6 — Cycle 6 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work + shipped
  C-2 fix in dirty tree: `governance/pii.py:138-151` separator-optional /
  case-insensitive phone/SSN/IBAN regexes, `tests/test_pii_separator.py`,
  `tests/test_pii_detection.py`; attack scripts only under
  /tmp/opencode/cycle6_adv/, no src/ modified by this review)
- Scope: forecast_ml_framework only; mandated C-2 payloads + key vectors
  (auth, consent, JWKS, pickle, extraction, OIDC sub)
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this
  subagent session); attacks executed directly against real code paths by
  adversarial tester (scripts /tmp/opencode/cycle6_adv/cycle6_c2.py,
  cycle6.py, cycle6_variants.py, cycle6_n1.py). No independent
  third-party verification claimed.
- Verdict: **APPROVED** — 0 blocking. All mandated C-2 payloads blocked
  on every write surface; all key vectors blocked in shipped forecast paths.

## Attack plan
C-2: 5551234567, int 5551234567, +15551234567, SSN spaced 123 45 6789,
IBAN lowercase (compact + spaced) — expect ValueError on detect/guard/
mask/materialize/set_features, no clear mask cell, no persisted write.
Key vectors: 1 auth default, 2 consent (declared/undeclared/revoked),
3 JWKS file/redirect, 4 OIDC no-sub, 5 pickle hostile, 6 extraction burst,
7 envelope invalid via file loader (bonus, as Cycle 4/5).
Expected blocked: RuntimeError/401/429/ValueError/ModelCorruptError,
no marker, no PII write for declared/detected PII.

## C-2 re-run (mandated payloads — ALL BLOCKED, cycle6_c2_output.txt)
Payloads (notes col, entity_id=e1): '5551234567', 5551234567 (int),
'+15551234567', '(555)1234567', '123 45 6789', '123456789',
'de89370400440532013000', 'de89 3704 0044 0532 0130 00'.
For EACH: detect_pii flags correct class (user_id/financial);
guard_write ['none'] AND None both BLOCKED; mask_pii ['none'] BLOCKED,
mask_pii [declared] hashed/REDACTED (clear=False, no clear cell);
OfflineFeatureStore.materialize DEFAULT and ['none'] both BLOCKED;
OnlineFeatureStore.set_features DEFAULT and ['none'] both BLOCKED.
Result: `C2 RESULT: ALL-BLOCKED` (cycle6_c2.py, 8 payloads x 7 surfaces).
Evidence: governance/pii.py:138-151 (separator-optional phone
`(?<!\d)(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b`,
SSN `\b\d{3}[- ]?\d{2}[- ]?\d{4}\b`, IBAN IGNORECASE compact+spaced);
guard_write pii.py:233-246; offline.py:56-94, online.py:166-201 pass
pii_classes through guard.
Clean controls: 'hello world','Paris','2024-01-01','555-123','DE89',
'123-45', numeric 42/2024/123.45 — guard ['none'] PASS-WRITE, no FP.
Close variants also blocked: float phone 5551234567.0, embedded
'call 5551234567 now', SSN int 123456789, mixed-case IBAN, dots/spaces/
paren phones (cycle6_variants_output.txt).

## Key vectors re-run (cycle6_output.txt — ALL BLOCKED)
- 1 Auth default: module import BIND=0.0.0.0 no key rc=1 (serving/app.py:278-311);
  Helm env rc=1; create_app 0.0.0.0/::/'0.0.0.0 '/'0'/env/short-key/partial-OIDC
  all RuntimeError (app.py:1175-1179); Helm+key POST no/wrong=401 good=200;
  Dockerfile validate_bind_host before uvicorn, FORECAST_REQUIRE_AUTH=1,
  BIND=0.0.0.0. Dev-mode bare create_app() 200 = R1 carried (documented,
  non-blocking).
- 2 Consent: declared email ledger=None/{} BLOCKED, undeclared email
  None/['none'] BLOCKED, revoked BLOCKED, granted control PASS
  (pii.py:233-280); Cycle 4/5 exact payloads (dashed phone, Title-case
  name, dashed SSN, compact upper IBAN, phone/full_name/ssn/iban cols)
  still BLOCKED.
- 3 JWKS: validate+fetcher blocked file/http/private/metadata/localhost/
  encoded/userinfo/::ffff/decimal/hex (auth.py:266-305,462-528);
  redirects to file/private/http/metadata blocked, seen=[initial only].
- 4 OIDC no-sub: real signed RS256 missing/empty/ws/int sub, +jti,
  +no-scope all BLOCKED ValueError missing sub (auth.py:733,834);
  with-sub control ALLOWED.
- 5 Pickle: os.system/eval REDUCE, pd.read_pickle chain, os.system func
  all BLOCKED ValueError, marker absent (core/artifact_store.py:790);
  legit LinearRegression ok; FileLoader hostile/.onnx ModelCorruptError
  (serving/model_loader.py:505-515).
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 637x200/13x429
  (token bucket 600/min app.py:99,481-483). Limiter works (R3 carried).
- 7 Envelope via file (bonus): bad label_set, missing/extra keys, None
  model, classifier-None, regressor+label all ModelCorruptError;
  valid controls unwrap correctly (model_loader.py:151-153 fix held).
- Shipped suites: 946 passed (pii-separator/detection, consent, jwks x2,
  oidc-sub, p2-default-auth, envelope, file-loader, governance);
  1162 passed (test_core+model_loader+app+governance+above).

## Successful (blocking)
- None. 0 blocking issues.

## Residuals / observations (non-blocking, documented)
- R1 Auth dev (carried): bare create_app() no host/no env => 200 dev mode
  (app.py WARNING); loopback by design. Shipped paths cover non-loopback.
- R2 JWKS DNS TOCTOU (carried, auth.py documented); no repro.
- R3 Extraction at-limit 600/min/process, 2 Helm replicas (carried).
- N1 Names/columns (carried from Cycle 5): 'alice smith'/'ALICE SMITH',
  UK '+44...' phone, generic customer_name/name cols with neutral values
  still PASS-WRITE — same family, FP tradeoffs; Lead adjudication (Cycle 5).
- N6 Dashed lowercase IBAN 'de89-3704-0044-0532-0130-00' not detected
  (cycle6_variants_output.txt) — IBAN canonical forms are compact/space-
  grouped (both blocked); dash-grouped is non-canonical. Disclosed for
  Lead adjudication; not a mandated payload, not counted blocking.

## Gate
Verdict APPROVED (0 blocking). Not accepted until anti-team-security
APPROVED. Scripts/output: /tmp/opencode/cycle6_adv/{cycle6_c2.py,
cycle6_c2_output.txt,cycle6.py,cycle6_output.txt,cycle6_variants.py,
cycle6_variants_output.txt,cycle6_n1.py,cycle6_n1_output.txt}
