# ADV-CYCLE7 — Cycle 7 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work + shipped
  C-2 fix in dirty tree: `governance/pii.py:138-141` separator-optional /
  case-insensitive phone regex `(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b`,
  `tests/test_pii_separator.py`, `tests/test_pii_detection.py`,
  `tests/test_pii_11digit.py`; attack scripts only under
  /tmp/opencode/cycle7_adv/, no src/ modified by this review)
- Scope: forecast_ml_framework only; mandated bare 11-digit phone payloads
  + key vectors (auth, consent, JWKS, pickle, extraction, OIDC sub,
  envelope)
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this
  subagent session); attacks executed directly against real code paths by
  adversarial tester (scripts /tmp/opencode/cycle7_adv/cycle7_11digit.py,
  cycle7_probe.py, cycle7_listfix.py, cycle7.py, cycle7_n1.py). No
  independent third-party verification claimed.
- Verdict: **APPROVED** — 0 blocking. All mandated bare 11-digit payloads
  blocked on every write surface; all key vectors blocked in shipped
  forecast paths.

## Attack plan
Bare 11-digit: '15551234567', int 15551234567, int64/float/Decimal forms,
spaced/dashed/dotted/paren/embedded/tel forms — expect ValueError on
detect/guard/mask/materialize/set_features, no clear mask cell, no
persisted write. Plus 10-digit/+1 regression (Cycle 6 mandated). Key
vectors: 1 auth default, 2 consent (declared/undeclared/revoked),
3 JWKS file/redirect, 4 OIDC no-sub, 5 pickle hostile, 6 extraction burst,
7 envelope invalid via file loader. Expected blocked:
RuntimeError/401/429/ValueError/ModelCorruptError, no marker, no PII
write for declared/detected PII.

## Bare 11-digit re-run (mandated payloads — ALL BLOCKED, cycle7_11digit_output.txt)
Payloads (notes col, entity_id=e1): '15551234567', 15551234567 (int),
np.int64, 15551234567.0 (float), Decimal, '1 555 123 4567',
'1-555-123-4567', '1.555.123.4567', '1 (555) 123-4567', '1(555)1234567',
'+1 (555) 123-4567', '1-5551234567', 'call 15551234567 now',
'tel:15551234567', '15551234567.', ' 15551234567', [15551234567],
b'15551234567'; regression: '5551234567', 5551234567, '+15551234567'.
For EACH: detect_pii flags user_id; guard_write ['none'] AND None both
BLOCKED; mask_pii ['none'] BLOCKED, mask_pii [user_id] hashed
(clear=False, no clear cell); OfflineFeatureStore.materialize DEFAULT
and ['none'] both BLOCKED; OnlineFeatureStore.set_features DEFAULT and
['none'] both BLOCKED; OnlineFeatureStore.materialize DEFAULT and
['none'] both BLOCKED (extra surface beyond Cycle 6 script).
Result: all blocked on all real surfaces. The 2 `FAIL-PASS-WRITE` lines
for the list payload in cycle7_11digit_output.txt are a script artifact
(sentinel return value in the harness, not a write); cycle7_listfix.py
proves the real `set_features({'notes':[15551234567]})` is BLOCKED
ValueError undeclared PII on DEFAULT and ['none'] (cycle7_listfix_output.txt).
Evidence: governance/pii.py:138-141; guard_write pii.py:233-246;
offline.py:91-94, online.py:201-203,249-252 pass pii_classes through guard.
Clean controls: 'hello world','Paris','2024-01-01','555-123','DE89',
'123-45', 42/2024/123.45, '1555123', '555-123-456' — guard ['none']
PASS-WRITE, no FP. Probes blocked: entity_id-as-phone, '+ 1 555...',
'++1555...', '001-555-123-4567', 'ID15551234567', '1555 1234567',
'1555123 4567', '1 5551234567' (cycle7_probe_output.txt).

## Key vectors re-run (cycle7_output.txt — ALL BLOCKED)
- 1 Auth default: module import BIND=0.0.0.0 no key rc=1 (serving/app.py:257-311);
  Helm env rc=1; create_app 0.0.0.0/::/'0.0.0.0 '/'0'/env/short-key/partial-OIDC
  all RuntimeError (app.py:1168-1179); Helm+key POST no/wrong=401 good=200;
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
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 641x200/9x429
  (token bucket 600/min app.py:99,511-528). Limiter works (R3 carried).
- 7 Envelope via file (bonus): bad label_set, missing/extra keys, None
  model, classifier-None, regressor+label all ModelCorruptError;
  valid controls unwrap correctly (model_loader.py:151-153 fix held).
- Shipped suites: 1161 passed (pii-11digit/separator/detection, consent,
  jwks x2, oidc-sub, p2-default-auth, envelope x2, file-loader,
  pickle-return, p2-oidc x2, governance); 216 passed
  (test_core+test_model_loader+test_app).

## Successful (blocking)
- None. 0 blocking issues.

## Residuals / observations (non-blocking, documented)
- R1 Auth dev (carried): bare create_app() no host/no env => 200 dev mode
  (app.py WARNING); loopback by design. Shipped paths cover non-loopback.
- R2 JWKS DNS TOCTOU (carried, auth.py documented); no repro.
- R3 Extraction at-limit 600/min/process, 2 Helm replicas (carried).
- N1 Names/columns (carried from Cycle 5/6, re-probed cycle7_n1):
  'alice smith'/'ALICE SMITH', UK '+44...' phone still PASS-WRITE —
  same family, FP tradeoffs; Lead adjudication (Cycle 5).
- N6 Dashed lowercase IBAN 'de89-3704-0044-0532-0130-00' not detected
  (carried from Cycle 6) — IBAN canonical forms are compact/space-
  grouped (both blocked); dash-grouped is non-canonical.
- N7 (new, disclosed): phone glued to a letter suffix '15551234567x' /
  '15551234567ext123' PASS-WRITE — `\b` in pii.py:140 fails when a word
  char follows the last digit. Not a mandated bare form (canonical
  embedded forms use separators and are blocked); if Lead wants
  letter-glued suffixes in scope, fix is `(?!\d)`-style boundary.
- N8 (new, disclosed): fullwidth / arabic-indic unicode digits
  '１５５５...' PASS-WRITE — phone regex is ASCII `1` + ASCII digit
  groups. Not a canonical stored form.
- N9 (new, disclosed): longer/non-US digit strings PASS-WRITE by shipped
  design: '25551234567' (11-digit non-1, explicitly unflagged in
  tests/test_pii_11digit.py:47-52), '155512345678' (12), '015551234567',
  '0115551234567' — `(?<!\d)`/`\b` boundaries prevent substring FP on
  numeric IDs. Same FP-tradeoff family as N1/N6.

## Gate
Verdict APPROVED (0 blocking). Not accepted until anti-team-security
APPROVED. Scripts/output: /tmp/opencode/cycle7_adv/{cycle7_11digit.py,
cycle7_11digit_output.txt,cycle7_probe.py,cycle7_probe_output.txt,
cycle7_listfix.py,cycle7_listfix_output.txt,cycle7.py,cycle7_output.txt,
cycle7_n1.py,cycle7_n1_output.txt}
