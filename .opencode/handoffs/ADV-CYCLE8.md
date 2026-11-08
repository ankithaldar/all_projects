# ADV-CYCLE8 — Cycle 8 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work + shipped
  C-2 fix in dirty tree: `governance/pii.py:138-141` separator-optional
  phone regex `(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b`,
  `tests/test_pii_separator.py`, `tests/test_pii_detection.py`,
  `tests/test_pii_11digit.py`; attack scripts only under
  /tmp/opencode/cycle8_adv/, no src/ modified by this review)
- Scope: forecast_ml_framework only (D-SCOPE-FORECAST-1); mandated bare
  11-digit phone payloads + key vectors (auth, consent, JWKS, pickle,
  extraction, OIDC sub, envelope). F2 (woe/gbq) + F6 (supply-chain
  periphery) OUT of forecast verdict per D-SCOPE-FORECAST-1 — neither
  re-tested nor cleared here.
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this
  subagent session); attacks executed directly against real code paths by
  adversarial tester (scripts /tmp/opencode/cycle8_adv/cycle8_11digit.py,
  cycle8_probe.py, cycle8_listfix.py, cycle8.py, cycle8_n1.py,
  cycle8_10digit_sanity.py; copied from Cycle 7 scripts with cycle8 path
  rename + one stricter harness fix, see below). No independent
  third-party verification claimed.
- Verdict: **APPROVED** — 0 blocking. All mandated bare 11-digit payloads
  blocked on every write surface; all key vectors blocked in shipped
  forecast paths. No code change since Cycle 7 on the attacked surfaces
  (same HEAD, same dirty shipped paths; `pii.py` phone regex line-identical).

## Attack plan
Bare 11-digit: '15551234567', int 15551234567, int64/float/Decimal forms,
spaced/dashed/dotted/paren/embedded/tel forms — expect ValueError on
detect/guard/mask/materialize/set_features, no clear mask cell, no
persisted write. Plus 10-digit/+1/SSN/IBAN regression (Cycle 6 mandated).
Key vectors: 1 auth default, 2 consent (declared/undeclared/revoked),
3 JWKS file/redirect, 4 OIDC no-sub, 5 pickle hostile, 6 extraction burst,
7 envelope invalid via file loader. Expected blocked:
RuntimeError/401/429/ValueError/ModelCorruptError, no marker, no PII
write for declared/detected PII.

## Bare 11-digit re-run (mandated payloads — ALL BLOCKED, cycle8_11digit_output.txt)
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
Result: `11DIGIT RESULT: ALL-BLOCKED` (cycle8_11digit_output.txt).
Cycle 8 harness fix (stricter, no weakening): the Cycle 7 list-cell
`set_features` check returned a sentinel string ("list-in-set_features-
skipped-direct", Cycle 7 script artifact); Cycle 8 `cycle8_11digit.py`
executes the real `set_features('e1', {'notes':[15551234567]}, ...)`
path instead — now BLOCKED ValueError on DEFAULT and ['none'] directly
in `cycle8_11digit_output.txt` (bare11-in-list section), closing the
artifact without needing `cycle8_listfix.py` (which independently
confirms: BLOCKED on DEFAULT and ['none']).
Evidence: governance/pii.py:138-141; guard_write pii.py:206-246;
offline.py:92, online.py:201,250 pass pii_classes through guard.
Clean controls: 'hello world','Paris','2024-01-01','555-123','DE89',
'123-45', 42/2024/123.45, '1555123', '555-123-456' — guard ['none']
PASS-WRITE, no FP. Probes blocked: entity_id-as-phone, '+ 1 555...',
'++1555...', '001-555-123-4567', 'ID15551234567', '1555 1234567',
'1555123 4567', '1 5551234567', '+1-555-123-4567' (cycle8_probe_output.txt).
C-2 regression (cycle8_10digit_output.txt): `C2 RESULT: ALL-BLOCKED`
(8 payloads: 10-digit str/int, +1 continuous, paren-no-sep, SSN spaced/
continuous, IBAN lowercase compact/spaced — detect + guard/mask/
offline/online all blocked; mask returns hashed/REDACTED, clear=False).

## Key vectors re-run (cycle8_output.txt — ALL BLOCKED)
- 1 Auth default: module import BIND=0.0.0.0 no key rc=1 (serving/app.py:257-311);
  Helm env rc=1; create_app 0.0.0.0/::/'0.0.0.0 '/'0'/env/short-key/partial-OIDC
  all RuntimeError (app.py:1168-1179); Helm+key POST no/wrong=401 good=200;
  Dockerfile validate_bind_host before uvicorn, FORECAST_REQUIRE_AUTH=1,
  BIND=0.0.0.0. Dev-mode bare create_app() 200 = R1 carried (documented,
  non-blocking, loopback by design).
- 2 Consent: declared email ledger=None/{} BLOCKED, undeclared email
  None/['none'] BLOCKED, revoked BLOCKED, granted control PASS
  (pii.py:206-280); Cycle 4/5 exact payloads (dashed phone, Title-case
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
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 639x200/11x429
  (token bucket 600/min app.py:99,511-528). Limiter works (R3 carried).
- 7 Envelope via file (bonus): bad label_set, missing/extra keys, None
  model, classifier-None, regressor+label all ModelCorruptError;
  valid controls unwrap correctly (model_loader.py:151-153 fix held).
- Shipped suites: 1408 passed, 2 skipped (pii-11digit/separator/
  detection, consent, jwks x2, oidc-sub, p2-default-auth, envelope x2,
  file-loader, pickle-return, p2-oidc x2, governance + p1 suites);
  216 passed (test_core+test_model_loader+test_app).

## Successful (blocking)
- None. 0 blocking issues.

## Residuals / observations (non-blocking, documented + ADR-accepted)
- R1 Auth dev (carried): bare create_app() no host/no env => 200 dev mode
  (app.py WARNING); loopback by design. Shipped paths cover non-loopback.
  (D-ACCEPTED-RISK-1 / ADV-CYCLE7:94-95; recorded by reference in
  D-ACCEPTED-RISK-2 consequences.)
- R2 JWKS DNS TOCTOU (carried, auth.py documented); no repro.
  Accepted by D-ACCEPTED-RISK-1.
- R3 Extraction at-limit 600/min/process, 2 Helm replicas (carried).
  Accepted by D-ACCEPTED-RISK-1.
- N1 Names/columns (carried, re-probed cycle8_n1): 'alice smith'/
  'ALICE SMITH' (lower/upper), UK '+447911123456' still PASS-WRITE —
  same FP-tradeoff family; 'Alice Smith' BLOCKED (canonical Title-case).
  Accepted by D-ACCEPTED-RISK-2.
- N6 Dashed lowercase IBAN 'de89-3704-0044-0532-0130-00' not detected
  (re-confirmed Cycle 8 direct detect_pii probe; dashed upper also
  undetected) — canonical compact/space-grouped IBAN forms are blocked;
  dash-grouped is non-canonical. Accepted by D-ACCEPTED-RISK-2.
- N7 (carried, re-probed): phone glued to a letter suffix
  '15551234567x' / '15551234567ext123' PASS-WRITE — `\b` in pii.py:140
  fails when a word char follows the last digit. Not a mandated bare
  form. Accepted by D-ACCEPTED-RISK-2. ('15551234567 ext.123' with
  separator IS blocked.)
- N8 (carried, re-probed): fullwidth / arabic-indic unicode digits
  PASS-WRITE — phone regex is ASCII-only. Not a canonical stored form.
  Accepted by D-ACCEPTED-RISK-2.
- N9 (carried, re-probed): longer/non-US digit strings PASS-WRITE by
  shipped design: '25551234567' (11-digit non-1, explicitly unflagged
  in tests/test_pii_11digit.py), '155512345678' (12), '015551234567',
  '0015551234567', '0115551234567' — `(?<!\d)`/`\b` boundaries prevent
  substring FP on numeric IDs. Accepted by D-ACCEPTED-RISK-2.

## Gate
Verdict APPROVED (0 blocking). Not accepted until anti-team-security
APPROVED. Scripts/output: /tmp/opencode/cycle8_adv/{cycle8_11digit.py,
cycle8_11digit_output.txt,cycle8_probe.py,cycle8_probe_output.txt,
cycle8_listfix.py,cycle8_listfix_output.txt,cycle8.py,cycle8_output.txt,
cycle8_n1.py,cycle8_n1_output.txt,cycle8_10digit_sanity.py,
cycle8_10digit_output.txt}
