# ADV-CYCLE11 — Cycle 11 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 2fbf8e9 (`git rev-parse --short HEAD` = 2fbf8e9), clean
  tree apart from this handoff. Diff vs Cycle 10 baseline 8158681 is
  docs-only: `git diff 8158681..2fbf8e9 -- forecast_ml_framework/src/`
  = 0 lines (only `.opencode/*` + forecast README/runbook/infra README).
  Zero `.py` change since Cycle 10; this cycle re-runs the key exploits
  against that identical code.
- Scope: forecast_ml_framework only (D-SCOPE-FORECAST-1); mandated bare
  11-digit phone payloads + key vectors (auth, consent, JWKS, pickle,
  extraction, OIDC sub, envelope) + new-surface probes (erasure, secret
  scrub, temporal split, purged CV). F2 (woe/gbq) + F6 (supply-chain
  periphery) OUT of forecast verdict per D-SCOPE-FORECAST-1 — neither
  re-tested nor cleared here.
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this
  subagent session); attacks executed directly against real code paths by
  adversarial tester (scripts /tmp/opencode/cycle11_adv/cycle11_11digit.py,
  cycle11_probe.py, cycle11_listfix.py, cycle11.py, cycle11_n1.py,
  cycle11_10digit_sanity.py, cycle11_new.py, cycle11_new2.py, cycle11_new3.py;
  Cycle 10 scripts copied with cycle11 path rename, no src/ modified by this
  review). No independent third-party verification claimed.
- Verdict: **APPROVED** — 0 blocking. All mandated bare 11-digit payloads
  blocked on every write surface; all key vectors blocked in shipped
  forecast paths; new P1/P3 surfaces probed with no successful exploit.

## Attack plan
Bare 11-digit: '15551234567', int 15551234567, int64/float/Decimal forms,
spaced/dashed/dotted/paren/embedded/tel forms — expect ValueError on
detect/guard/mask/materialize/set_features, no clear mask cell, no
persisted write. Plus 10-digit/+1/SSN/IBAN regression (Cycle 6 mandated).
Key vectors: 1 auth default, 2 consent (declared/undeclared/revoked),
3 JWKS file/redirect, 4 OIDC no-sub, 5 pickle hostile, 6 extraction burst,
7 envelope invalid via file loader. New surfaces: erasure (real store
delete + marker + no clear leak in return/audit; blank subject/actor,
unknown store, bad artifact shape rejected), secret scrub (api-key /
bearer / AWS sentinels redacted), temporal split (shuffled input,
train-max <= test-min; leaky presplit raises), purged CV (fold leakage
check). Expected blocked: RuntimeError/401/429/ValueError/ModelCorruptError,
no marker, no PII write for declared/detected PII.

## Bare 11-digit re-run (mandated payloads — ALL BLOCKED, cycle11_11digit_output.txt)
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
['none'] both BLOCKED. Result: `11DIGIT RESULT: ALL-BLOCKED`
(cycle11_11digit_output.txt, 190 BLOCKED lines — identical to Cycle 10).
List-cell real `set_features({'notes':[15551234567]})` BLOCKED on DEFAULT and
['none'] (cycle11_listfix_output.txt + bare11-in-list section of cycle11_11digit).
Evidence: governance/pii.py:139-141 phone regex
`(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?...`; guard_write pii.py:206-246;
offline.py:92, online.py:201,250 pass pii_classes through guard.
Clean controls: 'hello world','Paris','2024-01-01','555-123','DE89',
'123-45', 42/2024/123.45, '1555123', '555-123-456' — guard ['none']
PASS-WRITE, no FP. Probes blocked: entity_id-as-phone, '+ 1 555...',
'++1555...', '001-555-123-4567', 'ID15551234567', '1555 1234567',
'1555123 4567', '1 5551234567', '+1-555-123-4567'
(cycle11_probe_output.txt). C-2 regression (cycle11_10digit_output.txt):
`C2 RESULT: ALL-BLOCKED` (10-digit str/int, +1 continuous, paren-no-sep,
SSN spaced/continuous, IBAN lowercase compact/spaced — detect + guard/
mask/offline/online all blocked; mask returns hashed/REDACTED, clear=False).

## Key vectors re-run (cycle11_output.txt — ALL BLOCKED)
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
  with-sub control ALLOWED (principal_hash 99e13582a59bc73ad5bfef3176266b33b2c7e43ff9709bdbddd6af8198c290bd, identical to Cycle 9/10).
- 5 Pickle: os.system/eval REDUCE, pd.read_pickle chain, os.system func
  all BLOCKED ValueError, marker absent (core/artifact_store.py:790);
  legit LinearRegression ok; FileLoader hostile/.onnx ModelCorruptError
  (serving/model_loader.py:505-515).
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 643x200/7x429
  (token bucket 600/min app.py:99,511-528; exact 200/429 split varies
  with bucket timing across cycles — 636/14 in Cycle 10, 643/7 here —
  limiter works, R3 carried).
- 7 Envelope via file (bonus): bad label_set, missing/extra keys, None
  model, classifier-None, regressor+label all ModelCorruptError;
  valid controls unwrap correctly (model_loader.py:151-153 fix held).
- Shipped suites: full `uv run --frozen pytest tests/ -q` = **3513 passed,
  4 skipped, 0 failed** (93.14s, /tmp/opencode/cycle11_adv/cycle11_pytest_q.log)
  — identical counts to Cycle 10.

## New-surface probes (re-run on unchanged code)
- Erasure (cycle11_new.py): real OfflineFeatureStore row written then
  `erase_subject` — removed label `offline:<hash>`, re-read absent,
  fitted-registry `__meta__.erasure` + best-model `erasure` markers
  `action='retrain_required'`, clear subject AND actor absent from
  return/log/artifacts JSON. Blank subject, None subject, blank actor,
  unknown store, non-dict best_model artifact all ValueError. (erasure.py:159-223)
- Secret scrub (cycle11_new2/new3): scrub_text redacts api-key sentinel,
  `Authorization: Bearer <sentinel>`, and the AWS key AKIA... in
  `scrub_text`/`scrub_details` (new3 `PASS aws-secret-redacted`).
  NOTE: scrub is a *secret* scrubber, not a PII scrubber (scrub.py:3-10) —
  PII values passing through scrub_text unchanged is by design; PII is
  blocked at guard/mask (above), not here. First probe's "PII in
  scrub_text" FAILs were a wrong-dimension test, withdrawn, not a finding.
- Temporal split (cycle11_new2/new3): shuffled 100-row frame ->
  temporal_split 80/20, train max 2024-03-17 <= test min 2024-03-21,
  meta leakage_checked=True. Properly leaky presplit raises
  `ValueError: temporal leakage` (new3); clean presplit allowed.
  (First probe's "leaky presplit" FAIL used a non-leaky head — withdrawn.)
- Purged CV: 3 folds, `fold_leakage_check` passes, every fold
  train-max <= test-min. (data_split.py:610, purged_cv.py:22,126)

## Successful (blocking)
- None. 0 blocking issues.

## Residuals / observations (non-blocking, documented + ADR-accepted)
- R1 Auth dev (carried): bare create_app() no host/no env => 200 dev mode
  (app.py WARNING); loopback by design. Shipped paths cover non-loopback.
  (D-ACCEPTED-RISK-1; recorded by reference in D-ACCEPTED-RISK-2.)
- R2 JWKS DNS TOCTOU (carried, auth.py documented); no repro.
  Accepted by D-ACCEPTED-RISK-1.
- R3 Extraction at-limit 600/min/process, 2 Helm replicas (carried).
  Accepted by D-ACCEPTED-RISK-1.
- N1 Names/columns (carried, re-probed cycle11_n1): 'alice smith'/
  'ALICE SMITH' (lower/upper), UK '+447911123456' still PASS-WRITE —
  same FP-tradeoff family; 'Alice Smith' BLOCKED (canonical Title-case).
  Accepted by D-ACCEPTED-RISK-2.
- N6 Dashed lowercase IBAN 'de89-3704-0044-0532-0130-00' not detected
  (re-probed Cycle 11 direct detect_pii: dashed lower/upper [] vs compact
  BLOCKED financial) — canonical compact/space-grouped IBAN forms are blocked;
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
APPROVED. Scripts/output: /tmp/opencode/cycle11_adv/{cycle11_11digit.py,
cycle11_11digit_output.txt,cycle11_probe.py,cycle11_probe_output.txt,
cycle11_listfix.py,cycle11_listfix_output.txt,cycle11.py,cycle11_output.txt,
cycle11_n1.py,cycle11_n1_output.txt,cycle11_10digit_sanity.py,
cycle11_10digit_output.txt,cycle11_new.py,cycle11_new2.py,cycle11_new3.py,
cycle11_pytest_q.log}
