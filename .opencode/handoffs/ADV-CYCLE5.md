# ADV-CYCLE5 — Cycle 5 adversarial approval (forecast scope, 0-blocking bar)

- Baseline: HEAD 303a07c, dirty tree (pre-existing P1/P2/P3 work; attack
  scripts only under /tmp/opencode/cycle5_adv/, no src/ modified by this review)
- Scope: forecast_ml_framework only
- DEGRADED MODE: tester could NOT be tasked (nested subagent depth in this
  subagent session); attacks executed directly against real code paths by
  adversarial tester (scripts /tmp/opencode/cycle5_adv/cycle5.py,
  cycle5_extra.py, cycle5_extra2.py, cycle5_confirm.py). No independent
  third-party verification claimed.
- Verdict: **REJECTED** — 1 blocking (C-2 consent/PII detection residual,
  same class as Cycle 4 C-1). Vectors 1,3,4,5,6,7 blocked in shipped
  forecast paths; vector 2 exact Cycle 4 payloads now blocked, variants not.

## Attack plan
1 auth bypass default | 2 consent bypass | 3 JWKS file/redirect |
4 OIDC no-sub | 5 pickle hostile | 6 extraction burst |
7 envelope invalid via file loader. Expected blocked:
RuntimeError/401/429/ValueError/ModelCorruptError, no marker, no PII write
for declared/detected PII.

## Blocked (proven, executed)
- 1 Auth default: module import BIND=0.0.0.0 no key rc=1 (serving/app.py:257-284);
  Helm env rc=1; create_app 0.0.0.0/::/::/'0.0.0.0 '/env/short-key/partial-OIDC
  all RuntimeError (app.py:1175-1179); Helm+key POST no/wrong=401 good=200;
  host variants 0.0.0.0:8000/::ffff:0.0.0.0/empty/space/example.com/0 all
  refused; header variants empty/comma-join/trailing-space=401. Dev-mode
  bare create_app() 200 = R1 carried (documented, non-blocking).
- 3 JWKS: validate+fetcher blocked file/http/private/metadata/localhost/
  encoded/userinfo/::ffff/0.0.0.0/::1/CGNAT/multicast/octal/decimal (auth.py:266-305,462-528);
  redirects to file/private/http/metadata blocked, seen=[initial only];
  2-hop chain public->file blocked after hop 2 (auth.py:485-506).
- 4 OIDC no-sub: real signed RS256 missing/empty/ws/int/null/list/bool/dict
  sub, +jti, +no-scope all BLOCKED ValueError missing sub (auth.py:731-733,
  832-834 incl. lax-verifier re-check); with-sub control ALLOWED.
- 5 Pickle: os.system/eval REDUCE, pd.read_pickle chain, FunctionTransformer,
  Booster REDUCE/NEWOBJ_EX/NEWOBJ model_file, manual GLOBAL os.system,
  oversized cap100 all BLOCKED ValueError, marker absent
  (core/artifact_store.py:764-837); legit LinearRegression/XGBRegressor ok;
  FileLoader hostile/.onnx ModelCorruptError (serving/model_loader.py:505-515).
- 6 Extraction: unauth burst 20 => 0x200; auth burst 650 => 643x200/7x429
  (token bucket 600/min app.py:99,511-540). Limiter works (R3 carried).
- 7 Envelope via file: bad label_set, missing/extra keys, None model,
  classifier-None, regressor+label, erasure-bad, onnx-bad-sha,
  registry_ref-extra, label dup/swap all ModelCorruptError
  (model_loader.py:63-107,430-515; envelope.py:146-252). Cycle 4 R4 FIXED:
  load_model(path) valid now unwraps to LogisticRegression, not dict
  (model_loader.py:151-153).
- Shipped suites: 319 passed (pii/consent/jwks/oidc/envelope/file-loader/auth);
  1038 passed (test_core+test_model_loader+test_app+test_governance).

## Successful (blocking)
- C-2 [MED consent/PII, C-1 same class] Structured-ID detection still
  false-negatives on common stored forms; PII persisted+retrievable without
  consent via DEFAULT paths. Exact Cycle 4 payloads (dashed phone, Title-case
  name, dashed SSN, compact upper IBAN, phone/full_name/ssn/iban columns)
  are now BLOCKED (pii.py:38-51,134-147) — fix verified for those shapes.
  Residual bypasses (cycle5_extra_output.txt, cycle5_confirm_output.txt):
  - notes='5551234567' (10-digit, no separator), int 5551234567,
    '+15551234567': guard ['none']/None PASS-WRITE; OfflineFeatureStore.
    materialize() DEFAULT (no pii_classes) PASS-WRITE + retrievable;
    OnlineFeatureStore.set_features ['none'] PASS-WRITE; mask_pii
    ['user_id'] returns CLEAR cell (no finding => no mask).
  - Same root cause: SSN '123 45 6789' / '123456789', IBAN lowercase
    'de89370400440532013000' (compact+spaced): materialize DEFAULT PASS-WRITE.
  - Evidence: governance/pii.py:136-145 regexes require separators
    (phone), dashes (SSN), uppercase (IBAN); guard_write trusts ['none']/None
    when findings empty (pii.py:227-254); offline.py:56-94 defaults
    pii_classes=None, consent_ledger=None; online.py:167-201 same.
  - Severity rationale: 10-digit no-separator is the canonical stored form
    of a phone (not an "unusual shape" under the pii.py:11-17 accepted-risk
    header) — same reasoning Cycle 4 applied to the dashed form. Phone
    alone is sufficient for blocking; SSN/IBAN variants are supporting.
  - Fix (Lead/worker): make phone/SSN/IBAN regexes separator-optional and
    case-insensitive (mind FP on numeric features: scope 10-digit match to
    string/free-text columns or require explicit column inventory for
    ['none']/None, per ADV-FULL-REPO remediation 3 still open). Add
    regression tests: above payloads via materialize DEFAULT, guard_write
    ['none']/None, mask_pii ['user_id'].

## Residuals / observations (non-blocking, documented)
- R1 Auth dev (carried): bare create_app() no host/no env => 200 dev mode
  (app.py:318-319); loopback by design. Shipped paths cover non-loopback.
- R2 JWKS DNS TOCTOU (carried, auth.py:308-364 documented); no repro.
- R3 Extraction at-limit 600/min/process, 2 Helm replicas (carried).
- N1 Names: 'alice smith'/'ALICE SMITH'/'Alice J Smith', generic *_name/name/
  customer_name columns with neutral values, UK '+44...' phone PASS-WRITE —
  same family as C-2 but free-text/case/column-keyword gaps with FP tradeoffs;
  Lead to adjudicate whether included in C-2 fix. Not needed for verdict.
- N2 Booster direct pickle loads (PASS-RETURN Booster, same as Cycle 4);
  hostile Booster construction (REDUCE/NEWOBJ_EX model_file) blocked, no
  file-read/code-exec proven — not counted blocking.
- N3 JWKS validate allows trailing-dot host, public-IP:port, case variants —
  DNS/public-IP check still applies at fetch; operator-configured URL only.
- N4 OIDC padded sub ('  user-1  ') ALLOWED (hash differs) — requires a
  valid IdP-signed token; not attacker-controllable.
- N5 Auth over-strict: '127.0.0.1:8000'/'[::1]'/'localhost:8000' refused
  (fail-closed, availability only).

## Gate
Verdict REJECTED (1 blocking C-2). Not accepted until anti-team-security
APPROVED. Lead ruling needed: either fix C-2 (above) or explicitly
reclassify the residual as accepted risk in pii.py header + ADR — Cycle 4
left the same adjudication open for C-1 and it was fixed for exact payloads
only. Scripts/output: /tmp/opencode/cycle5_adv/{cycle5.py,cycle5_output.txt,
cycle5_extra.py,cycle5_extra_output.txt,cycle5_extra2.py,cycle5_extra2_output.txt,
cycle5_confirm.py,cycle5_confirm_output.txt}
