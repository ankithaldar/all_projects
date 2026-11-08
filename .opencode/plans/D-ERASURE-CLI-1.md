# ADR D-ERASURE-CLI-1 — Operator erasure CLI channel (secure env only)

> **Status:** frozen (anti-team `APPROVED` on FIX; 2026-11-08 — authoritative
> freeze record: `.opencode/context/decisions.md` D-ERASURE-CLI-1)
> **Date:** 2026-11-08
> **Owner:** `solution-architect` (contract) · gate: `anti-team-arch` · review: `security`
> **Owns:** `.opencode/plans/D-ERASURE-CLI-1.md` only (this file).
> **Contract of record:** this ADR for the CLI channel; frozen semantics
> live elsewhere and are not changed here — `erase_subject`
> (`.opencode/plans/contracts.md` §5.2(d)), sole `subject_hash` leaf
> (`governance/scrub.py`), D-SECRETS-PII-GDPR-1 (§5.2, secrets/PII/GDPR),
> D-MVP-READINESS-1 (readiness ≠ promotion, P11 only).
> **Blocker it closes:** `governance/retention_job.py` shipped while
> this ADR was unfrozen. P2 plan (`.opencode/plans/P2.md` P2-S1)
> requires this ADR (`solution-architect` + anti-team gate) before
> GREEN. P2 arch + security are `APPROVED`; reviewer `FAIL` is only
> on this unfrozen ADR. This ADR freezes the channel; it does not
> re-decide any frozen `erase_subject` semantics.

## Assumptions (stated up front)

- `Assumption:` the secure channel is a process environment variable
  injected from a cloud Secret mount / secret-manager reference
  (K8s Secret via Helm `secrets:` names/keys only, never a value in
  repo), because P2-S1 already wires secrets env-only out-of-band.
- `Assumption:` the operator actor is untrusted identity input, so
  it is hashed before audit (as the frozen enforcer already does).
- `Assumption:` feature-store row deletion is idempotent (missing
  entity is not an error), so an erasure retry is safe at the store
  layer even though each attempt appends an audit entry.

## Context and constraints

- Frozen `erase_subject(subject_id, *, stores, artifacts, actor, ...)`
  (`governance/erasure.py`) requires the **clear** subject id to
  locate and delete rows (`delete_entity` / `delete`) and to verify
  by re-read. A hash-only caller cannot complete that contract.
- The P2 CLI (`governance/retention_job.py`) is a new thin shell
  around two frozen surfaces: `enforce_retention` (sweep) and
  `erase_subject` (erasure). It must not introduce a second hash
  helper, a second marker shape, or a second persistence path.
- Clear ids and actor identities are sensitive: argv is visible in
  shell history, process lists, and CI logs; stdout/stderr/logs are
  retained. Env-injected values are the P2 secrets posture
  (D-SECRETS-PII-GDPR-1 §5.2(e): env only, no rotation code, no
  residency pin).
- Retention params must have exactly one source, or the CronJob and
  an operator could sweep with different windows. The frozen P2-S1
  source is `forecast_ml_framework/conf/base/platform.yml` under
  `governance_params.retention_days`, loaded via
  `--governance-params` (default that file). The earlier
  catalog-metadata claim is removed. The
  `conf/base/parameters/platform.yml` `governance_params` block
  (config-schema surface) is **not** read by the retention CLI.

## Alternatives considered

### A. Clear id from env only; CLI delegates to frozen `erase_subject` (chosen)

- Cost: one env var (`FORECAST_ERASE_SUBJECT_ID`), no new surface.
- Failure mode: env not set → exit 2 with no mutation (fail-closed).
- Verdict: **Chosen** — matches the env-only posture, keeps the
  clear id out of argv/logs, reuses the frozen contract unchanged.

### B. Clear id as an argv value (e.g. `--erase-subject <id>`)

- Cost: trivial to parse.
- Failure mode: the clear id leaks via shell history, `ps`, CI
  logs, and job manifests — exactly the channels the audit/hash
  rules exist to keep clean.
- Verdict: Rejected. Argv is not a secure channel.

### C. Hash-only CLI (`--erase-subject-hash <hash>`)

- Cost: no clear id in the operator channel.
- Failure mode: frozen `erase_subject` needs the clear id to find
  and delete rows and to verify by re-read; a hash cannot address
  `delete_entity(subject_id)` / `delete(subject_id)`. Shipping this
  flag would promise an erasure it cannot perform.
- Verdict: Rejected and **removed**: no `--erase-subject-hash`
  flag exists (P2-S1 acceptance grep guard asserts absence).

### D. Clear id via stdin or a file

- Cost: extra lifecycle (file permissions, cleanup, mount wiring)
  with no gain over a Secret-mounted env var.
- Verdict: Not authorized by this ADR. Any stdin/file channel
  requires a new ADR; env-only is the frozen channel.
- **Supersession recorded (LOW fix):** `.opencode/plans/P2.md`
  P2-S1 pseudocode previously described the secure channel as
  "env `FORECAST_ERASE_SUBJECT_ID` / stdin". This ADR
  **narrows and supersedes that stdin wording**: the frozen
  channel is **env only**; stdin is **not** an authorized
  channel under D-ERASURE-CLI-1 and must not be implemented
  on the strength of the P2.md text. Any stdin channel
  requires a new ADR + anti-team gate.

## Decision

1. **Secure channel — env only.** The clear subject id is read only
   from `FORECAST_ERASE_SUBJECT_ID`. It is never taken from argv,
   never printed, logged, audited, or returned. There is no
   `--erase-subject-hash` flag; `--erase` / `--erase-subject` are
   valueless switches (`action='store_true'`).
2. **Library delegation, frozen contract unchanged.**
   `run_erasure(*, subject_id, stores, artifacts, actor, log=None,
   at=None, now=None, clock=None)` validates non-blank subject and
   actor, then delegates to frozen `erase_subject`. Blank input
   raises `ValueError` before any store call. CLI `main()` maps
   validation failures to exit 2 with no store mutation.
3. **Markers — exact shape, single slot.** Per artifact, exactly
   `{subject_hash, requested_at, action}` with
   `action == 'retrain_required'`:
   - fitted registry: `__meta__.erasure`;
   - best-model envelope: top-level `erasure`.
   `subject_hash == governance.scrub.subject_hash(clear_id)` (the
   sole sha256 leaf; no second helper). A second erasure overwrites
   the single-slot marker; both hashes survive in the audit trail.
4. **Persistence — caller re-saves via existing writers;
   CLI is rows-only (HIGH fix).**
   `erase_subject` mutates the caller's in-memory artifact dicts in
   place only. The caller re-saves through the existing writers
   (registry JSON dataset; best-model restricted-pickle dataset,
   per contracts.md §5.2(d)). No new save path, no dataset I/O in
   the erasure module.
   **Scope of the CLI is rows-only, explicitly:**
   `retention_job.py:356-357` calls
   `run_erasure(subject_id=..., stores=stores, artifacts={}, actor=...)`.
   With `artifacts={}` the CLI **writes no markers** and performs
   **no artifact retrain-flagging**. A CLI `--erase` run therefore
   deletes feature rows only; it is **not**, on its own, a complete
   GDPR erasure of artifacts/models trained on the subject, and
   an operator must not treat a CLI exit-0 line alone as proof of
   complete erasure.
   **Operator duty (mandatory):** CLI erasure deletes feature
   rows; artifact retrain-flagging requires a library caller
   supplying artifact dicts + re-save via existing writers —
   mandatory operator step, see runbook (`docs/runbook-p2.md`,
   Retention/erasure ops, P2-R1). The operator must run that
   library-caller step (supply the registry/best-model artifact
   dicts, let `erase_subject` write the `retrain_required`
   markers, re-save via the existing writers) as part of the
   same erasure case, or record the artifact step as outstanding.
   **Follow-on owner:** Lead (dispatch) for the library-caller
   runbook step and for any future CLI artifact-flagging
   extension; design owner `solution-architect` — a CLI that
   loads/re-saves artifacts would need a new ADR + anti-team
   gate and is **not** authorized by this ADR. Until that
   follow-on, the CLI remains rows-only by design.
   Only a caller that supplies artifact dicts produces
   markers and must re-save them.
5. **Audit and output — hash only.** Audit entries carry the
   subject hash and hashed actor (`subject_hash(actor)`), hashed
   removal labels, and the legal-hold deferral; never clear values.
   CLI stdout carries hashes/counts only
   (`erased subject_hash=<64-hex> removed=<n> actor_hash=<64-hex>`).
   Store errors append a partial audit entry (`partial: True`,
   hash only) before propagating — a partial erasure is never
   claimed done. A not-found subject returns empty `removed` plus
   an audit entry, not an error.
6. **Retention params — sole source.** `conf/base/platform.yml`
   `governance_params.retention_days`, loaded only via
   `--governance-params` (default that file). Missing/unreadable
   file, missing key, non-int (bool/float rejected), or negative
   days → `ValueError` / exit 2 with no mutation. Never catalog
   metadata. The K8s CronJob runs `--retention` only (daily
   `0 3 * * *`); it never runs `--erase` and never injects the
   erasure env var.
7. **CLI guardrails.** `--erase` with missing/blank env → exit 2,
   no mutation. `--dry-run` combined with `--erase` → exit 2, no
   mutation (`--dry-run` plans with the pure checker only).
   Operator actor comes from `FORECAST_ACTOR`, else
   `FORECAST_OPERATOR`, else `forecast-ml-retention-job` — always
   hashed before audit/output.
8. **Store-targeting preconditions (MEDIUM fix).** A complete
   erasure requires the same artifact/Redis env stores were
   written with: the offline root (`FORECAST_ARTIFACT_BASE_PATH`
   / `artifact_base_path`, `build_stores`) and the online Redis
   URL (`FORECAST_REDIS_URL`) must point at the stores that hold
   the subject's rows. The operator (and the runbook) must set
   those env values to the production store locations before
   running `--erase`; running against a default/empty store
   (`./artifacts`, unset Redis) targets the wrong store.
   **`removed=0` on an empty or wrong store is not proof of
   absence** — it only means nothing was found in the store
   actually targeted. Absence may be claimed only after a
   re-read against the correct store(s) (the frozen
   `erase_subject` re-read verify), with the store location
   recorded (hash counts only, no clear id) in the runbook
   entry for the case.

## Module boundaries (dependency arrows one way)

| Module | Owner | Public interface | Must never |
|---|---|---|---|
| `governance/scrub.py` | `solution-architect` + `security` (frozen leaf) | `subject_hash(subject_id) -> 64-hex` | import telemetry/audit; be reimplemented anywhere |
| `governance/erasure.py` | frozen P1 (read-only in P2) | `erase_subject(...) -> report` | do dataset I/O; log clear ids |
| `governance/retention_job.py` | P2-S1 | `load_governance_params`, `build_stores`, `run_retention_sweep`, `run_erasure`, `main` | parse a subject from argv; print/log clear ids; invent a hash helper; persist artifacts; claim complete GDPR erasure from the rows-only CLI alone |
| `conf/base/platform.yml` | P2-S1 | `governance_params.retention_days: int` | hold secret values; duplicate the params source |

## Sequence (erasure happy path) and failure modes

Sequence: operator injects `FORECAST_ERASE_SUBJECT_ID` via Secret
mount, sets the artifact/Redis env to the stores the rows were
written with (Decision 8) → `main()` parses valueless `--erase` →
reads env, loads params → `build_stores()` → `run_erasure()`
(`artifacts={}`, rows-only) → frozen `erase_subject` (delete +
re-read verify per targeted store, append hashed audit; no
markers, because no artifact dicts were supplied) → CLI prints
hash/count line, exit 0 → **mandatory operator step:** library
caller supplies artifact dicts + re-save via existing writers
to write `retrain_required` markers, per the runbook
(`docs/runbook-p2.md` Retention/erasure ops). CLI exit 0 alone
is not complete GDPR erasure (Decision 4).

| Failure | Behavior | Evidence |
|---|---|---|
| Env missing/blank with `--erase` | Exit 2, no mutation | `test_main_erase_requires_env_subject_id` |
| `--erase` + `--dry-run` | Exit 2, no mutation | `test_main_erase_dry_run_combo_rejected` |
| Bad/missing params file | Exit 2, no store mutation | `test_main_retention_dry_run_and_bad_params` |
| Blank subject/actor (library) | `ValueError` before mutation | `test_run_erasure_rejects_blank_inputs` |
| Unknown store type | `ValueError`; sweep aborts with partial audit | `test_run_retention_sweep_unknown_store_aborts` |
| Store error mid-erasure | Partial audit (`partial: True`), error propagates | frozen `erase_subject` contract |
| Subject not found | `removed=[]` + audit entry, not an error | frozen `erase_subject` contract |
| CLI `--erase` completes | Rows deleted in targeted stores only; **no markers** (`artifacts={}`); not complete GDPR erasure until the library-caller retrain-flagging step (Decision 4) is done | `retention_job.py:356-357`; runbook |
| Wrong/empty store targeted | `removed=0` is **not** proof of absence; re-target correct artifact/Redis env and re-verify by re-read | Decision 8 |

Scaling/security notes: the erasure path is operator-invoked, not
a serving path — no throughput claim, no new scaling lever. Trust
boundary is the operator's secret mount: anyone who can set the
env var can erase that subject, so the Secret's readers are the
authorization boundary (no additional ACL is frozen here). The
env var must be unset after use by the operator runbook; an
injected value persisting in a long-lived env would re-erase on
every retry (idempotent at the store layer, but each attempt adds
an audit entry and overwrites the marker).

## Trade-off record

- **Gain:** the clear id never appears in argv, logs, audit, or
  output, while erasure still completes the frozen clear-id
  contract; one hash leaf, one marker shape, one params source.
- **Give up:** operators cannot script erasure from a hash
  (e.g. from an audit-trail lookup alone) — they must obtain the
  clear id through the secure channel; markers are single-slot,
  so multi-subject history lives only in the audit trail.
- **Give up (rows-only, HIGH fix):** the CLI does **not**
  retrain-flag artifacts — it passes `artifacts={}`
  (`retention_job.py:356-357`), so no `retrain_required`
  markers are written on the CLI path. The gain is a thin,
  no-dataset-I/O CLI with no new save path; the cost is that
  CLI erasure alone is **incomplete for GDPR** (rows gone,
  fitted/best-model artifacts not flagged). Completing an
  erasure therefore always requires the mandatory library-caller
  step (supply artifact dicts + re-save via existing writers,
  per the runbook). This split is deliberate: inventing
  artifact I/O in the CLI would create a second persistence
  path and bypass the frozen writers.
- **Reverse if:** a governed bulk-erasure service emerges that
  can hold clear ids in memory without an env hop, or legal
  requires statistical-erasure proof without retraining, or the
  Lead/`solution-architect` follow-on (new ADR + anti-team
  gate) authorizes CLI artifact loading/re-saving — either
  needs a new ADR (see also contracts.md §5.4 reversal note).
  Absent that follow-on, the rows-only scope stands.

## Acceptance (machine-checkable; all must pass)

- [ ] **Real-store erasure:** rows gone and marker equals the
  subject hash — `test_run_erasure_real_stores_marker_is_hash`
  (`forecast_ml_framework/tests/test_p2_retention_job.py`):
  offline store re-read empty, online store returns `{}`,
  `report['subject_hash'] == subject_hash(CLEAR)`,
  `registry['__meta__']['erasure']['subject_hash'] == digest`,
  `envelope['erasure']['subject_hash'] == digest`, and the clear
  id/actor are absent from report, audit log, and `caplog`.
- [ ] **Env-only CLI channel:** `test_main_erase_uses_env_subject_id`
  (clear id from env only; stdout carries the hash, never the
  clear id/actor; rows gone),
  `test_main_erase_requires_env_subject_id` (missing env → exit 2,
  no mutation), `test_main_erase_dry_run_combo_rejected`
  (→ exit 2, no mutation).
- [ ] **No hash-only flag / no second helper:**
  `test_retention_job_source_guards` (source contains
  `D-ERASURE-CLI-1` + `FORECAST_ERASE_SUBJECT_ID` +
  `subject_hash` + `--governance-params`; no `hashlib`, no
  `def subject_hash`, no `--erase-subject-hash`, no rotation
  wording), plus P2-S1 grep guard
  `grep -rn 'erase-subject-hash' forecast_ml_framework/src`
  returns empty (`.opencode/plans/P2.md` P2-S1 acceptance).
- [ ] **Sole params source:** `test_load_governance_params_from_platform_yml`
  + `test_platform_yml_shape` (**shape-only**: asserts the file
  carries only `governance_params.retention_days` as keys —
  `set(raw) == {'governance_params'}` and
  `set(raw['governance_params']) == {'retention_days'}`; it
  does **not** assert the value 365. The earlier value-365
  claim is corrected to shape-only; a value-365 assertion, if
  wanted, is a follow-on worker test note, not claimed here);
  `test_load_governance_params_rejects_bad_files` /
  `test_load_governance_params_invalid_yaml` (fail-closed).
- [ ] **CronJob retention-only (MEDIUM fix):** `test_cronjob_manifest`
  (daily `0 3 * * *`, `forecast-ml` service account,
  `retention_job` + `--governance-params`); **required test
  assertions (follow-on worker task, not this ADR task):**
  `'FORECAST_ERASE_SUBJECT_ID' not in text` (the manifest file
  text) and `'--erase' not in joined` (the joined container
  command). The current test does not assert these two
  conditions — the test edit itself is a follow-on P2-S1
  worker task owned by Lead dispatch; this ADR records them
  as required acceptance so the retention-only claim is
  machine-checkable.
- [ ] **CLI rows-only scope (HIGH fix):** acceptance exercises
  the split explicitly — (a) library path with real stores +
  supplied artifact dicts asserts rows gone **and** markers
  equal the subject hash (`test_run_erasure_real_stores_marker_is_hash`);
  (b) CLI path asserts rows gone with **no markers written**
  (source guard: `main()` passes `artifacts={}`, cf.
  `retention_job.py:356-357`); (c) operator runbook
  (`docs/runbook-p2.md` Retention/erasure ops) documents the
  mandatory library-caller retrain-flagging step as part of a
  complete erasure (follow-on owner: Lead / P2-R1, per
  Decision 4).
- [ ] **Store-targeting precondition (MEDIUM fix):** erasure
  acceptance for production use requires the artifact/Redis
  env to target the stores the rows were written with
  (Decision 8); `removed=0` on an empty/wrong store must not
  be reported as proof of absence — absence is claimed only
  via re-read verify against the correct store(s).
- [ ] **Gate before production use:** `anti-team-arch` records
  `APPROVED` on this ADR and the freeze is recorded in
  `.opencode/context/decisions.md` by a follow-on step owned by
  Lead / `solution-architect` (**not** this task — no
  `src/`/decisions edits here). **Production CLI erasure
  (rows-only) is enabled only after all of: the freeze, the
  `APPROVED`, and the operator runbook step for artifact
  retrain-flagging (library caller supplying artifact dicts +
  re-save via existing writers, `docs/runbook-p2.md`
  Retention/erasure ops) is documented and followed** — until
  then, erasure acceptance exercises the library path with
  real stores only, and no production CLI erasure runs as a
  claim of complete GDPR erasure. This freeze alone does not
  authorize production traffic, model promotion, or release
  (D-MVP-READINESS-1; P11 only).

## Pseudocode (contract sketch; workers write code)

```python
# retention_job.main: argv NEVER carries the subject.
def main(argv=None) -> int:
  args = parse(['--retention', '--erase', '--erase-subject',
                '--dry-run', '--governance-params'])
  if args.erase:
    if args.dry_run: return 2                # no mutation
    clear = environ.get('FORECAST_ERASE_SUBJECT_ID')
    if blank(clear): return 2                # no mutation
  params = load_governance_params(args.governance_params)  # sole source
  report = run_erasure(subject_id=clear, stores=build_stores(params),
                       artifacts={}, actor=actor_from_env())
  # artifacts={} -> rows-only: NO markers here; the library-caller
  # retrain-flagging step (runbook) is a mandatory operator step.
  print(f'erased subject_hash={report.subject_hash} ...')  # hash only

# run_erasure: validate, then delegate to frozen erase_subject.
def run_erasure(*, subject_id, stores, artifacts, actor, ...):
  require_non_blank(subject_id)              # ValueError before mutation
  require_non_blank(actor)
  return erase_subject(subject_id, stores=stores, artifacts=artifacts,
                       actor=actor, ...)     # caller re-saves markers
```

---

*Authored by `solution-architect`. Handoff: Lead runs the
`anti-team-arch` gate on this file; workers need no further design
— the contracts above are buildable as written.*
