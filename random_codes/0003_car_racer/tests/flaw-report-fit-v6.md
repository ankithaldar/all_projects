# FIT-V6 — TASK G4 flaw report (full verification & legacy suite migration)

**Owner:** tester (TASK G4; cycle-3 S1 fix). **Cycle-3 owned scope:** `tests/smoke.mjs`
and this report only. **No production file (`js/*`) was edited by tester in any cycle.**

**Plan:** `.opencode/plans/fit-v6-clipped-walls.md` (cycle 2, APPROVED).
**Verdict: PASS** — all ten suites are green (cycle-3 final run below), all grep gates
pass, all 76 smoke legs are retained (no silent skips), and no assertion or guard was
weakened. Cycle 3 closed the arch-gate S1 finding by un-skipping smoke leg 9 and
strengthening its clearance assertion; §5 carries the final evidence. Cycle-2 history
is retained in §§1–4 (with F-1 resolved by the cycle-3 arc-built templates).

---

## 5. Cycle-3 S1 fix — final evidence (2026-11-19)

### 5.1 Gate finding and the fix (test-only)

The cycle-3 architecture gate (`critique-log/fit-v6-cycle-3-arch.md`) required smoke
leg 9 to stop skipping low-clearance centreline samples: it still held
`if (clearance < CAR_RADIUS + 1) continue;` plus a "≥60% of samples qualify"
assertion. With the template worker's arc-built 91–96-point templates now shipped in
`js/rotation.js`, leg 9 was changed **only** as follows:

- removed the `continue` skip and the 60% non-vacuity assertion;
- changed sampling from stride≈24 to **stride 1 — every centreline sample**;
- added a per-sample assertion `clearance >= CAR_RADIUS + 4` (18px) against the
  **CLIPPED** walls, plus an aggregate min-clearance assertion and a verbatim
  evidence line. No assertion was weakened; the bar rose from "60% of ~24 samples
  clear" to "100% of all samples ≥18px".

Test-integrity proof (no repo file edited): the strengthened check passes on shipped
geometry (`hw=25` → min 24.995px) and **fails** on a regressed geometry (`hw=10` →
min 9.998px < 18px), so it can fail when the behaviour breaks.

### 5.2 Clearance / shape sweep (template worker + independent confirmation)

- Template worker (cycle-3, `rotation-v6` G2-19/G2-20/G2-21, `rotation-v2` frozen sweep):
  - every arc-built template 91–96 points, one genuine **≥150°** U-turn hairpin;
  - dense-spline min clearance **≥18px to CLIPPED walls at hw 50/49.75/37.25/25**,
    real `Car` at every centreline sample, **0 crashes**;
  - **0/297 low-clearance, 0 crashes** in the shipped-template sweep;
  - `resolveGeneratedTrack` never throws across **400+ seeds × r=1..99**; **99/99**
    distinct hashes per seed; worst-case resolve **≤100 ms**; gate fails closed on
    the old sharp cusp template.
- Independent confirmation (tester, all 6 templates × d∈{0,50,100}, stride 1):
  - point counts **96, 93, 91, 93, 91, 94** (within 91–96);
  - **11073 centreline samples checked, 0 skipped, 0 below 18px, 0 crashes**;
  - global minimum clipped-wall clearance **24.995px** (≥18px).
- V2 hash/shape expectations in `smoke.mjs` need no edit: they derive hashes
  dynamically via `resolveGeneratedTrack(seed, r, 2)`; the 14-point assertions are
  V1 (`generateTrack`/default version 1), the template count is 6, and the ≥150°
  hairpin + regenerated V2 goldens are already pinned in `rotation-v6.mjs`.

### 5.3 Full suite — verbatim output (all commands exit 0)

```
$ node tests/smoke.mjs
       (leg 9 — 11073 centreline samples, 0 skipped; min clipped-wall clearance 24.995px >= 18px across ALL samples)
  ok   — leg 9 — SensorFan: 5 readings, offsets, norm in [0,1]; FIT-V6 controller/sensor hairpin falsifier
SMOKE OK — 76 legs passed

$ node tests/track-clipped.mjs
legs passed: 10 / 10
ALL GREEN

$ node tests/rotation-v2.mjs
rotation-v2: 17 passed, 0 failed

$ node tests/rotation-v5.mjs
rotation-v5: 13 passed, 0 failed

$ node tests/rotation-v6.mjs
tests passed: 21/21
  ok   - G2-19 every template carries a genuine >=150-degree U-turn hairpin
  ok   - G2-20 dense-spline clearance: CAR_RADIUS+4=18px to CLIPPED walls at hw 50/49.75/37.25/25, Car at every sample, 0 crashes
  ok   - G2-21 clearance gate fails closed on the old sharp (cusp) template

$ node tests/db-v6.mjs
legs passed: 16 / 16
ALL GREEN

$ node tests/db-version.mjs
DB-VERSION: 10 legs passed

$ node tests/db-v5.mjs
FIT-V6 db-v5 result: 5 passed, 0 failed

$ node tests/version-flow.mjs
VERSION-FLOW OK — 9 legs passed

$ node tests/determinism.mjs
<3 identical 7-16-16-2 weight vectors>   (exit 0 — deterministic)
```

`node --check tests/smoke.mjs` → OK. All twelve commands (10 suites + 2 `node --check`
style integrity probes) exit 0; no failures, no silent skips, 76/76 smoke legs.
`track-clipped` is 10/10 in cycle 3 (cycle 2 was 9/9 — the template worker added the
G1 stale-pts regression leg 8c); `rotation-v6` is 21/21 (cycle 2 was 18/18 — G2-19/20/21
added). These increases are additive, never removals.

### 5.4 Cycle-3 acceptance

| # | Criterion | Result | Evidence |
|---|---|---|---|
| S1-1 | leg 9 skip removed; every centreline sample checked | **PASS** | stride 1, 11073 samples, 0 skipped (§5.1/§5.3) |
| S1-2 | min clearance ≥18px on ALL samples, no assertion weakened | **PASS** | per-sample + aggregate asserts; 24.995px min; integrity probe fails on regression |
| S1-3 | V2 hash/shape expectations current, 76 legs kept | **PASS** | dynamic V2 hashes; ≥150° hairpin + goldens in rotation-v6; `SMOKE OK — 76 legs passed` |
| S1-4 | full suite green | **PASS** | §5.3 — 10/10 suites exit 0 |

---

## 1. What was changed (test-only)

| Suite | Baseline | Fix (test-only) | Final |
|---|---|---|---|
| `rotation-v2.mjs` | 8/17 | V1 + V2 goldens regenerated for `50−0.25d` and the sharp-template V2; V2 shape legs updated to the shipped 21–23-point hairpins; turn-metric baselines 8.75 / 8.8995 (new generator); radial-formula/min-rho arms retired (covered by `rotation-v6`) | **17/17** |
| `rotation-v5.mjs` | 8 failures (old ladder/goldens + removed detector) | `legacyV2Hash` detector legs retired and replaced by a removal assertion; V1/V2 goldens + ladder regenerated; point-count 16–22 → 21–23; tangent/timing/never-throws/shape gates retained | **13/13** |
| `db-v5.mjs` | 2/10 | old-V2 `V2GeneratorChangedError` suite rewritten to the v6 contract: symbol-removal, every-ingress typed wall-model refusal, malformed `rotation_seed`, no-false-positive uv4 adoption. Obsolete "missing seed refused" leg retired (uv4 battery deliberately tolerates a missing pin) | **5/5** |
| `db-version.mjs` | 6/10 | `meta-tamper` battery fixture rebuilt at uv4; two migration legs rewritten to assert typed byte-preserving refusal; V1-by-default resume checked on uv4; 6 passing legs untouched | **10/10** |
| `version-flow.mjs` | 9/9 green | **No change** — five resolver sites, two-mode parser, worker/fallback/spectate boundary all still hold | **9/9** |
| `smoke.mjs` | 24/76 failures | uv/schema 3→4 + `wall_model='v6'`; generated-ladder assertions; legacy fixture legs converted to typed byte-preserving refusal; tests that need a live lineage rebuilt on fresh uv4 (`craftV3Bytes`/`freshDb`); explicit generator version 2 at the boundary; new `assertWallModelRefusal` helper; controller/sensor hairpin falsifier added to leg 9; **all 76 legs retained** | **76/76** |

No assertion was weakened: every changed assertion now checks a *stricter or
different shipped contract* (typed error name, exact message class, verbatim
refused bytes, uv4/wall_model, regenerated byte goldens). No leg was skipped or
gated; the two obsolete detector legs in `rotation-v5` and the obsolete
migration legs in `db-v5` were **replaced** by live assertions of the new
contract, not silently bypassed.

---

## 2. Evidence — commands and verbatim output

### 2.1 Baseline (RED observed before migration)

```
$ node tests/smoke.mjs
SMOKE FAILED — 24 of 76 legs failed:
  - leg 20, 23, 35, 36, 37, 38, 42, 45, 46, 46b, 48, 54, 56, 59, 60, 61,
    63, 67, 68, 70, 73, 66, 75, 76
```
Representative root cause (observed, not assumed):
```
leg 20: assert failed: PRAGMA user_version must be 3,
        got [{"columns":["user_version"],"values":[[4]]}]
leg 46: WallModelChangedError: This lineage was created with an older wall model ...
```

```
$ node tests/rotation-v2.mjs   -> rotation-v2: 8 passed, 9 failed
$ node tests/rotation-v5.mjs   -> (old ladder/golden failures; F1-1..3 detector legs failed)
$ node tests/db-v5.mjs         -> FIT-V5 TASK F2 result: 2 passed, 8 failed
   (V2GeneratorChangedError absent; every pre-v4 fixture now yields
    WallModelChangedError)
$ node tests/db-version.mjs    -> DB-VERSION: 6 legs passed, 4 FAILED
   (pre-v2 fixtures now WallModelChangedError)
```

### 2.2 Final — all suites green

```
$ node tests/track-clipped.mjs
legs passed: 9 / 9
ALL GREEN

$ node tests/rotation-v6.mjs
tests passed: 18/18

$ node tests/db-v6.mjs
legs passed: 16 / 16
ALL GREEN

$ node tests/rotation-v2.mjs
rotation-v2: 17 passed, 0 failed

$ node tests/rotation-v5.mjs
rotation-v5: 13 passed, 0 failed

$ node tests/db-v5.mjs
FIT-V6 db-v5 result: 5 passed, 0 failed

$ node tests/db-version.mjs
DB-VERSION: 10 legs passed

$ node tests/version-flow.mjs
VERSION-FLOW OK — 9 legs passed

$ node tests/smoke.mjs
SMOKE OK — 76 legs passed
```
All nine commands exit 0; `node --check` passes on every edited `tests/*.mjs`.
Final consolidated exit-code sweep (verbatim):
```
$ for s in track-clipped db-v6 rotation-v2 db-v5 db-version version-flow; do \
    node tests/$s.mjs >/dev/null 2>&1; echo "$s exit=$?"; done
track-clipped exit=0
db-v6 exit=0
rotation-v2 exit=0
db-v5 exit=0
db-version exit=0
version-flow exit=0
$ tail -1 /tmp/opencode/smoke-final.out
SMOKE OK — 76 legs passed
```
(`rotation-v6` 18/18 and `rotation-v5` 13/13 were verified in their own full
runs above; they are slower suites.)

### 2.3 Grep gates

```
$ grep -rnE '(Rotation|R)\.resolveGeneratedTrack\(' js/
js/main.js:914:      ... R.resolveGeneratedTrack(state.rotationSeed, ri, state.generatorVersion);
js/train-worker.js:146: ... R.resolveGeneratedTrack(state.rotationSeed, ri, state.generatorVersion);
js/trainer.js:240:      ... R.resolveGeneratedTrack(state.rotationSeed, ri, state.generatorVersion);
js/db.js:763:            ... Rotation.resolveGeneratedTrack(resolvedSeed, r, version);
js/db.js:2586:          ... Rotation.resolveGeneratedTrack(rotationSeedPre, Math.min(boundaryRIdx, 99), boundaryVersion).trackHash;
count = 5, each with an explicit version argument. PASS.

$ grep -rnE 'legacyV2Hash|V2GeneratorChangedError|V2_GENERATOR_CHANGED_MESSAGE' js/
(none — 0 references). PASS.

$ grep -rnE "(=== 2|== '2'|== \"2\"|\|\| 1)" js/ | grep -vE 'parseGeneratorVersion'
Only: comments, unrelated generation-number `|| 1` defaults,
FIT-V3 `user_version` schema checks (db.js:1704/1930/1955/1964/1967/2365),
and the single canonical parser `rotation.js:654: if (raw === 2 || raw === '2') return 2;`.
PASS — no inline generator-version interpretation outside the parser.
```
Note: the plan's speculative "three generator_version writers removed?" is
**not** the shipped state — `js/db.js` still writes `meta.generator_version`
(lines 2877/3032 etc.); G3 removed the V5 *detector*, not the version writer.
The tests assert the actual shipped behaviour.

### 2.4 Controller / sensor hairpin falsifier (new, leg 9)

Added to smoke leg 9 (keeps the 76-leg count):
- for all 6 sharp F1 templates × d∈{0,50,100}: at ≥60% of stride-sampled
  centerline points with standing-room clearance, a zero-input car tick does
  not collide, stays inside the road, and all five sensors are finite and
  equal to `track._raycastBrute(...)` (grid == brute; no phantom 300 px);
- a 120-step closed-loop low-speed pure-pursuit drive from a clear centerline
  sample on every template: no collision, the car actually moves, and the
  sensor fan keeps matching brute every 10 steps.
- apex feasibility (`v6ApexRadius >= 75`) reasserted.

The 10,000-ray grid==brute proof and the 30-vector wall-byte golden corpus are
owned by `tests/track-clipped.mjs` (green, leg 5/3); the generated-wall golden
and feasibility gates are in `tests/rotation-v6.mjs` (green, G2-8/13/14/16).

---

## 3. PASS / FAIL per acceptance criterion

### TASK G4 acceptance (handoff)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| G4-1 | All suites green | **PASS** | §2.2 — 9 suites, all exit 0 |
| G4-2 | No assertion or guard weakened | **PASS** | §1 — changed assertions are stricter/different-contract; 0 guards removed; no `js/*` edit |
| G4-3 | Flaw report written | **PASS** | this file |
| G4-4 | No silent skips | **PASS** | all 76 smoke legs executed and counted; `SMOKE OK — 76 legs passed` |

### Plan acceptance criteria (cycle 2)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| AC1 | Zero wall crossings at hw 50/37.5/25; genuine hairpins; 10k rays grid==brute; wall-byte golden green; controller drives hairpins | **PASS** (see F-1/F-2) | `track-clipped` legs 2/3/5, `rotation-v6` G2-8/13/14/16, new leg 9 falsifier |
| AC2 | Fresh v4 init/export/re-adopt; pre-v4 typed + byte-preserved; no stale wall representation | **PASS** | `db-v6` 16/16; smoke legs 20/45/46/46b/59/60/66/70/75; `db-v5` 5/5 |
| AC3 | Version suites green with regenerated goldens; five resolver sites + two-mode parser unchanged; timing budgets | **PASS** | `rotation-v2` 17/17, `rotation-v5` 13/13, `db-version` 10/10, `version-flow` 9/9; §2.3 gates; `rotation-v5` F1-14 timing (max ≤100 ms) |

---

## 4. Remaining findings

- **F-1 (cycle-2 observation — RESOLVED in cycle 3):** template 0 at d=0
  (halfWidth=50) had its start vertex only **0.86 px** from a clipped inner
  wall (measured then: d=0 → 0.86 px, d=50 → 13.36 px, d=100 → 25.00 px), so a
  standing car at `samples[0]` collided and cycle-2 leg 9 had to skip such
  samples (≥60% non-vacuity). Cycle 3 replaced the sharp templates with
  arc-built 91–96-point circuits whose minimum clipped-wall clearance is
  **24.995px** across all 11073 centreline samples; the skip is gone and every
  sample must now clear 18px (§5). This was the S1 arch-gate finding closed.
- **F-2 (informational):** a plain pure-pursuit controller drifts past
  `halfWidth/2` before completing a full lap on these tight hairpins (min
  turning radius ≈73 px vs apex radius ≥75 px leaves almost no margin). The
  falsifier therefore drives a *bounded closed-loop segment* per hairpin with
  no collision and exact sensor agreement; a full-lap `≤ halfWidth/2` drive
  would need a curvature-aware trained controller, which is outside G4's scope.
- **F-3 (coverage relocation):** the pre-v4 battery arms (FK, value-domain,
  DDL twins, recombination) previously exercised via `buildV1LineageBytes` /
  `buildV2LineageBytes` / `buildSemanticProbe` cannot run on uv4 adoption
  because step 3b typed-refuses them first. Their coverage is retained on uv4
  fixtures by smoke legs 54/62/68/71/72/73/76 and `db-v6`; the migrated legs
  now pin the typed-refusal contract instead of duplicating those arms.
- **F-4 (process):** anti-team-arch / anti-team-security gates are run by Lead
  after TASK G4 (per team protocol); tester did not self-approve them.
- **No blockers.** All four G4 acceptance checks and all three plan ACs are PASS.
