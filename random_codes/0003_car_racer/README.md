# Car Racer — Evolutionary Neural-Network Car Racer (FIT-V6)

Single-folder, no-build, no-dependency browser game. Everything below is verified against the code in this folder; nothing here describes a different build or a future feature. If you find a divergence between this file and the code, the code is correct and this file is stale: fix the README, not the facts.

**FIT-V6 in one paragraph.** Rotation is the headline: every **50 generations** the evaluated track rotates, and the replacement track is **generated deterministically from the lineage's rotation seed**. Rotation 0 is the lineage's pinned starting track (the frozen default circuit, or the editor/forked track at lineage creation). Rotation ≥ 1 tracks come from `resolveGeneratedTrack(seed, r, version)` and are byte-identical to the rows the pin/fork transaction pre-materializes. The shape is **versioned** by `meta.generator_version` (absent/`1` ⇒ V1, `2` ⇒ V2): V1 is the frozen 14-point sinusoid generator; V2 is one of six frozen arc-built asymmetric F1 circuits of **91–96 control points**, each with a genuine **≥150° hairpin**, long straights and corner complexes. A fresh, forked or zero-generation-re-pinned lineage is V2. There is **no migration of older files**: a pre-v4 `.sqlite` is refused byte-for-byte with a typed `WallModelChangedError`. Difficulty ramps monotonically with rotation index, capped at 100 (the 5000-generation cap spans 100 blocks, rotations 0–99). Resume adopts the stored `tracks` row of the resume generation's pinned track, never the editor/Manual track. The **wall model is byte-frozen**: the drivable-region boundary is a single canonical clipped `leftWall`/`rightWall` pair, the sole collision and sensor geometry, built fail-closed with `TrackWallError`. The shared half-width ladder is `50 − 0.25·d` (50 → 25). Fresh databases are `user_version = 4` / `schema_version = '4'` / `wall_model = 'v6'`. Fitness stays **signed**: net signed odometer plus lap bonus, with a `finished` outcome at 100 laps.

## 1. What this is

Car Racer is an evolutionary neural-network simulator: twenty policies are bred over generations to drive a car around a closed circuit, and the training lineage (every generation, every per-driver result, every best network's weights) is persisted into a **SQLite database that lives entirely inside the browser page** — no server, no account, no cloud. Fitness is signed: the net signed odometer (Σ speed·dt) plus a lap bonus (2 × the track length per lap, capped at 100 laps); an all-reverse run scores negative fitness, and a driver that completes 100 laps ends the run with outcome `finished`. At generation 51 there is a **one-time full restart**: all 20 children are fresh networks drawn from the current RNG stream and the elite clone is deliberately dropped. At every other generation — explicitly including 101, 151 and 201 — child 0 is an exact clone of the generation's best network and children 1–19 are Gaussian mutations of it. The database is a real `.sqlite` file you can export, import, fork and inspect with any SQLite tool.

What you see, screen by screen (five tabs; the Editor tab hides itself if `js/editor.js` fails to load — everything else keeps working):

- **Manual** — a canvas (1100×650) with a drivable car, a HUD showing speed (px/s), **Odometer (net)** (px) and the active track name, and a **Reset car** button.
- **Train** — a canvas plus the training controls: **Spectate (20 cars live)** runs the current generation on the main thread with all 20 cars rendered; **Start training** runs headless in a Web Worker; **Stop** aborts between drivers; a **Next generation** button appears after a spectate generation finishes; a controls panel (**Max generations**, 1–5000; lineage: **Fresh lineage (random seed)** or **Resume from generation N**); a live-stats panel (generation, **Best fitness / Avg fitness / Worst fitness** — `index.html:110-112`, sim seconds per driver, persist status, and a small **`R r · d d` rotation tag** next to the Generation stat — `index.html:109`, rendered by `js/main.js`, e.g. `R 1 · d 1`); a mini-log of the last five committed generations; and a persistent banner strip for DB events.
- **Replay** — pick a committed generation, watch its best driver drive it again **on the generation's own saved track snapshot** (rotation-aware: the snapshot is the track that generation actually ran on — `js/replay.js`). Optional, clearly-labeled override: **Use active track instead of the saved snapshot (results will differ)**. The Replay stats block also prints the generation's rotation and difficulty. The Replay HUD shows **Odometer (net)**; the expanded stats line ends with `run fitness <px>`, i.e. the replayed driver's fitness (net signed odometer + lap bonus; may be negative).
- **Logs** — a paged table (50 rows at a time, "Show 50 more") of committed generations. Columns: Generation, Best fitness (px), Avg fitness (px), Worst fitness (px), Best idx, Crashes, **Rotation**, **Difficulty**, Created. The Rotation column is the rotation index (one per 50-generation block, 0–99) derived code-side per row; the Difficulty column is that generation's track difficulty. The Best/Avg/Worst cells show fitness values and render negatives as-is. Click a row to expand the 20 per-driver results, in a table headed Driver / **Fitness** / Top speed (px/s) / Time alive (s) / Crashed / Stuck. Toolbar: Refresh, Export DB file, Import .sqlite, **Wipe all generations — start a NEW lineage**, **Fork to new track**.
- **Editor** — an in-browser track editor: click to add control points, drag to move them, right-click to delete; a road half-width slider; single-level undo; export/import `.cartrack`; and **Use as training track**, which forks the lineage when generations are committed (the confirm names how many generations will be deleted and that the new lineage starts at rotation 0 and moves to the generated schedule from rotation 1, generation 51 — `js/main.js`).

**Display and accessibility (implemented).** All four canvases (Manual, Train, Replay, Editor) share an immutable 1100×650 world space. CSS sizes each canvas to the largest uniform 1100:650 box that fits its wrapper and `100vh − 200px` (`--canvas-ratio`, `--canvas-viewport-inset`, `css/style.css:20-26,172-194`). JS mirrors that CSS box into a `devicePixelRatio`-scaled bitmap and installs one centred uniform transform with letterboxing, so circles stay circles and world coordinates are unchanged; `canvasContext` in `js/main.js:695-716` does this for the game canvases and `fitEditorCanvas` in `js/editor.js:83-101` does the same for the editor, including the inverse transform for pointer coordinates (`eventPos`, `js/editor.js:106-116`). A single rAF-debounced `resize` listener (`js/main.js:749-755,2001`) plus tab activation (`activateTab`, `js/main.js:1933-1958`) repaint the visible canvas from the last frame inputs (`redrawAllCanvases`). The five tabs use the WAI-ARIA APG tabs pattern: `role="tablist"`/`role="tab"`, a roving `tabindex` set in `activateTab`, and ArrowLeft/ArrowRight/Home/End navigation over `TAB_ORDER = ['manual','train','replay','logs','editor']` (`onTabsKeydown`, `js/main.js:1175-1207`); the Editor tab stays hidden if `js/editor.js` failed to load. The Train panel has an accessible `<progress>` indicator created from JS (`#train-progress`, `aria-label="Training progress — generations committed"`, text `<committed> / <max>`, rAF-throttled via `scheduleTrainProgress`, `js/main.js:1087-1131`). For rotation ≥ 1 the HUD, Train and Replay track fields show the truthful evaluation track (`generated · R<r> · <first 8 hash chars>…`) instead of a stale "default circuit" label, and the canvas aria-labels carry rotation and difficulty (`updateTrackLabel`/`updateCanvasLabels`, `js/main.js:1028-1085`). Replay and Logs show explicit empty-state messages when no generations are committed (`js/main.js:1160-1172`).

Header: **DB: starting…/file/browser cache/in-memory/unavailable** status pill plus **Open DB / Create DB / Export DB / Import DB** buttons. Footer: a status line with mode, DB state, latest committed generation, tick counter and fps.

## 2. How to run

### (a) Recommended — serve the folder over localhost

From this folder:

```sh
python3 -m http.server 8000
```

then open `http://localhost:8000/` in a browser. Any static file server works (nginx, `npx http-server`, `php -S`, …) — the app is plain HTML/CSS/JS with vendored WebAssembly; there is no server-side component and no build step. This is the recommended path: it exercises the exact code paths the app supports (classic `<script>` tags, vendored SQL, optional Web Worker) under a normal origin.

### (b) Double-click `index.html` (file://) — works too

Opening `index.html` directly from disk works in Chromium-based browsers and Firefox for the core game:

- Classic `<script>` tags are not blocked on `file://`.
- The vendored `vendor/sql-wasm-binary.js` embeds the entire SQLite WebAssembly binary as base64, so `sql.js` initializes with **zero `fetch()`/XHR** (Chromium refuses `fetch()`/XHR for `file://` reads). This is why `index.html` loads `sql-wasm-binary.js` **before** `sql-wasm.js`.
- What file:// does *not* give you: the Web Worker is refused from the `'null'` origin, so the Trainer transparently falls back to running the identical protocol in-thread via `MessageChannel` (the lineage replays identically). The File System Access pickers (Open DB / Create DB / Import via picker) exist but throw on `file://` — you get the banner `File pick unavailable here — this session is saved in your browser cache; use Export DB to save a .sqlite file`. On Firefox/Safari, where the File System Access API is absent entirely, `Import .sqlite` automatically uses a hidden `<input type=file>` fallback instead and `Create DB` creates a fresh browser-cache database.
- The Content-Security-Policy is exact: `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; object-src 'none'; base-uri 'none'` (`index.html:7`). `'wasm-unsafe-eval'` is there solely so sql.js's WebAssembly can compile — removing it blocks the database engine entirely. No inline scripts, no remote assets, no frames, no network.

### (c) No build, no dependencies, no server of your own

There is no `package.json`, no `npm install`, no bundler, no transpiler, no backend. The only runtime assets are the vendored files in `vendor/` (see §11) plus this folder's own HTML/CSS/JS. Node.js is needed **only** for the test suite (§9); the game itself needs nothing.

## 3. Controls (Manual tab)

- **ArrowUp** — gas (hold). At low speed this is throttle; near top speed the force fades to zero (throttle force = `gas · 550 · (1 − speed/600)`, clamped to [0, 1.25] — see `js/car.js`).
- **ArrowDown** — brake; from a standstill this becomes reverse (limited to 150 px/s).
- **ArrowLeft / ArrowRight** — steer left / right (speed-dependent steering angle; ~0.6 rad at zero speed fading to ~0.18 rad at max speed).
- **Reset car** button — teleports the car back to the track's start pose (arcPos 0) and clears speed/odometer/crash state.

Keys are mapped in `js/main.js` (`KEY_MAP`) and only registered when the event target is **not** an editable element (typing in "Max generations" or "Resume from generation" does not drive the car). Switching tabs clears the held-keys set, and so does window blur, so a stuck key can never phantom-drive after a focus change. A wall hit ends the run: the car is frozen in place (`crashed = true`, speed forced to 0) until you press **Reset car**.

## 4. The track model & the rotation (FIT-V6)

The track is a **closed, uniform Catmull–Rom spline** through a list of control points (`js/track.js`). Uniform (not centripetal) parameterization is deliberate: control points are roughly evenly spaced and uniform keeps the tangent field smooth and cheap.

The spline is sampled at ≤ ~4 px arc-length spacing; every sample carries `{ x, y, tangent (radians), arcPos (px from arcPos 0) }`. The start/finish line is arcPos 0, i.e. the first control point. A hard cap of **20 000 samples** (`MAX_TOTAL_SAMPLES`) is enforced before any array is allocated and again on the actual count; larger tracks are refused at construction.

**The drivable-region boundary is byte-frozen and clipped.** The raw left/right offset polylines (each sample offset along the unit normal by ±halfWidth) can self-cross at a hairpin. Each side is reduced to **one simple closed polyline** by repeatedly merging the innermost fold loop into the miter intersection of its two segments, or into an 8-point osculating arc when those segments are parallel (`js/track.js:177-414`). Every emitted coordinate is rounded with `Number(v.toFixed(9))`, array indices only are used, and the two independently clipped sides are brought to equal length by inserting exact midpoints into the longest segment. `segments` (each `{x1,y1,x2,y2,side}`) are then **rebuilt from the clipped walls**, not from the samples. There is no separate "outer boundary" object: the canonical `leftWall`/`rightWall` pair **is** the drivable-region boundary, and it is the **sole geometry used for collision and sensor rays**.

Any violation of that contract throws **`TrackWallError`** (a `ValidationError` subtype, `js/track.js:36-48`): fold-merge non-convergence, a non-finite or out-of-bounds coordinate, a collapsed wall, a broken length/`segments` invariant, or a per-segment AABB cell-span over 1024. The constructor fails closed — a hostile or degenerate track never produces partial geometry. A uniform 64 px spatial-hash grid is built over the clipped segments for the DDA raycaster; `_raycastGrid` and `_raycastBrute` share one segment-intersection helper, so results are bit-identical (asserted by `tests/track-clipped.mjs` over 10 000 seeded rays). Wall collision kills the run: the car is a circle of radius 14 px (`CAR_RADIUS`, `js/car.js`); a wall segment within 14 px of the car center crashes it.

**Rotation 0 — the pinned starting track.** Every lineage is pinned to a starting track in `meta.track_hash` at lineage creation. For a fresh lineage on the *default* circuit this is the frozen default (`(170,180) (300,120) (500,100) (700,120) (830,190) (880,340) (830,490) (650,570) (450,590) (250,560) (120,400)`, 11 points, halfWidth 70 px — `DEFAULT_CONTROL_POINTS`, `js/track.js:94-106`; `TRACK_DEFAULT_HALF_WIDTH = 70`). For a fresh lineage created while an **editor track** is active (or via **Use as training track** / **Fork to new track**), the pin is that editor track's hash. The editor can redraw it: click empty space to insert a point, drag to move, right-click to delete; minimum 3 points, maximum 512; half-width slider **1–2000 px**; single-level undo; `.cartrack` import/export (`Track.fromJSON` validated: ≤ 256 KiB, finite coordinates |x|,|y| ≤ 1e6, closed loop). Every structural change auto-saves to `localStorage` under `car_racer.track`. The active track is not hot-swapped while editing — it changes only via **Use as training track** or on the next page load.

**Rotation ≥ 1 — the generated curriculum.** Every `ROTATE_EVERY = 50` generations (`js/rotation.js:37`) the evaluation track rotates. The rotation for generation N is `floor((N − 1) / 50)` (`rotationForGeneration`, `js/rotation.js:88-93`); its difficulty is `min(r, 100)` (`difficultyForRotation`, `js/rotation.js:96-101`). Gens 1–50 are rotation 0 (difficulty 0); gens 4951–5000 are rotation 99 (difficulty 99). The generator is deterministic per (rotation_seed, rotationIndex) and draws all randomness from `mulberry32((rotationSeed ^ Math.imul(rotationIndex + 1, 0x85EBCA6B)) >>> 0)` (`js/rotation.js:121-130`). The default circuit is deliberately never read or reused by either generator.

**Shared ladder.** `halfWidthForDifficulty(d) = 50 − 0.25·d` (50 → 25) is shared by both generators (`js/rotation.js:58-61`). It was narrowed in FIT-V6 so that, with apex radius ≥ 75, arm separation `2R ≥ 150 > 2·50 + 20`, and the clipped inner wall cannot merge or vanish at a hairpin. The difficulty ladder is integer by construction; `validateDifficultyInput` accepts finite numbers in [0, 100].

**V1 — `generateTrack` (frozen).** 14 control points at `θ_i = 2π·i/14`, sinusoid superposition around `TRACK_CENTER = (550, 325)`: `ρ_i = 1 + Σ_k a_k·sin(k·θ_i + φ_k)`, with `a_k = AMPLITUDE × (1/k^(3/2)) / Σ_{j≤K} 1/j^(3/2)` (amplitudes sum exactly to AMPLITUDE) and `φ_k = 2π·(s() × 2⁻³²)`. V1 uses `harmonics(d) = 2 + floor(d/25)` (2 → 6) and `amplitude(d) = 0.08 + 0.34·(d/100)` (0.08 → 0.42), the `TRACK_MAX_RADIUS = 245` clamp, self-intersection as the only reject path with a fixed `TRACK_MAX_REJECTS = 200` cap, a regular 14-gon radius-170 fallback, and 3-decimal coordinate rounding (`js/rotation.js:188-251`). A rejected draw releases the stream and re-draws; after 200 rejects the regular polygon is returned — valid by construction, never a crash. The V1 **algorithm** is frozen; because the shared half-width ladder changed in FIT-V6, its golden bytes were regenerated for `50 − 0.25·d`.

**V2 — `generateTrackV2` (the shape model).** V2 no longer uses a radial/harmonic formula. It selects one of six **byte-frozen arc-built templates** (`V5_TEMPLATES`, `js/rotation.js:314-321`; 91–96 control points each) by `((rotationSeed ^ Math.imul(r + 1, 0x9E3779B1)) >>> 0) % 6`, then applies a rotation of up to `0.10·TAU` and a radial perturbation of amplitude `0.02 + 0.06·d/100` (one uint32 for θ, then two uint32s dx/dy per point). Uniform scale-to-fit with a `halfWidth + 20` margin and bbox-centred translation keeps the result inside the canvas. The older `THETAS36`/`harmonicsV2`/`amplitudeV2`/`generateTrackV2Fallback` exports still exist on the `Rotation` object but are **not used** by `generateTrackV2`.

Each candidate passes the FIT-V6 step-7 gate (`v5Valid`, `js/rotation.js:617-666`), in frozen order: finite coordinates; within canvas bounds; simple control polygon; start triple within 1° of collinear; apex radius ≥ 75 (minimum circumradius over a 3-step control-point window, `v6ApexRadius`); arm separation `2R > 2·halfWidth + 20`; a canonical clipped `Track` builds with sound walls and the required length/`segments` invariants; the spline turning radius through every apex is ≥ 75; and the minimum distance from **every centreline sample to every clipped wall segment is ≥ 18 px** (`CAR_RADIUS + 4`, `v6TrackClearance`, `js/rotation.js:548-609`). The code records why these two gates exist: an earlier sharp-template family had a 0.41 px spline radius and 0.86 px clearance at hw 50, which let a standing car collide at the apex. The accepted candidate reuses the single canonical clipped `Track` built during validation — it is never rebuilt. There is a fixed 200-redraw cap; the zero-perturbation, r-dependent fallback is validated by the **same** gate and throws a typed `ValidationError` if it fails (fail-closed, never a bad track). The cumulative generation budget is **99 × 200 = 19 800 attempts** (`MAX_PREMATERIALIZE_ATTEMPTS`).

**Version contract.** One two-mode parser serves every realm: `parseGeneratorVersion(raw)` (`js/rotation.js:726-732`) maps absent (`undefined`/`null`) ⇒ 1, `1`/`'1'` ⇒ 1, `2`/`'2'` ⇒ 2, and throws a typed `ValidationError` for `''`, whitespace and every other value — no silent downgrade. `requireGeneratorVersion(raw)` (`js/rotation.js:734-741`) is the strict boundary form: a missing or empty field throws `generator version is required` before delegating to the same parser. `Db` does not redefine either — `Db.parseGeneratorVersion` is a getter alias returning `Rotation.parseGeneratorVersion`. `GENERATORS_BY_VERSION` (`js/rotation.js:745`) is the only mapping from integer 1/2 to a generator; an omitted version defaults to 1 for legacy 2-argument callers, but every production call site passes it explicitly. The five explicit-version resolver sites are:

- `js/db.js:763` — curriculum pre-materialization;
- `js/db.js:2586` — the rotation-boundary pin-move guard;
- `js/main.js:914` — the rotation-aware training track (headless and spectate);
- `js/train-worker.js:146` — the Worker core;
- `js/trainer.js:240` — the MessageChannel fallback core.

The Worker, fallback and main-thread session handshake all require the integer with `requireGeneratorVersion`; a missing or empty field throws before any track is resolved. `tests/version-flow.mjs` statically gates these five sites. The old-V2 detector (`legacyV2Hash` / `V2GeneratorChangedError`) has been **removed** from production code.

**Hash.** `resolveGeneratedTrack(seed, r, version)` returns `{ controlPoints, halfWidth, difficulty, trackHash, track }`, where `trackHash = cyrb53(canonicalJSON({ points, halfWidth }))` — the same surrogate every consumer uses (`cyrb53`, `js/track.js:469`; `canonicalJSON` fixes key order `{points:[{x,y}], halfWidth}` and rounds every coordinate and the half-width via `toFixed(9)`, `js/track.js:492`). cyrb53 is a 64-bit public-domain string hash producing a 16-char lowercase hex string. It is a **consistency signal, NOT cryptographic**: same-geometry tracks hash identically across roundtrips; a tampered track JSON is detected by re-hashing. It is not an adversarial boundary.

**Pre-materialization.** At every fresh pin or fork, rotations 1..99 are resolved from `(seed, r, version)` and **pre-inserted** into the `tracks` table in the same transaction as the pin/wipe — 99 generated rows (plus the rotation-0 pin row) up front, byte-verified per row before any write (`preMaterializeCurriculum`, `js/db.js:704-765`): `cyrb53(canonicalJSON(...))` must equal the resolver-reported `trackHash`, or the caller rolls the transaction back (zero pin). The version comes from `meta.generator_version` through the canonical parser. Inserts use SELECT-then-split, never upsert; same-geometry pins and reforks are no-ops. The inserted rows carry `source = 'generated'`, `difficulty = r`, and `created_at = '2026-01-01 00:00:00'` (a sentinel, never `datetime('now')`). The post-materialization invariant is scoped to the current `rotation_seed`: for each r=1..99 the resolver at the selected version must match exactly one `source='generated'` row, and the other version must match none.

**The rotation-0 pin row** is inserted the same way: `source = 'default'` iff the hash equals the frozen default circuit's hash, else `'editor'`; `difficulty = 0`; `created_at = '2026-01-01 00:00:00'`. The fork flow ("Use as training track" / "Fork to new track" / "Wipe all generations") inserts the new pin row and pre-materializes the fork's curriculum from the new seed inside the same transaction as the wipe. **Tracks rows are never deleted** — wiped lineages leave their generated rows behind. A full fresh lineage carries exactly **100 tracks rows** (1 pin + 99 generated). Adoption refuses files carrying more than **1000 tracks rows** — refusal, never deletion; fork-history is permanent.

**Resume adopts the stored tracks row.** Resume never re-rolls the curriculum: `prepareResumeLineage` calls `Db.resumeState()` then `Db.adoptPinnedTrack(rs.trackHash)`, which reads the **stored `tracks` row of the resume generation's pinned track**, recombines `control_points + half_width` through `Track.fromJSON` verbatim (never string splicing), re-hashes and **must equal the row's `trackHash`**. Absence or mismatch ⇒ typed **`ResumeTrackError`** with **no generator fallback**: the persistent banner says the pinned track is not recoverable from this file and that the committed database on disk is untouched. The ambient editor/Manual track is never the resume track. On success the resumed Trainer evaluates the next generation on that stored track; the first rotation change past a 50-boundary comes from the seed.

**The pin moves at the boundary.** `logGeneration` enforces the rotation contract at write time: at the generation after a boundary (gen 51, 101, … 4951), the only legal next track is the byte-verified resolver output for this lineage's `rotation_seed`; a payload carrying the current pin, a stale editor row or any wrong hash is refused; the `tracks` row for it must exist with exactly `difficulty == min(boundaryRIdx, 100)` and `source = 'generated'`; and `meta.track_hash` moves **inside the single logging transaction**. Off-boundary re-pins are refused before `BEGIN`. A headless run whose worker/fallback core and this main-thread resolver ever disagree on the evaluation track's hash is halted with `LineageIntegrityError` (`code = 'FIT_V3_ROTATION_DIVERGENCE'`) before anything is logged.

**Difficulty is display/audit only** — never consumed by Selection or physics. The ladder is pure math; the Logs panel and the stats header render it for humans.

## 5. The neural network

Every driver is the same fixed architecture (`js/nn.js`, `NN_ARCH = '7-16-16-2'`):

```
7 inputs ──► 16 tanh ──► 16 tanh ──► 2 tanh outputs  →  [gas, steer]
```

**Inputs** (exact, ordered, all normalized to roughly [0,1] or [-1,1]):

| # | Input | Normalization |
|---|---|---|
| 1–5 | Five sensor distances | distance / `SENSOR_MAX` (300 px), clamped to [0,1]; 1 = no wall within range |
| 6 | Car speed | `speed / MAX_SPEED` (600), signed |
| 7 | Direction vs. track tangent | `wrapToPi(heading − tangentAngle) / π` ∈ [−1,1] |

**The five rays** are cast from the driver seat — in this arcade model the seat **is** the car's collision-circle center (there is no other seat offset; see `Car.driverSeat()`). Their directions are **[-70°, -35°, 0°, +35°, +70°] relative to the car's current heading** (`SENSOR_ANGLES_DEG`, `js/sensors.js`), on the canvas convention that y is down and positive angles are clockwise (steer +1 = right = clockwise). Each ray measures the distance to the **nearest clipped wall segment** using the same grid raycaster the preview uses; max ray length is **300 px** (`SENSOR_MAX`).

**Outputs**: `gas ∈ (−1,1)` — −1 full brake (and, from standstill, full reverse), +1 full throttle; `steer ∈ (−1,1)` — −1 full left, +1 full right. `tanh` saturates to exactly ±1 for |z| ≳ 19, so consumers tolerate [−1, 1] inclusive.

**Parameter count**: `7·16 + 16 + 16·16 + 16 + 16·2 + 2 = 434` floats (`WEIGHT_COUNT`). Biases start at 0; weights are initialized Xavier-like, `N(0, 1/√fanIn)`; random draw order is fixed (`w1 → b1 → w2 → b2 → w3 → b3`, row-major) and is part of the determinism contract. All "random" numbers flow through an explicit mulberry32 stream — `Math.random` is never called anywhere in the train path.

Where to change things: the 7-input layout is pinned by the architecture string — a change to inputs, layer sizes or count is a **new architecture**, which the database rejects (`architecture` must equal `'7-16-16-2'`) and which orphans every stored weight. The network class, weight validation (exactly 434 finite floats, |v| ≤ 1e6) and `mutate()` all live in `js/nn.js`.

## 6. The evolution

- **Population**: 20 networks (`POPULATION = 20` in `js/evolution.js`; the DB layer keeps a deliberately separate literal `POPULATION_SIZE = 20` in `js/db.js` — keep them in sync by hand).
- **Generation 1** is 20 randomly-initialized networks from the lineage's mulberry32 stream.
- **Evaluation** (headless or spectate): each of the 20 drivers is simulated in `SIM_DT = 1/120` s ticks against the frozen eval constants (`js/driver-sim.js`): a driver may live at most `EVAL_CAP_S = 45` simulated seconds; a driver that fails to make **+20 px of net forward centerline progress in 12 simulated seconds** (`STUCK_TIMEOUT_S`, `PROGRESS_ADVANCE_PX`) is killed as *stuck* — this is the anti-circling control, and it bounds only the no-net-progress case. Terminals, in the deterministic kill order — **crash > finished > stuck > time-cap**: `crashed` (wall touch), `finished` (laps reached `LAP_BONUS_LAPS`), `stuck` (stagnation kill), `time-cap` (`EVAL_CAP_S` reached). Stagnation never preempts `finished`.
- **Fitness — the canonical glossary** (owned by `js/driver-sim.js`; this is the normative wording for this app):
  1. **`car.odometer`** — NET SIGNED px travelled: Σ(speed·dt); can be negative (an all-reverse run). Display/debug only; never the selection currency by itself.
  2. **liveness `cumArc`** — seam-unwrapped signed centerline arcPos progress; drives the stagnation kill.
  3. **`cumArc` high-water (`maxCumArc`)** — ratcheted maximum of cumArc; `laps := floor(maxCumArc / track.length)`, capped at `LAP_BONUS_LAPS`.
  4. **fitness** — `car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length`. THE selection/logging/persistence currency. May be negative.
  5. **terminal outcomes** — `crashed` (wall touch), `stuck` (stagnation kill), `time-cap` (`EVAL_CAP_S`), `finished` (laps == `LAP_BONUS_LAPS`).
  6. **kill order (deterministic)** — crash > finished > stuck > time-cap. Stagnation never preempts `finished`.
  7. **fitness may be negative** — stored/rendered as-is; selection handles signs correctly.

  Selection is **selector-independent**: each driver's fitness is computed against the track alone — one car, one sim, no cross-driver terms, no population normalization — and `selectBest` only ranks the absolute numbers afterward. The constants: `LAP_BONUS_TRACK_LENGTHS = 2` and `LAP_BONUS_LAPS = 100` live in `js/driver-sim.js` (`DRIVER_SIM_CONFIG`, frozen).
- **Selection** (`Evolution.selectBest`, exact float comparisons, sign-honest): highest fitness wins; on an **exact** fitness tie a non-crashed driver beats a crashed one, then **higher time alive** wins. Top speed is reported only and never used by `selectBest` (`js/evolution.js:284-306`; `tests/f1-curriculum.mjs` leg 7).
- **Reproduction**: `Evolution.spawnNextGeneration()` calls the single `_spawnChildren(parent, nextGeneration)` routine with the **target** generation passed explicitly (`this.generation + 1`). For every target **except 51** — explicitly including 101, 151 and 201 — child 0 = `best.clone()` **exact deep clone, elitism anchor, zero RNG draws** + children 1–19 = `best.clone().mutate(MUTATION_RATE=0.25, MUTATION_SCALE=0.30, rng)`, where each weight is perturbed with probability 0.25 by a Gaussian sample (σ = 0.30, Box–Muller). At target **51 only** there is a **one-time full restart**: all 20 children are fresh `new NeuralNetwork(rng)` nets from the current stream (800 RNG draws each = **16,000 draws**; biases zero, Xavier-like weights), and the elite clone is deliberately dropped — the single escape from the oval-specialist local optimum. 101/151/201 are **ordinary mutant generations, not restart boundaries**: no re-diversification there. No crossover, no selection among non-best parents, no annealing. `resumeFromBest()` calls the same routine with `completedGeneration + 1`, so the resumed population is byte-identical to an uninterrupted run at either kind of target (`js/evolution.js:308-390`).
- **Until generation 5 000** (`MAX_GENERATIONS = 5000`, `js/evolution.js`); the Train tab's **Max generations** input is clamped to 1–5000.
- **Resume**: the Train tab's "Resume from generation N" continues from the **latest committed generation only** — never an earlier one. Picking an earlier N is refused. The resumed run continues on the resume generation's own pinned track — the stored `tracks` row, byte-verified and re-adopted. The committed prefix is never re-evaluated or rewritten: the `generations.generation` primary key enforces it, and a UNIQUE conflict during logging halts the run with `LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched`.
- **Determinism is per engine.** Same seed + same track + same JS engine ⇒ byte-identical populations and best weights for every generation. The rotation resolver is deterministic on the same engine for the same reason. Last-ulp cross-engine divergence is real (binary64 math, `Math.tanh` rounding) — do not expect byte-identity across machines or across Node major versions. `tests/determinism.mjs` prints the generation-1..3 best-weights JSON; `tests/smoke.mjs` leg 13 compares the stdout of two fresh node processes and demands byte-identity.
- **Seeding a fresh lineage** uses `crypto.getRandomValues` (a uint32). The RNG cursor (`rng_state`) is committed in the same transaction as each generation, so a resumed lineage re-derives the exact children an uninterrupted run would have produced. The **rotation seed** (`meta.rotation_seed`) is written exactly once at pin time (fresh lineage or fork), from the pin's crypto seed.

## 7. SQLite in the browser — the part everyone asks

There is **no server**, and there never was one. The database is a real SQLite database compiled to WebAssembly by **sql.js**, vendored into `vendor/` as plain files (`sql-wasm.js` loader, `sql-wasm.wasm` engine, `sql-wasm-binary.js` base64 embed, `SQL.js.LICENSE.txt` — MIT). It runs inside the page; the app opens, writes, verifies, exports and destroys that database entirely client-side. `js/db.js` is the whole story — read it if you want the full contract; this section is the honest summary.

**Where the bytes live** — exactly two places, never anywhere else:

1. **The browser cache** (the default): IndexedDB database **`car_racer`**, object store **`kv`**, under key **`car_racer.lastKnownGood`** — THE single browser-cache slot. It holds the full bytes of the last verified-good database plus a `{maxGeneration, trackHash, rngState, rotationIndex}` tag (the tag carries `rotationIndex` for observability only — never read for enforcement). A second slot, **`car_racer.lineageSnapshot`**, is a strictly-ahead-only recovery snapshot (written only when the new tag is strictly ahead of or on a different track from the stored tag; used only when the cache slot fails validation and the snapshot is provably newer). A third, unrelated database, **`car_racer-sessions`** (v2), stores the File System Access **handle** for a connected `.sqlite` file in a `handles` store — never the DB bytes.
2. **A real file you picked once** (Chrome/Edge/Opera, File System Access API): the `.sqlite` lives wherever you saved it; the app re-opens it through the stored handle at boot. Two-tab clobber protection: `createWritable({ mode: 'exclusive' })` holds an exclusive write lock on the target for the duration of the write; content before/after is guarded by the **freshness tripwire** — a recorded `{size, sha256}` fingerprint compared against the file **before every write**; a `navigator.locks` exclusive `'car_racer-db'` lock serializes cross-tab steps; and an in-instance promise chain (`_persistChain`) serializes same-tab writes unconditionally.

**The write pipeline** (`Db.persist()`, called once per committed generation): export the DB bytes **once** → verify them in a throwaway instance (integrity_check + uv==4 + `wall_model='v6'` + architecture pin + coherence tag) → fingerprint → [file mode: re-read the file and **tripwire-compare** — mismatch raises a blocking Reconcile banner] → write the **verified bytes** to `car_racer.lastKnownGood` → [file mode: exclusive write + post-close re-read and SHA-256 compare] → refresh the lineage snapshot under its tag rule. Any failure: the writable is aborted (no partial file), a persistent banner appears (`DB NOT PERSISTING — export manually`), training continues, and the banner never auto-hides.

**The browser-cache no-regress rule**: when IDB already holds a cached copy at generation ≥ the incoming tag's generation **on the same track**, the slot is not overwritten (a `cache-ahead` banner is raised and kept). Chromium's exclusive `createWritable` stages writes through a temporary `*.crswap` file; a crash mid-write can leave one behind. It is never adopted or read by this app — delete it.

**Exports and imports**: **Export DB** downloads `car_racer-gen<N>.sqlite` — a standard SQLite file readable by any SQLite tool. **Open DB / Create DB** pick or create a file via the File System Access API. **Import DB / Import .sqlite** runs the picked file through the **adoption battery**, the validation gauntlet in `Db.validateAdoptionBytes` (`js/db.js`). The battery runs, in order:

1. **Pre-parse gate** — bytes must be a non-empty `Uint8Array` ≤ **64 MiB** (`MAX_DB_BYTES`). Refusal before sql.js opens anything.
2. **Throwaway instance** — the bytes are opened in a scratch `SQL.Database`; the live working DB is never involved.
3. **`integrity_check == 'ok'`**.
3a. **`foreign_key_check` empty** — no orphans on any adopted bytes.
3b. **Wall-model break gate** — `user_version ∈ {1,2,3}` (a lineage built under the old wall model) is refused with a typed **`WallModelChangedError` carrying the refused bytes verbatim**; `user_version ∉ {1,2,3,4}` is a foreign file refused with a generic `ValidationError`; only `user_version = 4` proceeds. This runs **before** any scrub or DDL arm, so a pre-v4 file with a `tracks` table still gets the typed refusal.
3c. **DDL and wall-model readbacks** — for uv==4: `meta.schema_version == '4'` and the `generations` DDL must match this build's native twin **byte-for-byte after lowercasing, whitespace collapse, trailing-`;` strip and double-quote strip**; `meta.wall_model == 'v6'`; the `tracks` DDL must match the pinned `DDL_TRACKS` twin. A relabelled or stale-wall file is refused here.
4. **Exact table/column sets** — the five-table set (`meta`, `tracks`, `generations`, `driver_results`, `best_weights`) with exact columns.
5. **Architecture pin** — `meta.architecture == '7-16-16-2'`.
5b. **Generator-version allowlist** — `meta.generator_version` must be absent, `1` or `2`; `''`, whitespace, `'3'` and every other value throw a typed `ValidationError`.
5c. **V2-pinned curriculum rows** — for a V2 lineage with a pinned seed, each r=1..99 must resolve to exactly one `source = 'generated'` row at V2.
6. **Row-count caps** — at most 100 000 `generations` rows, 2 000 000 `driver_results` rows, and 1000 `tracks` rows.
7. **Weights payloads** — every `best_weights.weights_json` ≤ 16 384 chars and exactly 434 finite floats, plus a `best_weights` row for `MAX(generation)` is required.
8. **`track_json` ≤ 256 KiB**.
8b. **Self-hash arm** — for every generations row, the stored `track_json` must recombine to the row's `track_hash`.
9. **`sqlite_master` scrub** — EXACTLY the expected tables + `sqlite_autoindex_*`; zero triggers, views, foreign objects.
10a. **Semantics** — the declared FKs (children cascade to `generations`) plus the 17-arm `VALUE_DOMAIN_CHECKS` battery on every adopted `generations`/`driver_results` row (fitness finite but signed; integers in range; `crashed/stuck ∈ {0,1}`; `mutation_rate ∈ [0,1]`; etc.).
10b. **Tracks arms** — every `generations.track_hash` resolves in `tracks`; tracks value domains (typeof-strict); `control_points` content + whole-row recombination (recombined hash must equal the row's `trackHash`); the generations FK arm (exactly one FK to `tracks(track_hash)`, with a non-cascading `on_delete`).

Only then, and only with an explicit `confirm()`, does the battery's caller replace the working database. A file that fails any step is refused; the working database is untouched. **There is no schema migration**: a pre-v4 file is refused byte-for-byte and can be downloaded verbatim; it is never upgraded in place. The `Db.verifyForPersist` step-2 check pins uv==4 + `wall_model='v6'` on every persist, so a downgraded/corrupt export is refused before it is written.

**Wall-model refusal banner.** A typed `WallModelChangedError` is mapped, at every adoption door (open, import, cache restore, boot), to a persistent banner marked `WALL MODEL CHANGED` — verbatim:

> `WALL MODEL CHANGED — This lineage was created with an older wall model and cannot be resumed with this build. The original file was left untouched. Use Export .sqlite to download the refused bytes verbatim for your records, then start a fresh lineage. — The original was left untouched. Use Export .sqlite to download the refused bytes verbatim, then Create / Fork to start a fresh lineage.`

The banner carries a **Download refused bytes verbatim** button that saves `refused-lineage.sqlite`. The refusal is **byte-preserving**: neither the file on disk nor the browser-cache slot / handle is cleared, and `err.refusedBytes` holds the exact refused bytes. Use it only to start a fresh lineage; the refused file cannot be resumed by this build.

**Boot-time restore order** (`js/db-session.js`): a stored file handle with `readwrite` permission → re-adopt the file (battery included); otherwise (or on refusal) the battery-verified browser cache is tried (LSG first, then the strictly-ahead snapshot); otherwise a fresh in-memory session boots. On a typed wall-model refusal the handle and LSG slot are preserved and the refused bytes ride on the result. On a normal cache restore you will see, verbatim (with `<N>` the committed generation):

> `Loaded last session's database from this browser's verified cache (lastKnownGood, generation <N>) — Export to keep it`

or, when the snapshot slot passed:

> `Loaded last session's database from the browser cache snapshot (lineageSnapshot, generation <N> — battery-verified) — Export to keep it`

**`file://` caveat.** The `file://` origin rules are browser-defined and coarse; on some setups every local page you open can share one origin with every other local page, and IndexedDB is per-origin. Nothing in this app can stop a hostile local page from writing garbage into the shared cache slot. The defense is the adoption battery — every cached copy is re-validated at boot and refused bytes can never *load* — but the write end of the slot is only as trustworthy as the least careful local page you opened. **localhost for normal use** is the recommendation; localhost is not a security boundary either.

**Quota**: IndexedDB usage is browser-managed and evictable under storage pressure ("clearing site data clears the DB" — export to keep it). Both cache slots can hold a full copy at once, so the footprint is up to roughly twice the database size. The Logs tab shows `navigator.storage.estimate()` and the app calls `navigator.storage.persist()` once at boot in cache mode. Export is your real backup.

**Multi-tab semantics**: two tabs using the same file are serialized in-tab by the promise chain and cross-tab by `navigator.locks` where available; the tripwire detects any outside change at the next write (blocking Reconcile banner with four actions: **Reload from disk / Export my copy / Fork new lineage / Restore from browser cache**). Two tabs in browser-cache mode are last-put-wins except for the no-regress rule.

**Nothing is spying on you**: no telemetry, no analytics, no accounts, no network calls of any kind (no `fetch`/`XHR`/`WebSocket`/dynamic `import()` anywhere in `js/` or `tests/`). The `.sqlite` contains exactly the schema below plus your training data. Export regularly; keep the file in a folder you control.

## 8. The DB schema (the DDL below is copied from `js/db.js`; formatting normalized, semantics unchanged)

```sql
-- meta: key/value pin store (js/db.js:180)
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- tracks: one row per pinned/generated/forked track (js/db.js:206)
CREATE TABLE tracks(
  track_hash TEXT PRIMARY KEY,
  source TEXT NOT NULL CHECK (source IN ('default','generated','editor')),
  control_points TEXT NOT NULL,
  half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000),
  difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- generations: header row per committed generation (js/db.js:189)
CREATE TABLE generations(
  generation INTEGER PRIMARY KEY CHECK (generation >= 1),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0),
  best_fitness REAL NOT NULL,
  avg_fitness REAL NOT NULL,
  worst_fitness REAL NOT NULL,
  best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19),
  mutation_rate REAL NOT NULL,
  mutation_scale REAL NOT NULL,
  track_json TEXT NOT NULL,
  track_hash TEXT NOT NULL REFERENCES tracks(track_hash),
  sim_seconds REAL NOT NULL,
  rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295)
);

-- driver_results: 20 rows per generation, FK → generations
CREATE TABLE driver_results(
  generation INTEGER NOT NULL REFERENCES generations(generation) ON DELETE CASCADE,
  driver_index INTEGER NOT NULL CHECK (driver_index BETWEEN 0 AND 19),
  distance REAL NOT NULL,
  top_speed REAL NOT NULL,
  time_alive REAL NOT NULL,
  crashed INTEGER NOT NULL CHECK (crashed IN (0,1)),
  stuck INTEGER NOT NULL CHECK (stuck IN (0,1)),
  PRIMARY KEY (generation, driver_index)
);

-- best_weights: the tip weights per generation, FK → generations
CREATE TABLE best_weights(
  generation INTEGER PRIMARY KEY REFERENCES generations(generation) ON DELETE CASCADE,
  architecture TEXT NOT NULL CHECK (architecture = '7-16-16-2'),
  weights_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Five tables — nothing else may ever exist in a lineage database. The only indexes permitted are `sqlite_autoindex_*` entries from the PRIMARY KEY constraints; the adoption battery's `sqlite_master` scrub refuses any trigger, view, foreign table or non-conforming index. `tracks.control_points` holds the bare canonical JSON points array of exactly-two-keys `{x, y}` objects — NOT the `{points, halfWidth}` wrapper that `track_json` carries. `generations.track_hash` carries the declared `REFERENCES tracks(track_hash)`. The old non-negative-fitness CHECK on `best_fitness` is gone — fitness may be negative and must round-trip.

Fresh databases are **v4 from birth**: `PRAGMA user_version = 4` and `meta` is seeded with `schema_version='4'`, `wall_model='v6'`, `architecture=7-16-16-2`, `app_version=1.1.0`, `created_at=datetime('now')` — and deliberately **not** `rng_state`/`track_hash`/`rotation_seed`, which are created at lineage start by `ensureLineage()` (or re-pinned by `forkLineage()`). `meta.rng_state` is stored as decimal **TEXT** and is authoritative for resume; `generations.rng_state` is INTEGER, audit-only. `meta.generator_version` is likewise not seeded on fresh init: a lineage reaching its pin writes `'2'` exactly once before curriculum materialization. Both are committed in the same transaction as the generation rows. **No v1/v2/v3 file is migrated** — pre-v4 files are refused with `WallModelChangedError` (§16).

**Read-only inspection example** (the `sqlite3` CLI is the example client; every value below is illustrative):

```sh
$ sqlite3 car_racer-gen42.sqlite
sqlite> .tables
best_weights  driver_results  generations  meta  tracks
sqlite> SELECT generation, best_fitness, avg_fitness, worst_fitness, best_driver_index
   ...>   FROM generations WHERE generation = 42;
42|<committed best>|<avg>|<worst>|<best driver index 0-19>
sqlite> SELECT d.driver_index, d.distance, d.top_speed, d.time_alive, d.crashed, d.stuck
   ...>   FROM driver_results d WHERE d.generation = 42 ORDER BY d.driver_index;
0|...  1|...  2|...   (20 rows, one per driver, driver_index 0-19)
sqlite> SELECT COUNT(*) FROM tracks;
100
sqlite> SELECT key, value FROM meta WHERE key IN ('schema_version','wall_model');
schema_version|4
wall_model|v6
sqlite> PRAGMA user_version;
4
sqlite> PRAGMA integrity_check;
ok
sqlite> PRAGMA foreign_key_check;
sqlite> .quit
```

`COUNT(*) FROM tracks` is **100 for a fresh lineage** — 1 rotation-0 pin row + 99 pre-materialized generated rows; a forked lineage accumulates its fork curriculum on top. `PRAGMA user_version;` on a fresh or adopted file returns `4`. `PRAGMA foreign_key_check;` prints nothing — zero orphans, always (the battery refuses otherwise).

**Do not hand-set `PRAGMA user_version`.** Setting it to 4 on a pre-v4 file does not add `meta.schema_version='4'` or `meta.wall_model='v6'`, and the battery refuses the relabelled file at step 3c. There is no supported in-place v1 → v2 → v3 / v2 → v3 migration in this build. `PRAGMA foreign_keys = ON;` is the app's own default, readback-verified after every adoption.

The `best/avg/worst_fitness` and `driver_results.distance` numbers are signed fitness values (may be negative); `time_alive` is simulated seconds; `crashed`/`stuck` are 0/1 and persist no terminal outcome — `crashed=0, stuck=0` means transient, time-capped, or still running; the outcome is decided at sim time (kill order crash > finished > stuck > time-cap) and surfaced in the HUD/Replay stats and `onFinish.outcome`, never per-row. A run that reaches 100 laps ends with outcome `finished` and leaves both flags 0.

## 9. Tests

Run from this folder:

```sh
node tests/smoke.mjs
node tests/track-clipped.mjs
node tests/rotation-v2.mjs
node tests/rotation-v5.mjs
node tests/rotation-v6.mjs
node tests/db-v6.mjs
node tests/db-v5.mjs
node tests/db-version.mjs
node tests/version-flow.mjs
node tests/f1-curriculum.mjs
node tests/determinism.mjs
node tests/benchmark.mjs
```

All suites are pure Node — no browser, no DOM, no network. They load the app's `js/` files through guarded CommonJS exports and use the vendored `vendor/sql-wasm.js` + `vendor/sql-wasm.wasm` (bound as `{ wasmBinary }`, never fetched).

- `node tests/smoke.mjs` — **76 legs**. Exit 0 prints `SMOKE OK — 76 legs passed`; exit 1 prints the failing assertions. No leg is silently skipped. It covers the NN, the byte-frozen track model and raycasting, the car physics, the signed-fitness pipeline, the evolution/selection/elitism machinery, the persist pipeline, the browser-cache no-regress matrix, the vendor SHA-256 pins, the gen-50 rotation boundary and pin-move arms, fresh v4 schema init (uv=4, `schema_version='4'`, `wall_model='v6'`), the typed byte-preserving refusal arms, the `19 800`-attempt ceiling, and a controller/sensor hairpin falsifier: over all six arc-built templates at d∈{0,50,100}, every centreline sample (11 073 in the final sweep) clears the clipped walls by ≥18 px with 0 crashes and exact grid-equals-brute sensor agreement, plus a bounded 120-step low-speed drive per hairpin.
- `node tests/track-clipped.mjs` — **10 legs**: the constructor invariant (own wall/segment lengths, finite, 9-decimal rounding, `50 − 0.25·d` ladder), zero self-crossings on both clipped walls for all six templates at d∈{0,25,50,75,100}, a 30-vector byte-for-byte wall golden corpus, two-construction byte equality, 10 000 rays with grid == brute, `draw()` when wall length differs from sample length, adversarial `.cartrack` fuzz, the frozen miter + 8-point osculating-arc fallback, the sample budget on actual count, and the `TrackWallError` export/type.
- `node tests/rotation-v2.mjs` — **17 checks**: the two-mode parser allowlist and typed rejection of every other value, strict-boundary rejection of missing/empty fields, parser identity across realms, V1 constants and golden bytes regenerated for `50 − 0.25·d` (at version 1 and via the legacy 2-argument call), V2 golden hashes, resolver shape, version validation, determinism, V2 geometry, the regular 36-gon fallback, and the turn-metric corpus.
- `node tests/rotation-v5.mjs` — **13 checks**: the removal assertion for the old-V2 detector, V1/V2 golden bytes and ladder, independent shape/tangent/timing/never-throws gates over the arc-built F1 templates, and version plumbing.
- `node tests/rotation-v6.mjs` — **21 checks**: the six 91–96-point arc-built templates and their wall-byte goldens, the apex-radius and arm-separation feasibility gates, the dense-spline clearance gate (≥18 px to clipped walls at hw 50/49.75/37.25/25 with a real `Car` at every sample, 0 crashes), and the gate's fail-closed behavior on the old cusp template.
- `node tests/db-v6.mjs` — **16 legs**: fresh uv4 init (`user_version=4`, `schema_version='4'`, `wall_model='v6'`), export → re-adopt at uv4, typed `WallModelChangedError` on every ingress (adoption, init, open, import, cache offer, persist verify), byte-preserving refusal (`refusedBytes` equals the original; the working DB, handle and LSG are untouched), generic refusal for uv∉{1,2,3,4}, and no stale old-symbol references in `js/`.
- `node tests/db-v5.mjs` — **5 legs**: the removal assertion for `V2GeneratorChangedError`, a cross-check that pre-v4 bytes are typed-refused on every ingress, the strict-uint32 `rotation_seed` arm on uv4 V2-pinned bytes, and no-false-positive adoption of valid uv4 V1/V2 files.
- `node tests/db-version.mjs` — **10 legs**: the Db parser alias, `ensureLineage`/`forkLineage` writing `'2'` once, meta-tamper refused at uv4, current-seed V2 rows with retained foreign-seed rows tolerated, V1-by-default resume on uv4, and the version exposed on resume across lineage kinds.
- `node tests/version-flow.mjs` — **9 legs**: a static gate of exactly five `resolveGeneratedTrack` call sites each with an explicit version; Worker, MessageChannel fallback and `Trainer.start` refusing a missing/empty `generatorVersion`; Worker/fallback cores riding the V2 resolver byte-identically; and spectate using the V2 resolver hash at r≥1.
- `node tests/f1-curriculum.mjs` — **9 legs**, standalone evolution regression / falsifier (seed 4242); separate from the 76-leg smoke suite and does not edit any `js/` file. It runs generations 1–50 on the default circuit and 51–250 on the rotation-1 F1 track (`resolveGeneratedTrack(4242, 1, 2)`, hash `9a44a5c9dadd0597`, halfWidth `49.75`) with the real modules. Leg 2 asserts gen-250 best fitness ≥ **38 970** and the mean best over gens 201–250 ≥ **50 000** (and > the gens 51–100 mean); leg 3 requires at least 1 of the 20 trained drivers to complete ≥1 lap on R1 without crashing; leg 4 is the falsifier (stationary and full-gas/hard-steer policies complete 0 laps and fail the clean-lap criterion); leg 5 proves the one-time restart boundary set is exactly **{51}** and that 101/151/201 (and 50/52/100/102) stay exact-elite-clone + 19 mutants by byte-comparing each child; leg 6 proves byte-identical resume at 50→51 and 100→101 plus the four real-lineage boundaries {50,100,150,200}; legs 7–9 check tiebreaks, ranking, and that result objects carry exactly the five fields `crashed/distance/stuck/timeAlive/topSpeed`. Exit 0 prints `F1-CURRICULUM OK — all 9 legs passed`; exit 1 lists the failing legs.
- `node tests/determinism.mjs` — standalone probe; prints the gen-1..3 best-weights JSON on stdout; consumed as two fresh child processes by smoke leg 13.
- `node tests/benchmark.mjs` — **print-only** throughput probe for the headless eval path on the default track. No thresholds, no pass/fail, exit code always 0. Not a smoke leg.

There is no coverage threshold, no browser automation and no E2E suite in this folder; the smoke suite is the contract.

## 10. File inventory

Script/load order (as in `index.html`, which is load-bearing): `vendor/sql-wasm-binary.js` → `vendor/sql-wasm.js` → `js/driver-sim.js` → `js/track.js` → `js/rotation.js` → `js/car.js` → `js/sensors.js` → `js/nn.js` → `js/evolution.js` → `js/db.js` → `js/trainer.js` → `js/replay.js` → `js/main-db.js` → `js/spectate.js` → `js/db-session.js` → `js/editor.js` → `js/main.js`. The Web Worker (`js/train-worker.js`) loads `driver-sim → nn → track → rotation → car → sensors → evolution` itself.

| File | What it is / owns | Position / consumer |
|---|---|---|
| `index.html` | The single page: CSP meta, the four DB buttons + status pill, the persistent banner (`#persist-banner`), `#boot-error`, the five tab sections, script tags in load-bearing order. | Entry point; consumed by everything. |
| `css/style.css` | Dark theme; system font stack; no images, no CDN, no external fonts. | `index.html` only. |
| `js/driver-sim.js` | The single owner of the per-car simulation tick (`DriverSim`) and of the eval-loop constants: `SIM_DT=1/120`, `EVAL_CAP_S=45`, `STUCK_TIMEOUT_S=12`, `PROGRESS_ADVANCE_PX=20`, `LAP_BONUS_TRACK_LENGTHS=2`, `LAP_BONUS_LAPS=100`. It pins the kill order, the lap high-water ratchet and the fitness glossary (§6). | First of the scripts. |
| `js/track.js` | `Track` (uniform Catmull–Rom, byte-frozen clipped walls, 64 px grid raycaster), `cyrb53`, `canonicalJSON`, `TrackWallError`, `DEFAULT_CONTROL_POINTS`, `TRACK_DEFAULT_HALF_WIDTH = 70`, `Track.fromJSON`, `Track.drawTrackBackground`, `Track.VALIDATION`. | After `driver-sim.js`; required by `car.js` at load time. |
| `js/rotation.js` | The procedural-track rotation: `rotationForGeneration`/`difficultyForRotation`, the shared `50 − 0.25·d` half-width ladder, the frozen V1 `generateTrack` (14-point sinusoid, self-intersection rejection, radius-170 fallback), the V2 `generateTrackV2` (six frozen 91–96-point arc-built F1 templates with ≥150° hairpins and the FIT-V6 apex/clearance feasibility gates), the single `parseGeneratorVersion`/`requireGeneratorVersion` pair, the feasibility oracles (`v6ApexRadius`, `v6ArmSeparation`, `v5Valid`, `v6TrackClearance`, `v6SplineRadius`), `V5_TEMPLATES`, and `resolveGeneratedTrack(seed, r, version)` dispatching through `GENERATORS_BY_VERSION`. Pure logic; no DOM, no database. | After `track.js`, before `car.js`; consumed by the Worker, fallback, `main.js` and `db.js`. |
| `js/car.js` | Arcade physics (`MAX_SPEED=600`, reverse cap 150, brake 600, `CAR_RADIUS=14`), collision against clipped wall segments, `driverSeat()` (= car center), rendering, signed odometer. | After `track.js`; used by `driver-sim.js`, `main.js`. |
| `js/sensors.js` | `SensorFan` — the 5-ray fan (`[-70,-35,0,35,70]°`), `SENSOR_MAX=300`, norm = d/300. | After `track.js`, `car.js`; used by `driver-sim.js`. |
| `js/nn.js` | `NeuralNetwork` 7-16-16-2 (`NN_ARCH`), 434 params, `mulberry32`, `strictUint32`, `gaussianFromRng`, `NeuralNetwork.fromJSON`. | After `sensors.js`; used by `evolution.js`, `db.js`, `replay.js`, `trainer.js`. |
| `js/evolution.js` | `Evolution` — population, `selectBest`, `spawnNextGeneration` (elitism + Gaussian mutation), `resumeFromBest`, `wrapToPi`, `POPULATION/MAX_GENERATIONS/MUTATION_RATE/MUTATION_SCALE`. | After `nn.js`; used by the Worker, fallback, `spectate.js`, `main.js`. |
| `js/db.js` | The SQLite persistence layer: `Db` class; the v4 DDL (v3 twins reused verbatim; the wall-model break is `meta.wall_model='v6'`, not a new DDL literal); `WallModelChangedError` with byte-preserving refusal; the adoption battery (`validateAdoptionBytes`); the pre-materialization of rotations 1..99 with the current-seed post-invariant and 19 800-attempt ceiling; the `generator_version` meta arms and `Db.parseGeneratorVersion` alias; the persist pipeline; `forkLineage`/`ensureLineage`/`adoptPinnedTrack`/`resumeState`; `listGenerationSummaries`; `getPersistCost()`; IDB helpers. DOM-free. | After `nn.js`; used by `main-db.js`, `db-session.js`, `main.js`, `trainer.js`. |
| `js/trainer.js` | `Trainer` — main-thread controller for headless training: Worker (or MessageChannel fallback) handshake, logging through `opts.logger`, the LOG-FAILURE degrade policy. Requires `rotationSeed` and `generatorVersion` in the lineage spec; the version is gated by `requireGeneratorVersion`. | After `db.js`; consumed by `main.js`. |
| `js/train-worker.js` | The Web Worker: `importScripts`es the seven pure-logic files, owns the init/resume/start/next/stop protocol, re-derives the block's track from `resolveGeneratedTrack(rotationSeed, rotationIndex, generatorVersion)` once per 50-generation block. No DOM, no DB. | Loaded by `js/trainer.js`; imports the seven files itself. |
| `js/replay.js` | `Replay.loadGeneration` / `Replay.createRun` — read-only replay of a committed generation on its saved track snapshot (unless the labeled override is passed). | After `trainer.js`; used by `main.js`. |
| `js/main-db.js` | The banner system and every DB-lifecycle flow: export/import/create/open/fork/restore/reconnect; the reconcile banner's four actions; the DB status pill. | After `replay.js`; called by `main.js` and `spectate.js`. |
| `js/spectate.js` | Main-thread live eval of the same lineage spec: one `DriverSim` per population net on the train canvas, logging through the same `db.logGeneration` transaction. | After `main-db.js`; consumed by `main.js`. |
| `js/db-session.js` | Boot-time session restore: handle → cache → memory, battery-first everywhere; byte-preserving handling of typed wall-model refusals. Never throws to boot. | After `spectate.js`; used by `main.js`. |
| `js/editor.js` | The in-browser track editor (optional module): pointer editing, halfWidth slider, single-level undo, `.cartrack` import/export, auto-save to `localStorage 'car_racer.track'`, lineage-guarded "Use as training track". | After `db-session.js`; consumed by `main.js`. |
| `js/main.js` | The browser shell: boot contract, Manual mode, Train wiring (headless + spectate, rotation-aware track resolution, integer `generatorVersion` in the session handshake), Replay wiring, Logs table, tabs, rAF loop, boot. Owns the `state` object and the wall-model refusal banner with its verbatim-download action. | Last. |
| `tests/smoke.mjs` | The 76-leg node suite (see §9). | — |
| `tests/track-clipped.mjs` | Byte-frozen clipped-wall suite — 10 legs (see §9). | — |
| `tests/rotation-v2.mjs` / `tests/rotation-v5.mjs` / `tests/rotation-v6.mjs` | Generator/parser suites — 17 / 13 / 21 checks (see §9). | — |
| `tests/db-v6.mjs` / `tests/db-v5.mjs` | Schema-v4 / wall-model-refusal suites — 16 / 5 legs (see §9). | — |
| `tests/db-version.mjs` / `tests/version-flow.mjs` | Version-contract suites — 10 / 9 legs (see §9). | — |
| `tests/f1-curriculum.mjs` | Evolution curriculum regression / falsifier — 9 legs (see §9). | — |
| `tests/determinism.mjs` / `tests/benchmark.mjs` | Determinism probe; print-only throughput probe (see §9). | — |
| `vendor/sql-wasm.js` | sql.js emscripten loader (classic script; declares `window.initSqlJs`). SHA-256 pinned. | After `sql-wasm-binary.js`, before every `js/` file. |
| `vendor/sql-wasm.wasm` | SQLite compiled to WebAssembly. SHA-256 pinned. | Loaded by `js/db.js` — never `fetch`ed at runtime. |
| `vendor/sql-wasm-binary.js` | Base64 embed of the same `.wasm`; exists only so `file://` works with zero fetch. **DO NOT EDIT — regenerate.** | Before `sql-wasm.js`. |
| `vendor/SQL.js.LICENSE.txt` | MIT license for sql.js. | — |
| `README.md` | This file. | — |

## 11. vendor/ + integrity

`vendor/` is checked in on purpose: a fresh clone runs offline with zero install, and the exact engine build is part of the game's identity. Corrupting or swapping a vendor file must never be silent, so the suite pins both files by SHA-256 in `tests/smoke.mjs` (hard fail):

- `vendor/sql-wasm.js` — `9b96118841c6edfdf53c6f388c1419d1f473cf469df6a6fc2864b386d60bdb5e`
- `vendor/sql-wasm.wasm` — `43bbcdd44287e5a2db3b6b5af34b1a3485a7a441285d247976c1ebe3c61f8dbf`

The vendored build is sql.js; the enforcement is the pins above (the vendor files themselves carry no resolvable version string). `vendor/sql-wasm-binary.js` must contain only one leading comment and exactly one `globalThis.SQL_WASM_BINARY_BASE64="…";` assignment (enforced with an anchored regex; the base64 decodes to the pinned wasm hash). `index.html` must load it **before** `sql-wasm.js` — also asserted by the suite.

**DO NOT EDIT anything in `vendor/`.** To change the engine, obtain the new sql.js artifacts from a source you trust, replace the three files, regenerate `vendor/sql-wasm-binary.js` as one comment line plus one assignment line, recompute both SHA-256s and update `VENDOR_SQL_WASM_JS_SHA256` / `VENDOR_SQL_WASM_WASM_SHA256` in `tests/smoke.mjs`, then run the suite — the vendor leg must pass. The remaining legs exercise the engine (schema init, battery, persist, resume byte-identity) and catch API-shape breaks.

## 12. Threat model — honest summary

- **A stolen or leaked `.sqlite` is training data, not credentials.** It contains network weights, per-driver results, run metadata and the full curriculum of generated track rows. There are **no passwords, tokens, keys, API secrets or personal data anywhere in this app or its database**. Distribution of the file leaks weight matrices; it is never an account takeover.
- **Anyone with write access to this folder can swap the app.** Local HTML is code; CSP cannot help if `index.html` itself is hostile. The trust boundary is the filesystem: keep the folder somewhere only you write.
- **The CSP is wasm-scoped**: `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; object-src 'none'; base-uri 'none'`. `'wasm-unsafe-eval'` exists only so sql.js can compile WebAssembly; everything else is denied.
- **The adoption battery is the boundary for foreign bytes.** Any `.sqlite` presented to the app passes the numbered validation steps plus sub-arms (§7) before a byte enters your session; failure means zero adoption and a banner. Hand-crafted files (triggers, views, FK-less tables, out-of-domain values — including tracks rows with non-integer `difficulty` such as **99.9**, over-1000 tracks rows, and every pre-v4 lineage) are refused. Pre-v4 files get a typed, byte-preserving wall-model refusal; there is no migration path. The battery is a validator of *structure and values*, not a sandbox for "trust this file completely".
- **No telemetry, no analytics, no network.** The app makes zero outbound requests; there is nothing to leak and no server to leak it to.
- **Training cost is O(database file), every generation.** Each persist re-exports the whole database, verifies it, writes it to cache/file and re-fingerprints it. The file grows with committed generations; `js/db.js` notes the app-written file can reach **~25–80 MB** at the top of a long lineage, and the adoption pre-parse cap is 64 MiB (`MAX_DB_BYTES`). A maximal lineage may land at or above the import cap, which would make an *exported* full-lineage file un-importable — the in-session database and browser-cache copy are the durable paths at that size; **export early, not only at generation 5000**. No wall-clock is claimed here; per-persist cost is measured report-only (`Db.getPersistCost()`) and eval throughput by `node tests/benchmark.mjs`. Training a full lineage still takes **minutes to hours** on real hardware.
- **`VACUUM` is never issued — a wiped lineage keeps its file size.** The app contains zero `VACUUM` statements and never enables `auto_vacuum`. `forkLineage()` / "Wipe all generations" issues DELETEs; SQLite returns freed pages to its internal freelist — the `.sqlite` does not shrink. **Export DB is not a compaction** (the vendored `export()` copies the image untouched) and **Export → Import reclaims nothing** (the battery only validates bytes). The one compaction that works is external: `sqlite3 car_racer.sqlite 'VACUUM;'` on a copy. The `tracks` table is never deleted by a wipe, so a wipe does not free curriculum rows.
- **Elitism is exact apart from the one-time generation-51 restart, and the system is not self-improving forever**: after the single fresh-net restart, mutation never anneals, there is no crossover and no further re-diversification at 101/151/201, and fitness is the signed odometer plus lap bonus — see §15. Difficulty is display/audit only.

## 13. Failure modes

- **Boot: "Database required"** — `Db.createBrowser()` failed (vendor missing/corrupt, CSP blocked the wasm, atob/WASM failure): `js/main.js` shows `Database required — training disabled (Manual works). Reason: <why> — see README (serve over localhost or fix the vendor files).`, disables Train/Replay/Logs, and **Manual keeps working**. `#boot-error` (missing globals / wrong script order / truncated file) disables every control.
- **Pre-v4 file offered for adoption** — typed, byte-preserving refusal: the `WALL MODEL CHANGED` banner appears with a **Download refused bytes verbatim** button (`refused-lineage.sqlite`); the original file, handle and browser-cache slot are left untouched; nothing is opened as your working DB. The refused file cannot be resumed by this build — export the bytes for your records, then Create / Fork a fresh lineage.
- **File corruption (picked file or cache)** — adoption refusals are fail-closed. The battery refuses over-cap files, zero bytes, non-OK `integrity_check`, orphans, `user_version ∉ {1,2,3,4}`, a uv4 file whose `schema_version`/`wall_model` or `generations`/`tracks` DDL does not match, missing/renamed tables, wrong columns, wrong architecture, over-cap row counts, invalid weights, a missing tip-generation `best_weights`, `track_json` over 256 KiB, extra `sqlite_master` objects, FK mismatches, out-of-domain values (`difficulty` non-integer or outside [0,100] — e.g. **99.9**; `half_width` out of range or BLOB; `crashed=7`, `driver_index=99`, etc.) and any `track_json` that does not recombine to its `track_hash`.
- **Resume track unrecoverable** — a pinned `tracks` row that is absent or whose recombined hash disagrees with the pin is refused with **`ResumeTrackError`** and **no generator fallback**: the persistent banner says the pinned track is not recoverable from this file and that the committed database on disk is untouched.
- **Boundary payload refused** — at a 50-boundary (gen 51, 101, …) logging a payload whose `track_hash` is not the byte-verified resolver output is refused before `BEGIN`; nothing is logged and `meta.track_hash` does not move.
- **`.crswap` litter** — a browser crash during an exclusive write can leave a `*.crswap` beside your `.sqlite`. Deleting it is safe; the app never reads or adopts it.
- **Aliased/foreign paths** — Create DB re-stats the picked path and refuses to write if it is not empty; `createWritable({mode:'exclusive'})` and `navigator.locks` shrink the residual race; Open/Import always go through the battery.
- **Quota / eviction** — IndexedDB may refuse or evict the slots. Symptom: persistent banner `DB NOT PERSISTING — export manually`. The app calls `navigator.storage.persist()` once at boot in cache mode and shows usage/quota in Logs; **only Export DB is a backup**. Clearing site data clears both IndexedDB databases and the cached track.
- **Persistent notice: subtle crypto absent** — when `crypto.subtle.digest('SHA-256', …)` is unavailable, file corruption detection degrades to **size + first/last 4 KiB edge compare** (equal-length corruption between the edges is undetectable). The banner says, verbatim: `Browser without SHA-256 (subtle crypto) — file corruption detection reduced to size+edge compare`.
- **Firefox/Safari** — no File System Access API: **Import .sqlite** works via a hidden file input; **Create DB** creates a fresh browser-cache database; **Open DB** is refused with `This browser cannot open file handles (File System Access API missing) — use Import .sqlite instead`; **Export DB** works everywhere.
- **UNIQUE lineage conflict** — the transaction rolls back, `LineageIntegrityError` is raised, the run halts, the committed lineage is untouched: `LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched`.
- **Worker death** — a Web Worker failure before the first committed generation is respawned once on the MessageChannel fallback; after any commit a backend failure halts the run. On `file://` the worker is refused up front and the in-thread fallback is used transparently.
- **Replay/Resume against a corrupt committed row** — replay rebuilds the track from the row's `track_json`, re-hashes it and refuses on mismatch; best weights are re-validated through `NeuralNetwork.fromJSON`; replayed runs are read-only.

## 14. Rebuilding vendored sql.js

Do this only when the engine build must change; a re-vendor is a pinned-hash change and the suite will hard-fail until you re-pin:

1. Obtain the new sql.js artifacts — `sql-wasm.js`, `sql-wasm.wasm`, `SQL.js.LICENSE.txt` — from a source you trust. The pins defend against accidental local corruption, not a bad source.
2. Replace `vendor/sql-wasm.js`, `vendor/sql-wasm.wasm`, `vendor/SQL.js.LICENSE.txt`.
3. Regenerate `vendor/sql-wasm-binary.js`: exactly one comment line followed by exactly one line `globalThis.SQL_WASM_BINARY_BASE64="<base64 of the new sql-wasm.wasm>";` — nothing else.
4. Compute `sha256sum vendor/sql-wasm.js vendor/sql-wasm.wasm`; update `VENDOR_SQL_WASM_JS_SHA256` / `VENDOR_SQL_WASM_WASM_SHA256` in `tests/smoke.mjs`.
5. Run the suite. The vendor leg must pass; the other legs exercise the engine and catch API-shape breaks. If anything fails, the re-vendor is not done.
6. Keep the script order in `index.html` untouched (`sql-wasm-binary.js` strictly before `sql-wasm.js`) — asserted by the suite.

## 15. Known limitations

- **Fitness is signed — negatives are normal, and there IS a lap counter and a finish state.** A run's fitness is `car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length`. An all-reverse run scores negative fitness and is logged/rendered as-is — the schema dropped the non-negative-fitness CHECK on `generations.best_fitness` because fitness may be negative and must round-trip. At `LAP_BONUS_LAPS = 100` laps the run ends with outcome `finished`; the bonus is paid once per lap boundary via the ratcheted `maxCumArc` high-water — 2 × `track.length` per lap, capped at 100 laps — so an oscillator or infield flip-flop whose cumArc never grows pays nothing. The only anti-farming control is the stuck-kill (12 simulated seconds without +20 px of net forward centerline progress), which bounds *no-net-progress* driving only. The fitness bonus dwarfs the lap length, so a lineage can settle onto bonus-farming behavior; the fitness function was not designed to teach cornering technique. A full training run still takes **minutes to hours** on real hardware.
- **Wall-hugging micro-circles specifically survive** when they keep the "+20 px net progress per 12 simulated seconds" needle above the kill threshold while advancing nonsense — they earn odometer fitness, just not the lap bonus. The stuck-kill does not help there.
- **Lap accounting is a high-water ratchet** — `laps` is derived from `maxCumArc`, not from laps physically crossed; backwards driving after a high-water never un-pays an already-banked lap's bonus. It is simple, deterministic and gameable; it is not a race judge.
- **No crossover** — children are mutated clones of the single best parent, with one exception: the generation-51 full restart spawns all 20 nets fresh from the RNG stream (elite clone dropped). There is no recombination, and no diversity maintenance after generation 51.
- **Mutation never anneals** — `MUTATION_RATE = 0.25`, `MUTATION_SCALE = 0.30`, fixed forever; generation 51 is the only diversity injection, so late-generation diversity plateaus by construction. Rotation boundaries at 101/151/201 do **not** re-diversify.
- **Selection is exact and elitist** — child 0 is an exact clone except at the generation-51 restart; expect one lineage, occasionally boring. Resume is anchored on the same cloned child-0 weights (or the validated `rngState` cursor at the restart) — byte-identical resume is the feature.
- **Spectate is main-thread** — 20 cars step inside the rAF loop; it drops frames on slow machines rather than slowing the sim. Headless mode is the throughput path.
- **Replay uses the generation's saved `track_json`, never the current editor/active track** — unless the labeled "Use active track" toggle is ticked. The snapshot is re-hashed and cross-checked against the row's `track_hash` first. For gen 51+ it is the generated track the generation actually ran on, not the pinned one.
- **"Use as training track" is a lineage fork, destructively.** With committed generations the confirm names the number and states that the new lineage starts at rotation 0 (difficulty 0) and switches to the generated schedule from rotation 1 (generation 51). With zero committed generations it silently re-pins.
- **Resume is latest-only** — you cannot branch from an intermediate generation; the UNIQUE constraint enforces the immutable prefix. Resume adopts the stored tip `tracks` row, never the editor/Manual track or a generator output.
- **file:// (and multi-tab) semantics are browser-defined and coarse** — the `file://` origin can be shared with other local pages, IndexedDB is per-origin, and the cache slot's write end is only as trustworthy as the least careful local page in your profile. Multi-tab writes are serialized by `navigator.locks` where available plus the in-tab promise chain, with the tripwire as backstop and a blocking Reconcile banner as the failure mode. **localhost is the recommended origin** and still not a security boundary.
- **Determinism is per engine, not universal** — same process/engine byte-identity only; no cross-machine or cross-version promise. The rotation resolver inherits the same property.
- **Resume across spawn-mechanism versions can diverge** — a v4 database whose committed `rng_state` was written by an older build that re-diversified at *every* 50-boundary will not resume byte-identically once the resumed target is past generation 51 (101/151/…): this build spawns those targets as ordinary exact-clone + mutant generations instead of fresh-net restarts. This changes continued training only; the file is still adopted and validated normally. See `tests/f1-curriculum.mjs` leg 5 for the current boundary set.
- **`nearestCenterline` is O(#samples) per tick** — the default circuit has roughly 500 samples; at 120 Hz that is on the order of 60 000 comparisons per second per car; fine in practice, not sub-linear (the raycaster, by contrast, is grid-accelerated).
- **No maximum lineage length in practice, but a hard evidence ceiling in the battery** — adoption allows at most 100 000 generation rows, 2 000 000 driver-results rows and **1000 `tracks` rows** (`MAX_TRACK_ROWS`, an import-refusal bound, never a deletion bound: fork-history is permanent), plus a 64 MiB file cap (`MAX_DB_BYTES`). An exported full-lineage file may approach or exceed the cap and then cannot be re-imported.
- **Pre-existing out-of-domain NaN/Infinity or hand-crafted files can only be caught at adoption or at write time** — the battery validates structure and domains on the way in, but cannot rewrite history.
- **The database is rewritten whole, every generation** — lineage size and disk bandwidth grow together; the `tracks` table is copied with it.
- **The browser-cache safety net is two slots, and only one of them is durable.** The LSG slot holds only the last verified-good copy; the lineageSnapshot slot only catches a strict-ahead snapshot when LSG validation fails. Clearing site data destroys both. **Only Export DB is a backup.**
- **Wipe is absolute.** "Wipe all generations — start a NEW lineage" deletes everything in one transaction; there is no trash can or undo. `tracks` rows are not deleted by the wipe.
- **The editor's half-width slider max is 2000 px** and the sample budget is 20 000 samples — tracks needing more are refused at construction (intentional).
- **Changing the NN at all** invalidates every stored weight payload — `NeuralNetwork.fromJSON` refuses anything but exactly `'7-16-16-2'`.
- **Difficulty is display/audit only** — a file whose difficulties lie (e.g. **99.9**) is refused at adoption, but a log line that disagrees with the ladder is a display bug, not a sim bug.
- **Rotation stops at the generation cap** — 5000 generations ends at rotation 99; difficulty never exceeds `min(99, 100) = 99` within a capped lineage.
- **Cross-engine byte identity is not promised** — resume and resolver byte-verification are per-engine.

## 16. Schema v4 and the wall-model break

Fresh databases are **v4 from birth**: `PRAGMA user_version = 4`, `meta.schema_version = '4'`, `meta.wall_model = 'v6'`, `app_version = '1.1.0'`, and the `generations` DDL carries no non-negative-fitness CHECK and the declared `REFERENCES tracks(track_hash)`. The schema DDL is the v3 DDL reused verbatim; the wall-model break is carried by `meta.wall_model = 'v6'`, not by a new DDL literal. Fresh init runs the post-COMMIT verified set on its own path — one uniform gate for every instance the factory produces.

**Pre-v4 files are refused, not migrated.** In `Db.validateAdoptionBytes` step 3b, `user_version ∈ {1,2,3}` throws a typed `WallModelChangedError` carrying the exact refused bytes (`err.refusedBytes`), before any scrub or DDL arm. `user_version ∉ {1,2,3,4}` is a foreign file and throws a generic `ValidationError`. Only uv==4 proceeds, and it must additionally carry `schema_version='4'` and `wall_model='v6'` (step 3c). The same typed refusal is enforced at every other ingress — `init(bytes)`, `openExisting()`, `importBytes()`, the cache offer (`restoreFromLastKnownGood`) and the `verifyForPersist` step-2 check — so a downgraded file can never be adopted or written back. `_migrateIfNeeded` on a uv<4 file throws a `MigrationError` rather than migrating; that path is unreachable from the battery in normal use.

**Refusal is byte-preserving.** The refused bytes ride on the error, no storage is cleared, and `js/main.js` / `js/db-session.js` map the error to the persistent `WALL MODEL CHANGED` banner with the **Download refused bytes verbatim** action. The original `.sqlite`, the connected handle and the browser-cache slot are left untouched. To continue, export the refused bytes for your records and Create / Fork a fresh lineage; the old file cannot be resumed by this build.

**Why there is no migration.** The v1/v2/v3 wall model encoded a different drivable region than the byte-frozen clipped walls this build constructs. A stored `track_json` from the old model cannot reproduce the same collision geometry, so silently upgrading it would change a committed lineage's behavior. Refusing it byte-for-byte is the fail-closed choice. The adoption battery's remaining arms (FKs, value domains, DDL twins, recombination, self-hash) are exercised on uv4 fixtures by `tests/smoke.mjs` and `tests/db-v6.mjs`.

## 17. Scope and references

Everything in this README is verifiable inside this folder alone: the code (`index.html`, `css/`, `js/`, `tests/`, `vendor/`) is the only source consulted. No external build systems, no hidden config, no services, no references to files outside this app. This README deliberately contains no roadmap and no process notes — it documents what the code does, nothing else. If you find a divergence between this file and the code, the code is correct and this file is stale: fix the README, not the facts.

## 18. Glossary

- **adoption battery** — the validation gauntlet `Db.validateAdoptionBytes` every foreign-bytes adoption passes through before anything touches your session. It refuses structurally foreign files, semantically wrong files and every pre-v4 lineage (the latter with a typed, byte-preserving wall-model refusal).
- **apex radius / arm separation** — the minimum circumradius over a 3-step control-point window (`v6ApexRadius`) and twice that value; the generator requires radius ≥ 75 and `2R > 2·halfWidth + 20` so the clipped inner wall cannot merge at a hairpin.
- **byte-frozen clipped wall** — the single canonical `leftWall`/`rightWall` pair produced by folding each raw offset polyline into one simple closed polyline, with 9-decimal rounding and canonical-length alignment; rebuilding a track of the same geometry yields byte-identical walls.
- **clipped-wall clearance** — the minimum distance from every centreline sample to every clipped wall segment; the generator gate requires ≥ 18 px (`CAR_RADIUS + 4`).
- **curriculum** — rotations 1..99 pre-materialized into `tracks` at pin/fork time, each byte-verified against `resolveGeneratedTrack(seed, r, version)` before insertion.
- **Db.offline mode** — the `DbOfflineError` state ("Database required"): `Db.createBrowser()` failed; Train/Replay/Logs are disabled and Manual keeps working.
- **difficulty (display)-audit** — rendered for humans (Logs, the `R r · d d` tag) and audited by the battery, but never consumed by Selection, spawning or physics.
- **difficulty ladder** — `difficultyForRotation(r) = min(r, 100)` and `halfWidth(d) = 50 − 0.25·d` are shared by both generators. V1 adds `harmonics(d) = 2 + floor(d/25)` and `amplitude(d) = 0.08 + 0.34·(d/100)`.
- **file:// caveats** — the `'null'` origin can be shared per-browser; IndexedDB is per-origin; the worker is refused; the File System Access pickers throw; **localhost is the recommended origin**.
- **generator version** — `meta.generator_version`, the per-lineage selector: absent ⇒ V1, `1`/`'1'` ⇒ V1, `2`/`'2'` ⇒ V2. Written once before materialization as `'2'` for fresh/fork lineages; parsed by the single two-mode parser in `js/rotation.js`, re-exported as `Db.parseGeneratorVersion`.
- **lineageSnapshot** — the second (strictly-ahead-only) browser-cache recovery slot in `car_racer.lineageSnapshot`, used only when LSG fails validation and the snapshot is provably newer.
- **load-bearing asserts** — assertions whose failure halts the run or adoption rather than degrading it: the rotation-divergence assert (`FIT_V3_ROTATION_DIVERGENCE`), the boundary resolver-equality arms, and the adoption battery's typed-refusal arms.
- **lastKnownGood (LSG)** — THE single browser-cache slot (`car_racer` / `kv` / `car_racer.lastKnownGood`); full bytes of the last verified-good DB plus a tag; no-regress rule: an equal-or-ahead tag is never overwritten.
- **pin** — `meta.track_hash`: the lineage's rotation-0 starting track. Its `tracks` row carries `difficulty = 0`.
- **pin move** — at a 50-boundary the only legal next track is the byte-verified resolver output; `meta.track_hash` moves inside the single logging transaction.
- **quota / VACUUM** — IndexedDB is browser-managed and evictable; `VACUUM` is never issued by the app, exports do not compact, and wiped lineages keep their file size; the only compaction is external (`sqlite3 file 'VACUUM;'` on a copy).
- **resume "latest-only"** — resume continues from the latest committed generation; earlier prefixes are immutable and the UNIQUE constraint enforces it.
- **rotation index** — `rotationForGeneration(N) = floor((N − 1) / 50)`; the generation's position in the 50-generation blocks (0..99 across the cap).
- **rotation seed** — `meta.rotation_seed` (uint32), set once at pin time; deterministically derives every generated track; resume re-derives tracks from it, never re-rolls them.
- **TrackWallError** — a `ValidationError` subtype thrown for any violation of the byte-frozen clipped-wall contract; construction fails closed.
- **untrusted-import policy** — foreign bytes pass the adoption battery before they can touch the session; the battery is a validator of structure and values, not a sandbox.
- **verified set** — the post-COMMIT assertion battery `_assertMigratedState` that runs at every fresh init and every adoption: FAIL ⇒ the instance is discarded, zero adoption.
- **wall-model refusal** — the typed `WallModelChangedError` + byte-preserving banner + verbatim-download action taken for every pre-v4 lineage; no migration is performed.
- **.crswap** — Chromium's exclusive-`createWritable` staging temp file; never adopted, never listed, deletable.
