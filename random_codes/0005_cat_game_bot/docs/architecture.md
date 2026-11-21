# Architecture (as implemented)

This document describes the code in this repository only. Module names,
control flow, schemas and limits below are taken from the source files in
`cat_game_bot/`.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `main.py` | Thin wrapper: `raise SystemExit(main())` from `cat_game_bot.__main__`. |
| `cat_game_bot/__main__.py` | Canonical CLI. `parse_args` → `InteractionLog` → `_build_session` → `Session.run`; maps terminal results to exit codes. |
| `cat_game_bot/config.py` | Frozen `Config`, CLI parsing, numeric bounds, path confinement, region/offset JSON validation, canonical JSON encoder, `PROJECT_ROOT`, `SCHEDULE_PATH`, canonical id tuples. |
| `cat_game_bot/adb.py` | `AdbClient`: validated serial/package, discrete ADB subprocess calls with deadlines/output caps, bounded PNG/JPEG screenshot decoding, `is_connected`, `discover`. |
| `cat_game_bot/vision.py` | `TemplateLibrary` (bounded, confined manifest + images), `TemplateMatcher` (OpenCV correlation + IoU non-max suppression), `dominant_color`, `color_distance`. |
| `cat_game_bot/detection.py` | `ScreenDetector`: ordered state rules over template matches, LOADING brightness check; marker-name constants. |
| `cat_game_bot/ocr.py` | `OcrReader` protocol, `TesseractReader` (pytesseract wrapper), `parse_count`, `parse_duration`. |
| `cat_game_bot/inventory.py` | `InventoryReader`: reads all 23 canonical Bag counts from icon matches + adjacent count OCR. |
| `cat_game_bot/targets.py` | `TargetSpec`, `TARGET_IDS`, `load_targets`, `goal_met`, `verify_initial_state`, `item_template`. |
| `cat_game_bot/crafting.py` | `load_tree`, `load_schedule`, `CraftPlanner` (peek/advance cursor), `batch_cost`, `dialog_ready`, section map. |
| `cat_game_bot/gift.py` | `GiftCollector`: timer readiness, coin reads, gift-icon tap plan, coin-delta verification. |
| `cat_game_bot/navigator.py` | `Memory` (16-pixel-quantified visited matches, no-progress counter), `Navigator` (single action per tick). |
| `cat_game_bot/session.py` | `Session`: settle loop, three-phase gift protocol, action loop, memory transition, stop conditions, failure accounting. |
| `cat_game_bot/db.py` | `InteractionLog`: confined SQLite file, locked schema, canonical JSON payloads, parameterised writes. |
| `cat_game_bot/models.py` | Immutable data containers, enums and `Screen`. No I/O. |
| `cat_game_bot/errors.py` | `AdbError`, `ConfigError`, `VisionError`, `DbError`. |

Dependency direction is one-way for the leaf modules (`models.py`,
`errors.py` have no project imports). `config.py` imports from `adb.py`,
`errors.py` and `models.py`. Higher layers (`vision`, `inventory`, `crafting`,
`db`) import `config.py` helpers; `session.py` orchestrates the collaborators
and imports `config._json_default`, `models`, `navigator.Memory` and
`targets`.

## Control flow

1. `main(argv)` calls `_run(argv)` inside a `try`/`except DbError` and maps the
   result: `GOAL_MET` → `0`, any other result → `1`, `DbError` → `2`.
2. `_run` calls `parse_args(argv)`, opens `InteractionLog(cfg.db_path)`, then
   builds the session in a `try`/`finally` that always closes the log.
3. `_build_session` wires the graph: `AdbClient(serial)`;
   `TemplateLibrary(template_dir)` + `load(manifest.json)`;
   `TemplateMatcher(library, threshold)`; `ScreenDetector(matcher)`;
   `TesseractReader(tesseract_cmd, ocr_lang)`; `load_regions(ocr_regions_path)`;
   `InventoryReader(ocr, matcher, offsets, library)`;
   `load_targets(targets_path)`; `load_tree(tree_path, tree_path.parent)`;
   `CraftPlanner(tree, spec, {}, schedule_path=SCHEDULE_PATH)`;
   `GiftCollector(ocr, regions)`; `Navigator(detector)`; and finally `Session(…)`.
4. `Session.run()` opens a DB session with placeholder result `INTERRUPTED`,
   checks `adb.is_connected()` (returns `DISCONNECTED` if false), and in live
   mode launches `cfg.package`. In dry-run mode `launch` is skipped.
5. Each loop iteration:
   - `settle()` captures screens until the state plus sorted interactive-match
     signature `(name, x//16, y//16)` is stable across two consecutive polls;
     LOADING never counts as stable. On `settle_timeout` a `SETTLE_TIMEOUT`
     event is logged and the last pair is returned. Screenshots and detection
     still happen in dry-run mode.
   - Counters increment: `screenshots`, `states_seen[state]`. A
     `log_observation` row is written with a 16-hex-character SHA-256 prefix of
     the frame pixels.
   - Gift phases 1–3 run (all guarded by `not dry_run`).
   - On `BAG_DIALOG`, `inventory.read()` refreshes `self.counts` and
     `log_goal_progress` is called for all 11 `TARGET_IDS`.
   - `goal_met()` may set `GOAL_MET` and break.
   - A periodic Bag visit is inserted on HOME/CRAFTING when
     `actions > 0` and `actions % bag_visit_interval == 0`.
   - Otherwise one general action runs: collect a `craft_ready_check` tile
     (no schedule step consumed), place the peeked schedule step, or fall back
     to `Navigator.choose()`.
   - Memory marks matches visited at 16-pixel quantisation; a fresh match or
     state change resets `no_progress`, otherwise it increments.
   - Stop checks in order: `MAX_ACTIONS`, `NO_PROGRESS`, `ERROR_LIMIT`,
     `DISCONNECTED`, `DURATION`.
6. Exceptions: `KeyboardInterrupt`/`SystemExit` → `INTERRUPTED`;
   `AdbError`/`VisionError`/`OSError` or `DbError` → `ERROR_LIMIT` with a
   `RUN_FAULT` event. The `finally` block sets timing/result, calls
   `close_session`, then `close()` (idempotent).
7. `_execute()` makes no device call in dry-run (returns a successful
   `ActionResult` with detail `dry_run simulated`). In live mode it dispatches
   TAP → `adb.tap`, SWIPE → `adb.swipe`, BACK → `adb.back`, WAIT →
   `time.sleep(poll_delay)`, converting `AdbError`/`VisionError`/`OSError` into
   failed `ActionResult`s. Failed results increment `errors` and log an
   `ACTION_FAILURE` event.

`Navigator.choose()` precedence: GIFT_DIALOG → exactly one Collect tap (matched
`gift_collect`, else hardcoded `Point(540, 1625)`); POPUP/DIALOG → BACK
(`dismiss`, so the screen beneath stays open); BAG_DIALOG → BACK
(`close_bag`); RECIPE_DIALOG/CATALOG → `close_dialog*` tap, else first
unvisited match, else BACK; then supplied planner action; then first
unvisited match with target items sorted first; finally WAIT (`idle`).

Gift protocol in `Session.run()`:

1. **Open** — off the dialog, `gift.is_ready()` true → `read_coins()`; readable
   before value is stored and `gift.plan()` (tap at `gift_timer` region centre,
   reason `gift_icon`) is executed; unreadable → logged `gift_coins_unreadable`
   failure.
2. **Collect** — on GIFT_DIALOG with no prior collect: establish `before` if
   needed (unreadable → log `gift_no_before` and attempt `close_dialog` tap),
   then accept only a TAP whose reason is `gift_collect`; execute it once and
   set `_gift_collect_tapped`.
3. **Verify** — on a later non-dialog tick: `gift.verify(before, after)`;
   success → `log_gift` plus `gifts_collected += 1`; failure → logged
   `gift_verify_failed`; pending state cleared either way.

`CraftPlanner` cursor semantics: `next_step()` is a pure peek; the session
holds `_step` and `_step_remaining` and decrements per successful tap.
`mark_placed(item)` is the only mutation: it verifies the item matches the
current step (else `ConfigError`), advances within the row or to the next
populated row, and skips empty schedule rows.

## Data contracts

### In-memory types (`models.py`)

- `Point(x, y)`, `Rect(x, y, width, height)` — frozen; `Rect.center()` uses
  floor division, `contains()` is half-open, `area()` is width×height.
- `Screen(pixels: np.ndarray)` — BGR frame with `width`/`height` properties;
  `eq=False`.
- `Match(name, rect, score)` — frozen.
- `Detection(state: GameState, matches: tuple[Match, …], timestamp: float)` —
  frozen.
- `GameState` values: `UNKNOWN`, `LOADING`, `TAP_TO_START`, `TUTORIAL`, `HOME`,
  `CRAFTING`, `CATALOG`, `POPUP`, `RECIPE_DIALOG`, `BAG_DIALOG`, `GIFT_DIALOG`,
  `DIALOG`.
- `ActionKind` values: `TAP`, `SWIPE`, `WAIT`, `BACK`, `STOP`.
- `Action(kind, point, end, reason)` and
  `ActionResult(action, ok, detail)` — frozen.
- `GiftResult(ok, delta, reason)`, `InventoryItem(name, count, rect)`,
  `CraftStep(item, section, template, batch, cost)`,
  `SessionStats(actions, screenshots, errors, gifts_collected, states_seen,
  started_at, finished_at, result)`.

`config.Config` is a frozen dataclass holding the validated CLI values listed
in `README.md`. `targets.TargetSpec` is frozen with `items` (11 goal counts),
`initial_coins` and `initial_stash`.

### File schemas

- **`templates/manifest.json`** — object with `version == 1` (int, not bool)
  and `templates` object mapping name → filename. At most 200 entries; every
  filename resolved under the template directory and symlink-rejected.
- **`data/ocr-regions.json`** — top-level keys exactly `version`, `regions`,
  `offsets`; `version == 1`. `regions` keys exactly `coins`, `gift_timer`, each
  `{x, y, w, h}` strict ints. `offsets` keys exactly `item_count.<id>` for all
  23 canonical ids, each `{dx, dy, w, h}` strict ints.
- **`data/initial-state.json`** — object with `targets`, `initialCoins`,
  `initialStash`. `targets` must contain all 11 `TARGET_IDS` with non-negative
  ints; `initialCoins` a non-negative int; `initialStash` values non-negative
  ints. Extra stash keys are retained.
- **`data/crafting_tree.yaml`** — mapping with an `items` list; only dict
  entries are indexed by `id`. `recipe.coin_cost` (int or ≤12-digit ASCII
  decimal string) is bounded to `0…10**12`. Section names come from an item's
  `section` field, else `SECTION_BY_ITEM`, else `CRAFTING`.
- **Schedule CSV** (`SCHEDULE_PATH` constant) — header exactly `tick` followed
  by the 19 craftable ids in canonical order; `tick` values sequential from
  zero; every cell a non-negative ASCII decimal integer; file ≤ 8 MiB.

### Canonical JSON encoding

`config._json_default` serialises `Enum` → `.value`, `Path` → `str`, dataclasses
→ `asdict`, anything else → `str`. `db._canonical` dumps with
`ensure_ascii=True, sort_keys=True` and that default; the same default is used
for the session's `config_json` dump.

## SQLite schema (`cat_game_bot/db.py`)

```
sessions(id INTEGER PRIMARY KEY AUTOINCREMENT,
         started_at REAL NOT NULL, finished_at REAL,
         config_json TEXT NOT NULL, result TEXT NOT NULL)

events(id INTEGER PRIMARY KEY AUTOINCREMENT,
       session_id INTEGER NOT NULL REFERENCES sessions(id),
       ts REAL NOT NULL, event TEXT NOT NULL, state TEXT NOT NULL,
       detail_json TEXT NOT NULL)
CREATE INDEX idx_events_session_ts ON events(session_id, ts)

actions(id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        ts REAL NOT NULL, kind TEXT NOT NULL, target TEXT,
        x INTEGER, y INTEGER, ok INTEGER NOT NULL, detail TEXT NOT NULL)
CREATE INDEX idx_actions_session_ts ON actions(session_id, ts)

observations(id INTEGER PRIMARY KEY AUTOINCREMENT,
             session_id INTEGER NOT NULL REFERENCES sessions(id),
             ts REAL NOT NULL, state TEXT NOT NULL,
             matches_json TEXT NOT NULL, counts_json TEXT NOT NULL,
             screen_hash TEXT NOT NULL)
CREATE INDEX idx_observations_session_ts ON observations(session_id, ts)

goal_progress(id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL REFERENCES sessions(id),
              ts REAL NOT NULL, item TEXT NOT NULL, count INTEGER NOT NULL,
              required INTEGER NOT NULL, met INTEGER NOT NULL)
CREATE INDEX idx_goal_session_ts ON goal_progress(session_id, ts)

gifts(id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL REFERENCES sessions(id),
      ts REAL NOT NULL, coins_before INTEGER NOT NULL,
      coins_after INTEGER NOT NULL, delta INTEGER NOT NULL)
CREATE INDEX idx_gifts_session_ts ON gifts(session_id, ts)
```

All DDL uses `IF NOT EXISTS`. Writes are parameterised and committed
individually. For SWIPE actions the endpoint is encoded into `target` as
`"x,y"`; for other actions `x`/`y` come from `point`. `ok` is stored as an
integer. `log_gift` stores `delta = after - before`.

## Validation implemented in code

| Area | Enforcement |
| --- | --- |
| CLI / config | `parse_args` range-checks duration, max actions, Bag interval, threshold, poll delay, settle timeout, no-progress and error limits; `--serial` and `--package` use allow-list regexes; all paths pass `_resolve_path` (symlink rejected, must resolve under the fixed root/dir); log dir created `0700`. |
| OCR region / offsets | `_parse_region` requires exact keys, strict ints, non-negative coordinates, positive extents, bounds `x+w≤1080`, `y+h≤2400`, area `≤2592000`; `_parse_offset` requires exact keys, `dx,dy≥0`, `0<w≤200`, `0<h≤100`. |
| Screenshot | `adb.py`: PNG signature/SOF scan, dimension checks `1080×2400` and `2592000` pixels, `cv2.imdecode` failure → `VisionError`; command timeout 10 s, stdout cap 32 MiB, `shell=False`. |
| Templates | `vision.py`: manifest version 1, file caps (1 MiB manifest, 2 MiB per image, 2.592M pixels), path confinement, PIL decompression-bomb guards, threshold and IoU suppression with 3000-candidate bound. |
| Targets / tree | `targets.load_targets` and `crafting.load_tree` require exact top-level shapes and non-negative int counts; `_recipe_cost` bounds coin cost. |
| Schedule | 8 MiB cap, exact header, sequential ticks, strict non-negative decimal cells. |
| Gifts | Positive delta and `MAX_COIN_DELTA = 100000`; both balances must parse. |
| Database | Path confined to `<root>/logs`, directory `0700`, file `0600`, parameterised SQL, `:memory:` test path. |
| Runtime stops | Action/no-progress/error/limit, connectivity, duration checks in `Session.run`; exception paths log `RUN_FAULT` and terminate with `ERROR_LIMIT`. |

Errors are typed: `ConfigError` (bad config/data), `AdbError` (ADB failures),
`VisionError` (capture/decode/template failures), `DbError` (log failures).
