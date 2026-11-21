# Cat Game bot

A deterministic bot for the *Cat Game!* Android mobile game. It drives a device
or emulator through `adb`, recognises screens with OpenCV template matching and
Tesseract OCR, reads Bag counts, places crafts from a fixed schedule, collects
gift boxes only when a coin-delta check passes, and records every tick in a
SQLite database. Device actions are disabled by default (`--live` enables them).

## What is implemented

- **Perception.** Screenshots are taken with `adb exec-out screencap -p`,
  normalised for shell CR translation, checked for PNG/JPEG headers and bounded
  dimensions, then decoded to a BGR pixel array. Screen state is classified by
  an ordered, first-match-wins rule table over template matches (plus a
  brightness check for the near-black loading screen). OCR reads fixed screen
  rectangles for coins, the gift timer and item counts.
- **Crafting.** A fixed CSV schedule supplies the exact item order and batch
  sizes. A planner peeks the next step; only a successful placement advances
  its cursor. Completed craft tiles are collected without consuming a schedule
  step.
- **Gifts.** A three-phase protocol reads coins before opening the gift, taps
  Collect once on the gift dialog, then verifies the coin delta on a later
  tick before writing a gift row to the database.
- **Logging.** Each session writes an initial session row, one observation per
  settled screen, actions, goal progress for all 11 goal items, events and
  verified gifts to a confined SQLite file under `logs/`.

All default paths resolve relative to the directory containing the
`cat_game_bot/` package (the project root), not to the current working
directory.

## Install

```sh
cd /run/media/helder/Code_Dump/all_projects_all_branches/crazy_ideas_main/random_codes/0005_cat_game_bot
uv venv .venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` pins the runtime and test dependencies, including
`opencv-python`, `numpy`, `pillow`, `pytesseract`, `PyYAML`, `pytest` and
`pylint`.

**Tesseract.** The default OCR binary is `<project root>/.tools/bin/tesseract`.
That executable ships in the workspace's `.tools/` conda-style tree (`.tools/`
is git-ignored); point `--tesseract-cmd` elsewhere if your binary lives
somewhere else. OCR is invoked with language `eng` by default and page
segmentation mode 7.

**ADB.** The code calls the `adb` executable by name with `shell=False`, so
`adb` must be on `PATH`. A device serial is always required as a CLI argument.
`AdbClient.discover()` can list devices in the `device` state but is not wired
to the CLI.

## Run

```sh
# Dry run: analyse screens and log, but do not launch the app or send taps/swipes/back.
python main.py --serial <adb-serial>

# Live run: execute the planned device actions.
python main.py --serial <adb-serial> --live

# Equivalent package entry point.
python -m cat_game_bot --serial <adb-serial>
```

`main.py` is a thin wrapper that calls `cat_game_bot.__main__.main`. A live run
launches `--package` via `adb shell monkey`, then loops until a goal or stop
condition is reached. Dry-run mode still takes screenshots and checks
connectivity, but skips `launch` and all `tap`/`swipe`/`back` device calls;
planned actions are recorded as simulated successes.

The planner loads a fixed schedule CSV from the `SCHEDULE_PATH` constant in
`cat_game_bot/config.py` (an absolute path to `0004_cat_game_neat/data/best-schedule.csv`
outside this project). There is no CLI flag for it. The file must exist and
match the strict CSV schema or session construction fails with `ConfigError`
before the loop starts.

## CLI flags and defaults

All flags are defined in `cat_game_bot/config.py::parse_args`. Numeric values
are range-checked and all filesystem paths are confined under a fixed
subdirectory of the project root; symlinks are rejected.

| Flag | Default | Range / validation |
| --- | --- | --- |
| `--serial` | required | Allow-listed `[A-Za-z0-9][A-Za-z0-9._:-]*`. (The frozen `Config` dataclass has a `emulator-5554` fallback, but the CLI requires the flag.) |
| `--package` | `com.minogames.cats.client` | Dotted-package allow-list `[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+`. |
| `--duration` | `300.0` seconds | `1.0`–`3600.0`. |
| `--max-actions` | `500` | `1`–`10000`. |
| `--bag-visit-interval` | `10` actions | `1`–`100`. |
| `--templates` | `<root>/templates` | Confined under `<root>/templates`. |
| `--threshold` | `0.8` | `0.5`–`1.0`. Per-template score overrides apply to `craft_ready_check` (`0.97`) and `gift_collect` (`0.45`). |
| `--log-dir` | `<root>/logs` | Confined under `<root>/logs`; created with mode `0700`. |
| `--poll-delay` | `1.0` seconds | `0.2`–`5.0`. |
| `--settle-timeout` | `12.0` seconds | `2.0`–`60.0`. |
| `--live` | off (dry run is the default) | `store_false` on `dry_run`; with the flag, device actions execute. |
| `--no-progress-limit` | `15` ticks | `1`–`100`. |
| `--error-limit` | `5` errors | `1`–`20`. |
| `--tesseract-cmd` | `<root>/.tools/bin/tesseract` | Confined under `<root>/.tools`. |
| `--targets` | `<root>/data/initial-state.json` | Confined under `<root>/data`. |
| `--tree` | `<root>/data/crafting_tree.yaml` | Confined under `<root>/data`. |
| `--ocr-regions` | `<root>/data/ocr-regions.json` | Confined under `<root>/data`. |
| `--db` | `<root>/logs/bot.sqlite3` | Confined under `<root>/logs`. SQLite file forced to mode `0600`. |
| `--ocr-lang` | `eng` | Passed to Tesseract as `-l`. |

Invalid values, unsafe paths or an uncreatable/insecure log directory raise
`ConfigError`.

## Exact safety caps implemented in code

- **Config ranges:** duration `1.0`–`3600.0` s; max actions `1`–`10000`; Bag
  interval `1`–`100`; template threshold `0.5`–`1.0`; poll delay `0.2`–`5.0` s;
  settle timeout `2.0`–`60.0` s; no-progress limit `1`–`100`; error limit
  `1`–`20`.
- **Screenshot:** width ≤ `1080`, height ≤ `2400`, total pixels ≤ `2592000`;
  PNG/JPEG only; per-command ADB timeout `10.0` s; stdout cap `33554432` bytes
  (32 MiB); `shell=False` with discrete arguments.
- **OCR regions:** coordinates ≥ `0`, extent > `0`, `x+w ≤ 1080`, `y+h ≤ 2400`,
  area ≤ `2592000`. Count offsets: `dx,dy ≥ 0`, `0 < w ≤ 200`, `0 < h ≤ 100`.
- **Template library:** at most `200` templates; manifest ≤ `1048576` bytes;
  each template image ≤ `2097152` bytes and ≤ `2592000` pixels; manifest
  version must be `1`; all paths confined and symlinks rejected; decompression
  bombs rejected.
- **Template matching:** correlation threshold `0.8` by default, IoU
  suppression `0.5`, non-max suppression bounded to the top `3000` candidates
  in `64×64` spatial cells.
- **Schedule:** file ≤ `8` MiB; header must be `tick` followed by the 19
  craftable ids in canonical order; ticks sequential from zero; every cell a
  non-negative ASCII decimal integer.
- **Recipes:** `coin_cost` must be a non-negative integer ≤ `10**12`
  (at most 12 digits); batch cost is `round(init*b*(1+0.25*(b-1)))`.
- **Gifts:** a delta must be strictly positive and ≤ `100000` coins, with both
  before and after balances readable.
- **Database:** on-disk paths confined to `<project root>/logs`; directory
  `0700`, file `0600`; every SQL write is parameterised; JSON payloads are
  ASCII, key-sorted and encoded through one canonical encoder.

## How the implemented behaviours work

### Perception

`TemplateLibrary` loads `templates/manifest.json` (a version-1 name→filename
map) and decodes each bounded PNG into BGR. `TemplateMatcher` runs
`cv2.matchTemplate` with `TM_CCOEFF_NORMED`, keeps matches ≥ the threshold and
applies IoU-based non-max suppression. `ScreenDetector` walks `STATE_RULES` in
order and returns the first matching state: `GIFT_DIALOG` requires both
`banner_coins` and `gift_collect` on the same screen; `POPUP` uses the banner
markers plus `close_dialog`; `DIALOG` uses the banners only; `LOADING` is the
`loading_text` marker or mean brightness below `40.0`; otherwise `UNKNOWN`.
`loading_text` is excluded from the interactive match list.

`Session.settle()` captures screens until the state and the sorted set of
`(template name, x//16, y//16)` keys are identical on two consecutive polls;
a timeout logs one `SETTLE_TIMEOUT` event. `TesseractReader.read()` crops the
configured rectangle and calls pytesseract. `parse_count()` accepts digits with
commas and `K`/`M` suffixes; `parse_duration()` accepts `H:MM[:SS]` and
`NhNmNs` forms. `InventoryReader.read()` locates each `item_<id>` icon, takes
the highest-scoring match, reads the count rectangle at the configured offset
and returns `0` for missing/unreadable icons or counts.

### Crafting

`load_tree()` requires a mapping with an `items` list. `load_schedule()`
validates the CSV and returns rows of ordered `(item, batch)` steps; empty rows
are retained for tick alignment but skipped by the planner. `CraftPlanner`
keeps tree/targets/inventory only as metadata: `next_step()` peeks the current
step and builds a `CraftStep` with item, section, template, batch and computed
coin cost; `mark_placed(item)` is the only cursor mutation and raises
`ConfigError` if the item does not match the current step.

On each tick the session executes at most one general action. If a
`craft_ready_check` match is present, the tile is collected (the schedule step
is not consumed). Otherwise, when no batch is pending, the planner is peeked
and `_step_remaining` is set to the batch size; a successful tap decrements it,
and reaching zero calls `mark_placed`. Failed taps do not advance the cursor.
When no craft is possible, `Navigator.choose()` dismisses popups, closes the
Bag with BACK, taps the first unvisited target item, or emits a WAIT. On HOME
or CRAFTING, a Bag visit is inserted every `bag_visit_interval` actions. Goal
success is decided only from Bag OCR via `goal_met()` over the 11 ids in
`initial-state.json`; tree and inventory do not influence it.

### Gift collection

`GiftCollector.is_ready()` returns true only when the gift-timer region is
blank or parses to zero **and** the coins region is readable; unparseable text
or an unreadable coin counter blocks a gift cycle. `read_coins()` parses the
coins region. In the session loop:

1. **Open:** off the gift dialog, if a gift is ready, read coins; if readable,
   store the "before" value and tap the gift icon. If unreadable, log a
   `gift_coins_unreadable` failure.
2. **Collect:** on `GIFT_DIALOG`, with no prior Collect tap, establish the
   "before" value if needed (logging `gift_no_before` and attempting a close
   tap when it cannot be read), then require the navigator's chosen TAP to
   have reason `gift_collect`; execute exactly one such tap.
3. **Verify:** on a later non-dialog tick, read coins again and call
   `GiftCollector.verify()`. A positive delta ≤ `100000` writes a `gifts` row
   and increments the session's `gifts_collected` counter; otherwise a
   `gift_verify_failed` action is logged. Pending gift state is then cleared.

### SQLite logs

`InteractionLog` uses the schema in `cat_game_bot/db.py`. Tables: `sessions`
(id, started_at, finished_at, config_json, result), `events` (event, state,
detail_json), `actions` (kind, target, x, y, ok, detail), `observations`
(state, matches_json, counts_json, screen_hash), `goal_progress` (item, count,
required, met) and `gifts` (coins_before, coins_after, delta). Each child table
has a `(session_id, ts)` index. Event names emitted by the session are
`SETTLE_TIMEOUT`, `ACTION_FAILURE` and `RUN_FAULT`. Observation screen hashes
are the first 16 hex characters of SHA-256 over the screen pixels. With the
default configuration the database is `<project root>/logs/bot.sqlite3`.
`':memory:'` is supported for tests. The session result is written as
`INTERRUPTED` on open and overwritten with the terminal result on close.

## Terminal results and exit codes

`Session.run()` can end with `GOAL_MET`, `MAX_ACTIONS`, `NO_PROGRESS`,
`ERROR_LIMIT`, `DISCONNECTED`, `DURATION` or `INTERRUPTED`. Stop conditions are
checked in that order after each tick: action count ≥ `--max-actions`,
no-progress count ≥ `--no-progress-limit`, error count ≥ `--error-limit`, loss
of ADB connectivity, then elapsed time ≥ `--duration`.

`main()` maps results to process exit codes: `0` for `GOAL_MET`, `1` for any
other terminal result, and `2` for a `DbError` raised during setup.
