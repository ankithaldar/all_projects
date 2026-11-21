# random_codes

A collection of self-contained coding experiments. Each numbered folder is an
standalone project — the numeric prefix roughly tracks the order they were
built in. No shared build system; each project carries its own dependencies
and instructions (the root-level `requirements.txt` and `pyproject.toml` are
placeholders and intentionally empty).

Each entry below is a brief overview; see the project folder for the full
story and source.

## Projects

### [`0001_enigma`](0001_enigma/) — Enigma cipher machine in pure Python

A faithful software model of the German Enigma I cipher machine, plus
Bletchley Park–inspired tools for attacking it.

- `enigma.py` — full machine: standard Enigma I rotor wirings (I–V),
  Reflector B, turnover notches, **double-stepping**, plugboard, positions and
  ring settings, reciprocal encryption/decryption (decrypt = reset + encrypt).
  CLI for encrypt/decrypt (`--rotors`, `--positions`, `--plugboard`, `--mode`).
- `turing_bombe.py` — "Bombe": exhaustive search over all 26³ = 17,576 rotor
  position combinations for a ciphertext, or a fixed-position check
  (`--positions`). Logs every candidate plaintext; a crib filter is sketched
  (`crib_matches`) but not wired up as a CLI flag yet.
- `turing_decrypt.py` — decrypt a ciphertext with a known key, with an
  interactive prompt when `--text` is omitted.
- `test_enigma.py` — `unittest` suite for rotor stepping (advance, turnover,
  double-step).
- See [`0001_enigma/README.md`](0001_enigma/README.md) for CLI options and the
  full description.

Run: `python3 enigma.py --text "HELLO WORLD"` · test:
`python3 -m unittest test_enigma.py`

### [`0002_mandelbrot_set`](0002_mandelbrot_set/) — Mandelbrot set renderer

An interactive Mandelbrot set visualization using NumPy vectorization and
Matplotlib.

- Escape-time algorithm with **smooth (continuous) iteration counts**
  (`n + 1 − log2(log|z|)`) for banding-free coloring; interior drawn black via
  colormap `set_bad`.
- Two views, both exported as 1500×1200 px PNGs at 150 dpi: the full set
  (`mandelbrot_full.png`) and a Seahorse Valley zoom near the main cardioid
  (`mandelbrot_zoom.png`).
- Configurable at the top of `mandelbrot_set.py`: view ranges, `MAX_ITERATIONS
  = 3500`, escape radius 128, `inferno` colormap, figure size/DPI.
- Renders into interactive Matplotlib windows (toolbar zoom/pan); requires
  `numpy` and `matplotlib` (not pinned in this folder) and a display, or a
  non-interactive Matplotlib backend for headless runs.

Run: `python3 mandelbrot_set.py` · see
[`0002_mandelbrot_set/README.md`](0002_mandelbrot_set/README.md) for config
knobs and headless usage.

### [`0003_car_racer`](0003_car_racer/) — Evolutionary neural-network car racer

A single-folder, no-build, no-dependency browser game where 20 neural-network
drivers are bred over generations to race a closed circuit.

- Fixed architecture `7-16-16-2` (434 params), evolved by elitism + Gaussian
  mutation (no crossover, no annealing); fitness = net signed odometer +
  2 × track-length per lap (100-lap cap).
- The full training lineage — every generation, per-driver results and best
  weights — is persisted into a **real SQLite database inside the browser**
  (vendored sql.js/WebAssembly; `file://` works, `localhost` recommended).
  Export, import, fork and inspect the `.sqlite` with any SQLite tool.
- Manual driving mode, headless training in a Web Worker, live "spectate"
  mode, replay of any committed generation, a paged logs tab, and an
  in-browser track editor with a lineage-pinned track hash.
- Pure HTML/CSS/JS; Node.js is needed only for the test suite
  (`node tests/smoke.mjs` — 55 legs; plus `tests/determinism.mjs` and
  `tests/benchmark.mjs`).
- See [`0003_car_racer/README.md`](0003_car_racer/README.md) for the complete,
  exhaustively detailed documentation (schema, adoption battery, failure
  modes, v1→v2 migration, threat model).

Run: `cd 0003_car_racer && python3 -m http.server 8000`, then open
`http://localhost:8000/`

### [`0004_cat_game_neat`](0004_cat_game_neat/) — NEAT crafting-schedule search

A static browser app that uses **19 evolved NEAT policy networks** to find a
low-cost crafting schedule for a 23-item crafting game. Pure HTML/CSS/ES
modules — no build step, no Python application code, no backend; the simulation
and the NEAT engine run entirely in the browser.

- 23 items (4 unlimited tier-0 bases + 19 craftables) and 11 targets evaluated
  against the current stash each tick; batch cost scales super-linearly, craft
  time is rounded up to ticks, and income accrues every 5-minute tick over a
  28-day horizon.
- A hard manufacturing cap keyed by `manufactureRequired` ensures no craftable
  manufactures more than the gross amount needed to meet targets and replace
  downstream consumption; three starter modes (`PREDELIVERED_RESERVED`,
  `CONSUMABLE`, `FREE_INGREDIENTS`) change how the starter stash counts.
- Configurable seed (reproducible genomes/schedule), preset, population size and
  generation count, plus an editable initial-state panel (coins, targets and
  starter stash) and an 8,064-tick horizon.
- ES modules and the training Web Worker require `http://`; `file://` falls back
  to embedded config with training disabled. Tests live at `tests.html`.
- **Best schedule + saved initial state (data/):** `data/best-schedule.csv` — the app's best schedule
  (pivot of `result().schedule`, rows = ticks 0..3348, reuse run); `data/initial-state.json` — the saved,
  locally editable initial state (`{targets, initialCoins, initialStash}`; currently a custom snapshot;
  loaded by both the app's Reuse preset over `http://` and the generator);
  `tools/best-schedule.mjs` — generator (`node tools/best-schedule.mjs [reuse|empty]`) that reads
  `data/initial-state.json` and writes `data/best-schedule.csv`. `data/` is not committed.
- See [`0004_cat_game_neat/README.md`](0004_cat_game_neat/README.md) for game
  rules, the exact cap formula, controls and architecture notes.

Run: `cd 0004_cat_game_neat && ./start.sh` (or `start.bat` on Windows), or serve
the folder with `python3 -m http.server 8000` and open
`http://localhost:8000/index.html`
