# Entry points — 6 files

One live entry point, four legacy copies of it, and the environment script.

| file | LOC | status |
|---|---|---|
| [`src/sed_train.py`](#srcsed_trainpy) | 25 | **LIVE** |
| [`src/train.py`](#srctrainpy) | 25 | LEGACY |
| [`src/train_v2.py`](#srctrain_v2py) | 25 | LEGACY |
| [`src/train_v3.py`](#srctrain_v3py) | 25 | LEGACY |
| [`src/train_v4.py`](#srctrain_v4py) | 25 | LEGACY |
| [`startup.sh`](#startupsh) | 8 | **LIVE** |

The four legacy trainers are **byte-for-byte the same file** except for the
one import line that selects the engine.

---

## `src/sed_train.py`

**Status: LIVE** — the only documented way to train.

```python
def run(local_rank, config)
def main(hyperparams)
```

**Behaviour** (all 25 lines)

```python
from argparse import ArgumentParser, Namespace
from engine.sed_engine import SedEngine
import importlib, torch
import ignite.distributed as idist
torch.backends.cudnn.benchmark = True

def run(local_rank, config):
    pe = SedEngine(local_rank, config)
    pe.train(config.run_params)

def main(hyperparams):
    with idist.Parallel(**hyperparams.dist_params) as parallel:
        parallel.run(run, hyperparams)

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    parser.add_argument('--config', type=str)
    params = parser.parse_args()
    module = importlib.import_module(params.config, package=None)
    hyperparams = module.Parameters()
    main(hyperparams)
```

**How to run it**

```bash
cd src
python sed_train.py --config config_params.final_sed.final_sed_dense121_mix_fold0
```

**Gotchas**

- ⚠️ `cd src` is **mandatory**. The script imports `engine.sed_engine` as a
  top-level module, so `src/` must be the working directory (or on
  `sys.path`). There is no `__init__.py` anywhere — the tree works through
  implicit namespace packages.
- ⚠️ `--config` is a **dotted Python module path**, not a file path.
  `config_params/final_sed/x.py` will not work; it must be
  `config_params.final_sed.x`.
- ⚠️ `idist.Parallel(**hyperparams.dist_params)` — every shipped config sets
  `dist_params = {}`, so this is **single-process**. A multi-GPU run needs
  `torchrun` and a non-empty `dist_params`.
- ⚠️ `torch.backends.cudnn.benchmark = True` is set at import time, before any
  device check.
- ⚠️ `BaseEngine.train` swallows exceptions (see
  [engine.md](engine.md)), so **a crashed run exits 0**.
- `from argparse import ArgumentParser, Namespace` — `Namespace` is unused.
- The `parents=[]` argument is vestigial.

---

## `src/train.py` — v1

**Status: LEGACY.** Identical to `sed_train.py` except for the import:

```python
from engine.main_engine import MainEngine
```

and `pe = MainEngine(local_rank, config)`.

**Broken** — `MainEngine` fails at `AttributeError: 'MainEngine' object has no
attribute 'ls_fn'` on the first training step, and with a shipped config it
fails **earlier** in `_init_model` (no `sed_dense121att` branch). It is not
mentioned in the README.

---

## `src/train_v2.py` — v2

**Status: LEGACY.** Identical except for:

```python
from engine.main_engine_v2 import MainEngineV2
```

**Broken** — `AttributeError: 'MainEngineV2' object has no attribute 'ls_fn'`
on the first step, and the `"spec_aug"` augmentation branch raises
`ImportError` (via `augmentations/spec_augment.py`).

---

## `src/train_v3.py` — v3

**Status: LEGACY.** Identical except for:

```python
from engine.main_engine_v3 import MainEngineV3
```

**Broken** — the `"spec_aug"` augmentation branch raises `ImportError` (via
`augmentations/spec_augment.py`). The `ls_fn` bug does **not** exist here.

---

## `src/train_v4.py` — v4

**Status: LEGACY.** Identical except for:

```python
from engine.main_engine_v4 import MainEngineV4
```

**Broken** — `AttributeError: 'Parameters' object has no attribute
'loss_factors'` on the first training step, plus the four-argument
`criterion(...)` call that `BCEControlledStandardLoss` cannot accept, plus
`aug_name="secondary_default"` having no branch in `_init_augmentation`.

---

## `startup.sh`

**Status: LIVE.** The environment provisioning script, 8 lines.

**Behaviour.** Installs the pinned requirements, clones and installs NVIDIA
Apex for mixed precision, and runs the training entry point.

**Gotchas**

- ⚠️ **Apex is cloned and installed even though `USE_AMP` is hard-disabled.**
  `base_engine.py:6-13` imports `apex` and then sets `USE_AMP = False` in
  *both* branches. The install is pure cost — and the clone is from a moving
  upstream branch, so it is not reproducible.
- ⚠️ The Apex install is what the log message at `base_engine.py:8-9` refers
  to, and that message claims AMP is in use when it is not.
- ⚠️ The requirements install does **not** pin torch, torchvision or
  torchaudio — the script assumes they are preinstalled in the image.

---

## See also

- [modules/05 — Training engine](../modules/05-training-engine.md)
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [modules/12 — Kaggle and compute constraints](../modules/12-kaggle-constraints.md)
- [README.md](../../README.md) — the original upstream README, which documents `python sed_train.py --config config_params.example_config`
