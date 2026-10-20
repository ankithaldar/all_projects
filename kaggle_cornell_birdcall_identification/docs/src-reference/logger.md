# `src/logger/` — 6 files

Two loggers (Neptune, print) over one abstract base, with a shared helper
module for the Ignite event-filter pattern and one per logger for the actual
"what to write" closures.

| file | LOC | status |
|---|---|---|
| [`base/base_logger.py`](#basebase_loggerpy) | 35 | **LIVE** |
| [`base/utils.py`](#baseutilspy) | 12 | **LIVE** |
| [`neptune/neptune_logger.py`](#neptuneneptune_loggerpy) | 61 | **LIVE** |
| [`neptune/neptune_utils.py`](#neptuneneptune_utilspy) | 48 | **LIVE** |
| [`print/print_logger.py`](#printprint_loggerpy) | 51 | **LIVE** |
| [`print/print_utils.py`](#printprint_utilspy) | 50 | **LIVE** |

---

## `base/base_logger.py`

**Status: LIVE.** The abstract base both loggers extend.

```python
class BaseLogger(object):
    def __init__(self, log_every=10)
    def _init_logger(self, trainer, evaluator)
    def _add_train_handlers(self, iteration_events=[], completion_events=[])
    def _add_evaluation_handlers(self, iteration_events=[], completion_events=[])
    def _add_train_events(self, model=None, optimizer=None, scheduler=None, metrics={})   # NotImplementedError
    def _add_eval_events(self, model=None, optimizer=None, scheduler=None, metrics={})     # NotImplementedError
    def _add_custom_train_iteration_handler(self, iteration_event, log_every)
    def _add_custom_eval_iteration_handler(self, iteration_event, log_every)
```

**Behaviour.** The base class holds `self.log_every`, `self.trainer` and
`self.evaluator`, and delegates the event wiring to `base/utils.py`. The two
`_add_*_events` methods raise `NotImplementedError` — the subclasses implement
them.

**Gotchas**

- ⚠️ The default `log_every=10` here differs from the **5** in both
  subclasses' constructors, so the base default is never the one in effect.
- ⚠️ `self.logger` is only created on rank 0
  (`engine/base/base_engine.py:209-214`), so the logger's event handlers are
  attached by **one** process in a DDP run.

---

## `base/utils.py`

**Status: LIVE.** The 12-line file that contains the pattern with no direct
Lightning equivalent.

```python
def add_iteration_handlers(engine, method_events=[], log_every=5)
def add_completion_handlers(engine, method_events=[])
```

**Behaviour.** `add_iteration_handlers` registers each event on
`Events.ITERATION_COMPLETED(every=log_every)`; `add_completion_handlers`
registers each on `Events.EPOCH_COMPLETED`.

**⚠️ This is the `every=N` pattern.** Ignite's `every` filter is **per
handler** and applies uniformly to every event type. Lightning has no
equivalent: `Trainer(log_every_n_steps=N)` is a global logging knob, not a
per-handler event filter, and Keras has no event filter at all. Any port must
hand-roll a counter — the repo's own analysis calls this "a genuine
regression in expressiveness".

---

## `neptune/neptune_logger.py`

**Status: LIVE.** The default logger — every shipped config sets
`logger_name = "neptune"`.

```python
class MyNeptuneLogger(BaseLogger):
    def __init__(self, log_every=5, **kwargs)
    def _add_train_events(self, model=None, optimizer=None, scheduler=None, metrics={})
    def _add_eval_events(self, model=None, optimizer=None, scheduler=None, metrics={})
    def _end_of_training(self)
```

**Behaviour.** `__init__` builds **ignite's** `NeptuneLogger` with
`api_token=os.getenv('NEPTUNE_API_TOKEN')`, `project_name`, `name`, `params`
and `tags` taken from `hparams.logger_params`.

`_add_train_events` registers the per-iteration series (loss, every criterion
loss key, and the learning rate) plus a per-epoch completion handler;
`_add_eval_events` registers the per-iteration and per-epoch validation
series. `_end_of_training` calls `self.writer.experiment.stop()`.

**Gotchas**

- ⚠️ **The client version is a hard constraint.** The repo pins
  `neptune-client==0.4.107`, and ignite's `NeptuneLogger` speaks that 0.4.x
  API. Lightning's `NeptuneLogger` and `neptune-keras` both require
  `neptune>=1.0`. Any port must **upgrade the client** — an unavoidable
  dependency change, not a porting choice.
- ⚠️ The weight- and gradient-scalar handlers are **commented out** in
  `_add_train_events`; only `lr_iteration` is live.
- ⚠️ `_end_of_training` is called from `BaseEngine.train` (L222-223) — and
  because `train()` swallows exceptions, a crashed run still closes the
  Neptune experiment **normally**, so it looks finished in the tracker.

---

## `neptune/neptune_utils.py`

**Status: LIVE.** Four closure factories — the "what to write" half.

```python
def lr_iteration(optimizer, writer)
def training_iteration(writer)
def train_metrics_completion(writer)
def validation_metrics_completion(trainer, writer)
```

Each returns an Ignite handler `fn(engine)` that writes through
`writer.experiment.log_metric(name, step, value)`.

**Gotchas**

- ⚠️ `training_iteration` logs the **tensor**
  `engine.state.output['loss']`, not `.item()` — so the value written is a
  0-dim tensor, not a float.
- ⚠️ `validation_metrics_completion` takes the **trainer**, not the
  evaluator, so it reads the trainer's metric namespace.

---

## `print/print_logger.py`

**Status: LIVE.** The console logger, used by `logger_name = "print"`.

```python
class PrintLogger(BaseLogger):
    def __init__(self, log_every=5, **kwargs)
    def _add_train_events(self, ...)
    def _add_eval_events(self, ...)
    def _end_of_training(self)      # a no-op
```

**Behaviour.** Writes `train/loss` and `train/<dict_loss key>` every
`log_every` iterations, `train/<metric>` at each epoch end, and
`valid/<metric>` at each evaluation end — to **stdout**. It also posts to a
**Slack webhook**: one message per training epoch carrying
`metrics['train_avg_loss']`, and **one message per metric key** at validation
end.

**Gotchas**

- ⚠️ The Slack URL is read at **import time** from `os.environ["SLACK_URL"]`
  inside a bare `except` → `None` when unset. Changing the environment after
  import has no effect.
- ⚠️ The `lr_iteration` handler exists but is **commented out** — the print
  logger does not print the learning rate (the Neptune one does).
- `_end_of_training` is a no-op.

---

## `print/print_utils.py`

**Status: LIVE.** The same four closure names as the Neptune utils, writing to
stdout instead.

```python
def lr_iteration(optimizer, writer)
def training_iteration(writer)
def train_metrics_completion(writer)
def validation_metrics_completion(trainer, writer)
```

**Gotchas**

- ⚠️ The four names **shadow** the Neptune ones. Which is imported depends on
  the module path, and the two are easy to confuse when reading a diff.
- `writer` is `None` in the print logger (`self.writer = None`, L15) — the
  closures print rather than write, and ignore it.

---

## The logged series, in one place

| when | names | logger |
|---|---|---|
| every `log_every` iterations | `train/loss`, `train/<loss_key>`, `train/lr` | Neptune |
| every `log_every` iterations | `train/loss`, `train/<loss_key>` | print |
| end of each training epoch | `train/<metric>` | both |
| end of each evaluation | `valid/<metric>` | both |
| per training epoch | one Slack message with `train_avg_loss` | both |
| per validation end | one Slack message **per metric key** | both |

`logger_params` carries `project_name`, `log_every`, `name`, `prefix_name`,
`tags` and `params` — the last being the flattened model/loss/optimiser
configuration.

---

## See also

- [modules/10 — Metrics and tracking](../modules/10-metrics-and-tracking.md)
- [ports/pytorch-lightning-guide.md](../ports/pytorch-lightning-guide.md) §Phase 8 — the logging mapping and the `every=N` trap
- [ports/tensorflow-keras-guide.md](../ports/tensorflow-keras-guide.md) §Phase 9
