# Handoff P3-C — Guards and calibrate prohibition (worker)

## Task

P3-C: implement temporal-split guards and the calibrate
prohibition with TDD RED → GREEN. Transcribed from the worker's
completion report provided in the task dispatch.

## Owns

- `forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/data_split.py`
- `forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/calibration.py`
- `forecast_ml_framework/tests/test_p3_guards_calibrate.py`

## RED log

Tests were written first, before the production helpers existed.

Command:

```text
python -m pytest forecast_ml_framework/tests/test_p3_guards_calibrate.py -q
```

Output (from worker completion report):

```text
ImportError: cannot import name '_is_temporal_split'
from 'forecast_ml_framework.pipelines.modeling.calibration'
```

RED fails because the feature is missing (helper absent),
not because of a typo — collection fails on the missing
`_is_temporal_split` import.

## GREEN log

Implemented the guards in `data_split.py` and the
`_is_temporal_split` helper / calibrate prohibition in
`calibration.py`. GREEN is the minimal implementation that
makes the tests pass.

Command:

```text
python -m pytest forecast_ml_framework/tests/test_p3_guards_calibrate.py -q
```

Output (from worker completion report):

```text
32 passed
```

## Coverage commands

```text
python -m pytest --cov=forecast_ml_framework.pipelines.modeling.data_split --cov=forecast_ml_framework.pipelines.modeling.calibration forecast_ml_framework/tests/test_p3_guards_calibrate.py -q
```

Output (from worker completion report):

```text
forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/data_split.py 95.33%
forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/calibration.py 94.26%
```

## Assumptions

- RED/GREEN counts and coverage percentages are transcribed verbatim
  from the worker's completion report in the task dispatch; they were
  not independently re-run for this handoff log.
- Test and module paths above were verified to exist in the repo
  (`test_p3_guards_calibrate.py`, `data_split.py`, and `calibration.py`);
  no behavior claims beyond the reported counts are made here.

## Blockers

None.
