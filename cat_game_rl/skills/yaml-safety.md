# YAML Safety Skill

## Problem
`yaml.safe_load()` returns `None` for empty files or files with only whitespace/comments. Accessing keys on `None` causes `TypeError`, not `KeyError`.

## Rule
Always validate immediately after `yaml.safe_load()`:
```python
data = yaml.safe_load(f)
if data is None or "expected_key" not in data:
    raise ValueError(f"Invalid YAML: {path}")
```

## Learnings
- This was found in audit loop 1 across `items.py` and `target_provider.py`
- Both `CraftingTree.from_yaml()` and `TargetProvider._load()` now validate
- Dashboard `targets.py` also loads YAML — ensure validation there too
- Path traversal: `startswith(dir)` is insufficient — `"output_evil".startswith("output")` is True. Always append `os.sep`: `startswith(dir + os.sep) or path == dir`
- All simulation paths must cap batch at 20: env, GA, dashboard, baselines. The dashboard was missing the cap.
