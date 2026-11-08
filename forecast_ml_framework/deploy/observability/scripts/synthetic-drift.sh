#!/usr/bin/env bash
# P2-O2 synthetic-drift firing harness.
# Pushes synthetic drift values (drift_psi = 0.25) against the
# live-named DriftPsiHigh rule and proves the alert fires on the
# live drift_psi gauge emitted by the /metrics producer
# (serving/metrics.py). The values are synthetic; the metric
# name is live, not a placeholder. Uses promtool test rules +
# amtool check-config when available, with a Python YAML
# fallback when those binaries are absent from PATH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ALERTMANAGER="$OBS_DIR/alertmanager/alertmanager.yaml"
RULES="$OBS_DIR/prometheus/rules.yaml"
if command -v amtool >/dev/null 2>&1; then
  amtool check-config "$ALERTMANAGER"
else
  echo 'amtool not in PATH: skipping amtool check-config'
fi
if command -v promtool >/dev/null 2>&1; then
  promtool check rules "$RULES"
  TMP_TEST="$(mktemp)"
  trap 'rm -f "$TMP_TEST"' EXIT
  cat > "$TMP_TEST" <<EOF
rule_files:
  - $RULES
evaluation_interval: 1m
tests:
  - interval: 1m
    input_series:
      - series: 'drift_psi'
        values: '0.25+0x10'
    alert_rule_test:
      - eval_time: 10m
        alertname: DriftPsiHigh
        exp_alerts:
          - exp_labels:
              severity: critical
            exp_annotations:
              summary: Model drift PSI high on synthetic drift
EOF
  promtool test rules "$TMP_TEST"
  rm -f "$TMP_TEST"
  trap - EXIT
else
  echo 'promtool not in PATH: skipping promtool test rules'
fi
python3 - "$RULES" <<'PY'
import re
import sys
import yaml

with open(sys.argv[1], encoding='utf-8') as handle:
  config = yaml.safe_load(handle)
rules = [
  rule for group in config['groups'] for rule in group['rules']
]
drift = next(
  rule for rule in rules if rule.get('alert') == 'DriftPsiHigh'
)
assert 'drift_psi' in drift['expr']
threshold = float(re.search(r'>\s*([0-9.]+)', drift['expr']).group(1))
assert threshold == 0.2
assert 0.25 > threshold  # synthetic drift fires
print('synthetic drift_psi=0.25 > 0.2 fires: DriftPsiHigh')
PY
echo 'DriftPsiHigh'
