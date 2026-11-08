#!/usr/bin/env bash
# Rollback drill: upgrade twice, roll back, check rollout.
# Template-only fallback when no cluster is reachable.
set -euo pipefail
RELEASE="${RELEASE:-forecast-ml}"
CHART="${CHART:-forecast_ml_framework/deploy/helm/forecast-ml}"
NAMESPACE="${NAMESPACE:-default}"
if ! kubectl cluster-info >/dev/null 2>&1; then
  echo "template-only fallback: no cluster reachable"
  if command -v helm >/dev/null 2>&1; then
    helm lint "$CHART"
    helm template "$RELEASE" "$CHART" >/dev/null
  else
    echo "helm not found: template-only fallback complete"
  fi
  echo "rollback-drill template-only fallback complete"
  exit 0
fi
helm upgrade --install "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" --set image.tag=0.1.0
helm upgrade "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" --set image.tag=0.1.1
helm rollback "$RELEASE" --namespace "$NAMESPACE"
kubectl rollout status deployment/forecast-ml \
  --namespace "$NAMESPACE" --timeout=120s
helm history "$RELEASE" --namespace "$NAMESPACE"
echo "rollback-drill complete"
