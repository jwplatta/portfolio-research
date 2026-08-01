#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

export MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
export MLFLOW_PORT="${MLFLOW_PORT:-5000}"
export MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${REPO_ROOT}/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}"

mkdir -p "${MLFLOW_ARTIFACT_ROOT}"

echo "Starting MLflow at ${MLFLOW_TRACKING_URI}"
echo "Backend store: ${MLFLOW_BACKEND_STORE_URI}"
echo "Artifact root: ${MLFLOW_ARTIFACT_ROOT}"

cd "${REPO_ROOT}"
exec uv run mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
