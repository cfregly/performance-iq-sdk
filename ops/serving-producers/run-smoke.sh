#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash ops/serving-producers/run-smoke.sh launch-plan [extra args...]
  bash ops/serving-producers/run-smoke.sh preflight [extra args...]
  bash ops/serving-producers/run-smoke.sh smoke [extra args...]
  bash ops/serving-producers/run-smoke.sh no-submit [extra args...]

Modes:
  launch-plan  Print host-aware launch commands.
  preflight    Check all configured /v1/models endpoints for the smoke model.
  smoke        Submit producer runs and verify dashboard query surfaces.
  no-submit    Send requests and write artifacts without submitting runs.
USAGE
}

mode="${1:-preflight}"
case "$mode" in
  launch-plan|preflight|smoke|no-submit)
    shift || true
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-${repo_root}/python/src}"

model="${PIQ_SERVING_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
artifact_dir="${PIQ_ARTIFACT_DIR:-${repo_root}/performance-iq-output/serving-producers}"

common=(
  --model "$model"
  --artifact-dir "$artifact_dir"
  --vllm-url "${PIQ_VLLM_URL:-http://127.0.0.1:8000}"
  --sglang-url "${PIQ_SGLANG_URL:-http://127.0.0.1:30000}"
  --tensorrt-llm-url "${PIQ_TENSORRT_LLM_URL:-http://127.0.0.1:8001}"
)

case "$mode" in
  launch-plan)
    exec python -m performance_iq_sdk.serving_smoke --launch-plan-only --model "$model" "$@"
    ;;
  preflight)
    exec python -m performance_iq_sdk.serving_smoke --preflight-only "${common[@]}" "$@"
    ;;
  smoke)
    exec python -m performance_iq_sdk.serving_smoke "${common[@]}" --query-dashboard "$@"
    ;;
  no-submit)
    exec python -m performance_iq_sdk.serving_smoke "${common[@]}" --no-submit "$@"
    ;;
esac
