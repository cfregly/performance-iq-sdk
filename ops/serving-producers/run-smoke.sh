#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash ops/serving-producers/run-smoke.sh launch-plan [extra args...]
  bash ops/serving-producers/run-smoke.sh diagnostics [extra args...]
  bash ops/serving-producers/run-smoke.sh preflight [extra args...]
  bash ops/serving-producers/run-smoke.sh smoke [extra args...]
  bash ops/serving-producers/run-smoke.sh recorded-smoke [extra args...]
  bash ops/serving-producers/run-smoke.sh no-submit [extra args...]
  bash ops/serving-producers/run-smoke.sh verify-proof <proof-summary.json>
  bash ops/serving-producers/run-smoke.sh receipt-proxy [proxy args...]

Modes:
  launch-plan  Print host-aware launch commands.
  diagnostics  Print read-only host, cache, port, and endpoint diagnostics.
  preflight    Check all configured /v1/models endpoints for the smoke model.
  smoke        Submit producer runs and verify dashboard query surfaces.
  recorded-smoke
               Start receipt proxies, submit runs, and verify dashboards.
  no-submit    Send requests and write artifacts without submitting runs.
  verify-proof Verify a saved full three-engine proof bundle offline.
  receipt-proxy
               Proxy one engine endpoint and write JSONL request receipts.
USAGE
}

mode="${1:-preflight}"
case "$mode" in
  launch-plan|diagnostics|preflight|smoke|recorded-smoke|no-submit|verify-proof|receipt-proxy)
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
python_path_parts=("${repo_root}/python/src")
[[ -z "${PIQ_VLLM_SOURCE_PATH:-}" ]] || python_path_parts+=("${PIQ_VLLM_SOURCE_PATH}")
[[ -z "${PIQ_SGLANG_SOURCE_PATH:-}" ]] || python_path_parts+=("${PIQ_SGLANG_SOURCE_PATH}")
[[ -z "${PYTHONPATH:-}" ]] || python_path_parts+=("${PYTHONPATH}")
PYTHONPATH="$(IFS=:; echo "${python_path_parts[*]}")"
export PYTHONPATH

if [[ -n "${PIQ_SERVING_BIN_DIR:-}" ]]; then
  export PATH="${PIQ_SERVING_BIN_DIR}:${PATH}"
fi
python_bin="${PIQ_PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    echo "No Python interpreter found. Set PIQ_PYTHON_BIN to the Python executable for performance_iq_sdk." >&2
    exit 127
  fi
fi
if [[ "$python_bin" == */* ]]; then
  python_dir="$(cd "$(dirname "$python_bin")" && pwd)"
  export PATH="${python_dir}:${PATH}"
fi

model="${PIQ_SERVING_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
artifact_dir="${PIQ_ARTIFACT_DIR:-${repo_root}/performance-iq-output/serving-producers}"
receipt_log="${PIQ_SERVING_RECEIPT_LOG:-${artifact_dir}/request-receipts.jsonl}"

common=(
  --model "$model"
  --artifact-dir "$artifact_dir"
)

allow_partial_defaults=false
case "${PIQ_SERVING_ALLOW_PARTIAL:-false}" in
  1|true|TRUE|yes|YES)
    allow_partial_defaults=true
    ;;
esac

add_engine_url_arg() {
  local flag="$1"
  local env_name="$2"
  local default_url="$3"
  local value

  if value="$(printenv "$env_name")"; then
    common+=("$flag" "$value")
  elif [[ "$allow_partial_defaults" != "true" ]]; then
    common+=("$flag" "$default_url")
  fi
}

add_engine_url_arg --vllm-url PIQ_VLLM_URL http://127.0.0.1:8000
add_engine_url_arg --sglang-url PIQ_SGLANG_URL http://127.0.0.1:30000
add_engine_url_arg --tensorrt-llm-url PIQ_TENSORRT_LLM_URL http://127.0.0.1:8001

case "$mode" in
  launch-plan)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke --launch-plan-only --model "$model" "$@"
    ;;
  diagnostics)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke --diagnostics-only "${common[@]}" "$@"
    ;;
  preflight)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke --preflight-only "${common[@]}" "$@"
    ;;
  smoke)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke "${common[@]}" --query-dashboard "$@"
    ;;
  recorded-smoke)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke "${common[@]}" --query-dashboard --record-receipts --receipt-log "$receipt_log" "$@"
    ;;
  no-submit)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke "${common[@]}" --no-submit "$@"
    ;;
  verify-proof)
    exec "$python_bin" -m performance_iq_sdk.serving_smoke --verify-proof "$@"
    ;;
  receipt-proxy)
    exec "$python_bin" -m performance_iq_sdk.serving_receipts "$@"
    ;;
esac
