#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash ops/serving-producers/run-nvidia-e2e.sh validate [options]
  bash ops/serving-producers/run-nvidia-e2e.sh run [options] [-- smoke args...]

Options:
  --env-file PATH       Operator env file (default: .env.serving-producers).
  --skip-pull           Reuse locally cached images.
  --skip-manifests      Skip remote registry manifest checks.
  --keep-running        Leave engine and DCGM containers running after the run.
  -h, --help            Show this help.

The run mode requires Linux, a trusted NVIDIA Docker runtime, a reachable
PIQ_BASE_URL, and PIQ_SERVING_USD_PER_GPU_HOUR. It submits real producer rows
and runs the strict telemetry and real-runtime proof gates.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' is not available."
}

mode="${1:-run}"
case "$mode" in
  validate|run) shift ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
compose_file="${script_dir}/docker-compose.nvidia.yaml"
env_file="${repo_root}/.env.serving-producers"
skip_pull=false
skip_manifests=false
keep_running=false
smoke_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || die "--env-file requires a path."
      env_file="$2"
      shift 2
      ;;
    --skip-pull) skip_pull=true; shift ;;
    --skip-manifests) skip_manifests=true; shift ;;
    --keep-running) keep_running=true; shift ;;
    --) shift; smoke_args=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ "$env_file" = /* ]] || env_file="${repo_root}/${env_file}"
[[ -f "$env_file" ]] || die "Environment file not found: $env_file"

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

require_command docker
require_command curl
require_command jq
docker compose version >/dev/null

model="${PIQ_SERVING_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
model_revision="${PIQ_SERVING_MODEL_REVISION:-}"
[[ "$model_revision" =~ ^[0-9a-fA-F]{40}$ ]] || die "PIQ_SERVING_MODEL_REVISION must be a 40-character commit hash."
[[ "${PIQ_TENSORRT_LLM_BACKEND:-tensorrt}" == "tensorrt" ]] \
  || die "Strict TensorRT-LLM proof requires PIQ_TENSORRT_LLM_BACKEND=tensorrt."

dcgm_image="${PIQ_DCGM_EXPORTER_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:4.5.3-4.8.2-distroless@sha256:60d3b00ac80b4ae77f94dae2f943685605585ad9e92fdccda3154d009ae317cc}"
vllm_image="${PIQ_VLLM_IMAGE:-vllm/vllm-openai:v0.23.0@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f}"
sglang_image="${PIQ_SGLANG_IMAGE:-lmsysorg/sglang:v0.5.12@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356}"
trtllm_image="${PIQ_TENSORRT_LLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9}"
images=("$dcgm_image" "$vllm_image" "$sglang_image" "$trtllm_image")

for image in "${images[@]}"; do
  [[ -n "$image" ]] || die "All four NVIDIA stack images must be configured."
  [[ "$image" != *:latest ]] || die "Unpinned image is not allowed for proof: $image"
  [[ "$image" =~ @sha256:[0-9a-fA-F]{64}$ ]] || die "Strict proof image must be content-addressed with @sha256: $image"
  if [[ "$skip_manifests" != "true" ]]; then
    docker manifest inspect "$image" >/dev/null || die "Registry manifest is unavailable: $image"
  fi
done

required_dcgm_metrics=(
  DCGM_FI_DEV_POWER_USAGE
  DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION
  DCGM_FI_DEV_GPU_UTIL
  DCGM_FI_DEV_MEM_COPY_UTIL
  DCGM_FI_PROF_SM_ACTIVE
  DCGM_FI_PROF_DRAM_ACTIVE
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
  DCGM_FI_PROF_PIPE_FP64_ACTIVE
  DCGM_FI_PROF_PIPE_FP32_ACTIVE
  DCGM_FI_PROF_PIPE_FP16_ACTIVE
  DCGM_FI_DEV_PCIE_TX_THROUGHPUT
  DCGM_FI_DEV_PCIE_RX_THROUGHPUT
  DCGM_FI_PROF_PCIE_TX_BYTES
  DCGM_FI_PROF_PCIE_RX_BYTES
  DCGM_FI_DEV_PCIE_REPLAY_COUNTER
  DCGM_FI_PROF_NVLINK_TX_BYTES
  DCGM_FI_PROF_NVLINK_RX_BYTES
  DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL
  DCGM_FI_DEV_ENC_UTIL
  DCGM_FI_DEV_DEC_UTIL
  DCGM_FI_DEV_GPU_TEMP
  DCGM_FI_DEV_SM_CLOCK
  DCGM_FI_DEV_MEM_CLOCK
  DCGM_FI_DEV_FB_USED
  DCGM_FI_DEV_FB_FREE
  DCGM_FI_DEV_XID_ERRORS
  DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
  DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
  DCGM_FI_DEV_POWER_VIOLATION
  DCGM_FI_DEV_THERMAL_VIOLATION
)

for metric in "${required_dcgm_metrics[@]}"; do
  grep -qE "^${metric}," "${script_dir}/dcgm-counters.csv" || die "DCGM counter contract is missing $metric."
done

trtllm_compose_section="$(awk '/^  tensorrt-llm:/{capture=1} capture{print}' "$compose_file")"
grep -q -- '      - --revision' <<<"$trtllm_compose_section" \
  || die "TensorRT-LLM v1.2.1 launch must pin the model with --revision."
grep -q -- '      - --backend' <<<"$trtllm_compose_section" \
  || die "TensorRT-LLM launch must select an explicit backend."
grep -q -- '      - --config' <<<"$trtllm_compose_section" \
  || die "TensorRT-LLM launch must enable native performance metrics with --config."
if grep -qE -- '      - --(hf_revision|served_model_name)$' <<<"$trtllm_compose_section"; then
  die "TensorRT-LLM compose contains an unsupported v1.2.1 flag."
fi
grep -qE '^return_perf_metrics:[[:space:]]+true$' "${script_dir}/tensorrt-llm-config.yaml" \
  || die "TensorRT-LLM Prometheus and per-request performance metrics are not enabled."
grep -qE '^perf_metrics_max_requests:[[:space:]]+[1-9][0-9]*$' "${script_dir}/tensorrt-llm-config.yaml" \
  || die "TensorRT-LLM per-request performance metric retention is not configured."

compose=(docker compose --env-file "$env_file" -f "$compose_file")
"${compose[@]}" config --quiet

if [[ "$mode" == "validate" ]]; then
  echo "NVIDIA serving stack validation passed for $model@$model_revision."
  exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || die "Real NVIDIA proof requires a Linux host."
require_command nvidia-smi
nvidia-smi -L | grep -q '^GPU ' || die "nvidia-smi did not report a GPU."
docker info --format '{{json .Runtimes}}' | grep -q 'nvidia' || die "Docker does not report the NVIDIA runtime."
[[ -n "${PIQ_BASE_URL:-}" ]] || die "PIQ_BASE_URL is required for durable submission and dashboard queries."
[[ -n "${PIQ_SERVING_USD_PER_GPU_HOUR:-}" ]] || die "PIQ_SERVING_USD_PER_GPU_HOUR is required for strict cost provenance."

artifact_dir="${PIQ_ARTIFACT_DIR:-${repo_root}/performance-iq-output/serving-producers}"
[[ "$artifact_dir" = /* ]] || artifact_dir="${repo_root}/${artifact_dir}"
mkdir -p "$artifact_dir"
gpu_inventory_path="${artifact_dir}/nvidia-gpu-inventory.csv"
nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version \
  --format=csv,noheader,nounits >"$gpu_inventory_path"
gpu_name="$(awk -F',' 'NR == 1 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "$gpu_inventory_path")"
[[ -n "$gpu_name" ]] || die "Could not read the tested GPU name from nvidia-smi."
export PIQ_ARTIFACT_DIR="$artifact_dir"
export PIQ_SERVING_HARDWARE="$gpu_name"
export PIQ_SERVING_OPERATING_POINT="${PIQ_SERVING_OPERATING_POINT:-single-gpu-co-resident-three-engine}"
export PIQ_SERVING_GPU_INVENTORY_PATH="$gpu_inventory_path"
export PIQ_SERVING_RECEIPT_LOG="${PIQ_SERVING_RECEIPT_LOG:-${artifact_dir}/request-receipts.jsonl}"
export PIQ_SERVING_EVENT_LOG="${PIQ_SERVING_EVENT_LOG:-${artifact_dir}/serving-events.jsonl}"
export PIQ_VLLM_RUNTIME_BACKEND="${PIQ_VLLM_RUNTIME_BACKEND:-cuda}"
export PIQ_SGLANG_RUNTIME_BACKEND="${PIQ_SGLANG_RUNTIME_BACKEND:-cuda}"
export PIQ_TENSORRT_LLM_RUNTIME_BACKEND="tensorrt"

cleanup() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    "${compose[@]}" ps --all >"${artifact_dir}/nvidia-compose-ps.txt" 2>&1 || true
    "${compose[@]}" logs --no-color >"${artifact_dir}/nvidia-compose.log" 2>&1 || true
  fi
  if [[ "$keep_running" != "true" ]]; then
    "${compose[@]}" down >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

if [[ "$skip_pull" != "true" ]]; then
  "${compose[@]}" pull
fi
"${compose[@]}" up -d dcgm-exporter vllm sglang tensorrt-llm

startup_timeout="${PIQ_SERVING_STARTUP_TIMEOUT_SECONDS:-1800}"
poll_seconds="${PIQ_SERVING_STARTUP_POLL_SECONDS:-5}"

wait_for_text() {
  name="$1"
  url="$2"
  needle="$3"
  deadline=$((SECONDS + startup_timeout))
  while (( SECONDS < deadline )); do
    body="$(curl -fsS --max-time 10 "$url" 2>/dev/null || true)"
    if grep -q "$needle" <<<"$body"; then
      return 0
    fi
    sleep "$poll_seconds"
  done
  die "$name did not become ready at $url within ${startup_timeout}s."
}

wait_for_model() {
  name="$1"
  base_url="$2"
  deadline=$((SECONDS + startup_timeout))
  while (( SECONDS < deadline )); do
    body="$(curl -fsS --max-time 10 "${base_url}/v1/models" 2>/dev/null || true)"
    if jq -e --arg model "$model" 'any(.data[]?; .id == $model)' <<<"$body" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$poll_seconds"
  done
  die "$name did not advertise exact model '$model' within ${startup_timeout}s."
}

wait_for_text dcgm-exporter "${PIQ_VLLM_HARDWARE_METRICS_URL:-http://127.0.0.1:9400/metrics}" DCGM_FI_DEV_POWER_USAGE
wait_for_model vLLM "${PIQ_VLLM_URL:-http://127.0.0.1:8000}"
wait_for_model SGLang "${PIQ_SGLANG_URL:-http://127.0.0.1:30000}"
wait_for_model TensorRT-LLM "${PIQ_TENSORRT_LLM_URL:-http://127.0.0.1:8001}"

dcgm_body="$(curl -fsS "${PIQ_VLLM_HARDWARE_METRICS_URL:-http://127.0.0.1:9400/metrics}")"
for metric in "${required_dcgm_metrics[@]}"; do
  grep -qE "^${metric}(\{|[[:space:]])" <<<"$dcgm_body" || die "Running DCGM exporter did not expose $metric."
done

set_provenance() {
  service="$1"
  prefix="$2"
  distribution="$3"
  module="$4"
  container_id="$("${compose[@]}" ps -q "$service")"
  [[ -n "$container_id" ]] || die "No container ID found for $service."
  version_probe='import importlib.metadata as metadata, sys; distributions = metadata.packages_distributions().get(sys.argv[2]) or [sys.argv[1]]; print(metadata.version(distributions[0]))'
  framework_version="$(
    docker exec "$container_id" python3 -c "$version_probe" "$distribution" "$module" 2>/dev/null \
      || docker exec "$container_id" python -c "$version_probe" "$distribution" "$module" 2>/dev/null
  )" || die "Could not read the installed framework version from $service."
  [[ -n "$framework_version" ]] || die "Empty installed framework version from $service."
  entrypoint="$(docker inspect --format '{{json .Config.Entrypoint}}' "$container_id")"
  command="$(docker inspect --format '{{json .Config.Cmd}}' "$container_id")"
  printf -v "${prefix}_FRAMEWORK_VERSION" '%s' "$framework_version"
  printf -v "${prefix}_MODEL_REVISION" '%s' "$model_revision"
  printf -v "${prefix}_CONTAINER_ID" '%s' "$container_id"
  printf -v "${prefix}_PROCESS_ID" '%s' "$(docker inspect --format '{{.State.Pid}}' "$container_id")"
  printf -v "${prefix}_IMAGE_TAG" '%s' "$(docker inspect --format '{{.Config.Image}}' "$container_id")"
  printf -v "${prefix}_IMAGE_DIGEST" '%s' "$(docker inspect --format '{{.Image}}' "$container_id")"
  printf -v "${prefix}_SERVER_ARGS" '%s' \
    "$(jq -cn --argjson entrypoint "$entrypoint" --argjson command "$command" \
      '{entrypoint: $entrypoint, command: $command}')"
  printf -v "${prefix}_HOST_NAME" '%s' "$(hostname)"
  printf -v "${prefix}_NODE_NAME" '%s' "$(hostname)"
  export "${prefix}_FRAMEWORK_VERSION" "${prefix}_MODEL_REVISION" \
    "${prefix}_CONTAINER_ID" "${prefix}_PROCESS_ID" "${prefix}_IMAGE_TAG" \
    "${prefix}_IMAGE_DIGEST" "${prefix}_SERVER_ARGS" "${prefix}_HOST_NAME" "${prefix}_NODE_NAME"
}

set_provenance vllm PIQ_VLLM vllm vllm
set_provenance sglang PIQ_SGLANG sglang sglang
set_provenance tensorrt-llm PIQ_TENSORRT_LLM tensorrt_llm tensorrt_llm

if (( ${#smoke_args[@]} > 0 )); then
  bash "${script_dir}/run-smoke.sh" strict-recorded-smoke "${smoke_args[@]}"
else
  bash "${script_dir}/run-smoke.sh" strict-recorded-smoke
fi
echo "Strict NVIDIA serving proof completed. Artifacts: $artifact_dir"
