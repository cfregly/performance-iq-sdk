# Serving Producer Operations

This folder is the operator path for proving real vLLM, SGLang, and
TensorRT-LLM serving engines into Performance IQ.

The SDK adapters are endpoint-driven: each engine must expose an
OpenAI-compatible API with:

- `GET /v1/models`
- `POST /v1/chat/completions`

The smoke runner uses the same model and prompt across all three engines,
preflights `/v1/models` until the exact model appears in `data[].id`, sends
completion requests, writes per-engine
normalized summary artifacts and producer manifests, writes one overall smoke
proof summary, submits producer runs, and verifies the fixed Performance IQ
dashboard query surfaces.

## Files

- `performance-iq-serving.env.example` - environment contract for engines and
  Performance IQ.
- `docker-compose.nvidia.yaml` - NVIDIA/Linux local stack template for the
  three serving engines.
- `Dockerfile.smoke` - smoke-runner image for Kubernetes or remote operator
  hosts.
- `kubernetes-smoke-job.yaml` - cluster job template for running the smoke
  from inside the same network as deployed engines.
- `run-smoke.sh` - local wrapper around `performance_iq_sdk.serving_smoke`.

## Local NVIDIA Host

Copy the env template and set the TensorRT-LLM image:

```bash
cp ops/serving-producers/performance-iq-serving.env.example .env.serving-producers
$EDITOR .env.serving-producers
```

Start the three engines:

```bash
docker compose \
  --env-file .env.serving-producers \
  -f ops/serving-producers/docker-compose.nvidia.yaml \
  up -d vllm sglang tensorrt-llm
```

Run the model-aware endpoint preflight:

```bash
set -a
. ./.env.serving-producers
set +a

bash ops/serving-producers/run-smoke.sh preflight
```

When the serving frameworks are installed in a non-default Python environment,
set the runtime discovery knobs before running preflight or smoke:

```bash
export PIQ_PYTHON_BIN=/opt/miniconda3/bin/python
export PIQ_SERVING_BIN_DIR=/opt/miniconda3/bin
export PIQ_VLLM_SOURCE_PATH=/Users/admin/vllm
export PIQ_SGLANG_SOURCE_PATH=/Users/admin/sglang

bash ops/serving-producers/run-smoke.sh preflight
```

`PIQ_PYTHON_BIN` controls which Python executes the smoke CLI.
`PIQ_SERVING_BIN_DIR` is prepended to `PATH` for commands such as `vllm` and
`trtllm-serve`; use a colon-separated value when the serving framework CLIs live
in different virtualenvs. `PIQ_VLLM_SOURCE_PATH` and `PIQ_SGLANG_SOURCE_PATH` are
prepended to `PYTHONPATH` for source-build checkouts.

For one-engine debugging, set the other endpoint env vars to empty and pass
`--allow-missing-engines`. Alternatively, set `PIQ_SERVING_ALLOW_PARTIAL=true`
so `run-smoke.sh` only injects endpoint flags for explicitly configured
engines. Full proof runs must keep all three URLs configured.

```bash
export PIQ_SERVING_MODEL=meta-llama/Llama-3.2-1B-Instruct
export PIQ_SERVING_ALLOW_PARTIAL=true
export PIQ_VLLM_URL=http://127.0.0.1:8010

bash ops/serving-producers/run-smoke.sh no-submit \
  --allow-missing-engines \
  --repetitions 1
```

To preserve engine-side request receipts, run one recorder proxy per engine and
point the smoke URLs at the proxy ports:

```bash
export PIQ_SERVING_RECEIPT_LOG="$PIQ_ARTIFACT_DIR/request-receipts.jsonl"

bash ops/serving-producers/run-smoke.sh receipt-proxy \
  --engine vllm \
  --target-url http://127.0.0.1:8000 \
  --listen-port 18000 \
  --receipt-log "$PIQ_SERVING_RECEIPT_LOG"

bash ops/serving-producers/run-smoke.sh receipt-proxy \
  --engine sglang \
  --target-url http://127.0.0.1:30000 \
  --listen-port 18001 \
  --receipt-log "$PIQ_SERVING_RECEIPT_LOG"

bash ops/serving-producers/run-smoke.sh receipt-proxy \
  --engine tensorrt-llm \
  --target-url http://127.0.0.1:8001 \
  --listen-port 18002 \
  --receipt-log "$PIQ_SERVING_RECEIPT_LOG"

export PIQ_VLLM_URL=http://127.0.0.1:18000
export PIQ_SGLANG_URL=http://127.0.0.1:18001
export PIQ_TENSORRT_LLM_URL=http://127.0.0.1:18002
```

Submit producer runs and verify dashboard materialization:

```bash
bash ops/serving-producers/run-smoke.sh recorded-smoke
```

Use `smoke` only when request receipts are being captured by another layer.
`recorded-smoke` starts in-process receipt proxies, routes all engine traffic
through them, writes `PIQ_SERVING_RECEIPT_LOG`, submits the producer runs,
queries the dashboard surfaces, and writes the proof summary.

Verify the saved proof bundle offline:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  "$PIQ_ARTIFACT_DIR/serving-smoke-proof-<suffix>.json"
```

Partial local proofs are allowed only when the caller opts in. This keeps the
full three-engine gate strict while still letting a Mac operator preserve a real
vLLM-only or SGLang-only E2E proof:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  "$PIQ_ARTIFACT_DIR/serving-smoke-proof-<suffix>.json" \
  --allow-missing-engines
```

Success requires:

- all three `/v1/models` endpoints serve `PIQ_SERVING_MODEL`;
- each engine returns successful chat completions;
- each request has `x-performance-iq-request-id` trace evidence in the
  normalized summary artifact and producer manifest;
- each request ID has a matching receipt in `PIQ_SERVING_RECEIPT_LOG`;
- Performance IQ accepts all three producer runs;
- `price_performance`, `capacity_best`, `campaign_provenance`, and
  `run_details` row counts increase;
- the proof summary preserves row snapshots for those dashboard surfaces;
- submitted campaign IDs appear in both `campaign_provenance` and
  `run_details`.
- `serving-smoke-proof-<suffix>.json` exists under `PIQ_ARTIFACT_DIR` and
  preserves preflight, per-engine artifact and manifest paths, submissions,
  and dashboard row proof.
- `verify-proof` succeeds against that file, recomputing artifact hashes and
  checking the manifests, endpoint preflight, submitted campaigns, dashboard
  surfaces, and runtime framework provenance.

## Mac / Apple Silicon

Use the launch plan for local vLLM and SGLang source-build commands:

```bash
bash ops/serving-producers/run-smoke.sh launch-plan
```

If local source trees already exist, point the smoke wrapper at the same Python
and source paths you use to launch them:

```bash
export PIQ_PYTHON_BIN=/opt/miniconda3/bin/python
export PIQ_SERVING_BIN_DIR=/opt/miniconda3/bin
export PIQ_VLLM_SOURCE_PATH=/Users/admin/vllm

bash ops/serving-producers/run-smoke.sh preflight
```

For a local vLLM E2E on Apple Silicon, build vLLM in an isolated runtime venv
and launch it on a non-conflicting port:

```bash
uv venv /Users/admin/dev/cw/.runtime/vllm-macos --python 3.12
uv pip install --python /Users/admin/dev/cw/.runtime/vllm-macos/bin/python \
  -r /Users/admin/vllm/requirements/cpu.txt \
  --index-strategy unsafe-best-match
uv pip install --python /Users/admin/dev/cw/.runtime/vllm-macos/bin/python \
  setuptools-rust setuptools-scm
VLLM_TARGET_DEVICE=cpu CMAKE_BUILD_PARALLEL_LEVEL=6 MAX_JOBS=6 \
  uv pip install --python /Users/admin/dev/cw/.runtime/vllm-macos/bin/python \
  -e /Users/admin/vllm \
  --no-build-isolation \
  --no-deps

VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=nobind \
  /Users/admin/dev/cw/.runtime/vllm-macos/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8010 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 512 \
  --enforce-eager \
  --dtype float32
```

To include SGLang on the same Mac, use the Apple Metal/MLX source path and
launch it on the standard SGLang port:

```bash
git clone https://github.com/sgl-project/sglang.git /Users/admin/dev/cw/.runtime/sglang
cd /Users/admin/dev/cw/.runtime/sglang
/opt/homebrew/bin/uv venv -p 3.12 sglang-metal
sglang-metal/bin/python -m pip install --upgrade pip
rm -f python/pyproject.toml
mv python/pyproject_other.toml python/pyproject.toml
/opt/homebrew/bin/uv pip install --python sglang-metal/bin/python -e "python[all_mps]"

SGLANG_USE_MLX=1 sglang-metal/bin/python -m sglang.launch_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 30000 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --context-length 512 \
  --max-total-tokens 512
```

With vLLM on `8010` and SGLang on `30000`, submit a receipt-backed local proof
for both configured engines:

```bash
export PATH=/Users/admin/dev/cw/.runtime/vllm-macos/bin:/Users/admin/dev/cw/.runtime/sglang/sglang-metal/bin:$PATH

PIQ_BASE_URL=http://127.0.0.1:3002/workbench-apps/performance-iq \
PIQ_TOKEN=serving-producer-proof-token \
PIQ_PYTHON_BIN=/Users/admin/dev/cw/.runtime/vllm-macos/bin/python \
PIQ_SERVING_BIN_DIR=/Users/admin/dev/cw/.runtime/vllm-macos/bin \
PIQ_VLLM_SOURCE_PATH=/Users/admin/vllm \
PIQ_SGLANG_SOURCE_PATH=/Users/admin/dev/cw/.runtime/sglang/python \
PIQ_SERVING_ALLOW_PARTIAL=true \
PIQ_VLLM_URL=http://127.0.0.1:8010 \
PIQ_SGLANG_URL=http://127.0.0.1:30000 \
  bash ops/serving-producers/run-smoke.sh recorded-smoke \
  --repetitions 1 \
  --max-tokens 8 \
  --artifact-dir ./performance-iq-output/live-vllm-sglang-submit \
  --allow-missing-engines
```

Before installing or deleting anything, run diagnostics to see port listeners,
local runtime availability, free disk, and Hugging Face cache candidates for
the smoke model:

```bash
bash ops/serving-producers/run-smoke.sh diagnostics
```

TensorRT-LLM requires a Linux x86_64/aarch64 target with supported NVIDIA GPUs.
For a Mac workflow, run TensorRT-LLM remotely and set
`PIQ_TENSORRT_LLM_URL` to the reachable OpenAI-compatible endpoint.

## Remote Engines

When engines are already deployed, skip compose and set only the endpoint env:

```bash
export PIQ_BASE_URL=https://performance-iq.example.com
export PIQ_VLLM_URL=https://vllm.example.com
export PIQ_SGLANG_URL=https://sglang.example.com
export PIQ_TENSORRT_LLM_URL=https://trtllm.example.com
export PIQ_SERVING_MODEL=Qwen/Qwen2.5-0.5B-Instruct

bash ops/serving-producers/run-smoke.sh preflight
bash ops/serving-producers/run-smoke.sh smoke
```

Do not use `--skip-preflight` for proof runs. It exists only for targeted
debugging of nonstandard endpoints.

## Kubernetes Smoke Image

Build and push the smoke-runner image from the repository root:

```bash
docker build \
  -f ops/serving-producers/Dockerfile.smoke \
  -t performance-iq-sdk:serving-smoke .
```

For a remote cluster, retag and push that image to the cluster registry, update
`kubernetes-smoke-job.yaml`, and apply the job in the namespace that can reach
the three engine services. The job uses `--record-receipts` by default and
writes `/tmp/performance-iq-serving-producers/serving-smoke-proof.json`, so a
successful pod preserves endpoint preflight, request receipts, submitted
campaign IDs, artifact paths, and dashboard row evidence.
