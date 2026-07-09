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

## Telemetry Model

Streaming collection means the producer sends `stream: true` to the serving
engine, reads the SSE response as each `data:` frame arrives, and timestamps
first byte, first output token/content, every output chunk/token row, and
request completion from the measuring client. This is request-path measurement,
not Kafka streaming.

Do not put Kafka between the producer and the serving engine for TTFT/TPOT
measurement; a broker would add latency and change the measured path. Kafka can
be added later as an optional ingestion transport after the producer has already
captured timestamps, token details, raw operator artifacts, DCGM/native metrics,
hashes, receipts, and provenance.

Smoke runs now write a post-capture event log for Kafka-style ingestion:

```bash
export PIQ_SERVING_EVENT_LOG=$PIQ_ARTIFACT_DIR/serving-events.jsonl
export PIQ_SERVING_KAFKA_TOPIC=performance-iq.serving.telemetry.v1
```

Each JSONL record is a self-contained event with `schemaVersion`, `topic`,
`eventType`, `partitionKey`, `eventId`, campaign/run/request identifiers,
artifact and manifest paths, and a payload. Events include submissions,
artifact pointers, aggregate measurements, `serving_request_sample` rows,
`serving_token_timeline` rows, `serving_telemetry_coverage` rows, native
telemetry, DCGM telemetry, token-detail summaries, request receipts, and the
dashboard snapshot. This is the right
Kafka boundary: publish these already-timestamped events downstream, not the
live request stream used to measure TTFT/TPOT.

To publish that event log to Kafka after capture, install the optional Kafka
extra or use the smoke runner image, then opt in explicitly:

```bash
python -m pip install './python[kafka]'
export PIQ_SERVING_PUBLISH_KAFKA=true
export PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9092
export PIQ_SERVING_KAFKA_CLIENT_ID=performance-iq-serving-producer
```

When Kafka publication is enabled, the proof summary records a
`kafkaPublication` report with the event log path, published event count,
event-type counts, topic counts, client ID, and publication timestamp.

Token-level detail is captured when the engine exposes OpenAI-compatible
`logprobs`/`top_logprobs`. Enable it with:

```bash
export PIQ_SERVING_CAPTURE_TOKEN_DETAILS=true
export PIQ_SERVING_TOP_LOGPROBS=5
```

When logprob items include `token_id`, that engine-provided ID is used. When an
engine returns token text/logprob but omits IDs, strict smoke can resolve IDs
with the Hugging Face tokenizer and records `tokenIdSource` as tokenizer-backed
provenance:

```bash
export PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER=true
# Optional; defaults to PIQ_SERVING_MODEL when left blank.
export PIQ_SERVING_TOKENIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

DCGM hardware counters are captured from a Prometheus/DCGM exporter endpoint
when configured:

```bash
export PIQ_VLLM_HARDWARE_METRICS_URL=http://dcgm-exporter:9400/metrics
export PIQ_SGLANG_HARDWARE_METRICS_URL=http://dcgm-exporter:9400/metrics
export PIQ_TENSORRT_LLM_HARDWARE_METRICS_URL=http://dcgm-exporter:9400/metrics
```

Use `PIQ_SERVING_COLLECT_HARDWARE_METRICS=true` to read DCGM metrics from each
engine `/metrics` endpoint when DCGM is exposed there. Use
`PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY=true` for a strict proof gate that fails
metric completeness if configured engines do not produce hardware telemetry.

Native serving metrics are read from each engine Prometheus endpoint and are
promoted into queryable `serving_request_samples` rows when exposed: native
TTFT/TPOT/E2E/inter-token latency, queue/prefill/decode time, running/waiting
request counts, KV-cache usage, prefix-cache hit rate, cached/computed prompt
token deltas, and the metrics URL/source. Use
`PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY=true` for a strict proof gate that fails
when configured engines do not produce these native timing/cache/concurrency
fields. DCGM sensor counters promoted into the same request rows include power,
GPU/memory-copy utilization, temperature, SM and memory clocks, framebuffer
used/free, and energy.

TensorRT-LLM defaults to the Prometheus metrics endpoint
`/prometheus/metrics` for strict native telemetry. The smoke runner also reads
`PIQ_TENSORRT_LLM_JSON_METRICS_URL` (default `/metrics`) when available, so
TensorRT iteration-stat JSON fields such as iteration latency, GPU memory
usage, active requests, KV-cache block usage, and cache hit rate are preserved
without losing Prometheus queue/prefill/decode timing.

Runtime provenance has two layers. Dashboard rows get safe identifiers and
hashes: framework version, model revision, image tag/digest, server-args hash,
process/container/pod/node/host identifiers, and raw artifact links. Operator
artifacts retain the full raw request/response, token details, native telemetry,
hardware telemetry, full before/after native and DCGM metric snapshots, and full
runtime provenance for internal audit.

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

Start DCGM plus the three engines:

```bash
docker compose \
  --env-file .env.serving-producers \
  -f ops/serving-producers/docker-compose.nvidia.yaml \
  up -d dcgm-exporter vllm sglang tensorrt-llm
```

Run the model-aware endpoint preflight:

```bash
set -a
. ./.env.serving-producers
set +a

bash ops/serving-producers/run-smoke.sh preflight
```

Set the actual per-GPU hourly cost before a strict proof run. The strict wrapper
refuses to run without this because cost-per-token rows would otherwise be
incomplete:

```bash
export PIQ_SERVING_USD_PER_GPU_HOUR=<actual-blended-gpu-hourly-cost>
export PIQ_SERVING_GPU_COUNT=1
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

Submit strict producer runs and verify dashboard materialization:

```bash
bash ops/serving-producers/run-smoke.sh strict-recorded-smoke
```

Use `smoke` only when request receipts are being captured by another layer.
`recorded-smoke` starts in-process receipt proxies, routes all engine traffic
through them, writes `PIQ_SERVING_RECEIPT_LOG`, submits the producer runs,
queries the dashboard surfaces, writes `PIQ_SERVING_EVENT_LOG`, and writes the
proof summary.
Use `strict-smoke` or `strict-recorded-smoke` for product proof: these require
token IDs/logprobs, token ID provenance, native engine telemetry, DCGM hardware
counters, dashboard rows, request receipts, and configured GPU cost.

For CI and local contract checks without real runtimes, use deterministic fake
engines:

```bash
bash ops/serving-producers/run-smoke.sh fake-strict-smoke \
  --repetitions 1 \
  --max-tokens 8 \
  --artifact-dir ./performance-iq-output/fake-strict-serving
```

`fake-strict-smoke` starts local fake OpenAI-compatible vLLM, SGLang, and
TensorRT-LLM endpoints, routes traffic through receipt proxies, captures
stream timing, token IDs/logprobs, prompt token IDs, native metrics, DCGM
counters, operator-full artifacts, raw native/DCGM metric snapshots,
Kafka-ready event rows, and a synthetic dashboard row snapshot. The command
fails unless `verify-proof` is `ok` and `telemetryCoverage.allProven` is true.
This is local contract proof only; it does not replace `strict-recorded-smoke`
against real serving engines.

Verify the saved proof bundle offline:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  "$PIQ_ARTIFACT_DIR/serving-smoke-proof-<suffix>.json"
```

Read two fields in the verifier output separately:

- `ok` proves the proof bundle is internally valid: artifacts hash, manifests
  match, receipts line up, preflight passed, and dashboard rows are queryable.
- `telemetryCoverage.allProven` proves the full product telemetry set is
  present across required engines: client stream timing, native runtime
  telemetry, DCGM counters, tokenizer-exact prompt IDs, output token
  IDs/logprobs, operator-full raw artifacts, request receipts, runtime
  provenance, dashboard fine-grain rows, raw native/DCGM metric snapshots, and
  Kafka-ready event rows.

To inspect every generated row, add a proof-row dump. The output includes
submitted runs, dashboard row snapshots, request samples, token timeline rows,
producer measurement rows, producer telemetry coverage rows, verifier
telemetry coverage rows, native telemetry, DCGM telemetry, request trace rows,
request receipts, and Kafka-ready event rows, with campaign/run/engine
provenance attached to artifact-local rows:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  "$PIQ_ARTIFACT_DIR/serving-smoke-proof-<suffix>.json" \
  --dump-proof-rows "$PIQ_ARTIFACT_DIR/serving-proof-rows.json"
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
- `price_performance`, `capacity_best`, `campaign_provenance`, `run_details`,
  `serving_request_samples`, `serving_token_timeline`, and
  `serving_telemetry_coverage` row counts increase;
- the proof summary preserves row snapshots for those dashboard surfaces;
- submitted campaign IDs appear in both `campaign_provenance` and
  `run_details`, request IDs appear in `serving_request_samples` and
  `serving_token_timeline`, and coverage rows appear in
  `serving_telemetry_coverage`.
- `serving-smoke-proof-<suffix>.json` exists under `PIQ_ARTIFACT_DIR` and
  preserves preflight, per-engine artifact and manifest paths, submissions,
  and dashboard row proof.
- `verify-proof` succeeds against that file, recomputing artifact hashes and
  checking the manifests, endpoint preflight, submitted campaigns, dashboard
  surfaces, token timeline, native telemetry, DCGM counters, and runtime
  framework provenance.
- `serving-events.jsonl` exists and `verify-proof` accepts its Kafka-ready
  event schema, including request-sample and token-timeline events.

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

That Mac-local command is intentionally partial unless a remote TensorRT-LLM
endpoint and DCGM metrics endpoint are also configured. Full product proof uses
`strict-recorded-smoke` with all three engine URLs, native metrics URLs,
hardware metrics URLs, and token logprobs enabled.

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
bash ops/serving-producers/run-smoke.sh strict-recorded-smoke
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
the three engine services. The `performance-iq-serving-producer` secret must
include `base-url` and `usd-per-gpu-hour`; `token` is optional. The job uses
`--record-receipts` by default and writes
`/tmp/performance-iq-serving-producers/serving-smoke-proof.json`, so a
successful pod preserves endpoint preflight, request receipts, submitted
campaign IDs, artifact paths, fine-grain dashboard rows, and strict telemetry
evidence.
