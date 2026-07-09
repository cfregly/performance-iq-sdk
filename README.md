# performance-iq-sdk

Customer-facing SDKs for submitting benchmark results into Performance IQ.

This repo is intentionally separate from Workbench. The SDKs help producers
build, validate, hash, and submit results packets. Performance IQ server APIs
remain responsible for auth, SQL access, normalization, quote-readiness, and
confidentiality gates.

The v1 API is still pre-release. The SDKs preserve the current contract while
the server shape continues to evolve.

## Packages

- `js/` - TypeScript/JavaScript SDK and `piq` CLI.
- `python/` - Python SDK.
- `contracts/` - shared producer manifest schema.
- `ops/serving-producers/` - repeatable vLLM, SGLang, and TensorRT-LLM
  endpoint launch/smoke templates for producer proof runs.

## TypeScript

```ts
import { PerformanceIQ } from "performance-iq-sdk"

const piq = new PerformanceIQ({
  baseUrl: "https://performance-iq.example.com",
  token: process.env.PIQ_TOKEN,
})

await piq.submitRun({
  sourceType: "other-measured-producer",
  confidentiality: "operator-full",
  producer: { tool: "my-runner", repo: "acme/benchmarks", commitSha: "abc1234" },
  campaign: { campaignId: "campaign-123", runId: "run-123" },
  workload: { model: "llama-3.1-70b", hardware: "B200 SXM", operatingPoint: "peak" },
  runtime: { imageDigest: "sha256:" + "a".repeat(64) },
  artifacts: ["./run.log", "./summary.json"],
  measurements: [{ outputTpm: 1234 }],
})
```

CLI:

```bash
PIQ_BASE_URL=https://performance-iq.example.com PIQ_TOKEN=... \
  piq submit-manifest ./manifest.json
```

## Offline buyer verification (`piq verify-packet`)

`verify-packet` is the "don't trust us — replay it" primitive. A **buyer's**
own engineers can check a Performance IQ deal-proof packet with plain Node and
nothing else: **no server, no token, no license.** It is the demand-side lever
of the flywheel — the format spreads because buyers can verify it without
depending on the vendor.

```bash
piq verify-packet ./deal-packet.json
#   --max-age-days <n>   freshness window (default 30)
#   --allow-rehearsal    inspect non-measured packets instead of rejecting them
#   --artifacts <dir>    directory holding raw artifacts, to recompute hashes
#   --json               machine-readable result
```

It is fail-closed. It rejects example/template/rehearsal packets (only
`runClass: measured` is quote-grade), stale evidence, artifact hashes that do
not match shipped raw files, customer-facing packets that leak operator-only
detail (absolute paths, private hosts), and incomplete replay recipes. A
customer-safe packet that ships hashes but not raw files is treated as
declared-only, not a failure — re-run the producer to reproduce. Exit code is
`0` on pass, `1` on fail. See
`workbench/apps/data-platforms/performance-iq/RFP_LANGUAGE.md` for the
buyer-facing procurement clause that references it.

## Python

```py
from performance_iq_sdk import PerformanceIQ

piq = PerformanceIQ(
    base_url="https://performance-iq.example.com",
    token=os.environ["PIQ_TOKEN"],
)

piq.submit_run({
    "sourceType": "other-measured-producer",
    "confidentiality": "operator-full",
    "producer": {"tool": "my-runner", "repo": "acme/benchmarks", "commitSha": "abc1234"},
    "campaign": {"campaignId": "campaign-123", "runId": "run-123"},
    "workload": {"model": "llama-3.1-70b", "hardware": "B200 SXM", "operatingPoint": "peak"},
    "runtime": {"imageDigest": "sha256:" + "a" * 64},
    "artifacts": ["./run.log", "./summary.json"],
    "measurements": [{"outputTpm": 1234}],
})
```

## Serving Producers

The SDK includes producer adapters for OpenAI-compatible serving engines:
`vllm`, `sglang`, and `tensorrt-llm`. They send chat-completion requests to the
runtime, capture request/usage/latency/provenance, write a normalized summary
artifact, and submit a Performance IQ run.

TypeScript:

```ts
import { PerformanceIQ, laptopSmokeModel, runServingProducer } from "performance-iq-sdk"

const piq = new PerformanceIQ({
  baseUrl: "https://performance-iq.example.com",
  token: process.env.PIQ_TOKEN,
})

await runServingProducer({
  engine: { engine: "vllm", baseUrl: "http://127.0.0.1:8000" },
  request: {
    model: laptopSmokeModel(),
    messages: [{ role: "user", content: "Return a short acknowledgement." }],
    repetitions: 3,
  },
  performanceIq: piq,
  artifactDir: "./performance-iq-output",
  sourceType: "other-measured-producer",
  runClass: "measured",
  workload: {
    hardware: "local dev",
    operatingPoint: "laptop-smoke",
  },
  pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
})
```

Python:

```py
from performance_iq_sdk import PerformanceIQ, laptop_smoke_model, run_serving_producer

piq = PerformanceIQ("https://performance-iq.example.com", token=os.environ["PIQ_TOKEN"])

run_serving_producer(
    engine={"engine": "sglang", "baseUrl": "http://127.0.0.1:30000"},
    request={
        "model": laptop_smoke_model(),
        "messages": [{"role": "user", "content": "Return a short acknowledgement."}],
        "repetitions": 3,
    },
    performance_iq=piq,
    artifact_dir="./performance-iq-output",
    source_type="other-measured-producer",
    run_class="measured",
    workload={"hardware": "local dev", "operatingPoint": "laptop-smoke"},
    pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
)
```

Runtime defaults are intentionally thin: each adapter targets
`/v1/chat/completions` and relies on the runtime being launched with an
OpenAI-compatible API server. Use the same model across engines for comparable
rows. `Qwen/Qwen2.5-0.5B-Instruct` is the current laptop-smoke default.

Example endpoints:

- vLLM: `http://127.0.0.1:8000/v1/chat/completions`
- SGLang: `http://127.0.0.1:30000/v1/chat/completions`
- TensorRT-LLM: configure the OpenAI-compatible server URL for your deployment.

### Streaming collection and Kafka

Streaming collection means the producer sends OpenAI-compatible `stream: true`
requests to the serving engine, reads each SSE `data:` frame as it arrives, and
timestamps first byte, first output token/content, each output chunk/token row,
and request completion from the measuring client. These client-side stream
timestamps are what make TTFT, TTFOT, TPOT, inter-token latency, and E2E
latency portable across vLLM, SGLang, and TensorRT-LLM.

Kafka should not sit between the producer and the serving engine for those
measurements, because that would change the request path being measured.
Kafka is supported as a post-capture ingestion/export boundary: the smoke
runner writes Kafka-ready JSONL events after it has already captured request
timings, token details, raw operator artifacts, DCGM/native metrics, request
receipts, hashes, and provenance.

Use these knobs for strict product proof:

```bash
export PIQ_SERVING_CAPTURE_TOKEN_DETAILS=true
export PIQ_SERVING_TOP_LOGPROBS=5
export PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER=true
export PIQ_SERVING_COLLECT_HARDWARE_METRICS=true
export PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY=true
export PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY=true
export PIQ_SERVING_EVENT_LOG=./performance-iq-output/serving-producers/serving-events.jsonl
```

Enable Kafka publication only after local event-log capture is working:

```bash
export PIQ_SERVING_PUBLISH_KAFKA=true
export PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9092
export PIQ_SERVING_KAFKA_TOPIC=performance-iq.serving.telemetry.v1
```

The finest-grain dashboard surfaces are `serving_request_samples`,
`serving_token_timeline`, and `serving_telemetry_coverage`.
`serving_request_samples` is one row per request with latency, native/DCGM,
token-summary, provenance, and artifact fields. The request row exposes core
DCGM power/utilization/temperature/clock/memory/energy fields plus PCIe,
NVLink, encoder/decoder, SM/DRAM/tensor/FP pipe activity, XID/ECC, violation
time, and raw-metric-name provenance fields. `serving_token_timeline` is prompt
and output token/chunk detail with `tokenPhase`, token IDs, logprobs, hashes,
timing, and provenance. `serving_telemetry_coverage` is one row per
engine/category showing whether producer-local telemetry categories were
proven, missing, partial, or not configured. Restricted operator-full artifacts
also retain the full before/after native metrics and DCGM Prometheus snapshots;
dashboard rows expose bounded derived fields, raw metric counts/name hashes, and
artifact links rather than raw metric maps.

To exercise the full telemetry contract without real serving runtimes, use the
deterministic fake strict path:

```bash
bash ops/serving-producers/run-smoke.sh fake-strict-smoke \
  --repetitions 1 \
  --max-tokens 8 \
  --artifact-dir ./performance-iq-output/fake-strict-serving
```

This starts local fake OpenAI-compatible vLLM, SGLang, and TensorRT-LLM
endpoints, routes them through receipt proxies, captures stream timings,
token IDs/logprobs, native metrics, DCGM counters, prompt token IDs, raw
operator artifacts, raw native/DCGM metric snapshots, Kafka-ready events, and
proof rows, then verifies `strictTelemetryGate.ok`. It is CI/local
contract proof only and fails `realRuntimeProofGate.ok` when that gate is
required; real product proof still requires `strict-recorded-smoke` against
actual serving engines.

### Three-engine smoke

Use the Python smoke runner when you have real serving endpoints and want one
proof packet for all three frameworks:

```bash
export PIQ_BASE_URL=http://127.0.0.1:3002
export PIQ_VLLM_URL=http://127.0.0.1:8000
export PIQ_SGLANG_URL=http://127.0.0.1:30000
export PIQ_TENSORRT_LLM_URL=http://127.0.0.1:8001

python -m performance_iq_sdk.serving_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --repetitions 3 \
  --artifact-dir ./performance-iq-output/serving-producers \
  --record-receipts \
  --query-dashboard
```

From this source checkout, run the same command as:

```bash
PYTHONPATH=python/src python -m performance_iq_sdk.serving_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --repetitions 3 \
  --artifact-dir ./performance-iq-output/serving-producers \
  --record-receipts \
  --query-dashboard
```

The command sends the same chat-completion prompt to vLLM, SGLang, and
TensorRT-LLM, writes one normalized summary artifact and one producer manifest
per engine plus one overall smoke proof summary, submits three producer runs,
and checks the fixed dashboard surfaces `price_performance`, `capacity_best`,
`campaign_provenance`, and `run_details`.
Each engine request carries `x-performance-iq-*` trace headers, including
`x-performance-iq-request-id`. Each summary artifact includes those request
IDs, request samples, derived measurements, and the endpoint preflight evidence
used for that engine. Each manifest preserves the hashed artifact pointer and
request trace IDs submitted to Performance IQ.
Route traffic through `run-smoke.sh receipt-proxy` and set
`PIQ_SERVING_RECEIPT_LOG` when you need operator-visible proof that those exact
request IDs reached each backend serving endpoint.
The overall `serving-smoke-proof-<suffix>.json` file preserves the submitted
campaign IDs, per-engine artifact/manifest paths, preflight evidence, and
dashboard row proof in one place, including the fixed surface row snapshots
used to inspect the data behind the dashboard insights.
Verify that saved proof bundle offline before treating it as full E2E evidence:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  ./performance-iq-output/serving-producers/serving-smoke-proof-<suffix>.json
```

Dump all generated proof rows for inspection or handoff:

```bash
bash ops/serving-producers/run-smoke.sh verify-proof \
  ./performance-iq-output/serving-producers/serving-smoke-proof-<suffix>.json \
  --dump-proof-rows ./performance-iq-output/serving-producers/serving-proof-rows.json
```

The verifier requires all three producers, accepted submissions, matching
artifact hashes, producer manifests, model-aware endpoint preflight, dashboard
campaign rows, and runtime framework provenance.
Its JSON output also includes `telemetryCoverage`: `ok: true` means the proof
bundle is internally valid, while `telemetryCoverage.allProven: true` means all
telemetry categories configured by the run are proven across required engines.
Coverage categories include client stream timing, request receipts, dashboard
fine-grain rows, native runtime telemetry, DCGM hardware telemetry, prompt token
IDs, output token IDs/logprobs, operator-full artifacts, raw native/DCGM metric
snapshots, runtime provenance, and Kafka-ready event rows. `--dump-proof-rows`
also emits `telemetryCoverageRows`, one row per engine/category from the verifier.
Dashboard and Kafka-ready event-log coverage is evaluated per required engine
campaign, not from global row totals alone.
Native runtime telemetry, DCGM hardware telemetry, token timeline rows, and
operator-full raw metric snapshots are matched by request ID.
Add `--require-telemetry-coverage` to make `verify-proof` exit nonzero unless
`strictTelemetryGate.ok` is true. That gate requires every full-product category
to be configured and proven for every required engine, including prompt token
IDs and output token IDs/logprobs.
Add `--require-real-runtime-proof` for product proof. That separate gate fails
when proof-boundary fields declare fake, synthetic, fixture, or mock runtime
evidence. Fake strict smoke can pass `strictTelemetryGate.ok` while failing
`realRuntimeProofGate.ok`; real product proof should require both flags.
It fails fast unless all three URLs are configured and the configured endpoints
pass the model-aware `/v1/models` preflight. Use `--allow-missing-engines` only
for partial local debugging and `--skip-preflight` only when debugging a
nonstandard endpoint by hand.
When using `run-smoke.sh` for partial debugging, set
`PIQ_SERVING_ALLOW_PARTIAL=true` so the wrapper only injects endpoint flags for
explicitly configured engines, then pass `--allow-missing-engines`.
You can also set unused endpoint env vars to empty strings, for example
`PIQ_SGLANG_URL=` and `PIQ_TENSORRT_LLM_URL=`.

Run a non-mutating readiness check first when setting up real engines:

```bash
PYTHONPATH=python/src python -m performance_iq_sdk.serving_smoke \
  --preflight-only \
  --vllm-url http://127.0.0.1:8000 \
  --sglang-url http://127.0.0.1:30000 \
  --tensorrt-llm-url http://127.0.0.1:8001
```

The local wrapper resolves the smoke Python and runtime source paths for you.
Set these when framework packages live outside the default shell environment:

```bash
export PIQ_PYTHON_BIN=/opt/miniconda3/bin/python
export PIQ_SERVING_BIN_DIR=/opt/miniconda3/bin
export PIQ_VLLM_SOURCE_PATH=/Users/admin/vllm
export PIQ_SGLANG_SOURCE_PATH=/Users/admin/sglang

bash ops/serving-producers/run-smoke.sh preflight
```

`PIQ_PYTHON_BIN` selects the Python that runs the smoke CLI.
`PIQ_SERVING_BIN_DIR` is prepended to `PATH` for runtime commands.
`PIQ_VLLM_PYTHON_BIN` and `PIQ_SGLANG_PYTHON_BIN` point at runtime-specific
Python environments when the smoke CLI Python is not the same interpreter that
can import the serving framework.
`PIQ_VLLM_SOURCE_PATH` and `PIQ_SGLANG_SOURCE_PATH` are prepended to
`PYTHONPATH` so source-build checkouts are visible to preflight.

The preflight prints local binary/module status (`vllm`, `sglang`,
`trtllm-serve`, `nvidia-smi`), local free disk for source builds/model
downloads, and checks each configured endpoint at
`/v1/models`. The endpoint must return a standard OpenAI-compatible model list
with the configured smoke model in `data[].id`; otherwise the proof run fails
before any inference requests are sent. It does not send inference requests or
write Performance IQ runs.

To print the host-aware launch plan without checking endpoints:

```bash
PYTHONPATH=python/src python -m performance_iq_sdk.serving_smoke \
  --launch-plan-only \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

`runtimeDiscovery` in the launch plan reports usable per-engine Python
candidates and rewrites the vLLM/SGLang serve commands to the preferred working
binary when one is found. A missing `vllm` or `sglang` import in the smoke CLI
Python is not fatal when the corresponding runtime-specific candidate is usable.
When `--resolve-token-ids-with-tokenizer` is enabled, the producer also uses the
runtime-specific Python as an external Hugging Face tokenizer fallback, so
dashboard-safe token rows can still carry tokenizer-exact IDs when the smoke CLI
Python lacks `transformers`.

SGLang's Apple Silicon MPS/MLX path currently does not safely return decode
logprobs. Preflight reads SGLang `/server_info`; when `device` is `mps` and
token details are requested, the smoke runner records an
`outputTokenIdsLogprobs` capability gap and omits `logprobs` from the request so
the local server does not crash. Use a CUDA/Linux SGLang endpoint, or explicitly
set `PIQ_SGLANG_ALLOW_UNSAFE_TOKEN_DETAILS=true` for debugging that failure
mode.

To print read-only setup diagnostics before installing or deleting anything:

```bash
PYTHONPATH=python/src python -m performance_iq_sdk.serving_smoke \
  --diagnostics-only \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

Diagnostics include local runtime availability, default port listeners, free
disk, environment presence flags, and Hugging Face cache candidates for the
configured smoke model.

Reference launch shapes for the same smoke model:

```bash
# vLLM OpenAI-compatible server
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct

# SGLang OpenAI-compatible server
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct

# TensorRT-LLM OpenAI-compatible server; requires NVIDIA/CUDA.
trtllm-serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8001
```

Host notes:

- vLLM on Apple Silicon is a source-build path today; use the launch-plan
  output to keep the endpoint and model name aligned with the smoke runner.
- SGLang on Apple Silicon should run through its Metal/MLX path with
  `SGLANG_USE_MLX=1`.
- TensorRT-LLM requires a Linux x86_64/aarch64 target with supported NVIDIA
  GPUs. From a Mac, point `PIQ_TENSORRT_LLM_URL` at a reachable remote
  OpenAI-compatible TensorRT-LLM server.

Operational templates for NVIDIA hosts, Kubernetes jobs, and remote endpoint
proof runs live in `ops/serving-producers/`.

## Safety Rules

- SDKs never accept or forward caller-provided SQL, query names, or query lists.
- `customer-safe`, `public-safe`, and `redacted` writes fail closed until server-side governance is implemented.
- `fresh-run` and `other-measured-producer` are distinct source kinds and must not be collapsed into one proof label.
- `fresh-run` is reserved for Runner-owned proof; most customer integrations should start with `other-measured-producer`.
- Rehearsal packets can validate and submit, but cannot be promoted to live proof.
