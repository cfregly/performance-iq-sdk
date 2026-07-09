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

## Safety Rules

- SDKs never accept or forward caller-provided SQL, query names, or query lists.
- `customer-safe`, `public-safe`, and `redacted` writes fail closed until server-side governance is implemented.
- `preserved-snapshot`, `fresh-run`, and `other-measured-producer` are distinct source kinds and must not be collapsed into one proof label.
- `fresh-run` is reserved for Runner-owned proof; most customer integrations should start with `other-measured-producer`.
- Rehearsal packets can validate and submit, but cannot be promoted to live proof.
