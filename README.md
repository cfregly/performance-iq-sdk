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
  confidentiality: "internal-full",
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
    "confidentiality": "internal-full",
    "producer": {"tool": "my-runner", "repo": "acme/benchmarks", "commitSha": "abc1234"},
    "campaign": {"campaignId": "campaign-123", "runId": "run-123"},
    "workload": {"model": "llama-3.1-70b", "hardware": "B200 SXM", "operatingPoint": "peak"},
    "runtime": {"imageDigest": "sha256:" + "a" * 64},
    "artifacts": ["./run.log", "./summary.json"],
    "measurements": [{"outputTpm": 1234}],
})
```

## Safety Rules

- SDKs never accept or forward caller-provided SQL, query names, or query lists.
- `customer-safe`, `public-safe`, and `redacted` writes fail closed until server-side governance is implemented.
- `preserved-snapshot`, `fresh-run`, and `other-measured-producer` are distinct source kinds and must not be collapsed into one proof label.
- `fresh-run` is reserved for Runner-owned proof; most customer integrations should start with `other-measured-producer`.
- Rehearsal packets can validate and submit, but cannot be promoted to live proof.
