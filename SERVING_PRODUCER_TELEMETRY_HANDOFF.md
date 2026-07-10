# Serving Producer Telemetry Handoff

Generated: 2026-07-10

## Current status

The producer framework, fine-grain storage contract, durable local ingestion,
StarRocks query path, and browser dashboard are implemented and tested. Fresh
real local requests were sent to vLLM CPU and SGLang MLX with the same model,
`Qwen/Qwen2.5-0.5B-Instruct`. Their rows are committed to Nessie/Iceberg and
visible through the normal Workbench `starrocks` connector.

This is not yet strict three-engine NVIDIA proof. The remaining execution gate
is trusted access to a Linux/NVIDIA host for real TensorRT-LLM, DCGM, and an
SGLang CUDA runtime that supports decode logprobs. The configured Verda host is
presenting an unverified replacement SSH host key, so it was not used. The
Verda console session is currently signed out. The prior sign-in tab was no
longer present when the browser handoff was finalized, so the next session
must reopen the Verda console and authenticate.

Use `producers`, not `probes`, for this framework.

## Local runtime importability

The producer orchestrator does not need vLLM, SGLang, or TensorRT-LLM imported
into its own Python environment. It measures their OpenAI-compatible HTTP
endpoints and supports a dedicated interpreter for each runtime.

Current local preflight results:

```text
vLLM preferred Python:
/Users/admin/dev/cw/.runtime/vllm/.venv/bin/python
vllm import, vllm._C import, and vllm command: launch-ready

SGLang preferred Python:
/Users/admin/dev/cw/.runtime/sglang/sglang-metal/bin/python
sglang and sglang.launch_server imports: launch-ready

TensorRT-LLM:
not installed on macOS; requires the Linux/NVIDIA container path below
```

`/opt/miniconda3/bin/vllm` is a stale global console script whose interpreter
cannot import `vllm`; it is not the runtime used by the local proof. The smoke
preflight now discovers and reports the dedicated usable environments so that
global importability is not confused with endpoint/runtime readiness.

## Implemented

- OpenAI-compatible `stream: true` request collection for vLLM, SGLang, and
  TensorRT-LLM endpoints.
- Request timing: E2E, TTFB, TTFT, TTFOT, TPOT, inter-token latency, first/last
  output timestamps, stream chunks, prompt/output/total tokens, and status.
- Native telemetry: queue, prefill, decode, native TTFT/TPOT/E2E/ITL, running
  and waiting requests, KV/cache state, engine/model revision, image, server
  args, process/container/pod/node/host provenance.
- DCGM collection for power, utilization, SM/DRAM/tensor and FP activity,
  PCIe/NVLink throughput and counters, clocks, framebuffer, temperature,
  encoder/decoder, XID/ECC, power/thermal violations, raw metric inventory, and
  energy.
- Python and JS strict hardware gates now require the expanded DCGM contract;
  Python can no longer report full DCGM coverage from only power/util/temp.
- DCGM finite blank/unsupported sentinels are excluded from derived and
  queryable values, while the verbatim exposition and invalid-series reason
  remain in operator-full artifacts. Valid `DCGM_EXP_*` health rows are now
  queryable alongside `DCGM_FI_*` rows.
- Prompt/output token IDs, token hashes, output logprobs/top-logprobs when the
  runtime supports them, and tokenizer provenance.
- Operator-full raw request/response bodies and verbatim before/after native
  and DCGM metric exposition. Customer-facing rows contain hashes and redacted
  derived values.
- Queryable surfaces:
  - `serving_request_samples`
  - `serving_token_timeline`
  - `serving_metric_snapshots`
  - `serving_telemetry_coverage`
- Aggregate `run_details` percentiles for E2E, TTFB, TTFT, TTFOT, TPOT, ITL,
  queue, prefill, and decode.
- Request receipts, manifest/artifact hashes, content-addressed gzip bundles,
  Nessie commit hashes, Iceberg snapshot IDs, and Kafka-ready JSONL events.
- Deterministic gzip (`mtime=0`) plus canonical-payload validation for safe
  retries of legacy nondeterministic bundles.
- Typed ISO-8601 timestamp coercion before PyArrow/Iceberg writes.
- Platform Store source contracts and four serving fact models/views compile
  against the pinned production dbt versions.
- Local StarRocks bridge views expose Nessie `main` through the same
  `model_store.fact_performance_iq_serving_*` names used by registered queries.
- Desktop and 390px mobile UI proof, including exact sub-millisecond TPOT/ITL,
  hard-wrapped provenance, and unique timeline keys.
- Reproducible NVIDIA compose stack with content-addressed image refs for vLLM
  `v0.23.0`, SGLang `v0.5.12`, TensorRT-LLM `1.2.1`, and DCGM Exporter
  `4.5.3-4.8.2-distroless`, plus model revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Co-resident one-GPU defaults, explicit strict DCGM counters at 33 ms,
  TensorRT execution backend enforcement, full operator runtime provenance
  including live installed framework version, entrypoint, argv, and backend,
  and a one-command NVIDIA E2E runner.
- TensorRT-LLM `1.2.1` uses its exact CLI contract: `--revision`,
  `--backend tensorrt`, and a YAML config with `return_perf_metrics: true` and
  retained request metrics. Unsupported `--hf_revision` and
  `--served_model_name` flags were removed.
- The TensorRT producer reads `/prometheus/metrics`, `/metrics`, and the
  drain-on-read `/perf_metrics` endpoint. It persists native queue, prefill,
  TTFT, decode, E2E, optional TPOT/ITL, request-count/hash, iteration,
  concurrency, memory, and exact per-request KV block fields without treating
  cache blocks as prompt-token counts.
- The NVIDIA runner records the observed GPU name and a complete
  `nvidia-gpu-inventory.csv`; every engine's operator runtime configuration
  links the inventory path and SHA-256.
- Customer-safe request rows now retain endpoint, request start/completion,
  response ID/model, streaming mode, output bytes, runtime backend, operating
  point/basis, native JSON/per-request metrics URLs, redacted error hash, and
  latency alias in addition to the existing timing/token/hardware fields.
- The durable intake contract, dbt fact, governed DBO view, named query,
  in-process query adapter, frontend transform, and telemetry board expose the
  same request fields. The intake contract also accepts the complete serving
  aggregate row instead of rejecting expanded latency/DCGM/completeness data.
- Contract generation normalizes dbt `timestamp_ntz` to the ingestion
  service's canonical `timestamp` type, and maps `usdPer1m*` to the existing
  `usd_per_1m_*` cost columns used by Platform Store facts.

## Real local runtime proof

Artifact directory:

```text
/Users/admin/dev/cw/performance-iq-sdk/output/real-local-serving-metric-snapshots-20260710
```

Producer proof counts:

```text
real engine submissions                  2
serving_request_samples                  2
serving_token_timeline                  26
serving_metric_snapshots              1552
serving_telemetry_coverage              20
event log rows                        1621
```

Raw native metric capture:

```text
vLLM:   52,664 bytes before / 54,096 after; 394 / 409 numeric series
SGLang: 58,109 bytes before / 61,601 after; 360 / 389 numeric series
```

Two-engine verifier boundary at capture time:

```text
verification.ok = true
realRuntimeProofGate.ok = true
strictTelemetryGate.ok = false
strictTelemetryGate.missingCategories = ["dcgmHardwareTelemetry", "outputTokenLogprobs"]
```

Re-verification with the current stricter runtime-provenance gate still returns
`verification.ok = true` and `realRuntimeProofGate.ok = true`, but correctly
adds `runtimeProvenance` to `strictTelemetryGate.missingCategories`. These
historical Mac artifacts predate the new operator `runtimeConfiguration` block
and do not contain image/container/backend provenance. Do not use the older
`real-local-reverify.json` file as evidence for the current strict provenance
gate; it was generated before that gate was tightened.

The missing output-logprob category is SGLang MLX only. Tokenizer-exact SGLang
output IDs are present, but `tokenLogprob` is intentionally null and provenance
records `sglang-mps-mlx-logprobs-crash`. vLLM output IDs and logprobs are
present. DCGM is absent because this Mac has no NVIDIA device.

Important producer files:

```text
serving-smoke-summary.json
real-local-reverify.json
finest-grain-rows.json
serving-events.jsonl
serving-request-receipts.jsonl
vllm-Qwen-Qwen2.5-0.5B-Instruct-2026-07-10T03-26-28-748260Z-operator-full.json
sglang-Qwen-Qwen2.5-0.5B-Instruct-2026-07-10T03-27-54-661048Z-operator-full.json
```

## Durable row proof

The exact real envelopes were accepted by the durable FastAPI service and
committed through run-scoped Nessie branches into Iceberg `main`.

```text
vLLM Nessie commit:
13edda02e13003848dad37a489c6b19eb331da32e7abe4c4bfc36c882e416a63

SGLang Nessie commit:
be4f0d933840c510534e7ec6b9897eb87ccfd94a519ebf03240c856b13bbc862
```

Persisted real-row counts:

```text
intake_store.producer_runner_results                         2
intake_store.performance_iq_serving_request_sample           2
intake_store.performance_iq_serving_token_timeline          26
intake_store.performance_iq_serving_metric_snapshot       1,552
intake_store.performance_iq_serving_telemetry_coverage      20
intake_store.performance_iq_ingestion_runs                   2
```

Both idempotent POST replays returned the original receipt, and GET-by-run
returned byte-equivalent receipt content. All 1,552 metric rows carry model,
runtime framework, runtime engine, campaign, run, request, source, phase,
label hash, value, ordinal, raw exposition hash, and artifact provenance.

Durable evidence files:

```text
durable-iceberg-all-rows.json
durable-iceberg-row-summary.json
durable-persistence-proof.json
```

`durable-iceberg-all-rows.json` contains every persisted column for 1,600
fine-grain serving rows plus two result and two ledger rows (1,604 total).
Its SHA-256 is:

```text
7067fcb13b0a63c470add1152e957cf314623e4e8f7746ac8d329b1947f17b61
```

These real Mac rows predate the later TensorRT per-request fields, request
endpoint/start/end/backend columns, and GPU inventory hash. Their historical
durability remains valid, but they are not evidence that those newer columns
were populated by a real NVIDIA runtime.

## Dashboard proof

The Workbench named-query API returned the real rows with:

```text
x-piq-store-provider: starrocks
x-piq-deployment-mode: platform-hosted
x-piq-tenant: operator
```

Real-run rows observed through registered queries:

```text
serving_request_samples       2
serving_token_timeline       26
serving_metric_snapshots   1,552
serving_telemetry_coverage   20
```

Screenshots:

```text
/Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq/output/playwright/serving-producer-live-starrocks.png
/Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq/output/playwright/vllm-real-serving-request-row.png
/Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq/output/playwright/sglang-real-serving-request-row.png
/Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq/output/playwright/sglang-real-serving-request-row-mobile.png
```

Proof boundary: local StarRocks uses an idempotent bridge view over the durable
Nessie/Iceberg intake facts. Production still uses the compiled dbt models and
governed DBO views; they have not been deployed to a production environment in
this session.

## Validation

Passed after the final TensorRT, DCGM, image, runner, and runtime-provenance
strictness changes:

```text
Python SDK                                  80 passed
JavaScript SDK                              36 passed
JavaScript SDK type-check                   passed
Durable ingestion service                  47 passed
Workbench unit suite                        73 passed
Workbench full npm test                     passed
Workbench type-check                        passed
Serving producers dashboard proof           passed
Static data-contract/query-drift audit       passed
Generated ingestion contract reproducibility passed
Raw dbt timestamp variants in intake contract 0
git diff --check across all three repos      passed
Desktop browser console errors              0
390px serving panel overflow                 0
```

Pinned Platform Store compile:

```bash
uvx --python 3.12 --from dbt-core==1.9.3 \
  --with dbt-spark==1.9.3 --with pyspark==4.1.2 \
  dbt compile --project-dir dbt/store \
  --profiles-dir /tmp/piq-dbt-profiles \
  --select \
    fact_performance_iq_serving_request_sample \
    fact_performance_iq_serving_token_timeline \
    fact_performance_iq_serving_metric_snapshot \
    fact_performance_iq_serving_telemetry_coverage \
  --no-partial-parse
```

Result: dbt 1.9.3/Spark adapter 1.9.3 found 21 models and 268 data
tests; all four selected serving models compiled.

NVIDIA stack validation also passes locally without starting GPU containers:

```bash
bash ops/serving-producers/run-nvidia-e2e.sh validate \
  --env-file ops/serving-producers/performance-iq-serving.env.example
```

All four content-addressed registry manifests exist, the compose file renders,
the model revision is immutable, the strict DCGM counter inventory is
complete, and the runner rejects non-digest images or a TensorRT-LLM PyTorch
fallback. It also validates the exact TensorRT-LLM v1.2.1 revision/backend/
metrics-config contract.

The Linux/NVIDIA run path is covered by a controlled command fixture that
exercises GPU/runtime preflight, compose start/readiness, all three live
container provenance probes, GPU inventory capture, strict-smoke argument
forwarding, cleanup, diagnostic log capture, and nonzero smoke-exit
propagation. This is orchestration proof, not real GPU-runtime proof.

Fresh deterministic three-engine integration smoke:

```bash
bash ops/serving-producers/run-smoke.sh fake-strict-smoke \
  --repetitions 1 \
  --max-tokens 4 \
  --artifact-dir output/fake-schema-e2e-20260710
```

Result: all telemetry categories were proven for vLLM, SGLang, and
TensorRT-LLM, including receipts, token rows, native/DCGM rows, metric
snapshots, and complete runtime provenance. The fresh run produced three
request rows, 21 token rows, 495 metric-snapshot rows, and 30 coverage rows.
`strictTelemetryGate.ok = true` because the telemetry contract is complete;
the separate `realRuntimeProofGate.ok = false` correctly rejects the synthetic
proof boundary. This validates the full contract without misrepresenting it as
NVIDIA runtime evidence.

Every generated row is available at:

```text
/Users/admin/dev/cw/performance-iq-sdk/output/fake-schema-e2e-20260710/finest-grain-rows.json
/Users/admin/dev/cw/performance-iq-sdk/output/fake-schema-e2e-20260710/verification.json
```

The row dump contains 552 measurements (3 aggregate, 3 request, 21 token, 495
metric snapshot, and 30 producer coverage rows), 39 verifier coverage rows, 579
event rows, eight request receipts, three request traces, and three native plus
three hardware telemetry records.

A fresh TensorRT fake-envelope cross-repo smoke then projected and persisted
the exact SDK measurement set through the FastAPI `IngestionService` after
replacing only the synthetic repeated-digit image digest with the validated
TensorRT image digest. The persistence receipt covered 200 measurements:

```text
producer_runner_results                         1
performance_iq_serving_request_sample           1
performance_iq_serving_token_timeline           7
performance_iq_serving_metric_snapshot        181
performance_iq_serving_telemetry_coverage      10
```

The projected row preserved canonical cost fields, aggregate TTFT/completeness,
typed request timestamps, endpoint, backend, `/perf_metrics` URL, TensorRT KV
block fields/request hash, and GPU inventory hash. This proves schema and
transaction compatibility; it remains synthetic runtime evidence.

## Workspace state

The implementation is present but not committed. It spans three independent
Git repositories; `/Users/admin/dev/cw` itself is not the repository root:

```text
/Users/admin/dev/cw/performance-iq-sdk
/Users/admin/dev/cw/workbench
/Users/admin/dev/cw/platform-store
```

The temporary ingestion container, Next.js development server, and Playwright
browser used for verification were stopped after the evidence was captured.
The shared local Nessie, MinIO, and StarRocks services and their durable data
were not removed.

## Resume commands

Re-verify the current two-engine proof:

```bash
cd /Users/admin/dev/cw/performance-iq-sdk
PYTHONPATH=python/src /opt/miniconda3/bin/python \
  -m performance_iq_sdk.serving_smoke \
  --verify-proof \
    output/real-local-serving-metric-snapshots-20260710/serving-smoke-summary.json \
  --allow-missing-engines
```

Install the local StarRocks query bridge after the local lakehouse starts:

```bash
cd /Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq/services/ingestion
docker exec -i platform-starrocks mysql -h127.0.0.1 -P9030 -uroot \
  < scripts/local_starrocks_serving_views.sql
```

Run the app against StarRocks:

```bash
cd /Users/admin/dev/cw/workbench/apps/data-platforms/performance-iq
PIQ_STORE_PROVIDER=starrocks \
NEXT_PUBLIC_PIQ_STORE_PROVIDER=starrocks \
NEXT_PUBLIC_USE_SAMPLE_DATA=false \
PIQ_IDENTITY_MODE=none \
STARROCKS_HOST=127.0.0.1 \
STARROCKS_PORT=9030 \
STARROCKS_USERNAME=root \
STARROCKS_PASSWORD='' \
STARROCKS_DB=model_store \
npm run dev -- --hostname 127.0.0.1 --port 3000
```

## Remaining work

1. Verify the current Verda VM identity and SSH host key through the signed-in
   provider console or another authoritative control-plane channel. The host at
   `31.22.104.239` presents ED25519 fingerprint
   `SHA256:xd06KEymgO17malEm5UVUvzAak+hGzFuhbqx3BCkiJQ`, while the pinned key is
   different. Do not bypass strict checking or replace `known_hosts` without
   provider attestation.
   Rechecked on 2026-07-10: the presented and pinned fingerprints are unchanged.
   The Verda project URL now redirects to the sign-in form. Authenticate in the
   Verda console, inspect the VM identity/key through the provider control
   plane, and only then update `known_hosts` if the replacement key is attested.
2. On that trusted Linux/NVIDIA host, copy/configure `.env.serving-producers`
   with the reachable Performance IQ URL/token and actual GPU price, then run:

```bash
bash ops/serving-producers/run-nvidia-e2e.sh run \
  --env-file .env.serving-producers \
  --keep-running
```

   This starts pinned real vLLM, SGLang CUDA, TensorRT-LLM with the TensorRT
   backend, and DCGM against the same model; waits for exact-model readiness;
   captures actual image/container/process/host provenance; and runs the strict
   recorded smoke plus dashboard verifier.
3. Confirm the target B200 exposes every configured DCGM profiling/interconnect
   field. Unsupported DCGM sentinels now fail closed instead of being accepted
   as measurements.
4. Capture SGLang CUDA output logprobs and run all three engines without
   `--allow-missing-engines`.
5. Run the final gate:

```bash
PYTHONPATH=python/src python3 -m performance_iq_sdk.serving_smoke \
  --verify-proof <gpu-node-proof-summary.json> \
  --require-telemetry-coverage \
  --require-real-runtime-proof
```

6. Deploy/run the compiled dbt models and governed DBO views in the target
   production Platform Store/Airflow environment, including the additive
   Iceberg request/aggregate schema migration. The local durable and browser
   path is proven; production deployment is not claimed here.

No configured alternative NVIDIA endpoint was usable in this session: the
other SSH aliases either timed out or failed strict host-key verification.
