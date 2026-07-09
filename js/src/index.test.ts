import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

import {
  buildManifest,
  laptopSmokeModel,
  PerformanceIQ,
  runServingProducer,
  servingEngineLabel,
  validateRun,
  type PerformanceIQRunInput,
} from "./index"

const tmpDirs: string[] = []

function tmpArtifact(contents = "{\"ok\":true}\n"): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-sdk-test-"))
  tmpDirs.push(dir)
  const artifactPath = path.join(dir, "summary.json")
  fs.writeFileSync(artifactPath, contents)
  return artifactPath
}

function runInput(overrides: Partial<PerformanceIQRunInput> = {}): PerformanceIQRunInput {
  return {
    sourceType: "fresh-run",
    confidentiality: "operator-full",
    producer: {
      repo: "producer-runner",
      tool: "runner",
      commitSha: "1234567890abcdef",
    },
    campaign: {
      campaignId: "campaign-sdk-test",
      runId: "run-sdk-test",
    },
    workload: {
      model: "llama-3.1-70b",
      hardware: "B200 SXM",
      operatingPoint: "peak",
    },
    runtime: {
      imageDigest: `sha256:${"a".repeat(64)}`,
    },
    artifacts: [tmpArtifact()],
    measurements: [{ outputTpm: 1234 }],
    ...overrides,
  }
}

afterEach(() => {
  for (const dir of tmpDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
  vi.restoreAllMocks()
})

describe("performance-iq-sdk js", () => {
  it("builds a manifest with artifact hashes and pending SDK row proof", async () => {
    const manifest = await buildManifest(runInput())

    expect(manifest.schemaVersion).toBe("performance-iq.producer-manifest.v1")
    expect(manifest.sourceType).toBe("fresh-run")
    expect(manifest.artifacts[0]).toMatchObject({
      kind: "normalized-summary",
      sha256: "e5f1eb4d806641698a35efe20e098efd20d7d57a9b90ee69079d5bb650920726",
      sizeBytes: 12,
    })
    expect(manifest.store).toMatchObject({
      sourceTables: [
        "platform_store.object_store.producer_runner_result_bundles",
        "platform_store.iceberg.intake_store.producer_runner_results",
      ],
      modelTables: ["model_store.sdk_pending_ingest"],
      rowProof: [{ campaignId: "campaign-sdk-test", rowCount: 1 }],
    })
  })

  it("validates source kind, required fields, and live proof classification", async () => {
    const result = await validateRun(runInput())

    expect(result.ok).toBe(true)
    expect(result.liveProofReady).toBe(true)
    expect(result.freshRun).toBe(true)
    expect(result.producerBacked).toBe(true)
  })

  it("rejects non-producer source table names", async () => {
    const result = await validateRun(runInput({
      store: {
        sourceTables: ["model_store.synthetic_fixture"],
        modelTables: ["model_store.sdk_pending_ingest"],
        rowProof: [{ table: "model_store.sdk_pending_ingest", rowCount: 1 }],
      },
    }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("only use latest Producer Runner source tables")
  })

  it("fails closed for customer-safe submissions until governance is implemented", async () => {
    const result = await validateRun(runInput({ confidentiality: "customer-safe" }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("operator-full")
  })

  it("rejects arbitrary SQL/query keys anywhere in the payload", async () => {
    const result = await validateRun(runInput({
      measurements: [{ sql: "SELECT * FROM model_store.secret" }],
    }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("sql")
  })

  it("reports missing local artifacts during local validation", async () => {
    const result = await validateRun(runInput({ artifacts: ["/missing/performance-iq-artifact.json"] }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("ENOENT")
  })

  it("submits validated runs with auth and idempotency headers", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ id: "run-sdk-test", status: "accepted" }), {
      status: 202,
      headers: { "content-type": "application/json" },
    }))
    const client = new PerformanceIQ({
      baseUrl: "https://performance-iq.example",
      token: "service-token",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })

    const result = await client.submitRun(runInput(), { idempotencyKey: "idem-1" })

    expect(result).toEqual({ id: "run-sdk-test", status: "accepted" })
    expect(fetchImpl).toHaveBeenCalledWith("https://performance-iq.example/api/v1/runs", expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({
        authorization: "Bearer service-token",
        "idempotency-key": "idem-1",
      }),
    }))
  })

  it("captures OpenAI-compatible serving engine data as a producer run", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-test-"))
    tmpDirs.push(artifactDir)
    const requests: Array<{ body: unknown; headers: HeadersInit | undefined }> = []
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      requests.push({
        body: JSON.parse(String(init?.body)),
        headers: init?.headers,
      })
      const body = [
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [{ delta: { role: "assistant" } }] },
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [{ delta: { content: "o" } }] },
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [{ delta: { content: "k" }, finish_reason: "stop" }] },
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [], usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n"
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "vllm",
        baseUrl: "http://127.0.0.1:8000",
        frameworkVersion: "test",
        endpointPreflight: {
          url: "http://127.0.0.1:8000/v1/models",
          ok: true,
          modelAvailable: true,
        },
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
        repetitions: 2,
      },
      artifactDir,
      sourceType: "other-measured-producer",
      runClass: "measured",
      workload: {
        hardware: "local mock engine",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(requests.map((request) => request.body)).toEqual([
      expect.objectContaining({ model: laptopSmokeModel(), max_tokens: 64, stream: true }),
      expect.objectContaining({ model: laptopSmokeModel(), max_tokens: 64, stream: true }),
    ])
    expect(requests[0].headers).toMatchObject({
      "x-performance-iq-engine": "vllm",
      "x-performance-iq-request-id": expect.stringContaining("piq-vllm-"),
    })
    expect(result.engine).toBe("vllm")
    expect(result.manifest.producer.tool).toBe("vllm-serving-producer")
    expect(result.manifest.runtime.framework).toBe(servingEngineLabel("vllm"))
    expect(result.manifest.sourceType).toBe("other-measured-producer")
    expect(result.manifest.artifacts[0].path).toBe(result.artifactPath)
    expect(result.samples.every((sample) => sample.ok)).toBe(true)
    expect(result.samples.every((sample) => sample.streaming)).toBe(true)
    expect(result.samples[0]).toMatchObject({
      ttftSource: "client-stream-content",
      promptTokens: 12,
      completionTokens: 8,
      totalTokens: 20,
      outputTokenCount: 8,
      streamChunkCount: 5,
    })
    expect(result.samples[0].ttftMs).not.toBeNull()
    expect(result.samples[0].tpotMs).not.toBeNull()
    expect(result.samples[0].ttfotMs).not.toBeNull()
    expect(result.measurements[0]).toMatchObject({
      model: laptopSmokeModel(),
      runtimeFramework: "vLLM",
      runtimeEngine: "vllm",
      requestCount: 2,
      successCount: 2,
      completionTokens: 16,
      totalTokens: 40,
    })
    expect(result.measurements.filter((row) => row.surface === "serving_request_sample")).toHaveLength(2)
    expect(result.measurements.filter((row) => row.surface === "serving_token_timeline")).toHaveLength(4)
    expect(fs.existsSync(result.artifactPath)).toBe(true)
    expect(fs.existsSync(result.manifestPath)).toBe(true)
    const artifact = JSON.parse(fs.readFileSync(result.artifactPath, "utf-8"))
    const manifestArtifact = JSON.parse(fs.readFileSync(result.manifestPath, "utf-8"))
    expect(artifact.endpointPreflight).toMatchObject({
      url: "http://127.0.0.1:8000/v1/models",
      modelAvailable: true,
    })
    expect(artifact.samples[0].requestId).toBe((requests[0].headers as Record<string, string>)["x-performance-iq-request-id"])
    expect(artifact.samples[0].endpoint).toBe("http://127.0.0.1:8000/v1/chat/completions")
    expect(artifact.requestTrace[0].requestId).toBe(artifact.samples[0].requestId)
    expect(artifact.request.messages).toBeUndefined()
    expect(artifact.capturePolicy.mode).toBe("operator-full")
    expect(manifestArtifact.campaign.campaignId).toBe(result.manifest.campaign.campaignId)
    expect(manifestArtifact.artifacts[0].path).toBe(result.artifactPath)
    expect(manifestArtifact.artifacts[0].sha256).toBe(result.manifest.artifacts[0].sha256)
    expect(manifestArtifact.artifacts.some((entry: { kind: string }) => entry.kind === "operator-full-serving-raw")).toBe(true)
    expect(manifestArtifact.platform.requestTraceIds).toEqual(result.samples.map((sample) => sample.requestId))
    expect((await validateRun(result.runInput)).ok).toBe(true)
  })

  it("captures native Prometheus telemetry deltas when a serving metrics URL is configured", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-native-test-"))
    tmpDirs.push(artifactDir)
    const metricBodies = [
      [
        'vllm:time_to_first_token_seconds_count{model_name="qwen"} 2',
        'vllm:time_to_first_token_seconds_sum{model_name="qwen"} 0.4',
        'vllm:request_time_per_output_token_seconds_count{model_name="qwen"} 2',
        'vllm:request_time_per_output_token_seconds_sum{model_name="qwen"} 0.04',
        'vllm:request_queue_time_seconds_count{model_name="qwen"} 2',
        'vllm:request_queue_time_seconds_sum{model_name="qwen"} 0.01',
        'vllm:request_prefill_time_seconds_count{model_name="qwen"} 2',
        'vllm:request_prefill_time_seconds_sum{model_name="qwen"} 0.20',
        'vllm:request_decode_time_seconds_count{model_name="qwen"} 2',
        'vllm:request_decode_time_seconds_sum{model_name="qwen"} 0.30',
        'vllm:num_requests_running{model_name="qwen"} 1',
        'vllm:num_requests_waiting{model_name="qwen"} 0',
        'vllm:kv_cache_usage_perc{model_name="qwen"} 0.125',
        'vllm:prefix_cache_queries_total{model_name="qwen"} 10',
        'vllm:prefix_cache_hits_total{model_name="qwen"} 1',
        'vllm:prompt_tokens_cached_total{model_name="qwen"} 3',
        'vllm:request_prefill_kv_computed_tokens_sum{model_name="qwen"} 8',
      ].join("\n"),
      [
        'vllm:time_to_first_token_seconds_count{model_name="qwen"} 3',
        'vllm:time_to_first_token_seconds_sum{model_name="qwen"} 0.65',
        'vllm:request_time_per_output_token_seconds_count{model_name="qwen"} 3',
        'vllm:request_time_per_output_token_seconds_sum{model_name="qwen"} 0.06',
        'vllm:request_queue_time_seconds_count{model_name="qwen"} 3',
        'vllm:request_queue_time_seconds_sum{model_name="qwen"} 0.012',
        'vllm:request_prefill_time_seconds_count{model_name="qwen"} 3',
        'vllm:request_prefill_time_seconds_sum{model_name="qwen"} 0.26',
        'vllm:request_decode_time_seconds_count{model_name="qwen"} 3',
        'vllm:request_decode_time_seconds_sum{model_name="qwen"} 0.42',
        'vllm:num_requests_running{model_name="qwen"} 1',
        'vllm:num_requests_waiting{model_name="qwen"} 0',
        'vllm:kv_cache_usage_perc{model_name="qwen"} 0.25',
        'vllm:prefix_cache_queries_total{model_name="qwen"} 15',
        'vllm:prefix_cache_hits_total{model_name="qwen"} 2',
        'vllm:prompt_tokens_cached_total{model_name="qwen"} 6',
        'vllm:request_prefill_kv_computed_tokens_sum{model_name="qwen"} 16',
      ].join("\n"),
    ]
    const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const href = String(url)
      if (href.endsWith("/metrics")) {
        return new Response(metricBodies.shift() ?? "", {
          status: 200,
          headers: { "content-type": "text/plain" },
        })
      }
      expect(init?.method).toBe("POST")
      const body = [
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [{ delta: { content: "ok" } }] },
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [], usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 } },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "vllm",
        baseUrl: "http://127.0.0.1:8000",
        metricsUrl: "http://127.0.0.1:8000/metrics",
        frameworkVersion: "vllm-test",
        modelRevision: "revision-a",
        imageTag: "vllm:test",
        imageDigest: "sha256:abc",
        serverArgs: ["vllm", "serve", laptopSmokeModel()],
        processId: "1234",
        containerId: "container-a",
        podName: "pod-a",
        nodeName: "node-a",
        hostName: "host-a",
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
      },
      artifactDir,
      workload: {
        hardware: "local mock engine",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    expect(metricBodies).toHaveLength(0)
    expect(fetchImpl).toHaveBeenCalledTimes(3)
    const sample = result.samples[0]
    expect(sample.nativeTelemetryAvailable).toBe(true)
    expect(sample.nativeTelemetry?.nativeTtftMs as number).toBeCloseTo(250)
    expect(sample.nativeTelemetry?.nativeTpotMs as number).toBeCloseTo(20)
    expect(sample.nativeTelemetry?.queueWaitMs as number).toBeCloseTo(2)
    expect(sample.nativeTelemetry?.prefillMs as number).toBeCloseTo(60)
    expect(sample.nativeTelemetry?.decodeMs as number).toBeCloseTo(120)
    expect(sample.nativeTelemetry?.prefixCacheQueriesDelta).toBe(5)
    expect(sample.nativeTelemetry?.prefixCacheHitsDelta).toBe(1)
    expect(sample.nativeTelemetry?.cacheHitRate as number).toBeCloseTo(0.2)
    expect(sample.runningRequests).toBe(1)
    expect(sample.waitingRequests).toBe(0)
    expect(sample.kvCacheUsagePct).toBeCloseTo(0.25)
    expect(sample.promptTokensCachedDelta).toBe(3)
    expect(sample.promptTokensComputedDelta).toBe(8)
    expect(sample.engineVersion).toBe("vllm-test")
    expect(sample.modelRevision).toBe("revision-a")
    expect(sample.imageTag).toBe("vllm:test")
    expect(sample.imageDigest).toBe("sha256:abc")
    expect(sample.serverArgsSha256).toEqual(expect.any(String))
    expect(sample.processId).toBe("1234")
    expect(sample.containerId).toBe("container-a")
    expect(sample.podName).toBe("pod-a")
    expect(sample.nodeName).toBe("node-a")
    expect(sample.hostName).toBe("host-a")
    expect(sample.queueWaitMs).toBeCloseTo(2)
    expect(sample.prefillMs).toBeCloseTo(60)
    expect(sample.decodeMs).toBeCloseTo(120)
    expect(result.measurements[0].nativeTelemetryAvailableCount).toBe(1)
    expect(result.measurements[0].avgQueueWaitMs as number).toBeCloseTo(2)
    expect(result.measurements[0].avgPrefillMs as number).toBeCloseTo(60)
    expect(result.measurements[0].avgDecodeMs as number).toBeCloseTo(120)
    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    expect(sampleRows[0].nativeTtftMs).toBeCloseTo(250)
    expect(sampleRows[0].runningRequests).toBe(1)
    expect(sampleRows[0].promptTokensCachedDelta).toBe(3)
    expect(sampleRows[0].engineVersion).toBe("vllm-test")
    expect(sampleRows[0].containerId).toBe("container-a")
  })

  it("captures response token logprobs and DCGM hardware metrics", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-token-dcgm-test-"))
    tmpDirs.push(artifactDir)
    const metricBodies = [
      [
        'DCGM_FI_DEV_POWER_USAGE{gpu="0"} 100',
        'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 40',
        'DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 12',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 60',
        'DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1800',
        'DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5000',
        'DCGM_FI_DEV_FB_USED{gpu="0"} 4096',
        'DCGM_FI_DEV_FB_FREE{gpu="0"} 8192',
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 1000',
      ].join("\n"),
      [
        'DCGM_FI_DEV_POWER_USAGE{gpu="0"} 120',
        'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 50',
        'DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 20',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 61',
        'DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1801',
        'DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5001',
        'DCGM_FI_DEV_FB_USED{gpu="0"} 4097',
        'DCGM_FI_DEV_FB_FREE{gpu="0"} 8191',
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 2500',
      ].join("\n"),
    ]
    const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const href = String(url)
      if (href.endsWith("/dcgm")) {
        return new Response(metricBodies.shift() ?? "", {
          status: 200,
          headers: { "content-type": "text/plain" },
        })
      }
      expect(init?.method).toBe("POST")
      const parsed = JSON.parse(String(init?.body))
      expect(parsed.logprobs).toBe(true)
      expect(parsed.top_logprobs).toBe(2)
      const body = [
        {
          id: "chatcmpl-test",
          model: laptopSmokeModel(),
          choices: [{
            delta: { content: "o" },
            logprobs: {
              content: [{
                token: "o",
                token_id: 101,
                logprob: -0.1,
                top_logprobs: [
                  { token: "o", token_id: 101, logprob: -0.1 },
                  { token: "O", token_id: 102, logprob: -2.0 },
                ],
              }],
            },
          }],
        },
        {
          id: "chatcmpl-test",
          model: laptopSmokeModel(),
          choices: [{
            delta: { content: "k" },
            finish_reason: "stop",
            logprobs: { content: [{ token: "k", token_id: 202, logprob: -0.2 }] },
          }],
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "vllm",
        baseUrl: "http://127.0.0.1:8000",
        hardwareMetricsUrl: "http://127.0.0.1:9400/dcgm",
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
        captureTokenDetails: true,
        topLogprobs: 2,
      },
      artifactDir,
      workload: {
        hardware: "local dcgm engine",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    expect(metricBodies).toHaveLength(0)
    const sample = result.samples[0]
    expect(sample.hardwareTelemetryAvailable).toBe(true)
    expect(sample.tokenDetailsAvailable).toBe(true)
    expect(sample.tokenIdsAvailable).toBe(true)
    expect(sample.logprobsAvailable).toBe(true)
    expect(sample.tokenDetailCount).toBe(2)
    expect(sample.tokenIdSource).toBe("response-logprobs")
    expect(sample.tokenTimeline?.[0].tokenId).toBe(101)
    expect(sample.tokenTimeline?.[0].tokenIdSource).toBe("response-logprobs")
    expect(sample.tokenTimeline?.[0].tokenLogprob).toBeCloseTo(-0.1)
    expect(sample.avgPowerWatts).toBeCloseTo(120)
    expect(sample.avgPowerWattsPerGpu).toBeCloseTo(120)
    expect(sample.gpuUtilizationPct).toBeCloseTo(50)
    expect(sample.memoryCopyUtilizationPct).toBeCloseTo(20)
    expect(sample.gpuTemperatureC).toBeCloseTo(61)
    expect(sample.smClockMHz).toBeCloseTo(1801)
    expect(sample.memoryClockMHz).toBeCloseTo(5001)
    expect(sample.fbUsedMiB).toBeCloseTo(4097)
    expect(sample.fbFreeMiB).toBeCloseTo(8191)
    expect(sample.energyJoules).toBeCloseTo(1.5)
    expect(result.measurements[0].dcgmGrounded).toBe(true)
    expect(result.measurements[0].hardwareTelemetryAvailableCount).toBe(1)
    expect(result.measurements[0].tokenDetailsAvailableCount).toBe(1)
    expect(result.measurements[0].tokenIdsAvailableCount).toBe(1)
    expect(result.measurements[0].logprobsAvailableCount).toBe(1)
    expect(result.measurements[0].powerSource).toBe("dcgm")
    const timelineRows = result.measurements.filter((row) => row.surface === "serving_token_timeline")
    expect(timelineRows[0].tokenId).toBe(101)
    expect(timelineRows[0].tokenIdSource).toBe("response-logprobs")
    expect(timelineRows[1].tokenLogprob as number).toBeCloseTo(-0.2)
    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    expect(sampleRows[0].tokenIdSource).toBe("response-logprobs")
    expect(sampleRows[0].gpuTemperatureC).toBeCloseTo(61)
    expect(sampleRows[0].smClockMHz).toBeCloseTo(1801)
    expect(sampleRows[0].fbUsedMiB).toBeCloseTo(4097)
  })

  it("resolves missing token IDs from a configured token map", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-token-map-test-"))
    tmpDirs.push(artifactDir)
    const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(init?.method).toBe("POST")
      const body = [
        {
          id: "chatcmpl-test",
          model: laptopSmokeModel(),
          choices: [{
            delta: { content: "o" },
            logprobs: {
              content: [{
                token: "o",
                logprob: -0.1,
                top_logprobs: [
                  { token: "o", logprob: -0.1 },
                  { token: "O", logprob: -2.0 },
                ],
              }],
            },
          }],
        },
        {
          id: "chatcmpl-test",
          model: laptopSmokeModel(),
          choices: [{
            delta: { content: "k" },
            finish_reason: "stop",
            logprobs: { content: [{ token: "k", logprob: -0.2 }] },
          }],
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "vllm",
        baseUrl: "http://127.0.0.1:8000",
        tokenIdMap: { o: 101, O: 102, k: 202 },
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
        captureTokenDetails: true,
        topLogprobs: 2,
      },
      artifactDir,
      workload: {
        hardware: "local token map engine",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    const sample = result.samples[0]
    expect(sample.tokenIdsAvailable).toBe(true)
    expect(sample.tokenIdSource).toBe("configured-token-id-map")
    expect(sample.tokenTimeline?.[0].tokenId).toBe(101)
    expect(sample.tokenTimeline?.[0].tokenIdSource).toBe("configured-token-id-map")
    const topLogprobs = JSON.parse(String(sample.tokenTimeline?.[0].topLogprobsJson))
    expect(topLogprobs[1].tokenId).toBe(102)
    expect(topLogprobs[1].tokenIdSource).toBe("configured-token-id-map")
    const timelineRows = result.measurements.filter((row) => row.surface === "serving_token_timeline")
    expect(timelineRows[1].tokenId).toBe(202)
    expect(timelineRows[1].tokenIdSource).toBe("configured-token-id-map")
  })
})
