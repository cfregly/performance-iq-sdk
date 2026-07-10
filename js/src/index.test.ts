import fs from "node:fs"
import { createHash } from "node:crypto"
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
        "platform_store.iceberg.intake_platform_store.hpc_perftest_raw",
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
        runtimeBackend: "unit-test-backend",
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
      p50TtftMs: expect.any(Number),
      p99TpotMs: expect.any(Number),
    })
    expect(result.measurements.filter((row) => row.surface === "serving_request_sample")).toHaveLength(2)
    expect(result.measurements.filter((row) => row.surface === "serving_token_timeline")).toHaveLength(4)
    const coverageRows = result.measurements.filter((row) => row.surface === "serving_telemetry_coverage")
    expect(coverageRows).toHaveLength(11)
    expect(coverageRows.map((row) => row.coverageCategory)).toContain("clientStreamTiming")
    expect(coverageRows).toContainEqual(expect.objectContaining({
      coverageCategory: "kafkaEventLog",
      coverageStatus: "proven",
      provenCount: 1,
      expectedCount: 1,
    }))
    expect(fs.existsSync(result.artifactPath)).toBe(true)
    expect(fs.existsSync(result.manifestPath)).toBe(true)
    expect(typeof result.eventLogPath).toBe("string")
    expect(fs.existsSync(result.eventLogPath!)).toBe(true)
    const artifact = JSON.parse(fs.readFileSync(result.artifactPath, "utf-8"))
    const manifestArtifact = JSON.parse(fs.readFileSync(result.manifestPath, "utf-8"))
    const eventLog = fs.readFileSync(result.eventLogPath!, "utf-8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line))
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
    expect(manifestArtifact.artifacts.some((entry: { kind: string }) => entry.kind === "serving-telemetry-event-log")).toBe(true)
    expect(eventLog.every((event) => event.schemaVersion === "performance-iq.serving-telemetry-event.v1")).toBe(true)
    expect(eventLog.every((event) => event.topic === "performance-iq.serving.telemetry.v1")).toBe(true)
    expect(eventLog.every((event) => typeof event.eventId === "string" && event.eventId.length === 64)).toBe(true)
    expect(eventLog).toContainEqual(expect.objectContaining({
      eventType: "serving.producer_run",
      payload: expect.objectContaining({
        campaignId: result.manifest.campaign.campaignId,
        servingRequestSampleCount: 2,
        servingTokenTimelineCount: 4,
      }),
    }))
    expect(eventLog.filter((event) => event.eventType === "serving.measurement.serving_request_sample")).toHaveLength(2)
    expect(eventLog.filter((event) => event.eventType === "serving.measurement.serving_token_timeline")).toHaveLength(4)
    const rawArtifact = manifestArtifact.artifacts.find((entry: { kind: string; path: string }) => entry.kind === "operator-full-serving-raw")
    expect(rawArtifact).toBeDefined()
    if (!rawArtifact) throw new Error("operator-full-serving-raw artifact missing")
    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    expect(sampleRows[0]).toMatchObject({
      requestEndpoint: "http://127.0.0.1:8000/v1/chat/completions",
      responseId: "chatcmpl-test",
      responseModel: laptopSmokeModel(),
      runtimeBackend: "unit-test-backend",
      operatingPoint: "laptop-smoke",
      basis: "per_request",
      streaming: true,
      outputBytes: 2,
      errorSha256: null,
    })
    expect(sampleRows[0].requestStartedAtUtc).toEqual(expect.any(String))
    expect(sampleRows[0].requestCompletedAtUtc).toEqual(expect.any(String))
    expect(sampleRows[0].latencyMs).toBe(sampleRows[0].e2eLatencyMs)
    expect(sampleRows[0].rawArtifactPath).toBe(rawArtifact.path)
    expect(coverageRows.find((row) => row.coverageCategory === "operatorFullArtifacts")?.proofPath).toBe(rawArtifact.path)
    expect(manifestArtifact.platform.requestTraceIds).toEqual(result.samples.map((sample) => sample.requestId))
    expect((await validateRun(result.runInput)).ok).toBe(true)
  })

  it("fails closed when a streaming serving response is interrupted", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-interrupted-test-"))
    tmpDirs.push(artifactDir)
    const encoder = new TextEncoder()
    const fetchImpl = vi.fn(async () => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({
          id: "chatcmpl-interrupted",
          model: laptopSmokeModel(),
          choices: [{ delta: { role: "assistant" } }],
        })}\n\n`))
        controller.error(new Error("stream interrupted after role chunk"))
      },
    }), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }))

    const result = await runServingProducer({
      engine: {
        engine: "vllm",
        baseUrl: "http://127.0.0.1:8000",
        frameworkVersion: "test",
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
        repetitions: 1,
      },
      artifactDir,
      sourceType: "other-measured-producer",
      runClass: "measured",
      workload: {
        hardware: "local stream engine",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })

    const sample = result.samples[0]
    expect(sample.ok).toBe(false)
    expect(sample.status).toBe(0)
    expect(sample.streaming).toBe(true)
    expect(sample.e2eLatencyMs).toEqual(expect.any(Number))
    expect(sample.timeToFirstByteMs).toBeNull()
    expect(sample.ttftMs).toBeNull()
    expect(sample.ttfotMs).toBeNull()
    expect(sample.tpotMs).toBeNull()
    expect(sample.streamChunkCount).toBe(0)
    expect(sample.outputTokenCount).toBe(0)
    expect(sample.error).toContain("stream interrupted")

    expect(result.measurements[0]).toMatchObject({
      successCount: 0,
      errorCount: 1,
    })
    expect(result.measurements[0].metricCompleteness as number).toBeLessThan(1)
    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    const timelineRows = result.measurements.filter((row) => row.surface === "serving_token_timeline")
    const coverageRows = result.measurements.filter((row) => row.surface === "serving_telemetry_coverage")
    expect(sampleRows[0]).toMatchObject({ ok: false, ttftMs: null, tpotMs: null, ttfotMs: null })
    expect(sampleRows[0].errorSha256).toMatch(/^[0-9a-f]{64}$/)
    expect(fs.existsSync(String(sampleRows[0].rawArtifactPath))).toBe(true)
    expect(timelineRows).toHaveLength(0)
    expect(coverageRows.find((row) => row.coverageCategory === "clientStreamTiming")).toMatchObject({
      coverageStatus: "missing",
    })
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
        nativeTelemetry: {
          trtllmIterationLatencyMs: 7,
          trtllmGpuMemoryBytes: 2000,
          trtllmKvCacheUsedBlocks: 4,
          trtllmKvCacheMaxBlocks: 10,
        },
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Say ok." }],
        promptTokenIds: [11, 22, 33],
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
    expect(sample.nativeIterationLatencyMs).toBe(7)
    expect(sample.nativeGpuMemoryBytes).toBe(2000)
    expect(sample.nativeKvCacheUsedBlocks).toBe(4)
    expect(sample.nativeKvCacheMaxBlocks).toBe(10)
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
    expect(sample.promptTokenIdsAvailable).toBe(true)
    expect(sample.promptTokenDetailCount).toBe(3)
    expect(sample.promptTokenIdSource).toBe("configured-prompt-token-ids")
    expect(sample.promptTokenIdsSha256).toEqual(expect.any(String))
    expect(sample.tokenTimeline?.filter((row) => row.tokenPhase === "prompt")).toHaveLength(3)
    expect(sample.tokenTimeline?.find((row) => row.tokenPhase === "prompt")?.tokenId).toBe(11)
    expect(result.measurements[0].nativeTelemetryAvailableCount).toBe(1)
    expect(result.measurements[0].promptTokenDetailsRequired).toBe(true)
    expect(result.measurements[0].promptTokenIdsAvailableCount).toBe(1)
    expect(result.measurements[0].avgQueueWaitMs as number).toBeCloseTo(2)
    expect(result.measurements[0].avgPrefillMs as number).toBeCloseTo(60)
    expect(result.measurements[0].avgDecodeMs as number).toBeCloseTo(120)
    expect(result.measurements[0].avgNativeIterationLatencyMs).toBe(7)
    expect(result.measurements[0].avgNativeGpuMemoryBytes).toBe(2000)
    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    expect(sampleRows[0].nativeTtftMs).toBeCloseTo(250)
    expect(sampleRows[0].runningRequests).toBe(1)
    expect(sampleRows[0].promptTokensCachedDelta).toBe(3)
    expect(sampleRows[0].nativeIterationLatencyMs).toBe(7)
    expect(sampleRows[0].nativeGpuMemoryBytes).toBe(2000)
    expect(sampleRows[0].engineVersion).toBe("vllm-test")
    expect(sampleRows[0].containerId).toBe("container-a")
    expect(sampleRows[0].promptTokenIdsAvailable).toBe(true)
    expect(sampleRows[0].promptTokenIdSource).toBe("configured-prompt-token-ids")
    const promptTimelineRows = result.measurements.filter((row) => row.surface === "serving_token_timeline" && row.tokenPhase === "prompt")
    expect(promptTimelineRows).toHaveLength(3)
    expect(promptTimelineRows[0].tokenId).toBe(11)
    const artifact = JSON.parse(fs.readFileSync(result.artifactPath, "utf-8"))
    const rawArtifact = JSON.parse(fs.readFileSync(artifact.capturePolicy.rawArtifactPath, "utf-8"))
    expect(rawArtifact.captures[0].nativeMetricsRaw.before.available).toBe(true)
    expect(rawArtifact.captures[0].nativeMetricsRaw.after.available).toBe(true)
    expect(rawArtifact.captures[0].nativeMetricsRaw.before.metrics).toHaveProperty("vllm:time_to_first_token_seconds_count")
  })

  it("derives late-emitted SGLang native metrics", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-sglang-late-metrics-test-"))
    tmpDirs.push(artifactDir)
    const metricBodies = [
      [
        "sglang:num_running_reqs 0",
        "sglang:num_queue_reqs 0",
        "sglang:token_usage 0",
        "sglang:cache_hit_rate 0",
      ].join("\n"),
      [
        "sglang:time_to_first_token_seconds_count 1",
        "sglang:time_to_first_token_seconds_sum 0.11",
        "sglang:inter_token_latency_seconds_count 3",
        "sglang:inter_token_latency_seconds_sum 0.03",
        "sglang:e2e_request_latency_seconds_count 1",
        "sglang:e2e_request_latency_seconds_sum 0.14",
        "sglang:queue_time_seconds_count 1",
        "sglang:queue_time_seconds_sum 0.005",
        'sglang:per_stage_req_latency_seconds_count{stage="prefill_forward"} 1',
        'sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward"} 0.1',
        "sglang:num_running_reqs 0",
        "sglang:num_queue_reqs 0",
        "sglang:token_usage 0",
        "sglang:cache_hit_rate 0",
        "sglang:uncached_prompt_tokens_histogram_sum 32",
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
        { id: "chatcmpl-test", model: laptopSmokeModel(), choices: [], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "sglang",
        baseUrl: "http://127.0.0.1:30000",
        metricsUrl: "http://127.0.0.1:30000/metrics",
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
    })

    expect(metricBodies).toHaveLength(0)
    const sample = result.samples[0]
    expect(sample.nativeTelemetryAvailable).toBe(true)
    expect(sample.nativeTelemetry?.nativeTtftMs as number).toBeCloseTo(110)
    expect(sample.nativeTelemetry?.nativeTpotMs as number).toBeCloseTo(10)
    expect(sample.nativeTelemetry?.nativeInterTokenLatencyMs as number).toBeCloseTo(10)
    expect(sample.nativeTelemetry?.nativeE2eLatencyMs as number).toBeCloseTo(140)
    expect(sample.nativeTelemetry?.queueWaitMs as number).toBeCloseTo(5)
    expect(sample.nativeTelemetry?.prefillMs as number).toBeCloseTo(100)
    expect(sample.nativeTelemetry?.decodeMs as number).toBeCloseTo(30)
    expect(sample.promptTokensCachedDelta).toBe(0)
    expect(sample.promptTokensComputedDelta).toBe(32)
    expect(result.measurements[0].nativeTelemetryRequired).toBe(true)
    const nativeCoverage = result.measurements.find((row) => row.surface === "serving_telemetry_coverage" && row.coverageCategory === "nativeRuntimeTelemetry")
    expect(nativeCoverage).toMatchObject({
      coverageStatus: "proven",
      provenCount: 1,
      expectedCount: 1,
    })
    const rawCoverage = result.measurements.find((row) => row.surface === "serving_telemetry_coverage" && row.coverageCategory === "rawMetricSnapshots")
    expect(rawCoverage).toMatchObject({
      coverageStatus: "proven",
      provenCount: 1,
      expectedCount: 1,
    })
    const rawArtifact = JSON.parse(fs.readFileSync(String(rawCoverage?.proofPath), "utf-8"))
    expect(rawArtifact.captures[0].nativeMetricsRaw.before.available).toBe(true)
    expect(rawArtifact.captures[0].nativeMetricsRaw.after.available).toBe(true)
  })

  it("collects TensorRT-LLM Prometheus, iteration, and per-request metrics", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-trt-metrics-url-test-"))
    tmpDirs.push(artifactDir)
    const metricUrls: string[] = []
    let perfMetricReads = 0
    const fetchImpl = vi.fn(async (url: string | URL | Request) => {
      const href = String(url)
      if (href.endsWith("/prometheus/metrics")) {
        metricUrls.push(href)
        return new Response("", { status: 200, headers: { "content-type": "text/plain" } })
      }
      if (href.endsWith("/metrics")) {
        metricUrls.push(href)
        return new Response('[{"gpuMemUsage": 2000, "iterLatencyMS": 7, "kvCacheStats": {"usedNumBlocks": 4, "maxNumBlocks": 10, "cacheHitRate": 0.4}, "numActiveRequests": 1}]', {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      }
      if (href.endsWith("/perf_metrics")) {
        metricUrls.push(href)
        perfMetricReads += 1
        return new Response(perfMetricReads === 1 ? "[]" : JSON.stringify([{
          request_id: 99,
          perf_metrics: {
            timing_metrics: {
              arrival_time: 100,
              first_scheduled_time: 100.005,
              first_token_time: 100.2,
              last_token_time: 100.7,
            },
            kv_cache_metrics: {
              num_total_allocated_blocks: 10,
              num_new_allocated_blocks: 8,
              num_reused_blocks: 2,
              num_missed_blocks: 8,
            },
          },
        }]), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      }
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
        engine: "tensorrt-llm",
        baseUrl: "http://127.0.0.1:8001",
        collectNativeMetrics: true,
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
    })

    expect(metricUrls).toEqual([
      "http://127.0.0.1:8001/prometheus/metrics",
      "http://127.0.0.1:8001/metrics",
      "http://127.0.0.1:8001/perf_metrics",
      "http://127.0.0.1:8001/prometheus/metrics",
      "http://127.0.0.1:8001/metrics",
      "http://127.0.0.1:8001/perf_metrics",
    ])
    expect(result.samples[0].nativeTelemetry?.queueWaitMs as number).toBeCloseTo(5)
    expect(result.samples[0].nativeTelemetry?.prefillMs as number).toBeCloseTo(195)
    expect(result.samples[0].nativeTelemetry?.decodeMs as number).toBeCloseTo(500)
    expect(result.samples[0].nativeTelemetry?.nativeTpotMs as number).toBeCloseTo(500)
    expect(result.samples[0].trtllmPerfRecordCount).toBe(1)
    expect(result.samples[0].trtllmPerfRequestIdSha256).toMatch(/^[0-9a-f]{64}$/)
    expect(result.measurements.some((row) => row.surface === "serving_metric_snapshot" && row.metricSource === "native-perf-json")).toBe(true)
  })

  it("captures response token logprobs and DCGM hardware metrics", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-token-dcgm-test-"))
    tmpDirs.push(artifactDir)
    const metricBodies = [
      [
        'DCGM_FI_DEV_POWER_USAGE{gpu="0"} 100',
        'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 40',
        'DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 12',
        'DCGM_FI_PROF_SM_ACTIVE{gpu="0"} 0.4',
        'DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.1',
        'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0"} 0.6',
        'DCGM_FI_PROF_PIPE_FP64_ACTIVE{gpu="0"} 0.01',
        'DCGM_FI_PROF_PIPE_FP32_ACTIVE{gpu="0"} 0.3',
        'DCGM_FI_PROF_PIPE_FP16_ACTIVE{gpu="0"} 0.5',
        'DCGM_FI_DEV_PCIE_TX_THROUGHPUT{gpu="0"} 111',
        'DCGM_FI_DEV_PCIE_RX_THROUGHPUT{gpu="0"} 333',
        'DCGM_FI_PROF_PCIE_TX_BYTES{gpu="0"} 1000',
        'DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0"} 2000',
        'DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="0"} 3',
        'DCGM_FI_PROF_NVLINK_TX_BYTES{gpu="0"} 3000',
        'DCGM_FI_PROF_NVLINK_RX_BYTES{gpu="0"} 4000',
        'DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL{gpu="0"} 200',
        'DCGM_FI_DEV_ENC_UTIL{gpu="0"} 7',
        'DCGM_FI_DEV_DEC_UTIL{gpu="0"} 10',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 60',
        'DCGM_FI_DEV_MEMORY_TEMP{gpu="0"} 9223372036854775794',
        'DCGM_EXP_GPU_HEALTH_STATUS{gpu="0"} 0',
        'DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1800',
        'DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5000',
        'DCGM_FI_DEV_FB_USED{gpu="0"} 4096',
        'DCGM_FI_DEV_FB_FREE{gpu="0"} 8192',
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 1000',
        'DCGM_FI_DEV_XID_ERRORS{gpu="0"} 0',
        'DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{gpu="0"} 10',
        'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0"} 4',
        'DCGM_FI_DEV_POWER_VIOLATION{gpu="0"} 100',
        'DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0"} 200',
      ].join("\n"),
      [
        'DCGM_FI_DEV_POWER_USAGE{gpu="0"} 120',
        'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 50',
        'DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 20',
        'DCGM_FI_PROF_SM_ACTIVE{gpu="0"} 0.7',
        'DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.3',
        'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0"} 0.8',
        'DCGM_FI_PROF_PIPE_FP64_ACTIVE{gpu="0"} 0.02',
        'DCGM_FI_PROF_PIPE_FP32_ACTIVE{gpu="0"} 0.4',
        'DCGM_FI_PROF_PIPE_FP16_ACTIVE{gpu="0"} 0.6',
        'DCGM_FI_DEV_PCIE_TX_THROUGHPUT{gpu="0"} 222',
        'DCGM_FI_DEV_PCIE_RX_THROUGHPUT{gpu="0"} 444',
        'DCGM_FI_PROF_PCIE_TX_BYTES{gpu="0"} 7000',
        'DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0"} 10000',
        'DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="0"} 8',
        'DCGM_FI_PROF_NVLINK_TX_BYTES{gpu="0"} 15000',
        'DCGM_FI_PROF_NVLINK_RX_BYTES{gpu="0"} 17000',
        'DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL{gpu="0"} 300',
        'DCGM_FI_DEV_ENC_UTIL{gpu="0"} 8',
        'DCGM_FI_DEV_DEC_UTIL{gpu="0"} 11',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 61',
        'DCGM_FI_DEV_MEMORY_TEMP{gpu="0"} 9223372036854775794',
        'DCGM_EXP_GPU_HEALTH_STATUS{gpu="0"} 0',
        'DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1801',
        'DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5001',
        'DCGM_FI_DEV_FB_USED{gpu="0"} 4097',
        'DCGM_FI_DEV_FB_FREE{gpu="0"} 8191',
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 2500',
        'DCGM_FI_DEV_XID_ERRORS{gpu="0"} 1',
        'DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{gpu="0"} 12',
        'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0"} 5',
        'DCGM_FI_DEV_POWER_VIOLATION{gpu="0"} 130',
        'DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0"} 240',
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
        frameworkVersion: "0.23.0",
        runtimeBackend: "cuda",
        modelRevision: "model-revision",
        imageTag: "vllm/vllm-openai:v0.23.0",
        imageDigest: "sha256:runtime-image",
        serverArgs: ["--model", laptopSmokeModel(), "--gpu-memory-utilization", "0.20"],
        containerId: "container-vllm",
        hostName: "gpu-host",
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
    const dcgmMetricNames = [
      "DCGM_EXP_GPU_HEALTH_STATUS",
      "DCGM_FI_DEV_DEC_UTIL",
      "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
      "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
      "DCGM_FI_DEV_ENC_UTIL",
      "DCGM_FI_DEV_FB_FREE",
      "DCGM_FI_DEV_FB_USED",
      "DCGM_FI_DEV_GPU_TEMP",
      "DCGM_FI_DEV_GPU_UTIL",
      "DCGM_FI_DEV_MEM_CLOCK",
      "DCGM_FI_DEV_MEM_COPY_UTIL",
      "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
      "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
      "DCGM_FI_DEV_PCIE_RX_THROUGHPUT",
      "DCGM_FI_DEV_PCIE_TX_THROUGHPUT",
      "DCGM_FI_DEV_POWER_USAGE",
      "DCGM_FI_DEV_POWER_VIOLATION",
      "DCGM_FI_DEV_SM_CLOCK",
      "DCGM_FI_DEV_THERMAL_VIOLATION",
      "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
      "DCGM_FI_DEV_XID_ERRORS",
      "DCGM_FI_PROF_DRAM_ACTIVE",
      "DCGM_FI_PROF_NVLINK_RX_BYTES",
      "DCGM_FI_PROF_NVLINK_TX_BYTES",
      "DCGM_FI_PROF_PCIE_RX_BYTES",
      "DCGM_FI_PROF_PCIE_TX_BYTES",
      "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
      "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
      "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
      "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
      "DCGM_FI_PROF_SM_ACTIVE",
    ].sort()
    const dcgmMetricNamesSha256 = createHash("sha256").update(dcgmMetricNames.join("\n")).digest("hex")
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
    expect(sample.smActivePct).toBeCloseTo(70)
    expect(sample.dramActivePct).toBeCloseTo(30)
    expect(sample.tensorActivePct).toBeCloseTo(80)
    expect(sample.fp64ActivePct).toBeCloseTo(2)
    expect(sample.fp32ActivePct).toBeCloseTo(40)
    expect(sample.fp16ActivePct).toBeCloseTo(60)
    expect(sample.pcieTxThroughputKiBps).toBeCloseTo(222)
    expect(sample.pcieRxThroughputKiBps).toBeCloseTo(444)
    expect(sample.pcieTxBytesDelta).toBe(6000)
    expect(sample.pcieRxBytesDelta).toBe(8000)
    expect(sample.pcieReplayDelta).toBe(5)
    expect(sample.nvlinkTxBytesDelta).toBe(12000)
    expect(sample.nvlinkRxBytesDelta).toBe(13000)
    expect(sample.nvlinkBandwidthTotalMBps).toBeCloseTo(300)
    expect(sample.encoderUtilizationPct).toBeCloseTo(8)
    expect(sample.decoderUtilizationPct).toBeCloseTo(11)
    expect(sample.gpuTemperatureC).toBeCloseTo(61)
    expect(sample.smClockMHz).toBeCloseTo(1801)
    expect(sample.memoryClockMHz).toBeCloseTo(5001)
    expect(sample.fbUsedMiB).toBeCloseTo(4097)
    expect(sample.fbFreeMiB).toBeCloseTo(8191)
    expect(sample.energyJoules).toBeCloseTo(1.5)
    expect(sample.xidErrors).toBe(1)
    expect(sample.xidErrorsDelta).toBe(1)
    expect(sample.eccSbeVolatileTotalDelta).toBe(2)
    expect(sample.eccDbeVolatileTotalDelta).toBe(1)
    expect(sample.powerViolationTimeUsDelta).toBe(30)
    expect(sample.thermalViolationTimeUsDelta).toBe(40)
    expect(sample.hardwareRawMetricCount).toBe(31)
    expect(sample.hardwareRawMetricNamesSha256).toBe(dcgmMetricNamesSha256)
    expect(result.measurements[0].dcgmGrounded).toBe(true)
    expect(result.measurements[0].hardwareTelemetryAvailableCount).toBe(1)
    expect(result.measurements[0].avgTensorActivePct).toBeCloseTo(80)
    expect(result.measurements[0].avgPcieTxBytesDelta).toBe(6000)
    expect(result.measurements[0].avgNvlinkRxBytesDelta).toBe(13000)
    expect(result.measurements[0].avgEccDbeVolatileTotalDelta).toBe(1)
    expect(result.measurements[0].hardwareRawMetricCountMin).toBe(31)
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
    expect(sampleRows[0].tensorActivePct).toBeCloseTo(80)
    expect(sampleRows[0].pcieTxBytesDelta).toBe(6000)
    expect(sampleRows[0].nvlinkRxBytesDelta).toBe(13000)
    expect(sampleRows[0].xidErrors).toBe(1)
    expect(sampleRows[0].eccDbeVolatileTotalDelta).toBe(1)
    expect(sampleRows[0].hardwareRawMetricCount).toBe(31)
    expect(sampleRows[0].hardwareRawMetricNamesSha256).toBe(dcgmMetricNamesSha256)
    expect(sampleRows[0].smClockMHz).toBeCloseTo(1801)
    expect(sampleRows[0].fbUsedMiB).toBeCloseTo(4097)
    const metricSnapshotRows = result.measurements.filter((row) => row.surface === "serving_metric_snapshot")
    expect(metricSnapshotRows).toHaveLength(62)
    expect(new Set(metricSnapshotRows.map((row) => row.metricName))).toContain("DCGM_EXP_GPU_HEALTH_STATUS")
    expect(metricSnapshotRows.some((row) => row.metricName === "DCGM_FI_DEV_MEMORY_TEMP")).toBe(false)
    expect(metricSnapshotRows.every((row) => row.metricSource === "dcgm-prometheus")).toBe(true)
    expect(new Set(metricSnapshotRows.map((row) => row.snapshotPhase))).toEqual(new Set(["before", "after"]))
    expect(metricSnapshotRows.every((row) => typeof row.metricLabelsSha256 === "string" && row.metricLabelsSha256.length === 64)).toBe(true)
    expect(metricSnapshotRows.every((row) => typeof row.rawMetricTextSha256 === "string" && row.rawMetricTextSha256.length === 64)).toBe(true)
    const metricCoverage = result.measurements.find((row) => row.surface === "serving_telemetry_coverage" && row.coverageCategory === "metricSnapshots")
    expect(metricCoverage).toMatchObject({ coverageStatus: "proven", provenCount: 1, expectedCount: 1 })
    const rawArtifact = JSON.parse(fs.readFileSync(String((JSON.parse(fs.readFileSync(result.artifactPath, "utf8")) as { capturePolicy: { rawArtifactPath: string } }).capturePolicy.rawArtifactPath), "utf8"))
    const hardwareBefore = rawArtifact.captures[0].hardwareMetricsRaw.before
    expect(hardwareBefore.rawMetricsText).toContain('DCGM_FI_DEV_POWER_USAGE{gpu="0"} 100')
    expect(hardwareBefore.rawMetricsText).toContain('DCGM_FI_DEV_MEMORY_TEMP{gpu="0"} 9223372036854775794')
    expect(hardwareBefore.invalidMetricCount).toBe(1)
    expect(hardwareBefore.invalidMetricSeries[0].invalidReason).toBe("dcgm-blank-sentinel")
    expect(rawArtifact.runtimeConfiguration.runtimeBackend).toBe("cuda")
    expect(rawArtifact.runtimeConfiguration.serverArgs[0]).toBe("--model")
    expect(rawArtifact.runtimeConfiguration.containerId).toBe("container-vllm")
    const hardwareCoverage = result.measurements.find((row) => row.surface === "serving_telemetry_coverage" && row.coverageCategory === "dcgmHardwareTelemetry")
    expect(hardwareCoverage).toMatchObject({
      coverageStatus: "proven",
      provenCount: 1,
      expectedCount: 1,
    })
  })

  it("resolves prompt and output token IDs with an external tokenizer Python", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-external-tokenizer-test-"))
    tmpDirs.push(artifactDir)
    const tokenizerBin = path.join(artifactDir, "fake-tokenizer-python.js")
    fs.writeFileSync(tokenizerBin, [
      "#!/usr/bin/env node",
      "const mode = process.argv[4]",
      "const payload = JSON.parse(process.argv[6])",
      "if (mode === 'prompt') {",
      "  console.log(JSON.stringify({ ok: true, tokenIds: [501, 502], tokenTexts: ['Return', ' ok'], mode: 'prompt-text' }))",
      "} else if (mode === 'token') {",
      "  const ids = { o: 101, O: 102, k: 202 }",
      "  console.log(JSON.stringify({ ok: true, tokenId: ids[payload] ?? null }))",
      "} else {",
      "  console.log(JSON.stringify({ ok: false, error: 'bad mode' }))",
      "}",
    ].join("\n"))
    fs.chmodSync(tokenizerBin, 0o755)
    const tokenizerPythonBinSha256 = createHash("sha256").update(fs.readFileSync(tokenizerBin)).digest("hex")
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.method).toBe("POST")
      const body = [
        {
          id: "chatcmpl-external-tokenizer",
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
          id: "chatcmpl-external-tokenizer",
          model: laptopSmokeModel(),
          choices: [{
            delta: { content: "k" },
            finish_reason: "stop",
            logprobs: { content: [{ token: "k", logprob: -0.2 }] },
          }],
          usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
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
        tokenizerModel: laptopSmokeModel(),
        tokenizerPythonBin: tokenizerBin,
        resolveTokenIdsWithTokenizer: true,
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Return ok." }],
        captureTokenDetails: true,
        topLogprobs: 2,
        resolveTokenIdsWithTokenizer: true,
      },
      artifactDir,
      workload: {
        hardware: "local tokenizer runtime",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    const sample = result.samples[0]
    expect(sample.promptTokenIdsAvailable).toBe(true)
    expect(sample.promptTokenIdSource).toBe("external-hf-tokenizer")
    expect(sample.promptTokenDetailCount).toBe(2)
    expect(sample.tokenIdsAvailable).toBe(true)
    expect(sample.tokenIdSource).toBe("external-hf-tokenizer")
    expect(sample.tokenizerModel).toBe(laptopSmokeModel())
    expect(sample.tokenizerPythonBinSha256).toBe(tokenizerPythonBinSha256)
    const promptRows = sample.tokenTimeline?.filter((row) => row.tokenPhase === "prompt") ?? []
    const outputRows = sample.tokenTimeline?.filter((row) => row.tokenPhase === "output") ?? []
    expect(promptRows.map((row) => row.tokenId)).toEqual([501, 502])
    expect(promptRows.every((row) => row.tokenIdSource === "external-hf-tokenizer")).toBe(true)
    expect(promptRows.every((row) => row.tokenizerModel === laptopSmokeModel())).toBe(true)
    expect(promptRows.every((row) => row.tokenizerPythonBinSha256 === tokenizerPythonBinSha256)).toBe(true)
    expect(outputRows[0].tokenId).toBe(101)
    expect(outputRows[0].tokenIdSource).toBe("external-hf-tokenizer")
    expect(outputRows[0].tokenizerModel).toBe(laptopSmokeModel())
    expect(outputRows[0].tokenizerPythonBinSha256).toBe(tokenizerPythonBinSha256)
    const topLogprobs = JSON.parse(String(outputRows[0].topLogprobsJson))
    expect(topLogprobs[0].tokenId).toBe(101)
    expect(topLogprobs[0].tokenIdSource).toBe("external-hf-tokenizer")

    const sampleRows = result.measurements.filter((row) => row.surface === "serving_request_sample")
    expect(sampleRows[0].tokenizerModel).toBe(laptopSmokeModel())
    expect(sampleRows[0].tokenizerPythonBinSha256).toBe(tokenizerPythonBinSha256)
    expect(sampleRows[0].promptTokenIdSource).toBe("external-hf-tokenizer")
    const promptMeasurementRows = result.measurements.filter((row) => row.surface === "serving_token_timeline" && row.tokenPhase === "prompt")
    expect(promptMeasurementRows.map((row) => row.tokenId)).toEqual([501, 502])
    expect(promptMeasurementRows.every((row) => row.tokenizerModel === laptopSmokeModel())).toBe(true)
    const outputMeasurementRows = result.measurements.filter((row) => row.surface === "serving_token_timeline" && row.tokenPhase === "output")
    expect(outputMeasurementRows[0].tokenIdSource).toBe("external-hf-tokenizer")
    expect(outputMeasurementRows[0].tokenizerPythonBinSha256).toBe(tokenizerPythonBinSha256)
  })

  it("tokenizes output text when response logprobs are unavailable", async () => {
    const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-serving-output-tokenizer-test-"))
    tmpDirs.push(artifactDir)
    const tokenizerBin = path.join(artifactDir, "fake-output-tokenizer-python.js")
    fs.writeFileSync(tokenizerBin, [
      "#!/usr/bin/env node",
      "const mode = process.argv[4]",
      "if (mode === 'prompt') {",
      "  console.log(JSON.stringify({ ok: true, tokenIds: [501, 502], tokenTexts: ['Return', ' ok'], mode: 'prompt-text' }))",
      "} else if (mode === 'text') {",
      "  console.log(JSON.stringify({ ok: true, tokenIds: [101, 202], tokenTexts: ['o', 'k'], mode: 'output-text' }))",
      "} else {",
      "  console.log(JSON.stringify({ ok: false, error: 'bad mode' }))",
      "}",
    ].join("\n"))
    fs.chmodSync(tokenizerBin, 0o755)
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.method).toBe("POST")
      const body = [
        {
          id: "chatcmpl-output-tokenizer",
          model: laptopSmokeModel(),
          choices: [{ delta: { content: "o" } }],
        },
        {
          id: "chatcmpl-output-tokenizer",
          model: laptopSmokeModel(),
          choices: [{ delta: { content: "k" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
        },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      })
    })

    const result = await runServingProducer({
      engine: {
        engine: "sglang",
        baseUrl: "http://127.0.0.1:30000",
        tokenizerModel: laptopSmokeModel(),
        tokenizerPythonBin: tokenizerBin,
        resolveTokenIdsWithTokenizer: true,
      },
      request: {
        model: laptopSmokeModel(),
        messages: [{ role: "user", content: "Return ok." }],
        resolveTokenIdsWithTokenizer: true,
      },
      artifactDir,
      workload: {
        hardware: "local tokenizer runtime",
        operatingPoint: "laptop-smoke",
      },
      pricing: { usdPerGpuHour: 1, gpuCount: 1, powerWattsPerGpu: 100 },
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => new Date("2026-07-09T12:00:00Z"),
    })

    const sample = result.samples[0]
    expect(sample.tokenDetailsAvailable).toBe(true)
    expect(sample.tokenIdsAvailable).toBe(true)
    expect(sample.logprobsAvailable).toBe(false)
    expect(sample.tokenIdSource).toBe("external-hf-tokenizer")
    expect(sample.tokenDetailSource).toBe("external-hf-tokenizer-output-text")
    const outputRows = sample.tokenTimeline?.filter((row) => row.tokenPhase === "output") ?? []
    expect(outputRows.map((row) => row.tokenId)).toEqual([101, 202])
    expect(outputRows.every((row) => row.tokenIdSource === "external-hf-tokenizer")).toBe(true)
    expect(outputRows.every((row) => row.tokenLogprob == null)).toBe(true)
    expect(outputRows.every((row) => row.chunkIndex == null)).toBe(true)
    const outputMeasurementRows = result.measurements.filter((row) => row.surface === "serving_token_timeline" && row.tokenPhase === "output")
    expect(outputMeasurementRows.map((row) => row.tokenId)).toEqual([101, 202])
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
