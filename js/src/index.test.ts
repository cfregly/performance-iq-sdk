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
})
