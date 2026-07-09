import { mkdir, writeFile } from "node:fs/promises"
import path from "node:path"

import {
  buildManifest,
  type Confidentiality,
  type PerformanceIQ,
  type PerformanceIQRunInput,
  type ProducerIdentity,
  type ProducerRunManifest,
  type RunClass,
  type SourceType,
} from "./index"

export type ServingEngineId = "vllm" | "sglang" | "tensorrt-llm"
export type ChatMessage = { role: "system" | "user" | "assistant"; content: string }

export interface ServingEngineConfig {
  engine: ServingEngineId
  baseUrl: string
  apiKey?: string
  requestPath?: string
  frameworkVersion?: string
  imageDigest?: string
  imageTag?: string
}

export interface ServingRequestConfig {
  model: string
  messages: ChatMessage[]
  maxTokens?: number
  temperature?: number
  topP?: number
  repetitions?: number
}

export interface ServingProducerConfig {
  engine: ServingEngineConfig
  request: ServingRequestConfig
  performanceIq?: PerformanceIQ
  submit?: boolean
  artifactDir?: string
  producer?: Partial<ProducerIdentity>
  campaign?: Partial<PerformanceIQRunInput["campaign"]>
  workload?: Partial<PerformanceIQRunInput["workload"]>
  sourceType?: SourceType
  runClass?: RunClass
  confidentiality?: Confidentiality
  pricing?: {
    usdPerGpuHour?: number
    gpuCount?: number
    powerWattsPerGpu?: number
  }
  fetchImpl?: typeof fetch
  now?: () => Date
}

export interface ServingRequestSample {
  requestIndex: number
  status: number
  ok: boolean
  latencyMs: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  responseId?: string
  responseModel?: string
  finishReason?: string
  error?: string
}

export interface ServingProducerResult {
  engine: ServingEngineId
  manifest: ProducerRunManifest
  runInput: PerformanceIQRunInput
  artifactPath: string
  samples: ServingRequestSample[]
  measurements: Record<string, unknown>[]
  submission?: unknown
}

const ENGINE_LABELS: Record<ServingEngineId, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  "tensorrt-llm": "TensorRT-LLM",
}

const DEFAULT_IMAGE_DIGEST = `sha256:${"0".repeat(64)}`

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "")
}

function nowIso(now: () => Date): string {
  return now().toISOString()
}

function percentile(values: number[], p: number): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))
  return sorted[index]
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function metricCompleteness(row: Record<string, unknown>): number {
  const required = [
    row.outputTpm,
    row.totalTpm,
    row.usdPer1mOutputTokens,
    row.usdPer1mTotalTokens,
    row.tokensPerWatt,
  ]
  return required.filter((value) => typeof value === "number" && finite(value)).length / required.length
}

function requestPayload(config: ServingRequestConfig) {
  return {
    model: config.model,
    messages: config.messages,
    max_tokens: config.maxTokens ?? 64,
    temperature: config.temperature ?? 0,
    ...(config.topP == null ? {} : { top_p: config.topP }),
  }
}

async function sendChatCompletion(
  config: ServingProducerConfig,
  requestIndex: number,
): Promise<ServingRequestSample> {
  const fetchImpl = config.fetchImpl ?? fetch
  const endpoint = `${normalizeBaseUrl(config.engine.baseUrl)}${config.engine.requestPath ?? "/v1/chat/completions"}`
  const started = performance.now()
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
      },
      body: JSON.stringify(requestPayload(config.request)),
    })
    const latencyMs = performance.now() - started
    const body = await response.json().catch(() => ({})) as Record<string, any>
    const usage = body.usage ?? {}
    return {
      requestIndex,
      status: response.status,
      ok: response.ok,
      latencyMs,
      promptTokens: Number(usage.prompt_tokens ?? usage.promptTokens ?? 0),
      completionTokens: Number(usage.completion_tokens ?? usage.completionTokens ?? 0),
      totalTokens: Number(usage.total_tokens ?? usage.totalTokens ?? 0),
      responseId: typeof body.id === "string" ? body.id : undefined,
      responseModel: typeof body.model === "string" ? body.model : undefined,
      finishReason: body.choices?.[0]?.finish_reason,
      error: response.ok ? undefined : JSON.stringify(body),
    }
  } catch (error) {
    return {
      requestIndex,
      status: 0,
      ok: false,
      latencyMs: performance.now() - started,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      error: error instanceof Error ? error.message : String(error),
    }
  }
}

function buildMeasurements(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  capturedAtUtc: string,
): Record<string, unknown>[] {
  const engineLabel = ENGINE_LABELS[config.engine.engine]
  const successful = samples.filter((sample) => sample.ok)
  const durationSeconds = Math.max(sum(successful.map((sample) => sample.latencyMs)) / 1000, 0.001)
  const outputTokens = sum(successful.map((sample) => sample.completionTokens))
  const totalTokens = sum(successful.map((sample) => sample.totalTokens))
  const promptTokens = sum(successful.map((sample) => sample.promptTokens))
  const outputTpm = outputTokens / (durationSeconds / 60)
  const totalTpm = totalTokens / (durationSeconds / 60)
  const gpuCount = config.pricing?.gpuCount ?? Number(config.workload?.parallelism ?? 1)
  const usdPerGpuHour = config.pricing?.usdPerGpuHour
  const powerWattsPerGpu = config.pricing?.powerWattsPerGpu
  const costUsd = finite(usdPerGpuHour) ? (durationSeconds / 3600) * usdPerGpuHour * gpuCount : null
  const usdPer1mOutputTokens = costUsd != null && outputTokens > 0 ? costUsd / (outputTokens / 1_000_000) : null
  const usdPer1mTotalTokens = costUsd != null && totalTokens > 0 ? costUsd / (totalTokens / 1_000_000) : null
  const tokensPerWatt = finite(powerWattsPerGpu) && powerWattsPerGpu > 0
    ? (totalTokens / durationSeconds) / (powerWattsPerGpu * gpuCount)
    : null
  const avgLatencyMs = successful.length ? sum(successful.map((sample) => sample.latencyMs)) / successful.length : null
  const p95LatencyMs = percentile(successful.map((sample) => sample.latencyMs), 95)
  const row: Record<string, unknown> = {
    surface: "result",
    model: config.request.model,
    runtimeFramework: engineLabel,
    runtimeEngine: config.engine.engine,
    operatingPoint: config.workload?.operatingPoint ?? "laptop-smoke",
    basis: "per_engine",
    requestCount: samples.length,
    successCount: successful.length,
    errorCount: samples.length - successful.length,
    promptTokens,
    completionTokens: outputTokens,
    totalTokens,
    outputTpm,
    totalTpm,
    avgLatencyMs,
    p95LatencyMs,
    usdPer1mOutputTokens,
    usdPer1mTotalTokens,
    avgPowerWattsPerGpu: powerWattsPerGpu ?? null,
    tokensPerWatt,
    campaignCount: 1,
    latestCapturedAtUtc: capturedAtUtc,
    experimentFamily: "serving-producer",
    experimentStatus: successful.length === samples.length ? "accepted" : "partial",
    verdictTier: successful.length === samples.length ? "request-captured" : "request-errors",
    solRigor: config.runClass === "measured" ? "l3" : "smoke",
    plotReadyPoints: 0,
    dcgmGrounded: false,
    tags: [
      "serving-producer",
      config.engine.engine,
      engineLabel,
      config.request.model,
      config.runClass ?? "measured",
      config.sourceType ?? "other-measured-producer",
    ].join(","),
  }
  row.metricCompleteness = metricCompleteness(row)
  return [row]
}

async function writeSummaryArtifact(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  measurements: Record<string, unknown>[],
  capturedAtUtc: string,
): Promise<string> {
  const artifactDir = config.artifactDir ?? path.join(process.cwd(), ".performance-iq", "serving-producers")
  await mkdir(artifactDir, { recursive: true })
  const safeModel = config.request.model.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-|-$/g, "")
  const artifactPath = path.join(
    artifactDir,
    `${config.engine.engine}-${safeModel}-${capturedAtUtc.replace(/[:.]/g, "-")}.json`,
  )
  await writeFile(artifactPath, JSON.stringify({
    schemaVersion: "performance-iq.serving-producer-summary.v1",
    capturedAtUtc,
    engine: config.engine.engine,
    engineLabel: ENGINE_LABELS[config.engine.engine],
    baseUrl: config.engine.baseUrl,
    requestPath: config.engine.requestPath ?? "/v1/chat/completions",
    model: config.request.model,
    request: requestPayload(config.request),
    samples,
    measurements,
  }, null, 2) + "\n")
  return artifactPath
}

export async function runServingProducer(config: ServingProducerConfig): Promise<ServingProducerResult> {
  const now = config.now ?? (() => new Date())
  const capturedAtUtc = nowIso(now)
  const repetitions = Math.max(1, config.request.repetitions ?? 1)
  const samples: ServingRequestSample[] = []
  for (let index = 0; index < repetitions; index += 1) {
    samples.push(await sendChatCompletion(config, index))
  }

  const measurements = buildMeasurements(config, samples, capturedAtUtc)
  const artifactPath = await writeSummaryArtifact(config, samples, measurements, capturedAtUtc)
  const engineLabel = ENGINE_LABELS[config.engine.engine]
  const campaignId = config.campaign?.campaignId ?? `serving-${config.engine.engine}-${config.request.model}`
  const runId = config.campaign?.runId ?? `${campaignId}-${capturedAtUtc.replace(/[:.]/g, "-")}`
  const runInput: PerformanceIQRunInput = {
    sourceType: config.sourceType ?? "other-measured-producer",
    runClass: config.runClass ?? "measured",
    confidentiality: config.confidentiality ?? "operator-full",
    producer: {
      repo: "performance-iq-sdk",
      tool: `${config.engine.engine}-serving-producer`,
      commitSha: "local-serving-producer",
      ...config.producer,
    },
    campaign: {
      campaignId,
      runId,
      capturedAtUtc,
      completedAtUtc: nowIso(now),
      ...config.campaign,
    },
    workload: {
      model: config.request.model,
      hardware: "unknown",
      operatingPoint: "serving-smoke",
      scenario: `OpenAI-compatible chat completions through ${engineLabel}`,
      ...config.workload,
    },
    runtime: {
      imageDigest: config.engine.imageDigest ?? DEFAULT_IMAGE_DIGEST,
      imageTag: config.engine.imageTag,
      framework: engineLabel,
    },
    artifacts: [{ kind: "normalized-summary", path: artifactPath }],
    measurements,
    platform: {
      decisionBriefPath: "performance-iq://serving-producer",
    },
    methodology: [
      `${engineLabel} producer sent ${repetitions} OpenAI-compatible chat completion request(s)`,
      `to ${normalizeBaseUrl(config.engine.baseUrl)}${config.engine.requestPath ?? "/v1/chat/completions"}`,
      `for model ${config.request.model}.`,
      "Metrics are derived from response usage fields and wall-clock request latency.",
    ].join(" "),
    limitations: [
      "Serving producer captures request-path, usage, latency, and provenance; hardware-level power/kernel counters require engine-side or cluster instrumentation.",
      ...(samples.some((sample) => !sample.ok) ? ["One or more serving requests failed; see normalized-summary artifact for per-request errors."] : []),
    ],
  }
  if (config.engine.frameworkVersion) {
    runInput.runtime.imageTag = runInput.runtime.imageTag ?? `${config.engine.engine}:${config.engine.frameworkVersion}`
  }
  const manifest = await buildManifest(runInput)
  const submission = config.performanceIq && config.submit !== false
    ? await config.performanceIq.submitRun(runInput, { idempotencyKey: manifest.campaign.runId })
    : undefined

  return {
    engine: config.engine.engine,
    manifest,
    runInput,
    artifactPath,
    samples,
    measurements,
    submission,
  }
}

export function laptopSmokeModel(): string {
  return "Qwen/Qwen2.5-0.5B-Instruct"
}

export function servingEngineLabel(engine: ServingEngineId): string {
  return ENGINE_LABELS[engine]
}
