import { createHash } from "node:crypto"
import { existsSync } from "node:fs"
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
  metricsUrl?: string
  nativeJsonMetricsUrl?: string
  jsonMetricsUrl?: string
  collectNativeMetrics?: boolean
  hardwareMetricsUrl?: string
  dcgmMetricsUrl?: string
  collectHardwareMetrics?: boolean
  requireNativeTelemetry?: boolean
  requireHardwareTelemetry?: boolean
  nativeTelemetry?: Record<string, unknown>
  hardwareTelemetry?: Record<string, unknown>
  tokenIdMap?: Record<string, number | string>
  tokenIdResolver?: (token: string, item: Record<string, unknown>, engine: ServingEngineConfig, request: ServingRequestConfig) => number | string | null | undefined
  promptTokenIds?: Array<number | string>
  frameworkVersion?: string
  modelRevision?: string
  imageDigest?: string
  imageTag?: string
  serverArgs?: unknown
  processId?: string | number
  pid?: string | number
  containerId?: string
  podName?: string
  nodeName?: string
  hostName?: string
  hostname?: string
  endpointPreflight?: Record<string, unknown>
}

export interface ServingRequestConfig {
  model: string
  messages: ChatMessage[]
  maxTokens?: number
  temperature?: number
  topP?: number
  repetitions?: number
  stream?: boolean
  captureTokenDetails?: boolean
  logprobs?: boolean
  topLogprobs?: number
  promptTokenIds?: Array<number | string>
  tokenizerModel?: string
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
  requestId: string
  requestIndex: number
  endpoint: string
  requestStartedAtUtc: string
  requestCompletedAtUtc: string
  status: number
  ok: boolean
  latencyMs: number
  e2eLatencyMs: number
  timeToFirstByteMs: number | null
  ttftMs: number | null
  ttfotMs: number | null
  tpotMs: number | null
  interTokenLatencyMs: number | null
  firstChunkAtUtc?: string | null
  firstOutputAtUtc?: string | null
  lastOutputAtUtc?: string | null
  streamChunkCount: number
  outputTokenCount: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  tokenCountSource?: string
  responseId?: string
  responseModel?: string
  finishReason?: string
  ttftSource: string
  streaming: boolean
  promptSha256?: string
  requestPayloadSha256?: string
  outputSha256?: string
  outputBytes?: number
  nativeTelemetry?: Record<string, unknown>
  nativeTelemetryAvailable?: boolean
  hardwareTelemetry?: Record<string, unknown>
  hardwareTelemetryAvailable?: boolean
  nativeTelemetrySource?: string | null
  nativeMetricsUrl?: string | null
  nativeTtftMs?: number | null
  nativeTpotMs?: number | null
  nativeE2eLatencyMs?: number | null
  nativeInterTokenLatencyMs?: number | null
  nativeIterationLatencyMs?: number | null
  nativeGpuMemoryBytes?: number | null
  nativeKvCacheUsedBlocks?: number | null
  nativeKvCacheMaxBlocks?: number | null
  runningRequests?: number | null
  waitingRequests?: number | null
  kvCacheUsagePct?: number | null
  cacheHitRate?: number | null
  prefixCacheQueriesDelta?: number | null
  prefixCacheHitsDelta?: number | null
  promptTokensCachedDelta?: number | null
  promptTokensComputedDelta?: number | null
  hardwareTelemetrySource?: string | null
  hardwareMetricsUrl?: string | null
  avgPowerWatts?: number | null
  avgPowerWattsPerGpu?: number | null
  gpuUtilizationPct?: number | null
  memoryCopyUtilizationPct?: number | null
  gpuTemperatureC?: number | null
  smClockMHz?: number | null
  memoryClockMHz?: number | null
  fbUsedMiB?: number | null
  fbFreeMiB?: number | null
  energyJoules?: number | null
  tokenDetailsAvailable?: boolean
  tokenIdsAvailable?: boolean
  logprobsAvailable?: boolean
  tokenDetailCount?: number
  tokenDetailSource?: string
  tokenIdSource?: string | null
  promptTokenIdsAvailable?: boolean
  promptTokenDetailCount?: number
  promptTokenIdSource?: string | null
  promptTokenIdsSha256?: string | null
  promptTokenizationSource?: string | null
  promptTokenizerModel?: string | null
  queueWaitMs?: number | null
  prefillMs?: number | null
  decodeMs?: number | null
  engineVersion?: unknown
  modelRevision?: unknown
  imageTag?: unknown
  imageDigest?: unknown
  serverArgsSha256?: string | null
  processId?: unknown
  containerId?: unknown
  podName?: unknown
  nodeName?: unknown
  hostName?: unknown
  tokenTimeline?: ServingTokenTimelineChunk[]
  error?: string
}

export interface ServingTokenTimelineChunk {
  requestId: string
  tokenPhase?: string
  chunkIndex: number | null
  receivedAtUtc: string
  relativeMs: number
  contentBytes: number | null
  contentSha256: string | null
  isFirstOutput: boolean
  tokenIndex?: number | null
  tokenId?: number | null
  tokenIdSource?: string | null
  tokenLogprob?: number | null
  tokenTextSha256?: string | null
  topLogprobsJson?: string | null
  tokenDetailSource?: string
}

type PromptTokenSummary = Pick<
  ServingRequestSample,
  | "promptTokenIdsAvailable"
  | "promptTokenDetailCount"
  | "promptTokenIdSource"
  | "promptTokenIdsSha256"
  | "promptTokenizationSource"
  | "promptTokenizerModel"
>

type PromptTokenCaptureResult = {
  summary: PromptTokenSummary
  details: Array<Record<string, unknown>>
}

export interface ServingProducerResult {
  engine: ServingEngineId
  manifest: ProducerRunManifest
  runInput: PerformanceIQRunInput
  artifactPath: string
  manifestPath: string
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

function safeSlug(value: string): string {
  return value.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-|-$/g, "") || "value"
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

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

function sha256Json(value: unknown): string {
  return sha256Text(JSON.stringify(value))
}

function sha256OptionalJson(value: unknown): string | null {
  return value == null ? null : sha256Json(value)
}

function nestedValue(source: Record<string, unknown>, parent: string, keys: string[]): unknown {
  const candidate = source[parent]
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return undefined
  const nested = candidate as Record<string, unknown>
  for (const key of keys) {
    if (nested[key] != null) return nested[key]
  }
  return undefined
}

function firstDefined(source: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (source[key] != null) return source[key]
  }
  return undefined
}

function runtimeProvenance(config: ServingProducerConfig, telemetry: Record<string, unknown> = {}): Record<string, unknown> {
  const engine = config.engine as unknown as Record<string, unknown>
  const serverArgs = telemetry.serverArgs ?? engine.serverArgs
  return {
    engineVersion: telemetry.engineVersion ?? engine.frameworkVersion,
    modelRevision: telemetry.modelRevision ?? engine.modelRevision,
    imageTag: firstDefined(engine, ["imageTag", "containerImageTag"]),
    imageDigest: firstDefined(engine, ["imageDigest", "containerImageDigest"]),
    serverArgsSha256: sha256OptionalJson(serverArgs),
    processId: firstDefined(engine, ["processId", "pid"]) ?? nestedValue(engine, "process", ["pid", "processId"]),
    containerId: firstDefined(engine, ["containerId"]) ?? nestedValue(engine, "container", ["id", "containerId"]),
    podName: firstDefined(engine, ["podName"]) ?? nestedValue(engine, "container", ["podName"]),
    nodeName: firstDefined(engine, ["nodeName"]) ?? nestedValue(engine, "container", ["nodeName"]),
    hostName: firstDefined(engine, ["hostName", "hostname"]) ?? nestedValue(engine, "process", ["hostName", "hostname"]),
  }
}

function metricCompleteness(row: Record<string, unknown>): number {
  const required = [
    row.outputTpm,
    row.totalTpm,
    row.usdPer1mOutputTokens,
    row.usdPer1mTotalTokens,
    row.tokensPerWatt,
    row.p50LatencyMs,
    row.p95LatencyMs,
    row.p99LatencyMs,
    row.avgTimeToFirstByteMs,
    row.avgTtftMs,
    row.p50TtftMs,
    row.p95TtftMs,
    row.p99TtftMs,
    row.avgTpotMs,
    row.p50TpotMs,
    row.p95TpotMs,
    row.p99TpotMs,
    row.avgTtfotMs,
    row.p50TtfotMs,
    row.p95TtfotMs,
    row.p99TtfotMs,
    row.requestCount === row.successCount ? row.requestCount : null,
    row.streamingRequestCount === row.successCount ? row.streamingRequestCount : null,
    row.hardwareProvenance === "configured" ? 1 : null,
  ]
  if (row.nativeTelemetryRequired) {
    required.push(row.nativeTelemetryAvailableCount === row.successCount ? row.nativeTelemetryAvailableCount : null)
    required.push(
      row.avgQueueWaitMs,
      row.p50QueueWaitMs,
      row.p95QueueWaitMs,
      row.p99QueueWaitMs,
      row.avgPrefillMs,
      row.p50PrefillMs,
      row.p95PrefillMs,
      row.p99PrefillMs,
      row.avgDecodeMs,
      row.p50DecodeMs,
      row.p95DecodeMs,
      row.p99DecodeMs,
    )
  }
  if (row.hardwareTelemetryRequired) {
    required.push(row.hardwareTelemetryAvailableCount === row.successCount ? row.hardwareTelemetryAvailableCount : null)
    required.push(
      row.avgPowerWatts,
      row.avgPowerWattsPerGpu,
      row.avgGpuUtilizationPct,
      row.avgMemoryCopyUtilizationPct,
      row.totalEnergyJoules,
    )
  }
  if (row.tokenDetailsRequired) {
    required.push(row.logprobsAvailableCount === row.successCount ? row.logprobsAvailableCount : null)
  }
  if (row.promptTokenDetailsRequired) {
    required.push(row.promptTokenIdsAvailableCount === row.successCount ? row.promptTokenIdsAvailableCount : null)
  }
  return required.filter((value) => typeof value === "number" && finite(value)).length / required.length
}

function requestPayload(config: ServingRequestConfig, stream?: boolean) {
  const streamEnabled = stream ?? config.stream
  return {
    model: config.model,
    messages: config.messages,
    max_tokens: config.maxTokens ?? 64,
    temperature: config.temperature ?? 0,
    ...(config.topP == null ? {} : { top_p: config.topP }),
    ...(streamEnabled ? { stream: true, stream_options: { include_usage: true } } : {}),
    ...(config.captureTokenDetails || config.logprobs || config.topLogprobs != null
      ? {
          logprobs: config.logprobs ?? true,
          ...(config.topLogprobs == null ? {} : { top_logprobs: config.topLogprobs }),
        }
      : {}),
  }
}

function redactedRequest(payload: Record<string, unknown>) {
  const messages = Array.isArray(payload.messages) ? payload.messages as Array<{ content?: unknown }> : []
  const promptText = messages.map((message) => String(message.content ?? "")).join("\n")
  return {
    model: payload.model,
    messageCount: messages.length,
    max_tokens: payload.max_tokens,
    temperature: payload.temperature,
    top_p: payload.top_p,
    stream: payload.stream ?? false,
    promptBytes: Buffer.byteLength(promptText),
    promptSha256: sha256Text(promptText),
    requestPayloadSha256: sha256Json(payload),
  }
}

function promptText(payload: Record<string, unknown>): string {
  const messages = Array.isArray(payload.messages) ? payload.messages as Array<{ content?: unknown }> : []
  return messages.map((message) => String(message.content ?? "")).join("\n")
}

function estimatedTokenCount(text: string): number {
  return text ? Math.max(1, Math.floor(Buffer.byteLength(text) / 4)) : 0
}

function requestTraceId(engine: ServingEngineConfig, runId: string, requestIndex: number): string {
  return [
    "piq",
    engine.engine,
    safeSlug(runId),
    `request-${requestIndex + 1}`,
  ].join("-")
}

function traceHeaders(
  engine: ServingEngineConfig,
  campaignId: string,
  runId: string,
  requestId: string,
): Record<string, string> {
  return {
    "x-performance-iq-engine": engine.engine,
    "x-performance-iq-campaign-id": campaignId,
    "x-performance-iq-run-id": runId,
    "x-performance-iq-request-id": requestId,
  }
}

function choice(body: Record<string, any>): Record<string, any> {
  return Array.isArray(body.choices) && body.choices[0] && typeof body.choices[0] === "object"
    ? body.choices[0]
    : {}
}

function choiceContent(body: Record<string, any>): string {
  const first = choice(body)
  if (typeof first.delta?.content === "string") return first.delta.content
  if (typeof first.message?.content === "string") return first.message.content
  if (typeof first.text === "string") return first.text
  return ""
}

function nativeTelemetry(config: ServingProducerConfig, body?: Record<string, any>): Record<string, unknown> {
  const engineExtras = config.engine as unknown as Record<string, unknown>
  const configured = engineExtras.nativeTelemetry
  if (configured && typeof configured === "object" && !Array.isArray(configured)) {
    return { available: true, source: "engine-config", ...configured as Record<string, unknown> }
  }
  for (const key of ["nativeTelemetry", "native_telemetry", "metrics", "timings"]) {
    const candidate = body?.[key]
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      return { available: true, source: `response.${key}`, ...candidate as Record<string, unknown> }
    }
  }
  return {
    available: false,
    source: "not-exposed-by-openai-compatible-response",
    engineVersion: config.engine.frameworkVersion,
    modelRevision: engineExtras.modelRevision,
    serverArgs: engineExtras.serverArgs,
    queueWaitMs: null,
    prefillMs: null,
    decodeMs: null,
    batchSize: null,
    concurrency: null,
    kvCacheUsagePct: null,
    cacheHitRate: null,
  }
}

function defaultNativeMetricsUrl(engine: ServingEngineConfig): string {
  const baseUrl = normalizeBaseUrl(engine.baseUrl)
  return engine.engine === "tensorrt-llm" ? `${baseUrl}/prometheus/metrics` : `${baseUrl}/metrics`
}

function defaultNativeJsonMetricsUrl(engine: ServingEngineConfig): string | null {
  return engine.engine === "tensorrt-llm" ? `${normalizeBaseUrl(engine.baseUrl)}/metrics` : null
}

function metricsUrl(config: ServingProducerConfig): string | null {
  if (config.engine.metricsUrl?.trim()) return config.engine.metricsUrl.trim()
  return config.engine.collectNativeMetrics ? defaultNativeMetricsUrl(config.engine) : null
}

function nativeJsonMetricsUrl(config: ServingProducerConfig): string | null {
  if (config.engine.nativeJsonMetricsUrl?.trim()) return config.engine.nativeJsonMetricsUrl.trim()
  if (config.engine.jsonMetricsUrl?.trim()) return config.engine.jsonMetricsUrl.trim()
  return config.engine.collectNativeMetrics ? defaultNativeJsonMetricsUrl(config.engine) : null
}

function parsePrometheusMetrics(text: string): Record<string, number> {
  const metrics: Record<string, number> = {}
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)/)
    if (!match) continue
    const value = Number(match[3])
    if (!Number.isFinite(value)) continue
    const metricName = match[1]
    metrics[metricName] = (metrics[metricName] ?? 0) + value
    metrics[`${metricName}__sample_count`] = (metrics[`${metricName}__sample_count`] ?? 0) + 1
    const labels = Object.fromEntries(Array.from((match[2] ?? "").matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"/g), (item) => [item[1], item[2]]))
    for (const labelName of ["stage", "mode", "source", "reason", "finished_reason"]) {
      const labelValue = labels[labelName]
      if (!labelValue) continue
      const labelledName = `${metricName}{${labelName}=${labelValue}}`
      metrics[labelledName] = (metrics[labelledName] ?? 0) + value
      metrics[`${labelledName}__sample_count`] = (metrics[`${labelledName}__sample_count`] ?? 0) + 1
      for (const suffix of ["_sum", "_count"]) {
        if (!metricName.endsWith(suffix)) continue
        const labelledHistogramName = `${metricName.slice(0, -suffix.length)}{${labelName}=${labelValue}}${suffix}`
        metrics[labelledHistogramName] = (metrics[labelledHistogramName] ?? 0) + value
        metrics[`${labelledHistogramName}__sample_count`] = (metrics[`${labelledHistogramName}__sample_count`] ?? 0) + 1
      }
    }
  }
  return metrics
}

function flattenNumericJsonMetrics(value: unknown, prefix = ""): Record<string, number> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value as Record<string, unknown>).reduce<Record<string, number>>((acc, [key, nested]) => {
      const childPrefix = prefix ? `${prefix}.${key}` : key
      return { ...acc, ...flattenNumericJsonMetrics(nested, childPrefix) }
    }, {})
  }
  if (Array.isArray(value)) {
    return value.reduce<Record<string, number>>((acc, nested, index) => {
      const childPrefix = prefix ? `${prefix}.${index}` : String(index)
      return { ...acc, ...flattenNumericJsonMetrics(nested, childPrefix) }
    }, {})
  }
  return typeof value === "number" && Number.isFinite(value) && prefix ? { [prefix]: value } : {}
}

function parseNativeJsonMetrics(text: string): Record<string, number> {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return {}
  }
  const records = Array.isArray(parsed) ? parsed : [parsed]
  const numericRecords = records
    .filter((record) => record && typeof record === "object" && !Array.isArray(record))
    .map((record) => flattenNumericJsonMetrics(record))
    .filter((record) => Object.keys(record).length > 0)
  if (!numericRecords.length) return {}
  const latest = numericRecords[numericRecords.length - 1] ?? {}
  const summary: Record<string, number> = Object.fromEntries(
    Object.entries(latest).map(([key, value]) => [`latest.${key}`, value]),
  )
  const keys = new Set(numericRecords.flatMap((record) => Object.keys(record)))
  for (const key of keys) {
    const values = numericRecords.map((record) => record[key]).filter((value): value is number => Number.isFinite(value))
    if (!values.length) continue
    summary[`avg.${key}`] = values.reduce((sum, value) => sum + value, 0) / values.length
    summary[`max.${key}`] = Math.max(...values)
  }
  return summary
}

async function readNativeMetrics(config: ServingProducerConfig): Promise<Record<string, unknown>> {
  const url = metricsUrl(config)
  const jsonUrl = nativeJsonMetricsUrl(config)
  if (!url && !jsonUrl) return { available: false, source: "metrics-url-not-configured" }
  const fetchImpl = config.fetchImpl ?? fetch
  const metrics: Record<string, number> = {}
  const jsonMetrics: Record<string, number> = {}
  const sources: string[] = []
  const errors: Array<Record<string, unknown>> = []
  if (url) {
    try {
      const response = await fetchImpl(url, {
        method: "GET",
        headers: {
          accept: "text/plain",
          ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
        },
      })
      if (!response.ok) {
        errors.push({ source: "prometheus-unavailable", metricsUrl: url, status: response.status })
      } else {
        const text = await response.text()
        Object.assign(metrics, parsePrometheusMetrics(text))
        Object.assign(jsonMetrics, parseNativeJsonMetrics(text))
        if (Object.keys(metrics).length) sources.push("prometheus-snapshot")
        if (Object.keys(jsonMetrics).length) sources.push("native-json-snapshot")
      }
    } catch (error) {
      errors.push({
        source: "prometheus-unavailable",
        metricsUrl: url,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
  if (jsonUrl && jsonUrl !== url) {
    try {
      const response = await fetchImpl(jsonUrl, {
      method: "GET",
      headers: {
        accept: "application/json, text/plain",
        ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
      },
    })
    if (!response.ok) {
        errors.push({ source: "native-json-unavailable", nativeJsonMetricsUrl: jsonUrl, status: response.status })
      } else {
        const parsed = parseNativeJsonMetrics(await response.text())
        if (Object.keys(parsed).length) {
          Object.assign(jsonMetrics, parsed)
          sources.push("native-json-snapshot")
        } else if (!Object.keys(metrics).length) {
          errors.push({ source: "native-json-empty", nativeJsonMetricsUrl: jsonUrl })
        }
      }
    } catch (error) {
      errors.push({
        source: "native-json-unavailable",
        nativeJsonMetricsUrl: jsonUrl,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
  if (!Object.keys(metrics).length && !Object.keys(jsonMetrics).length) {
    return {
      available: false,
      source: typeof errors[0]?.source === "string" ? errors[0].source : "native-metrics-empty",
      metricsUrl: url,
      ...(jsonUrl ? { nativeJsonMetricsUrl: jsonUrl } : {}),
      ...(errors.length ? { errors } : {}),
    }
  }
  return {
    available: true,
    source: Array.from(new Set(sources)).join("+") || "native-metrics-snapshot",
    metricsUrl: url,
    ...(jsonUrl ? { nativeJsonMetricsUrl: jsonUrl } : {}),
    metrics,
    ...(Object.keys(jsonMetrics).length ? { jsonMetrics } : {}),
    ...(errors.length ? { errors } : {}),
    capturedAtUtc: new Date().toISOString(),
  }
}

function hardwareMetricsUrl(config: ServingProducerConfig): string | null {
  const engine = config.engine
  if (engine.hardwareMetricsUrl?.trim()) return engine.hardwareMetricsUrl.trim()
  if (engine.dcgmMetricsUrl?.trim()) return engine.dcgmMetricsUrl.trim()
  return engine.collectHardwareMetrics ? `${normalizeBaseUrl(engine.baseUrl)}/metrics` : null
}

function hardwareTelemetry(config: ServingProducerConfig): Record<string, unknown> {
  const configured = config.engine.hardwareTelemetry
  if (configured && typeof configured === "object" && !Array.isArray(configured)) {
    return { available: true, source: "engine-config", ...configured }
  }
  return { available: false, source: "not-configured" }
}

async function readHardwareMetrics(config: ServingProducerConfig): Promise<Record<string, unknown>> {
  const url = hardwareMetricsUrl(config)
  if (!url) return { available: false, source: "hardware-metrics-url-not-configured" }
  const fetchImpl = config.fetchImpl ?? fetch
  try {
    const response = await fetchImpl(url, {
      method: "GET",
      headers: {
        accept: "text/plain",
        ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
      },
    })
    if (!response.ok) {
      return { available: false, source: "hardware-prometheus-unavailable", metricsUrl: url, status: response.status }
    }
    const metrics = Object.fromEntries(
      Object.entries(parsePrometheusMetrics(await response.text())).filter(([key]) => key.startsWith("DCGM_FI_")),
    )
    if (!Object.keys(metrics).length) return { available: false, source: "dcgm-prometheus-empty", metricsUrl: url }
    return {
      available: true,
      source: "dcgm-prometheus-snapshot",
      metricsUrl: url,
      metrics,
      capturedAtUtc: new Date().toISOString(),
    }
  } catch (error) {
    return {
      available: false,
      source: "hardware-prometheus-unavailable",
      metricsUrl: url,
      error: error instanceof Error ? error.message : String(error),
    }
  }
}

function metricValue(metrics: Record<string, number>, candidates: string[]): number | null {
  for (const name of candidates) {
    const value = metrics[name]
    if (Number.isFinite(value)) return value
  }
  return null
}

function metricAverage(metrics: Record<string, number>, candidates: string[]): number | null {
  for (const name of candidates) {
    const value = metrics[name]
    if (!Number.isFinite(value)) continue
    const count = metrics[`${name}__sample_count`]
    return Number.isFinite(count) && count > 0 ? value / count : value
  }
  return null
}

function jsonMetricValue(metrics: Record<string, number>, candidates: string[]): number | null {
  return metricValue(metrics, candidates)
}

function firstNumber(...values: Array<number | null>): number | null {
  for (const value of values) {
    if (value != null && Number.isFinite(value)) return value
  }
  return null
}

function counterDelta(before: Record<string, number>, after: Record<string, number>, candidates: string[]): number | null {
  const beforeValue = metricValue(before, candidates)
  const afterValue = metricValue(after, candidates)
  if (beforeValue == null || afterValue == null) return null
  const delta = afterValue - beforeValue
  return delta >= 0 ? delta : null
}

function histogramDeltaMeanMs(before: Record<string, number>, after: Record<string, number>, bases: string[]): number | null {
  for (const base of bases) {
    const sumDelta = counterDelta(before, after, [`${base}_sum`])
    const countDelta = counterDelta(before, after, [`${base}_count`])
    if (sumDelta != null && countDelta != null && countDelta > 0) return (sumDelta / countDelta) * 1000
  }
  return null
}

function nativeMetricsDelta(
  config: ServingProducerConfig,
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): Record<string, unknown> {
  if (!before.available || !after.available) {
    return {
      available: false,
      source: "prometheus-delta-unavailable",
      metricsUrl: before.metricsUrl ?? after.metricsUrl ?? metricsUrl(config),
      before,
      after,
    }
  }
  const beforeMetrics = before.metrics && typeof before.metrics === "object" && !Array.isArray(before.metrics)
    ? before.metrics as Record<string, number>
    : {}
  const afterMetrics = after.metrics && typeof after.metrics === "object" && !Array.isArray(after.metrics)
    ? after.metrics as Record<string, number>
    : {}
  const afterJsonMetrics = after.jsonMetrics && typeof after.jsonMetrics === "object" && !Array.isArray(after.jsonMetrics)
    ? after.jsonMetrics as Record<string, number>
    : {}
  const prefixQueries = counterDelta(beforeMetrics, afterMetrics, [
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
    "sglang:prefix_cache_queries_total",
    "sglang:prefix_cache_queries",
    "sglang_prefix_cache_queries_total",
    "sglang_prefix_cache_queries",
    "trtllm_prefix_cache_queries",
    "trtllm:prefix_cache_queries_total",
    "trtllm_prefix_cache_queries_total",
  ])
  const prefixHits = counterDelta(beforeMetrics, afterMetrics, [
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
    "sglang:prefix_cache_hits_total",
    "sglang:prefix_cache_hits",
    "sglang_prefix_cache_hits_total",
    "sglang_prefix_cache_hits",
    "trtllm_prefix_cache_hits",
    "trtllm:prefix_cache_hits_total",
    "trtllm_prefix_cache_hits_total",
  ])
  const nativeE2eLatencyMs = histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:e2e_request_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang_e2e_request_latency_seconds",
    "trtllm:e2e_request_latency_seconds",
    "trtllm_e2e_request_latency_seconds",
  ])
  const queueWaitMs = histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_queue_time_seconds",
    "sglang:queue_time_seconds",
    "sglang:request_queue_time_seconds",
    "sglang_request_queue_time_seconds",
    "trtllm:request_queue_time_seconds",
    "trtllm_request_queue_time_seconds",
    "trtllm_queue_time_seconds",
    "trtllm:queue_time_seconds",
  ])
  const prefillMs = histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_prefill_time_seconds",
    "sglang:per_stage_req_latency_seconds{stage=prefill_forward}",
    "sglang:per_stage_req_latency_seconds{mode=prefill_forward}",
    "sglang:request_prefill_time_seconds",
    "sglang_request_prefill_time_seconds",
    "trtllm:request_prefill_time_seconds",
    "trtllm_request_prefill_time_seconds",
    "trtllm:context_time_seconds",
    "trtllm_context_time_seconds",
  ])
  let decodeMs = histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_decode_time_seconds",
    "sglang:per_stage_req_latency_seconds{stage=decode_forward}",
    "sglang:per_stage_req_latency_seconds{mode=decode}",
    "sglang:request_decode_time_seconds",
    "sglang_request_decode_time_seconds",
    "trtllm:request_decode_time_seconds",
    "trtllm_request_decode_time_seconds",
    "trtllm:generation_time_seconds",
    "trtllm_generation_time_seconds",
  ])
  if (decodeMs == null && nativeE2eLatencyMs != null && queueWaitMs != null && prefillMs != null) {
    const derivedDecodeMs = nativeE2eLatencyMs - queueWaitMs - prefillMs
    if (derivedDecodeMs >= 0) decodeMs = derivedDecodeMs
  }
  const trtllmKvUsedBlocks = jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.usedNumBlocks", "max.kvCacheStats.usedNumBlocks"])
  const trtllmKvMaxBlocks = jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.maxNumBlocks", "max.kvCacheStats.maxNumBlocks"])
  const trtllmKvUsage = trtllmKvUsedBlocks != null && trtllmKvMaxBlocks != null && trtllmKvMaxBlocks > 0
    ? trtllmKvUsedBlocks / trtllmKvMaxBlocks
    : null
  const prefixCacheHitRate = prefixHits != null && prefixQueries != null && prefixQueries > 0 ? prefixHits / prefixQueries : null
  const values: Record<string, number | null> = {
    nativeTtftMs: histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
      "vllm:time_to_first_token_seconds",
      "sglang:time_to_first_token_seconds",
      "sglang_time_to_first_token_seconds",
      "trtllm:time_to_first_token_seconds",
      "trtllm_time_to_first_token_seconds",
    ]),
    nativeTpotMs: histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
      "vllm:request_time_per_output_token_seconds",
      "vllm:time_per_output_token_seconds",
      "sglang:request_time_per_output_token_seconds",
      "sglang:time_per_output_token_seconds",
      "sglang_request_time_per_output_token_seconds",
      "sglang_time_per_output_token_seconds",
      "trtllm:request_time_per_output_token_seconds",
      "trtllm:time_per_output_token_seconds",
      "trtllm_request_time_per_output_token_seconds",
      "trtllm_time_per_output_token_seconds",
    ]),
    nativeInterTokenLatencyMs: histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
      "vllm:inter_token_latency_seconds",
      "sglang:inter_token_latency_seconds",
      "sglang_inter_token_latency_seconds",
      "trtllm:inter_token_latency_seconds",
      "trtllm_inter_token_latency_seconds",
    ]),
    nativeE2eLatencyMs,
    queueWaitMs,
    prefillMs,
    decodeMs,
    runningRequests: firstNumber(
      metricValue(afterMetrics, ["vllm:num_requests_running", "sglang:num_running_reqs", "sglang_num_running_reqs", "trtllm:num_requests_running", "trtllm_num_requests_running", "trtllm:num_active_requests", "trtllm_num_active_requests"]),
      jsonMetricValue(afterJsonMetrics, ["latest.numActiveRequests", "avg.numActiveRequests", "max.numActiveRequests"]),
    ),
    waitingRequests: metricValue(afterMetrics, ["vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang_num_queue_reqs", "trtllm:num_requests_waiting", "trtllm_num_requests_waiting", "trtllm:num_queued_requests", "trtllm_num_queued_requests"]),
    kvCacheUsagePct: firstNumber(
      metricValue(afterMetrics, ["vllm:kv_cache_usage_perc", "sglang:token_usage", "sglang_token_usage", "trtllm:kv_cache_usage_perc", "trtllm_kv_cache_usage_perc", "trtllm:kv_cache_utilization", "trtllm_kv_cache_utilization"]),
      trtllmKvUsage,
    ),
    trtllmIterationLatencyMs: jsonMetricValue(afterJsonMetrics, ["avg.iterLatencyMS", "latest.iterLatencyMS"]),
    trtllmGpuMemoryBytes: jsonMetricValue(afterJsonMetrics, ["latest.gpuMemUsage", "max.gpuMemUsage"]),
    trtllmKvCacheUsedBlocks: trtllmKvUsedBlocks,
    trtllmKvCacheMaxBlocks: trtllmKvMaxBlocks,
    prefixCacheQueriesDelta: prefixQueries,
    prefixCacheHitsDelta: prefixHits,
    cacheHitRate: firstNumber(
      prefixCacheHitRate,
      metricValue(afterMetrics, ["sglang:cache_hit_rate", "sglang_cache_hit_rate", "trtllm:kv_cache_hit_rate", "trtllm_kv_cache_hit_rate"]),
      jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.cacheHitRate", "avg.kvCacheStats.cacheHitRate"]),
    ),
    promptTokensCachedDelta: counterDelta(beforeMetrics, afterMetrics, [
      "vllm:prompt_tokens_cached_total",
      "sglang:prompt_tokens_cached_total",
      "sglang_prompt_tokens_cached_total",
      "trtllm:prompt_tokens_cached_total",
      "trtllm_prompt_tokens_cached_total",
    ]),
    promptTokensComputedDelta: counterDelta(beforeMetrics, afterMetrics, [
      "vllm:request_prefill_kv_computed_tokens_sum",
      "sglang:request_prefill_kv_computed_tokens_sum",
      "sglang_request_prefill_kv_computed_tokens_sum",
      "trtllm:request_prefill_kv_computed_tokens_sum",
      "trtllm_request_prefill_kv_computed_tokens_sum",
    ]),
  }
  const availableValues = Object.fromEntries(Object.entries(values).filter(([, value]) => value != null))
  const deltaSources = [
    ...(Object.keys(beforeMetrics).length && Object.keys(afterMetrics).length ? ["prometheus-delta"] : []),
    ...(Object.keys(afterJsonMetrics).length ? ["native-json-snapshot"] : []),
  ]
  return {
    available: Object.keys(availableValues).length > 0,
    source: deltaSources.join("+") || "native-metrics-delta",
    metricsUrl: after.metricsUrl ?? before.metricsUrl,
    nativeJsonMetricsUrl: after.nativeJsonMetricsUrl ?? before.nativeJsonMetricsUrl,
    beforeCapturedAtUtc: before.capturedAtUtc,
    afterCapturedAtUtc: after.capturedAtUtc,
    ...availableValues,
  }
}

function hardwareMetricsDelta(
  config: ServingProducerConfig,
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): Record<string, unknown> {
  const configured = hardwareTelemetry(config)
  if (configured.available) return configured
  if (!before.available || !after.available) {
    return {
      available: false,
      source: "dcgm-delta-unavailable",
      metricsUrl: before.metricsUrl ?? after.metricsUrl ?? hardwareMetricsUrl(config),
      before,
      after,
    }
  }
  const beforeMetrics = before.metrics && typeof before.metrics === "object" && !Array.isArray(before.metrics)
    ? before.metrics as Record<string, number>
    : {}
  const afterMetrics = after.metrics && typeof after.metrics === "object" && !Array.isArray(after.metrics)
    ? after.metrics as Record<string, number>
    : {}
  const energyMj = counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"])
  const values: Record<string, number | null> = {
    powerWatts: metricValue(afterMetrics, ["DCGM_FI_DEV_POWER_USAGE"]),
    powerWattsPerGpu: metricAverage(afterMetrics, ["DCGM_FI_DEV_POWER_USAGE"]),
    gpuUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE"]),
    memoryCopyUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_PROF_DRAM_ACTIVE"]),
    gpuTemperatureC: metricAverage(afterMetrics, ["DCGM_FI_DEV_GPU_TEMP"]),
    smClockMHz: metricAverage(afterMetrics, ["DCGM_FI_DEV_SM_CLOCK"]),
    memoryClockMHz: metricAverage(afterMetrics, ["DCGM_FI_DEV_MEM_CLOCK"]),
    fbUsedMiB: metricValue(afterMetrics, ["DCGM_FI_DEV_FB_USED"]),
    fbFreeMiB: metricValue(afterMetrics, ["DCGM_FI_DEV_FB_FREE"]),
    energyJoules: energyMj == null ? null : energyMj / 1000,
  }
  const availableValues = Object.fromEntries(Object.entries(values).filter(([, value]) => value != null))
  return {
    available: Object.keys(availableValues).length > 0,
    source: "dcgm-prometheus-delta",
    metricsUrl: after.metricsUrl ?? before.metricsUrl,
    beforeCapturedAtUtc: before.capturedAtUtc,
    afterCapturedAtUtc: after.capturedAtUtc,
    ...availableValues,
  }
}

function combineNativeTelemetry(...items: Array<Record<string, unknown>>): Record<string, unknown> {
  const combined: Record<string, unknown> = {}
  const availableSources: string[] = []
  const fallbackSources: string[] = []
  for (const item of items) {
    if (typeof item.source === "string") {
      if (item.available) availableSources.push(item.source)
      else fallbackSources.push(item.source)
    }
    Object.assign(combined, item)
  }
  combined.available = items.some((item) => Boolean(item.available))
  const sources = availableSources.length ? availableSources : fallbackSources
  if (sources.length) combined.source = Array.from(new Set(sources)).join("+")
  return combined
}

function numberFrom(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function nativeIterationFields(telemetry: Record<string, unknown>): Pick<
  ServingRequestSample,
  "nativeIterationLatencyMs" | "nativeGpuMemoryBytes" | "nativeKvCacheUsedBlocks" | "nativeKvCacheMaxBlocks"
> {
  return {
    nativeIterationLatencyMs: numberFrom(telemetry.nativeIterationLatencyMs) ?? numberFrom(telemetry.trtllmIterationLatencyMs),
    nativeGpuMemoryBytes: numberFrom(telemetry.nativeGpuMemoryBytes) ?? numberFrom(telemetry.trtllmGpuMemoryBytes),
    nativeKvCacheUsedBlocks: numberFrom(telemetry.nativeKvCacheUsedBlocks) ?? numberFrom(telemetry.trtllmKvCacheUsedBlocks),
    nativeKvCacheMaxBlocks: numberFrom(telemetry.nativeKvCacheMaxBlocks) ?? numberFrom(telemetry.trtllmKvCacheMaxBlocks),
  }
}

function tokenIdFrom(item: Record<string, any>): number | null {
  for (const key of ["token_id", "tokenId", "id"]) {
    const value = item[key]
    if (Number.isInteger(value)) return value
    if (typeof value === "string" && /^\d+$/.test(value)) return Number(value)
  }
  return null
}

function tokenIdWithSource(
  item: Record<string, any>,
  token: string,
  config: ServingProducerConfig,
): { tokenId: number | null; tokenIdSource: string | null } {
  const responseTokenId = tokenIdFrom(item)
  if (responseTokenId != null) return { tokenId: responseTokenId, tokenIdSource: "response-logprobs" }

  const mapped = config.engine.tokenIdMap?.[token]
  if (Number.isInteger(mapped)) return { tokenId: Number(mapped), tokenIdSource: "configured-token-id-map" }
  if (typeof mapped === "string" && /^\d+$/.test(mapped)) {
    return { tokenId: Number(mapped), tokenIdSource: "configured-token-id-map" }
  }

  const resolved = config.engine.tokenIdResolver?.(token, item, config.engine, config.request)
  if (Number.isInteger(resolved)) return { tokenId: Number(resolved), tokenIdSource: "configured-token-id-resolver" }
  if (typeof resolved === "string" && /^\d+$/.test(resolved)) {
    return { tokenId: Number(resolved), tokenIdSource: "configured-token-id-resolver" }
  }

  return { tokenId: null, tokenIdSource: null }
}

function sanitizeTopLogprobs(value: unknown, config: ServingProducerConfig): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return []
    const raw = item as Record<string, any>
    const token = typeof raw.token === "string" ? raw.token : ""
    const { tokenId, tokenIdSource } = tokenIdWithSource(raw, token, config)
    return [{
      tokenSha256: token ? sha256Text(token) : null,
      tokenBytes: token ? Buffer.byteLength(token) : null,
      tokenId,
      tokenIdSource,
      logprob: numberFrom(raw.logprob),
    }]
  })
}

function choiceTokenDetails(body: Record<string, any>, config: ServingProducerConfig): Array<Record<string, any>> {
  const content = choice(body).logprobs?.content
  if (!Array.isArray(content)) return []
  return content.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return []
    const raw = item as Record<string, any>
    const token = typeof raw.token === "string" ? raw.token : ""
    const bytes = Array.isArray(raw.bytes) ? raw.bytes : null
    const { tokenId, tokenIdSource } = tokenIdWithSource(raw, token, config)
    return [{
      tokenText: token,
      tokenSha256: token ? sha256Text(token) : null,
      tokenBytes: token ? Buffer.byteLength(token) : bytes?.length ?? null,
      tokenId,
      tokenIdSource,
      logprob: numberFrom(raw.logprob),
      topLogprobs: sanitizeTopLogprobs(raw.top_logprobs ?? raw.topLogprobs, config),
    }]
  })
}

function tokenDetailSummary(details: Array<Record<string, any>>, requested: boolean): Record<string, unknown> {
  if (!details.length) {
    return {
      tokenDetailsAvailable: false,
      tokenIdsAvailable: false,
      logprobsAvailable: false,
      tokenDetailCount: 0,
      tokenDetailSource: requested ? "requested-not-exposed" : "not-requested",
      tokenIdSource: null,
    }
  }
  const tokenIdSources = Array.from(new Set(
    details
      .filter((detail) => detail.tokenId != null && typeof detail.tokenIdSource === "string")
      .map((detail) => String(detail.tokenIdSource)),
  ))
  return {
    tokenDetailsAvailable: true,
    tokenIdsAvailable: details.some((detail) => detail.tokenId != null),
    logprobsAvailable: details.some((detail) => detail.logprob != null),
    tokenDetailCount: details.length,
    tokenDetailSource: "response-logprobs",
    tokenIdSource: tokenIdSources.length ? tokenIdSources.join("+") : null,
  }
}

function coerceTokenIds(value: unknown): number[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (Number.isInteger(item)) return [Number(item)]
    if (typeof item === "string" && /^\d+$/.test(item)) return [Number(item)]
    return []
  })
}

function promptTokenizerModel(config: ServingProducerConfig): string | null {
  const engineExtras = config.engine as unknown as Record<string, unknown>
  const model = config.request.tokenizerModel ?? engineExtras.tokenizerModel ?? engineExtras.tokenizer_model
  return typeof model === "string" && model.trim() ? model.trim() : null
}

function promptTokenDetailsRequired(config: ServingProducerConfig): boolean {
  const engineExtras = config.engine as unknown as Record<string, unknown>
  const requestExtras = config.request as unknown as Record<string, unknown>
  return Boolean(
    (config.request.promptTokenIds?.length ?? 0) > 0
      || (config.engine.promptTokenIds?.length ?? 0) > 0
      || promptTokenizerModel(config)
      || requestExtras.resolveTokenIdsWithTokenizer
      || engineExtras.resolveTokenIdsWithTokenizer
  )
}

function promptTokenCapture(config: ServingProducerConfig, payload: Record<string, unknown>): PromptTokenCaptureResult {
  const explicitTokenIds = coerceTokenIds(
    (config.request.promptTokenIds?.length ? config.request.promptTokenIds : undefined)
      ?? (config.engine.promptTokenIds?.length ? config.engine.promptTokenIds : undefined),
  )
  const tokenizerModel = promptTokenizerModel(config)
  const promptDigest = sha256Text(promptText(payload))
  if (!explicitTokenIds.length) {
    return {
      summary: {
        promptTokenIdsAvailable: false,
        promptTokenDetailCount: 0,
        promptTokenIdSource: null,
        promptTokenIdsSha256: null,
        promptTokenizationSource: promptTokenDetailsRequired(config) ? "js-tokenizer-not-configured" : "tokenizer-not-configured",
        promptTokenizerModel: tokenizerModel,
      },
      details: [],
    }
  }
  const source = "configured-prompt-token-ids"
  return {
    summary: {
      promptTokenIdsAvailable: true,
      promptTokenDetailCount: explicitTokenIds.length,
      promptTokenIdSource: source,
      promptTokenIdsSha256: sha256Json(explicitTokenIds),
      promptTokenizationSource: source,
      promptTokenizerModel: tokenizerModel,
    },
    details: explicitTokenIds.map((tokenId, tokenIndex) => ({
      tokenPhase: "prompt",
      tokenIndex,
      tokenId,
      tokenIdSource: source,
      tokenDetailSource: source,
      promptSha256: promptDigest,
      tokenLogprob: null,
      tokenTextSha256: null,
      topLogprobsJson: null,
    })),
  }
}

function promptTokenTimeline(
  requestId: string,
  details: Array<Record<string, unknown>>,
  receivedAtUtc: string,
): ServingTokenTimelineChunk[] {
  return details.map((detail, index) => ({
    requestId,
    tokenPhase: "prompt",
    chunkIndex: null,
    tokenIndex: numberFrom(detail.tokenIndex) ?? index,
    receivedAtUtc,
    relativeMs: 0,
    contentBytes: null,
    contentSha256: null,
    isFirstOutput: false,
    tokenId: numberFrom(detail.tokenId),
    tokenIdSource: typeof detail.tokenIdSource === "string" ? detail.tokenIdSource : null,
    tokenLogprob: null,
    tokenTextSha256: typeof detail.tokenTextSha256 === "string" ? detail.tokenTextSha256 : null,
    topLogprobsJson: null,
    tokenDetailSource: typeof detail.tokenDetailSource === "string" ? detail.tokenDetailSource : "configured-prompt-token-ids",
  }))
}

async function readSseEvents(response: Response, started: number): Promise<Array<Record<string, any>>> {
  const events: Array<Record<string, any>> = []
  const decoder = new TextDecoder()
  let buffer = ""
  const pushEvent = (frame: string, receivedMs: number, receivedAtUtc: string) => {
    const dataLines = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
    for (const data of dataLines) {
      if (!data) continue
      if (data === "[DONE]") {
        events.push({ done: true, raw: data, receivedMs, receivedAtUtc })
        continue
      }
      let body: Record<string, unknown>
      try {
        body = JSON.parse(data)
      } catch {
        body = { _parseError: data }
      }
      events.push({ body, raw: data, receivedMs, receivedAtUtc })
    }
  }

  if (!response.body) {
    const text = await response.text()
    const receivedMs = performance.now() - started
    const receivedAtUtc = new Date().toISOString()
    for (const frame of text.split(/\n\n/)) pushEvent(frame, receivedMs, receivedAtUtc)
    return events
  }

  const reader = response.body.getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    const receivedMs = performance.now() - started
    const receivedAtUtc = new Date().toISOString()
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split(/\n\n/)
    buffer = frames.pop() ?? ""
    for (const frame of frames) pushEvent(frame, receivedMs, receivedAtUtc)
  }
  buffer += decoder.decode()
  if (buffer.trim()) {
    pushEvent(buffer, performance.now() - started, new Date().toISOString())
  }
  return events
}

async function sendChatCompletion(
  config: ServingProducerConfig,
  requestIndex: number,
  campaignId: string,
  runId: string,
): Promise<ServingRequestSample & { rawCapture?: Record<string, unknown> }> {
  const fetchImpl = config.fetchImpl ?? fetch
  const endpoint = `${normalizeBaseUrl(config.engine.baseUrl)}${config.engine.requestPath ?? "/v1/chat/completions"}`
  const requestId = requestTraceId(config.engine, runId, requestIndex)
  const stream = config.request.stream !== false
  const payload = requestPayload(config.request, stream)
  const promptTokens = promptTokenCapture(config, payload)
  const nativeBefore = await readNativeMetrics(config)
  const hardwareBefore = await readHardwareMetrics(config)
  const tokenDetailsRequested = Boolean((payload as Record<string, unknown>).logprobs)
  const started = performance.now()
  const requestStartedAtUtc = new Date().toISOString()
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...traceHeaders(config.engine, campaignId, runId, requestId),
        ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
      },
      body: JSON.stringify(payload),
    })
    if (stream) {
      const events = await readSseEvents(response, started)
      const latencyMs = performance.now() - started
      const requestCompletedAtUtc = new Date().toISOString()
      const outputChunks: Array<{
        chunkIndex: number
        content: string
        receivedMs: number
        receivedAtUtc: string
        tokenDetails: Array<Record<string, any>>
      }> = []
      let responseId: string | undefined
      let responseModel: string | undefined
      let finishReason: string | undefined
      let usage: Record<string, any> = {}
      let lastBody: Record<string, any> = {}
      let telemetry = nativeTelemetry(config)
      const rawTokenDetails: Array<Record<string, any>> = []
      for (const event of events) {
        const body = event.body && typeof event.body === "object" ? event.body as Record<string, any> : {}
        if (Object.keys(body).length) {
          lastBody = body
          responseId = responseId ?? (typeof body.id === "string" ? body.id : undefined)
          responseModel = responseModel ?? (typeof body.model === "string" ? body.model : undefined)
          finishReason = finishReason ?? choice(body).finish_reason
          if (body.usage && typeof body.usage === "object") usage = body.usage
          const maybeTelemetry = nativeTelemetry(config, body)
          if (maybeTelemetry.available) telemetry = maybeTelemetry
        }
        const content = choiceContent(body)
        if (content) {
          const tokenDetails = choiceTokenDetails(body, config)
          rawTokenDetails.push(...tokenDetails.map((detail) => ({ ...detail, chunkIndex: outputChunks.length })))
          outputChunks.push({
            chunkIndex: outputChunks.length,
            content,
            receivedMs: Number(event.receivedMs),
            receivedAtUtc: String(event.receivedAtUtc),
            tokenDetails,
          })
        }
      }
      const nativeAfter = await readNativeMetrics(config)
      const hardwareAfter = await readHardwareMetrics(config)
      telemetry = combineNativeTelemetry(telemetry, nativeMetricsDelta(config, nativeBefore, nativeAfter))
      const hwTelemetry = hardwareMetricsDelta(config, hardwareBefore, hardwareAfter)
      const firstChunk = events[0]
      const firstOutput = outputChunks[0]
      const lastOutput = outputChunks[outputChunks.length - 1]
      const outputText = outputChunks.map((chunk) => chunk.content).join("")
      const tokenCountSource = Object.keys(usage).length ? "response-usage" : "client-estimate"
      const inputTokens = Number(usage.prompt_tokens ?? usage.promptTokens ?? 0) || estimatedTokenCount(promptText(payload))
      const completionTokens = Number(usage.completion_tokens ?? usage.completionTokens ?? 0) || outputChunks.length
      const totalTokens = Number(usage.total_tokens ?? usage.totalTokens ?? 0) || (inputTokens + completionTokens)
      const outputTokenCount = completionTokens || outputChunks.length
      const tpotMs = firstOutput && lastOutput
        ? (lastOutput.receivedMs - firstOutput.receivedMs) / Math.max(outputTokenCount - 1, 1)
        : null
      const gaps = outputChunks.slice(1).map((chunk, index) => chunk.receivedMs - outputChunks[index].receivedMs)
      const tokenSummary = tokenDetailSummary(rawTokenDetails, tokenDetailsRequested)
      const tokenTimeline: ServingTokenTimelineChunk[] = promptTokenTimeline(requestId, promptTokens.details, requestStartedAtUtc)
      let tokenIndex = 0
      for (const chunk of outputChunks) {
        const details = Array.isArray((chunk as Record<string, any>).tokenDetails)
          ? (chunk as Record<string, any>).tokenDetails as Array<Record<string, any>>
          : []
        if (details.length) {
          for (const detail of details) {
            const topLogprobs = Array.isArray(detail.topLogprobs) ? detail.topLogprobs : []
            tokenTimeline.push({
              requestId,
              tokenPhase: "output",
              chunkIndex: chunk.chunkIndex,
              tokenIndex,
              receivedAtUtc: chunk.receivedAtUtc,
              relativeMs: chunk.receivedMs,
              contentBytes: detail.tokenBytes ?? Buffer.byteLength(chunk.content),
              contentSha256: detail.tokenSha256 ?? sha256Text(chunk.content),
              isFirstOutput: tokenIndex === 0,
              tokenId: numberFrom(detail.tokenId),
              tokenIdSource: typeof detail.tokenIdSource === "string" ? detail.tokenIdSource : null,
              tokenLogprob: numberFrom(detail.logprob),
              tokenTextSha256: typeof detail.tokenSha256 === "string" ? detail.tokenSha256 : null,
              topLogprobsJson: topLogprobs.length ? JSON.stringify(topLogprobs) : null,
              tokenDetailSource: String(tokenSummary.tokenDetailSource),
            })
            tokenIndex += 1
          }
        } else {
          tokenTimeline.push({
            requestId,
            tokenPhase: "output",
            chunkIndex: chunk.chunkIndex,
            tokenIndex: null,
            receivedAtUtc: chunk.receivedAtUtc,
            relativeMs: chunk.receivedMs,
            contentBytes: Buffer.byteLength(chunk.content),
            contentSha256: sha256Text(chunk.content),
            isFirstOutput: chunk.chunkIndex === 0,
            tokenId: null,
            tokenIdSource: null,
            tokenLogprob: null,
            tokenTextSha256: null,
            topLogprobsJson: null,
            tokenDetailSource: String(tokenSummary.tokenDetailSource),
          })
        }
      }
      const redacted = redactedRequest(payload)
      const provenance = runtimeProvenance(config, telemetry)
      return {
        requestId,
        requestIndex,
        endpoint,
        requestStartedAtUtc,
        requestCompletedAtUtc,
        status: response.status,
        ok: response.ok,
        latencyMs,
        e2eLatencyMs: latencyMs,
        timeToFirstByteMs: numberFrom(firstChunk?.receivedMs),
        ttftMs: numberFrom(firstOutput?.receivedMs),
        ttfotMs: numberFrom(firstOutput?.receivedMs),
        tpotMs,
        interTokenLatencyMs: gaps.length ? sum(gaps) / gaps.length : tpotMs,
        firstChunkAtUtc: typeof firstChunk?.receivedAtUtc === "string" ? firstChunk.receivedAtUtc : null,
        firstOutputAtUtc: firstOutput?.receivedAtUtc ?? null,
        lastOutputAtUtc: lastOutput?.receivedAtUtc ?? null,
        streamChunkCount: events.length,
        outputTokenCount,
        promptTokens: inputTokens,
        completionTokens,
        totalTokens,
        tokenCountSource,
        responseId,
        responseModel,
        finishReason,
        ttftSource: "client-stream-content",
        streaming: true,
        promptSha256: redacted.promptSha256,
        requestPayloadSha256: redacted.requestPayloadSha256,
        outputSha256: sha256Text(outputText),
        outputBytes: Buffer.byteLength(outputText),
        nativeTelemetry: telemetry,
        nativeTelemetryAvailable: Boolean(telemetry.available),
        hardwareTelemetry: hwTelemetry,
        hardwareTelemetryAvailable: Boolean(hwTelemetry.available),
        nativeTelemetrySource: typeof telemetry.source === "string" ? telemetry.source : null,
        nativeMetricsUrl: typeof telemetry.metricsUrl === "string" ? telemetry.metricsUrl : null,
        nativeTtftMs: numberFrom(telemetry.nativeTtftMs),
        nativeTpotMs: numberFrom(telemetry.nativeTpotMs),
        nativeE2eLatencyMs: numberFrom(telemetry.nativeE2eLatencyMs),
        nativeInterTokenLatencyMs: numberFrom(telemetry.nativeInterTokenLatencyMs),
        ...nativeIterationFields(telemetry),
        runningRequests: numberFrom(telemetry.runningRequests),
        waitingRequests: numberFrom(telemetry.waitingRequests),
        kvCacheUsagePct: numberFrom(telemetry.kvCacheUsagePct),
        cacheHitRate: numberFrom(telemetry.cacheHitRate),
        prefixCacheQueriesDelta: numberFrom(telemetry.prefixCacheQueriesDelta),
        prefixCacheHitsDelta: numberFrom(telemetry.prefixCacheHitsDelta),
        promptTokensCachedDelta: numberFrom(telemetry.promptTokensCachedDelta),
        promptTokensComputedDelta: numberFrom(telemetry.promptTokensComputedDelta),
        hardwareTelemetrySource: typeof hwTelemetry.source === "string" ? hwTelemetry.source : null,
        hardwareMetricsUrl: typeof hwTelemetry.metricsUrl === "string" ? hwTelemetry.metricsUrl : null,
        avgPowerWatts: numberFrom(hwTelemetry.powerWatts),
        avgPowerWattsPerGpu: numberFrom(hwTelemetry.powerWattsPerGpu),
        gpuUtilizationPct: numberFrom(hwTelemetry.gpuUtilizationPct),
        memoryCopyUtilizationPct: numberFrom(hwTelemetry.memoryCopyUtilizationPct),
        gpuTemperatureC: numberFrom(hwTelemetry.gpuTemperatureC),
        smClockMHz: numberFrom(hwTelemetry.smClockMHz),
        memoryClockMHz: numberFrom(hwTelemetry.memoryClockMHz),
        fbUsedMiB: numberFrom(hwTelemetry.fbUsedMiB),
        fbFreeMiB: numberFrom(hwTelemetry.fbFreeMiB),
        energyJoules: numberFrom(hwTelemetry.energyJoules),
        ...(tokenSummary as {
          tokenDetailsAvailable: boolean
          tokenIdsAvailable: boolean
          logprobsAvailable: boolean
          tokenDetailCount: number
          tokenDetailSource: string
          tokenIdSource: string | null
        }),
        ...promptTokens.summary,
        queueWaitMs: numberFrom(telemetry.queueWaitMs),
        prefillMs: numberFrom(telemetry.prefillMs),
        decodeMs: numberFrom(telemetry.decodeMs),
        ...provenance,
        tokenTimeline,
        error: response.ok ? undefined : JSON.stringify(lastBody),
        rawCapture: {
          requestId,
          endpoint,
          requestPayload: payload,
          responseEvents: events,
          outputText,
          tokenDetails: rawTokenDetails,
          promptTokenDetails: promptTokens.details,
          nativeTelemetry: telemetry,
          hardwareTelemetry: hwTelemetry,
          runtimeProvenance: provenance,
        },
      }
    }

    const latencyMs = performance.now() - started
    const requestCompletedAtUtc = new Date().toISOString()
    const body = await response.json().catch(() => ({})) as Record<string, any>
    const usage = body.usage ?? {}
    const outputText = choiceContent(body)
    const nativeAfter = await readNativeMetrics(config)
    const hardwareAfter = await readHardwareMetrics(config)
    const telemetry = combineNativeTelemetry(
      nativeTelemetry(config, body),
      nativeMetricsDelta(config, nativeBefore, nativeAfter),
    )
    const hwTelemetry = hardwareMetricsDelta(config, hardwareBefore, hardwareAfter)
    const rawTokenDetails = choiceTokenDetails(body, config)
    const tokenSummary = tokenDetailSummary(rawTokenDetails, tokenDetailsRequested)
    const tokenTimeline: ServingTokenTimelineChunk[] = [
      ...promptTokenTimeline(requestId, promptTokens.details, requestStartedAtUtc),
      ...rawTokenDetails.map((detail, index): ServingTokenTimelineChunk => ({
        requestId,
        tokenPhase: "output",
        chunkIndex: 0,
        tokenIndex: index,
        receivedAtUtc: requestCompletedAtUtc,
        relativeMs: latencyMs,
        contentBytes: numberFrom(detail.tokenBytes) ?? 0,
        contentSha256: typeof detail.tokenSha256 === "string" ? detail.tokenSha256 : "",
        isFirstOutput: index === 0,
        tokenId: numberFrom(detail.tokenId),
        tokenIdSource: typeof detail.tokenIdSource === "string" ? detail.tokenIdSource : null,
        tokenLogprob: numberFrom(detail.logprob),
        tokenTextSha256: typeof detail.tokenSha256 === "string" ? detail.tokenSha256 : null,
        topLogprobsJson: Array.isArray(detail.topLogprobs) && detail.topLogprobs.length ? JSON.stringify(detail.topLogprobs) : null,
        tokenDetailSource: String(tokenSummary.tokenDetailSource),
      })),
    ]
    const redacted = redactedRequest(payload)
    const provenance = runtimeProvenance(config, telemetry)
    return {
      requestId,
      requestIndex,
      endpoint,
      requestStartedAtUtc,
      requestCompletedAtUtc,
      status: response.status,
      ok: response.ok,
      latencyMs,
      e2eLatencyMs: latencyMs,
      timeToFirstByteMs: null,
      ttftMs: null,
      ttfotMs: null,
      tpotMs: null,
      interTokenLatencyMs: null,
      streamChunkCount: 0,
      outputTokenCount: Number(usage.completion_tokens ?? usage.completionTokens ?? 0),
      promptTokens: Number(usage.prompt_tokens ?? usage.promptTokens ?? 0),
      completionTokens: Number(usage.completion_tokens ?? usage.completionTokens ?? 0),
      totalTokens: Number(usage.total_tokens ?? usage.totalTokens ?? 0),
      tokenCountSource: Object.keys(usage).length ? "response-usage" : "none",
      responseId: typeof body.id === "string" ? body.id : undefined,
      responseModel: typeof body.model === "string" ? body.model : undefined,
      finishReason: choice(body).finish_reason,
      ttftSource: "not-streamed",
      streaming: false,
      promptSha256: redacted.promptSha256,
      requestPayloadSha256: redacted.requestPayloadSha256,
      outputSha256: sha256Text(outputText),
      outputBytes: Buffer.byteLength(outputText),
      nativeTelemetry: telemetry,
      nativeTelemetryAvailable: Boolean(telemetry.available),
      hardwareTelemetry: hwTelemetry,
      hardwareTelemetryAvailable: Boolean(hwTelemetry.available),
      nativeTelemetrySource: typeof telemetry.source === "string" ? telemetry.source : null,
      nativeMetricsUrl: typeof telemetry.metricsUrl === "string" ? telemetry.metricsUrl : null,
      nativeTtftMs: numberFrom(telemetry.nativeTtftMs),
      nativeTpotMs: numberFrom(telemetry.nativeTpotMs),
      nativeE2eLatencyMs: numberFrom(telemetry.nativeE2eLatencyMs),
      nativeInterTokenLatencyMs: numberFrom(telemetry.nativeInterTokenLatencyMs),
      ...nativeIterationFields(telemetry),
      runningRequests: numberFrom(telemetry.runningRequests),
      waitingRequests: numberFrom(telemetry.waitingRequests),
      kvCacheUsagePct: numberFrom(telemetry.kvCacheUsagePct),
      cacheHitRate: numberFrom(telemetry.cacheHitRate),
      prefixCacheQueriesDelta: numberFrom(telemetry.prefixCacheQueriesDelta),
      prefixCacheHitsDelta: numberFrom(telemetry.prefixCacheHitsDelta),
      promptTokensCachedDelta: numberFrom(telemetry.promptTokensCachedDelta),
      promptTokensComputedDelta: numberFrom(telemetry.promptTokensComputedDelta),
      hardwareTelemetrySource: typeof hwTelemetry.source === "string" ? hwTelemetry.source : null,
      hardwareMetricsUrl: typeof hwTelemetry.metricsUrl === "string" ? hwTelemetry.metricsUrl : null,
      avgPowerWatts: numberFrom(hwTelemetry.powerWatts),
      avgPowerWattsPerGpu: numberFrom(hwTelemetry.powerWattsPerGpu),
      gpuUtilizationPct: numberFrom(hwTelemetry.gpuUtilizationPct),
      memoryCopyUtilizationPct: numberFrom(hwTelemetry.memoryCopyUtilizationPct),
      gpuTemperatureC: numberFrom(hwTelemetry.gpuTemperatureC),
      smClockMHz: numberFrom(hwTelemetry.smClockMHz),
      memoryClockMHz: numberFrom(hwTelemetry.memoryClockMHz),
      fbUsedMiB: numberFrom(hwTelemetry.fbUsedMiB),
      fbFreeMiB: numberFrom(hwTelemetry.fbFreeMiB),
      energyJoules: numberFrom(hwTelemetry.energyJoules),
      ...(tokenSummary as {
        tokenDetailsAvailable: boolean
        tokenIdsAvailable: boolean
        logprobsAvailable: boolean
        tokenDetailCount: number
        tokenDetailSource: string
        tokenIdSource: string | null
      }),
      ...promptTokens.summary,
      queueWaitMs: numberFrom(telemetry.queueWaitMs),
      prefillMs: numberFrom(telemetry.prefillMs),
      decodeMs: numberFrom(telemetry.decodeMs),
      ...provenance,
      tokenTimeline,
      error: response.ok ? undefined : JSON.stringify(body),
      rawCapture: {
        requestId,
        endpoint,
        requestPayload: payload,
        responseBody: body,
        outputText,
        tokenDetails: rawTokenDetails,
        promptTokenDetails: promptTokens.details,
        nativeTelemetry: telemetry,
        hardwareTelemetry: hwTelemetry,
        runtimeProvenance: provenance,
      },
    }
  } catch (error) {
    const latencyMs = performance.now() - started
    const telemetry = nativeTelemetry(config)
    const hwTelemetry = hardwareTelemetry(config)
    return {
      requestId,
      requestIndex,
      endpoint,
      requestStartedAtUtc,
      requestCompletedAtUtc: new Date().toISOString(),
      status: 0,
      ok: false,
      latencyMs,
      e2eLatencyMs: latencyMs,
      timeToFirstByteMs: null,
      ttftMs: null,
      ttfotMs: null,
      tpotMs: null,
      interTokenLatencyMs: null,
      streamChunkCount: 0,
      outputTokenCount: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      tokenCountSource: "none",
      ttftSource: stream ? "client-stream-content" : "not-streamed",
      streaming: stream,
      nativeTelemetry: telemetry,
      nativeTelemetryAvailable: false,
      hardwareTelemetry: hwTelemetry,
      hardwareTelemetryAvailable: false,
      nativeTelemetrySource: typeof telemetry.source === "string" ? telemetry.source : null,
      hardwareTelemetrySource: typeof hwTelemetry.source === "string" ? hwTelemetry.source : null,
      ...runtimeProvenance(config, telemetry),
      tokenDetailsAvailable: false,
      tokenIdsAvailable: false,
      logprobsAvailable: false,
      tokenDetailCount: 0,
      tokenDetailSource: "error",
      tokenIdSource: null,
      promptTokenIdsAvailable: false,
      promptTokenDetailCount: 0,
      promptTokenIdSource: null,
      promptTokenIdsSha256: null,
      promptTokenizationSource: "error",
      promptTokenizerModel: promptTokenizerModel(config),
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
  const durationSeconds = Math.max(sum(successful.map((sample) => sample.e2eLatencyMs)) / 1000, 0.001)
  const outputTokens = sum(successful.map((sample) => sample.completionTokens))
  const totalTokens = sum(successful.map((sample) => sample.totalTokens))
  const promptTokens = sum(successful.map((sample) => sample.promptTokens))
  const outputTpm = outputTokens / (durationSeconds / 60)
  const totalTpm = totalTokens / (durationSeconds / 60)
  const avg = (values: number[]) => values.length ? sum(values) / values.length : null
  const gpuCount = config.pricing?.gpuCount ?? Number(config.workload?.parallelism ?? 1)
  const usdPerGpuHour = config.pricing?.usdPerGpuHour
  const configuredPowerWattsPerGpu = config.pricing?.powerWattsPerGpu
  const observedPowerWattsPerGpu = avg(successful.map((sample) => sample.avgPowerWattsPerGpu ?? null).filter(finite))
  const powerWattsPerGpu = finite(configuredPowerWattsPerGpu) ? configuredPowerWattsPerGpu : observedPowerWattsPerGpu
  const costUsd = finite(usdPerGpuHour) ? (durationSeconds / 3600) * usdPerGpuHour * gpuCount : null
  const usdPer1mOutputTokens = costUsd != null && outputTokens > 0 ? costUsd / (outputTokens / 1_000_000) : null
  const usdPer1mTotalTokens = costUsd != null && totalTokens > 0 ? costUsd / (totalTokens / 1_000_000) : null
  const tokensPerWatt = finite(powerWattsPerGpu) && powerWattsPerGpu > 0
    ? (totalTokens / durationSeconds) / (powerWattsPerGpu * gpuCount)
    : null
  const latencies = successful.map((sample) => sample.e2eLatencyMs).filter(finite)
  const ttfts = successful.map((sample) => sample.ttftMs).filter(finite)
  const ttfots = successful.map((sample) => sample.ttfotMs).filter(finite)
  const tpots = successful.map((sample) => sample.tpotMs).filter(finite)
  const firstBytes = successful.map((sample) => sample.timeToFirstByteMs).filter(finite)
  const interTokenLatencies = successful.map((sample) => sample.interTokenLatencyMs).filter(finite)
  const queueWaits = successful.map((sample) => sample.queueWaitMs ?? null).filter(finite)
  const prefills = successful.map((sample) => sample.prefillMs ?? null).filter(finite)
  const decodes = successful.map((sample) => sample.decodeMs ?? null).filter(finite)
  const avgLatencyMs = successful.length ? sum(successful.map((sample) => sample.e2eLatencyMs)) / successful.length : null
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
    p50LatencyMs: percentile(latencies, 50),
    p95LatencyMs: percentile(latencies, 95),
    p99LatencyMs: percentile(latencies, 99),
    avgTimeToFirstByteMs: avg(firstBytes),
    p50TimeToFirstByteMs: percentile(firstBytes, 50),
    p95TimeToFirstByteMs: percentile(firstBytes, 95),
    p99TimeToFirstByteMs: percentile(firstBytes, 99),
    avgTtftMs: avg(ttfts),
    p50TtftMs: percentile(ttfts, 50),
    p95TtftMs: percentile(ttfts, 95),
    p99TtftMs: percentile(ttfts, 99),
    avgTtfotMs: avg(ttfots),
    p50TtfotMs: percentile(ttfots, 50),
    p95TtfotMs: percentile(ttfots, 95),
    p99TtfotMs: percentile(ttfots, 99),
    avgTpotMs: avg(tpots),
    p50TpotMs: percentile(tpots, 50),
    p95TpotMs: percentile(tpots, 95),
    p99TpotMs: percentile(tpots, 99),
    avgInterTokenLatencyMs: avg(interTokenLatencies),
    p50InterTokenLatencyMs: percentile(interTokenLatencies, 50),
    p95InterTokenLatencyMs: percentile(interTokenLatencies, 95),
    p99InterTokenLatencyMs: percentile(interTokenLatencies, 99),
    avgQueueWaitMs: avg(queueWaits),
    p50QueueWaitMs: percentile(queueWaits, 50),
    p95QueueWaitMs: percentile(queueWaits, 95),
    p99QueueWaitMs: percentile(queueWaits, 99),
    avgPrefillMs: avg(prefills),
    p50PrefillMs: percentile(prefills, 50),
    p95PrefillMs: percentile(prefills, 95),
    p99PrefillMs: percentile(prefills, 99),
    avgDecodeMs: avg(decodes),
    p50DecodeMs: percentile(decodes, 50),
    p95DecodeMs: percentile(decodes, 95),
    p99DecodeMs: percentile(decodes, 99),
    avgNativeIterationLatencyMs: avg(successful.map((sample) => sample.nativeIterationLatencyMs ?? null).filter(finite)),
    avgNativeGpuMemoryBytes: avg(successful.map((sample) => sample.nativeGpuMemoryBytes ?? null).filter(finite)),
    avgNativeKvCacheUsedBlocks: avg(successful.map((sample) => sample.nativeKvCacheUsedBlocks ?? null).filter(finite)),
    avgNativeKvCacheMaxBlocks: avg(successful.map((sample) => sample.nativeKvCacheMaxBlocks ?? null).filter(finite)),
    usdPer1mOutputTokens,
    usdPer1mTotalTokens,
    avgPowerWatts: avg(successful.map((sample) => sample.avgPowerWatts ?? null).filter(finite)),
    avgPowerWattsPerGpu: powerWattsPerGpu ?? null,
    powerSource: finite(configuredPowerWattsPerGpu) ? "pricing-config" : observedPowerWattsPerGpu != null ? "dcgm" : "unknown",
    avgGpuUtilizationPct: avg(successful.map((sample) => sample.gpuUtilizationPct ?? null).filter(finite)),
    avgMemoryCopyUtilizationPct: avg(successful.map((sample) => sample.memoryCopyUtilizationPct ?? null).filter(finite)),
    totalEnergyJoules: sum(successful.map((sample) => sample.energyJoules ?? 0)),
    tokensPerWatt,
    campaignCount: Math.max(successful.length, 1),
    latestCapturedAtUtc: capturedAtUtc,
    experimentFamily: "serving-producer",
    experimentStatus: successful.length === samples.length ? "accepted" : "partial",
    verdictTier: successful.length === samples.length ? "request-captured" : "request-errors",
    solRigor: config.runClass === "measured" ? "l3" : "smoke",
    plotReadyPoints: 0,
    dcgmGrounded: successful.length > 0 && successful.every((sample) => sample.hardwareTelemetryAvailable),
    streamingRequestCount: successful.filter((sample) => sample.streaming).length,
    nativeTelemetryAvailableCount: successful.filter((sample) => sample.nativeTelemetryAvailable).length,
    nativeTelemetryRequired: Boolean(config.engine.requireNativeTelemetry),
    hardwareTelemetryAvailableCount: successful.filter((sample) => sample.hardwareTelemetryAvailable).length,
    hardwareTelemetryRequired: Boolean(config.engine.requireHardwareTelemetry || hardwareMetricsUrl(config)),
    tokenDetailsAvailableCount: successful.filter((sample) => sample.tokenDetailsAvailable).length,
    tokenIdsAvailableCount: successful.filter((sample) => sample.tokenIdsAvailable).length,
    logprobsAvailableCount: successful.filter((sample) => sample.logprobsAvailable).length,
    tokenDetailsRequired: Boolean(requestPayload(config.request, config.request.stream !== false).logprobs),
    promptTokenIdsAvailableCount: successful.filter((sample) => sample.promptTokenIdsAvailable).length,
    promptTokenDetailsRequired: promptTokenDetailsRequired(config),
    hardwareProvenance: config.workload?.hardware && config.workload.hardware !== "unknown" ? "configured" : "unknown",
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
  const sampleRows = samples.map((sample): Record<string, unknown> => ({
    surface: "serving_request_sample",
    model: config.request.model,
    hardware: config.workload?.hardware ?? "unknown",
    runtimeFramework: engineLabel,
    runtimeEngine: config.engine.engine,
    operatingPoint: config.workload?.operatingPoint ?? "laptop-smoke",
    basis: "per_request",
    requestId: sample.requestId,
    requestIndex: sample.requestIndex,
    status: sample.status,
    ok: sample.ok,
    e2eLatencyMs: sample.e2eLatencyMs,
    latencyMs: sample.latencyMs,
    timeToFirstByteMs: sample.timeToFirstByteMs,
    ttftMs: sample.ttftMs,
    ttfotMs: sample.ttfotMs,
    tpotMs: sample.tpotMs,
    interTokenLatencyMs: sample.interTokenLatencyMs,
    promptTokens: sample.promptTokens,
    completionTokens: sample.completionTokens,
    totalTokens: sample.totalTokens,
    tokenCountSource: sample.tokenCountSource,
    outputTokenCount: sample.outputTokenCount,
    streamChunkCount: sample.streamChunkCount,
    finishReason: sample.finishReason,
    ttftSource: sample.ttftSource,
    promptSha256: sample.promptSha256,
    requestPayloadSha256: sample.requestPayloadSha256,
    outputSha256: sample.outputSha256,
    nativeTelemetryAvailable: sample.nativeTelemetryAvailable,
    hardwareTelemetryAvailable: sample.hardwareTelemetryAvailable,
    nativeTelemetrySource: sample.nativeTelemetrySource,
    nativeMetricsUrl: sample.nativeMetricsUrl,
    nativeTtftMs: sample.nativeTtftMs,
    nativeTpotMs: sample.nativeTpotMs,
    nativeE2eLatencyMs: sample.nativeE2eLatencyMs,
    nativeInterTokenLatencyMs: sample.nativeInterTokenLatencyMs,
    nativeIterationLatencyMs: sample.nativeIterationLatencyMs,
    nativeGpuMemoryBytes: sample.nativeGpuMemoryBytes,
    nativeKvCacheUsedBlocks: sample.nativeKvCacheUsedBlocks,
    nativeKvCacheMaxBlocks: sample.nativeKvCacheMaxBlocks,
    runningRequests: sample.runningRequests,
    waitingRequests: sample.waitingRequests,
    kvCacheUsagePct: sample.kvCacheUsagePct,
    cacheHitRate: sample.cacheHitRate,
    prefixCacheQueriesDelta: sample.prefixCacheQueriesDelta,
    prefixCacheHitsDelta: sample.prefixCacheHitsDelta,
    promptTokensCachedDelta: sample.promptTokensCachedDelta,
    promptTokensComputedDelta: sample.promptTokensComputedDelta,
    hardwareTelemetrySource: sample.hardwareTelemetrySource,
    hardwareMetricsUrl: sample.hardwareMetricsUrl,
    avgPowerWatts: sample.avgPowerWatts,
    avgPowerWattsPerGpu: sample.avgPowerWattsPerGpu,
    gpuUtilizationPct: sample.gpuUtilizationPct,
    memoryCopyUtilizationPct: sample.memoryCopyUtilizationPct,
    gpuTemperatureC: sample.gpuTemperatureC,
    smClockMHz: sample.smClockMHz,
    memoryClockMHz: sample.memoryClockMHz,
    fbUsedMiB: sample.fbUsedMiB,
    fbFreeMiB: sample.fbFreeMiB,
    energyJoules: sample.energyJoules,
    tokenDetailsAvailable: sample.tokenDetailsAvailable,
    tokenIdsAvailable: sample.tokenIdsAvailable,
    logprobsAvailable: sample.logprobsAvailable,
    tokenDetailCount: sample.tokenDetailCount,
    tokenDetailSource: sample.tokenDetailSource,
    tokenIdSource: sample.tokenIdSource,
    promptTokenIdsAvailable: sample.promptTokenIdsAvailable,
    promptTokenDetailCount: sample.promptTokenDetailCount,
    promptTokenIdSource: sample.promptTokenIdSource,
    promptTokenIdsSha256: sample.promptTokenIdsSha256,
    promptTokenizationSource: sample.promptTokenizationSource,
    promptTokenizerModel: sample.promptTokenizerModel,
    queueWaitMs: sample.queueWaitMs,
    prefillMs: sample.prefillMs,
    decodeMs: sample.decodeMs,
    engineVersion: sample.engineVersion,
    modelRevision: sample.modelRevision,
    imageTag: sample.imageTag,
    imageDigest: sample.imageDigest,
    serverArgsSha256: sample.serverArgsSha256,
    processId: sample.processId,
    containerId: sample.containerId,
    podName: sample.podName,
    nodeName: sample.nodeName,
    hostName: sample.hostName,
    latestCapturedAtUtc: capturedAtUtc,
  }))
  const timelineRows = samples.flatMap((sample) => (sample.tokenTimeline ?? []).map((chunk): Record<string, unknown> => ({
    surface: "serving_token_timeline",
    model: config.request.model,
    runtimeFramework: engineLabel,
    runtimeEngine: config.engine.engine,
    requestId: sample.requestId,
    tokenPhase: chunk.tokenPhase ?? "output",
    chunkIndex: chunk.chunkIndex,
    receivedAtUtc: chunk.receivedAtUtc,
    relativeMs: chunk.relativeMs,
    contentBytes: chunk.contentBytes,
    contentSha256: chunk.contentSha256,
    isFirstOutput: chunk.isFirstOutput,
    tokenIndex: chunk.tokenIndex,
    tokenId: chunk.tokenId,
    tokenIdSource: chunk.tokenIdSource,
    tokenLogprob: chunk.tokenLogprob,
    tokenTextSha256: chunk.tokenTextSha256,
    topLogprobsJson: chunk.topLogprobsJson,
    tokenDetailSource: chunk.tokenDetailSource,
    latestCapturedAtUtc: capturedAtUtc,
  })))
  return [row, ...sampleRows, ...timelineRows]
}

const PRODUCER_COVERAGE_DESCRIPTIONS: Record<string, string> = {
  clientStreamTiming: "Client stream=true timing for E2E, TTFB, TTFT, TTFOT, TPOT, and output token timeline rows.",
  nativeRuntimeTelemetry: "Native runtime timing/cache/concurrency fields exposed by vLLM, SGLang, or TensorRT-LLM metrics.",
  dcgmHardwareTelemetry: "DCGM hardware counters for power, utilization, clocks, memory, temperature, and energy.",
  promptTokenIds: "Tokenizer-exact prompt/input token IDs and prompt token provenance.",
  outputTokenIdsLogprobs: "Output token IDs, token logprobs, top-logprobs, and token provenance.",
  operatorFullArtifacts: "Operator-full raw request/response artifacts retained outside customer-safe rows.",
  runtimeProvenance: "Engine version, model revision, image, server args, process, container, pod, node, or host provenance.",
}

function coverageStatus(proven: number, expected: number): string {
  if (expected <= 0) return "not_configured"
  if (proven >= expected) return "proven"
  if (proven > 0) return "partial"
  return "missing"
}

function hasRuntimeProvenance(sample: ServingRequestSample): boolean {
  return [
    sample.engineVersion,
    sample.modelRevision,
    sample.imageDigest,
    sample.serverArgsSha256,
    sample.processId,
    sample.containerId,
    sample.podName,
    sample.nodeName,
    sample.hostName,
  ].some((value) => value != null && value !== "")
}

function producerCoverageRows(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  aggregateRow: Record<string, unknown>,
  rawArtifactPath: string,
  capturedAtUtc: string,
): Record<string, unknown>[] {
  const successful = samples.filter((sample) => sample.ok)
  const expectedSamples = successful.length || samples.length || 1
  const tokenRequired = Boolean(requestPayload(config.request, config.request.stream !== false).logprobs)
  const promptRequired = Boolean(aggregateRow.promptTokenDetailsRequired)
  const nativeExpected = aggregateRow.nativeTelemetryRequired ? expectedSamples : 0
  const hardwareExpected = aggregateRow.hardwareTelemetryRequired ? expectedSamples : 0
  const specs: Array<[string, number, number, string[]]> = []

  const streamProven = successful.filter((sample) =>
    sample.streaming === true &&
    finite(sample.e2eLatencyMs) &&
    finite(sample.timeToFirstByteMs) &&
    finite(sample.ttftMs) &&
    finite(sample.ttfotMs) &&
    finite(sample.tpotMs) &&
    Boolean(sample.tokenTimeline?.some((row) => (row.tokenPhase ?? "output") === "output")),
  ).length
  specs.push([
    "clientStreamTiming",
    streamProven,
    expectedSamples,
    streamProven === expectedSamples ? [] : ["stream timing or output token timeline rows missing"],
  ])

  const nativeProven = successful.filter((sample) =>
    sample.nativeTelemetryAvailable === true &&
    finite(sample.nativeTtftMs) &&
    finite(sample.nativeTpotMs) &&
    finite(sample.nativeE2eLatencyMs) &&
    finite(sample.queueWaitMs) &&
    finite(sample.prefillMs) &&
    finite(sample.decodeMs),
  ).length
  specs.push([
    "nativeRuntimeTelemetry",
    nativeProven,
    nativeExpected,
    nativeExpected === 0 || nativeProven === nativeExpected ? [] : ["native runtime metrics missing"],
  ])

  const hardwareProven = successful.filter((sample) =>
    sample.hardwareTelemetryAvailable === true &&
    finite(sample.avgPowerWatts) &&
    finite(sample.gpuUtilizationPct) &&
    finite(sample.gpuTemperatureC) &&
    finite(sample.energyJoules),
  ).length
  specs.push([
    "dcgmHardwareTelemetry",
    hardwareProven,
    hardwareExpected,
    hardwareExpected === 0 || hardwareProven === hardwareExpected ? [] : ["DCGM hardware counters missing"],
  ])

  const promptProven = successful.filter((sample) =>
    sample.promptTokenIdsAvailable === true &&
    typeof sample.promptTokenIdsSha256 === "string" &&
    typeof sample.promptTokenIdSource === "string",
  ).length
  specs.push([
    "promptTokenIds",
    promptProven,
    promptRequired ? expectedSamples : 0,
    !promptRequired || promptProven === expectedSamples ? [] : ["prompt token IDs missing"],
  ])

  const outputProven = successful.filter((sample) =>
    sample.tokenDetailsAvailable === true &&
    sample.tokenIdsAvailable === true &&
    sample.logprobsAvailable === true &&
    typeof sample.tokenIdSource === "string",
  ).length
  specs.push([
    "outputTokenIdsLogprobs",
    outputProven,
    tokenRequired ? expectedSamples : 0,
    !tokenRequired || outputProven === expectedSamples ? [] : ["output token IDs/logprobs missing"],
  ])

  const rawPresent = rawArtifactPath && existsSync(rawArtifactPath) ? 1 : 0
  specs.push([
    "operatorFullArtifacts",
    rawPresent,
    1,
    rawPresent ? [] : ["operator-full raw artifact missing"],
  ])

  const runtimeProven = successful.filter(hasRuntimeProvenance).length
  specs.push([
    "runtimeProvenance",
    runtimeProven,
    expectedSamples,
    runtimeProven === expectedSamples ? [] : ["runtime provenance missing or partial"],
  ])

  const allProven = specs.every(([, proven, expected]) => ["proven", "not_configured"].includes(coverageStatus(proven, expected)))
  return specs.map(([category, proven, expected, missing]) => ({
    surface: "serving_telemetry_coverage",
    model: config.request.model,
    hardware: config.workload?.hardware ?? "unknown",
    runtimeFramework: ENGINE_LABELS[config.engine.engine],
    runtimeEngine: config.engine.engine,
    coverageSource: "producer-submit",
    coverageCategory: category,
    coverageStatus: coverageStatus(proven, expected),
    provenCount: proven,
    expectedCount: expected,
    missingJson: JSON.stringify(missing),
    description: PRODUCER_COVERAGE_DESCRIPTIONS[category],
    allProven,
    latestCapturedAtUtc: capturedAtUtc,
  }))
}

async function writeSummaryArtifact(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  measurements: Record<string, unknown>[],
  capturedAtUtc: string,
  rawArtifactPath: string,
): Promise<string> {
  const artifactDir = config.artifactDir ?? path.join(process.cwd(), ".performance-iq", "serving-producers")
  await mkdir(artifactDir, { recursive: true })
  const safeModel = safeSlug(config.request.model)
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
    endpointPreflight: config.engine.endpointPreflight,
    model: config.request.model,
    capturePolicy: {
      mode: "operator-full",
      rawArtifactPath,
      summaryPayload: "redacted-hashes-and-derived-fields",
    },
    request: redactedRequest(requestPayload(config.request, config.request.stream !== false)),
    requestTrace: samples.map((sample) => ({
      requestId: sample.requestId,
      requestIndex: sample.requestIndex,
      endpoint: sample.endpoint,
      requestStartedAtUtc: sample.requestStartedAtUtc,
      requestCompletedAtUtc: sample.requestCompletedAtUtc,
      firstChunkAtUtc: sample.firstChunkAtUtc,
      firstOutputAtUtc: sample.firstOutputAtUtc,
      lastOutputAtUtc: sample.lastOutputAtUtc,
      responseId: sample.responseId,
      responseModel: sample.responseModel,
    })),
    samples,
    tokenTimeline: samples.flatMap((sample) => sample.tokenTimeline ?? []),
    nativeTelemetry: samples.map((sample) => ({
      requestId: sample.requestId,
      ...(sample.nativeTelemetry ?? {}),
    })),
    hardwareTelemetry: samples.map((sample) => ({
      requestId: sample.requestId,
      ...(sample.hardwareTelemetry ?? {}),
    })),
    tokenDetails: samples.map((sample) => ({
      requestId: sample.requestId,
      tokenDetailsAvailable: sample.tokenDetailsAvailable,
      tokenIdsAvailable: sample.tokenIdsAvailable,
      logprobsAvailable: sample.logprobsAvailable,
      tokenDetailCount: sample.tokenDetailCount,
      tokenDetailSource: sample.tokenDetailSource,
      tokenIdSource: sample.tokenIdSource,
    })),
    promptTokenDetails: samples.map((sample) => ({
      requestId: sample.requestId,
      promptTokenIdsAvailable: sample.promptTokenIdsAvailable,
      promptTokenDetailCount: sample.promptTokenDetailCount,
      promptTokenIdSource: sample.promptTokenIdSource,
      promptTokenIdsSha256: sample.promptTokenIdsSha256,
      promptTokenizationSource: sample.promptTokenizationSource,
      promptTokenizerModel: sample.promptTokenizerModel,
    })),
    measurements,
  }, null, 2) + "\n")
  return artifactPath
}

async function writeRawArtifact(
  config: ServingProducerConfig,
  rawCaptures: Record<string, unknown>[],
  capturedAtUtc: string,
): Promise<string> {
  const artifactDir = config.artifactDir ?? path.join(process.cwd(), ".performance-iq", "serving-producers")
  await mkdir(artifactDir, { recursive: true })
  const safeModel = safeSlug(config.request.model)
  const rawPath = path.join(
    artifactDir,
    `${config.engine.engine}-${safeModel}-${capturedAtUtc.replace(/[:.]/g, "-")}-operator-full.json`,
  )
  await writeFile(rawPath, JSON.stringify({
    schemaVersion: "performance-iq.serving-operator-full-raw.v1",
    confidentiality: "operator-full",
    capturedAtUtc,
    engine: config.engine.engine,
    engineLabel: ENGINE_LABELS[config.engine.engine],
    requestPayload: requestPayload(config.request, config.request.stream !== false),
    captures: rawCaptures,
  }, null, 2) + "\n")
  return rawPath
}

async function writeManifestArtifact(
  config: ServingProducerConfig,
  manifest: ProducerRunManifest,
  capturedAtUtc: string,
): Promise<string> {
  const artifactDir = config.artifactDir ?? path.join(process.cwd(), ".performance-iq", "serving-producers")
  await mkdir(artifactDir, { recursive: true })
  const safeModel = safeSlug(config.request.model)
  const manifestPath = path.join(
    artifactDir,
    `${config.engine.engine}-${safeModel}-${capturedAtUtc.replace(/[:.]/g, "-")}-manifest.json`,
  )
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n")
  return manifestPath
}

export async function runServingProducer(config: ServingProducerConfig): Promise<ServingProducerResult> {
  const now = config.now ?? (() => new Date())
  const capturedAtUtc = nowIso(now)
  const repetitions = Math.max(1, config.request.repetitions ?? 1)
  const campaignId = config.campaign?.campaignId ?? `serving-${config.engine.engine}-${config.request.model}`
  const runId = config.campaign?.runId ?? `${campaignId}-${capturedAtUtc.replace(/[:.]/g, "-")}`
  const samples: ServingRequestSample[] = []
  const rawCaptures: Record<string, unknown>[] = []
  for (let index = 0; index < repetitions; index += 1) {
    const sample = await sendChatCompletion(config, index, campaignId, runId)
    if (sample.rawCapture) rawCaptures.push(sample.rawCapture)
    const { rawCapture: _rawCapture, ...sanitizedSample } = sample
    samples.push(sanitizedSample)
  }

  const measurements = buildMeasurements(config, samples, capturedAtUtc)
  const rawArtifactPath = await writeRawArtifact(config, rawCaptures, capturedAtUtc)
  measurements.push(...producerCoverageRows(config, samples, measurements[0], rawArtifactPath, capturedAtUtc))
  const artifactPath = await writeSummaryArtifact(config, samples, measurements, capturedAtUtc, rawArtifactPath)
  const engineLabel = ENGINE_LABELS[config.engine.engine]
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
    artifacts: [
      { kind: "normalized-summary", path: artifactPath },
      { kind: "operator-full-serving-raw", path: rawArtifactPath },
    ],
    measurements,
    platform: {
      decisionBriefPath: "performance-iq://serving-producer",
      requestTraceIds: samples.map((sample) => sample.requestId),
    },
    methodology: [
      `${engineLabel} producer sent ${repetitions} OpenAI-compatible chat completion request(s)`,
      `to ${normalizeBaseUrl(config.engine.baseUrl)}${config.engine.requestPath ?? "/v1/chat/completions"}`,
      `for model ${config.request.model}.`,
      "Each request includes x-performance-iq-* trace headers.",
      "Metrics are derived from client-side streaming SSE timings, response usage fields, response token logprobs/IDs when exposed, native telemetry when exposed, and DCGM hardware metrics when a hardware metrics endpoint is configured.",
    ].join(" "),
    limitations: [
      "Serving producer captures client stream timing, request-path, usage, latency, token logprobs/IDs when exposed, and provenance; hardware-level DCGM counters require a reachable DCGM/Prometheus metrics endpoint or configured hardware telemetry.",
      ...(samples.some((sample) => !sample.ok) ? ["One or more serving requests failed; see normalized-summary artifact for per-request errors."] : []),
    ],
  }
  if (config.engine.frameworkVersion) {
    runInput.runtime.imageTag = runInput.runtime.imageTag ?? `${config.engine.engine}:${config.engine.frameworkVersion}`
  }
  const manifest = await buildManifest(runInput)
  const manifestPath = await writeManifestArtifact(config, manifest, capturedAtUtc)
  const submission = config.performanceIq && config.submit !== false
    ? await config.performanceIq.submitRun(runInput, { idempotencyKey: manifest.campaign.runId })
    : undefined

  return {
    engine: config.engine.engine,
    manifest,
    runInput,
    artifactPath,
    manifestPath,
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
