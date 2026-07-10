import { createHash } from "node:crypto"
import { spawnSync } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { mkdir, writeFile } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

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
  metricsUrlAutoConfigured?: boolean
  nativeJsonMetricsUrl?: string
  nativeJsonMetricsUrlAutoConfigured?: boolean
  nativePerfMetricsUrl?: string
  nativePerfMetricsUrlAutoConfigured?: boolean
  perfMetricsUrl?: string
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
  tokenizerModel?: string
  tokenizerPythonBin?: string
  tokenizerResolveTimeoutSeconds?: number
  tokenizerTrustRemoteCode?: boolean
  trustRemoteCode?: boolean
  resolveTokenIdsWithTokenizer?: boolean
  frameworkVersion?: string
  runtimeBackend?: string
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
  hardwareInventoryPath?: string
  hardwareInventorySha256?: string
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
  resolveTokenIdsWithTokenizer?: boolean
}

export interface ServingProducerConfig {
  engine: ServingEngineConfig
  request: ServingRequestConfig
  performanceIq?: PerformanceIQ
  submit?: boolean
  artifactDir?: string
  eventLogPath?: string
  writeEventLog?: boolean
  eventTopic?: string
  /** @deprecated Use eventTopic. The stable SDK does not publish to Kafka. */
  kafkaTopic?: string
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
  trtllmPerfKvAllocatedBlocks?: number | null
  trtllmPerfKvNewBlocks?: number | null
  trtllmPerfKvReusedBlocks?: number | null
  trtllmPerfKvMissedBlocks?: number | null
  trtllmPerfRecordCount?: number | null
  trtllmPerfRequestIdSha256?: string | null
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
  smActivePct?: number | null
  dramActivePct?: number | null
  tensorActivePct?: number | null
  fp64ActivePct?: number | null
  fp32ActivePct?: number | null
  fp16ActivePct?: number | null
  pcieTxThroughputKiBps?: number | null
  pcieRxThroughputKiBps?: number | null
  pcieTxBytesDelta?: number | null
  pcieRxBytesDelta?: number | null
  pcieReplayDelta?: number | null
  nvlinkTxBytesDelta?: number | null
  nvlinkRxBytesDelta?: number | null
  nvlinkBandwidthTotalMBps?: number | null
  encoderUtilizationPct?: number | null
  decoderUtilizationPct?: number | null
  gpuTemperatureC?: number | null
  smClockMHz?: number | null
  memoryClockMHz?: number | null
  fbUsedMiB?: number | null
  fbFreeMiB?: number | null
  energyJoules?: number | null
  xidErrors?: number | null
  xidErrorsDelta?: number | null
  eccSbeVolatileTotalDelta?: number | null
  eccDbeVolatileTotalDelta?: number | null
  powerViolationTimeUsDelta?: number | null
  thermalViolationTimeUsDelta?: number | null
  hardwareRawMetricCount?: number | null
  hardwareRawMetricNamesSha256?: string | null
  tokenDetailsAvailable?: boolean
  tokenIdsAvailable?: boolean
  logprobsAvailable?: boolean
  tokenDetailCount?: number
  tokenDetailSource?: string
  tokenIdSource?: string | null
  tokenDetailsCapabilityStatus?: string | null
  tokenDetailsUnsupportedReason?: string | null
  tokenizerModel?: string | null
  tokenizerPythonBinSha256?: string | null
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
  runtimeBackend?: unknown
  modelRevision?: unknown
  imageTag?: unknown
  imageDigest?: unknown
  serverArgsSha256?: string | null
  processId?: unknown
  containerId?: unknown
  podName?: unknown
  nodeName?: unknown
  hostName?: unknown
  hardwareInventorySha256?: unknown
  tokenTimeline?: ServingTokenTimelineChunk[]
  metricSnapshots?: ServingMetricSnapshot[]
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
  tokenizerModel?: string | null
  tokenizerPythonBinSha256?: string | null
}

export interface ServingMetricSnapshot {
  requestId: string
  metricSource: string
  snapshotPhase: "before" | "after"
  metricName: string
  metricLabelsSha256: string
  metricValue: number
  metricSampleOrdinal: number
  capturedAtUtc?: string
  rawMetricTextSha256?: string | null
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
  eventLogPath?: string
  samples: ServingRequestSample[]
  measurements: Record<string, unknown>[]
  submission?: unknown
}

const ENGINE_LABELS: Record<ServingEngineId, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  "tensorrt-llm": "TensorRT-LLM",
}

const DEFAULT_IMAGE_DIGEST = `sha256:${sha256Text("performance-iq-sdk:uncontainerized-local:v1")}`
const DEFAULT_PRODUCER_COMMIT = `uncommitted-worktree:${createHash("sha256")
  .update(readFileSync(fileURLToPath(import.meta.url)))
  .digest("hex")}`
const SERVING_EVENT_SCHEMA_VERSION = "performance-iq.serving-telemetry-event.v1"
const SERVING_EVENT_DEFAULT_TOPIC = "performance-iq.serving.telemetry.v1"
const EXTERNAL_TOKENIZER_CACHE = new Map<string, number | null>()
const EXTERNAL_PROMPT_TOKENIZER_CACHE = new Map<string, {
  tokenIds: number[]
  tokenTexts: Array<string | null>
  mode: string
} | null>()
const EXTERNAL_TEXT_TOKENIZER_CACHE = new Map<string, {
  tokenIds: number[]
  tokenTexts: Array<string | null>
  mode: string
} | null>()

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "")
}

function safeSlug(value: string): string {
  return value.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-|-$/g, "") || "value"
}

function artifactDir(config: ServingProducerConfig): string {
  return config.artifactDir ?? path.join(process.cwd(), ".performance-iq", "serving-producers")
}

function artifactBaseName(config: ServingProducerConfig, capturedAtUtc: string): string {
  return `${config.engine.engine}-${safeSlug(config.request.model)}-${capturedAtUtc.replace(/[:.]/g, "-")}`
}

function summaryArtifactPath(config: ServingProducerConfig, capturedAtUtc: string): string {
  return path.join(artifactDir(config), `${artifactBaseName(config, capturedAtUtc)}.json`)
}

function rawArtifactPath(config: ServingProducerConfig, capturedAtUtc: string): string {
  return path.join(artifactDir(config), `${artifactBaseName(config, capturedAtUtc)}-operator-full.json`)
}

function manifestArtifactPath(config: ServingProducerConfig, capturedAtUtc: string): string {
  return path.join(artifactDir(config), `${artifactBaseName(config, capturedAtUtc)}-manifest.json`)
}

function eventLogEnabled(config: ServingProducerConfig): boolean {
  return config.writeEventLog !== false
}

function servingEventLogPath(config: ServingProducerConfig, capturedAtUtc: string): string | null {
  if (!eventLogEnabled(config)) return null
  const configuredPath = config.eventLogPath?.trim()
  if (configuredPath) return configuredPath
  return path.join(artifactDir(config), `${artifactBaseName(config, capturedAtUtc)}-events.jsonl`)
}

function eventTopic(config: ServingProducerConfig): string {
  return config.eventTopic?.trim() || config.kafkaTopic?.trim() || SERVING_EVENT_DEFAULT_TOPIC
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

function stableJson(value: unknown): string {
  return JSON.stringify(stableJsonValue(value)) ?? "null"
}

function stableJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => stableJsonValue(item) ?? null)
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value as Record<string, unknown>)
        .sort()
        .flatMap((key) => {
          const normalized = stableJsonValue((value as Record<string, unknown>)[key])
          return normalized === undefined ? [] : [[key, normalized]]
        }),
    )
  }
  return value
}

function sha256StableJson(value: unknown): string {
  return sha256Text(stableJson(value))
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

function tokenizerResolutionRequested(config: ServingProducerConfig): boolean {
  return Boolean(config.request.resolveTokenIdsWithTokenizer || config.engine.resolveTokenIdsWithTokenizer)
}

function tokenizerModel(config: ServingProducerConfig): string | null {
  const engineExtras = config.engine as unknown as Record<string, unknown>
  const explicit = config.request.tokenizerModel ?? engineExtras.tokenizerModel ?? engineExtras.tokenizer_model
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim()
  return tokenizerResolutionRequested(config) ? config.request.model : null
}

function tokenizerPythonBin(config: ServingProducerConfig): string | null {
  const engineExtras = config.engine as unknown as Record<string, unknown>
  const value = engineExtras.tokenizerPythonBin ?? engineExtras.tokenizer_python_bin
  return typeof value === "string" && value.trim() ? value.trim() : null
}

function tokenizerProvenance(config: ServingProducerConfig): Record<string, string | null> {
  const pythonBin = tokenizerPythonBin(config)
  return {
    tokenizerModel: tokenizerModel(config),
    tokenizerPythonBinSha256: pythonBin ? createHash("sha256").update(readFileSync(pythonBin)).digest("hex") : null,
  }
}

function runtimeProvenance(config: ServingProducerConfig, telemetry: Record<string, unknown> = {}): Record<string, unknown> {
  const engine = config.engine as unknown as Record<string, unknown>
  const serverArgs = telemetry.serverArgs ?? engine.serverArgs
  return {
    engineVersion: telemetry.engineVersion ?? engine.frameworkVersion,
    runtimeBackend: telemetry.runtimeBackend ?? engine.runtimeBackend,
    modelRevision: telemetry.modelRevision ?? engine.modelRevision,
    imageTag: firstDefined(engine, ["imageTag", "containerImageTag"]),
    imageDigest: firstDefined(engine, ["imageDigest", "containerImageDigest"]),
    serverArgsSha256: sha256OptionalJson(serverArgs),
    processId: firstDefined(engine, ["processId", "pid"]) ?? nestedValue(engine, "process", ["pid", "processId"]),
    containerId: firstDefined(engine, ["containerId"]) ?? nestedValue(engine, "container", ["id", "containerId"]),
    podName: firstDefined(engine, ["podName"]) ?? nestedValue(engine, "container", ["podName"]),
    nodeName: firstDefined(engine, ["nodeName"]) ?? nestedValue(engine, "container", ["nodeName"]),
    hostName: firstDefined(engine, ["hostName", "hostname"]) ?? nestedValue(engine, "process", ["hostName", "hostname"]),
    hardwareInventorySha256: engine.hardwareInventorySha256,
    ...tokenizerProvenance(config),
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
    typeof row.successCount === "number" &&
    row.successCount > 0 &&
    row.runtimeProvenanceAvailableCount === row.successCount
      ? row.runtimeProvenanceAvailableCount
      : null,
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
      row.avgSmActivePct,
      row.avgDramActivePct,
      row.avgTensorActivePct,
      row.avgPcieTxBytesDelta,
      row.avgPcieRxBytesDelta,
      row.avgNvlinkTxBytesDelta,
      row.avgNvlinkRxBytesDelta,
      row.avgEncoderUtilizationPct,
      row.avgDecoderUtilizationPct,
      row.avgXidErrorsDelta,
      row.avgEccDbeVolatileTotalDelta,
      row.hardwareRawMetricCountMin,
      row.totalEnergyJoules,
    )
  }
  if (row.tokenDetailsRequired) {
    required.push(row.logprobsAvailableCount === row.successCount ? row.logprobsAvailableCount : null)
  }
  if (row.promptTokenDetailsRequired) {
    required.push(row.promptTokenIdsAvailableCount === row.successCount ? row.promptTokenIdsAvailableCount : null)
  }
  if (row.eventLogRequired) {
    required.push(row.eventLogWritten ? 1 : null)
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

function defaultNativePerfMetricsUrl(engine: ServingEngineConfig): string | null {
  return engine.engine === "tensorrt-llm" ? `${normalizeBaseUrl(engine.baseUrl)}/perf_metrics` : null
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

function nativePerfMetricsUrl(config: ServingProducerConfig): string | null {
  if (config.engine.nativePerfMetricsUrl?.trim()) return config.engine.nativePerfMetricsUrl.trim()
  if (config.engine.perfMetricsUrl?.trim()) return config.engine.perfMetricsUrl.trim()
  return config.engine.collectNativeMetrics ? defaultNativePerfMetricsUrl(config.engine) : null
}

function nativeTelemetryExpected(config: ServingProducerConfig): boolean {
  if (config.engine.requireNativeTelemetry) return true
  if (config.engine.metricsUrlAutoConfigured || config.engine.nativeJsonMetricsUrlAutoConfigured || config.engine.nativePerfMetricsUrlAutoConfigured) return false
  return Boolean(metricsUrl(config) || nativeJsonMetricsUrl(config) || nativePerfMetricsUrl(config))
}

const DCGM_BLANK_VALUE_THRESHOLD = 9e18

function isDcgmBlankValue(metricName: string, value: number): boolean {
  return metricName.startsWith("DCGM_") && Math.abs(value) >= DCGM_BLANK_VALUE_THRESHOLD
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
    if (isDcgmBlankValue(metricName, value)) continue
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

function parsePrometheusMetricSeries(text: string): Array<Record<string, unknown>> {
  const series: Array<Record<string, unknown>> = []
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)/)
    if (!match) continue
    const value = Number(match[3])
    if (!Number.isFinite(value)) continue
    const metricName = match[1]
    if (isDcgmBlankValue(metricName, value)) continue
    const labels = Object.fromEntries(Array.from((match[2] ?? "").matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"/g), (item) => [item[1], item[2]]))
    series.push({
      metricName,
      labels,
      labelsSha256: sha256StableJson(labels),
      value,
    })
  }
  return series
}

function parseInvalidDcgmMetricSeries(text: string): Array<Record<string, unknown>> {
  const invalid: Array<Record<string, unknown>> = []
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)/)
    if (!match) continue
    const value = Number(match[3])
    const metricName = match[1]
    if (!Number.isFinite(value) || !isDcgmBlankValue(metricName, value)) continue
    const labels = Object.fromEntries(Array.from((match[2] ?? "").matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"/g), (item) => [item[1], item[2]]))
    invalid.push({
      metricName,
      labels,
      labelsSha256: sha256StableJson(labels),
      rawValue: match[3],
      invalidReason: "dcgm-blank-sentinel",
    })
  }
  return invalid
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

function parseNativeJsonMetricSeries(text: string): Array<Record<string, unknown>> {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return []
  }
  const records = Array.isArray(parsed) ? parsed : [parsed]
  return records.flatMap((record, recordIndex) => {
    if (!record || typeof record !== "object" || Array.isArray(record)) return []
    const labels = { record: String(recordIndex) }
    const labelsSha256 = sha256StableJson(labels)
    return Object.entries(flattenNumericJsonMetrics(record)).map(([metricName, value]) => ({
      metricName,
      labels,
      labelsSha256,
      value,
    }))
  })
}

function parseNativePerfMetadata(text: string): Record<string, unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return {}
  }
  const records = (Array.isArray(parsed) ? parsed : [parsed]).filter(
    (record): record is Record<string, unknown> => Boolean(record) && typeof record === "object" && !Array.isArray(record),
  )
  if (!records.length) return {}
  const requestId = records[records.length - 1].request_id
  return {
    recordCount: records.length,
    ...(requestId != null ? { requestIdSha256: sha256Text(String(requestId)) } : {}),
  }
}

async function readNativeMetrics(config: ServingProducerConfig): Promise<Record<string, unknown>> {
  const url = metricsUrl(config)
  const jsonUrl = nativeJsonMetricsUrl(config)
  const perfUrl = nativePerfMetricsUrl(config)
  if (!url && !jsonUrl && !perfUrl) return { available: false, source: "metrics-url-not-configured" }
  const fetchImpl = config.fetchImpl ?? fetch
  const metrics: Record<string, number> = {}
  const jsonMetrics: Record<string, number> = {}
  const perfJsonMetrics: Record<string, number> = {}
  const metricSeries: Array<Record<string, unknown>> = []
  const jsonMetricSeries: Array<Record<string, unknown>> = []
  const perfMetricSeries: Array<Record<string, unknown>> = []
  const perfMetadata: Record<string, unknown> = {}
  const rawMetricsText: Record<string, string> = {}
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
        metricSeries.push(...parsePrometheusMetricSeries(text))
        Object.assign(jsonMetrics, parseNativeJsonMetrics(text))
        jsonMetricSeries.push(...parseNativeJsonMetricSeries(text))
        rawMetricsText.metrics = text
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
        const text = await response.text()
        const parsed = parseNativeJsonMetrics(text)
        if (Object.keys(parsed).length) {
          Object.assign(jsonMetrics, parsed)
          jsonMetricSeries.push(...parseNativeJsonMetricSeries(text))
          sources.push("native-json-snapshot")
        } else if (!Object.keys(metrics).length) {
          errors.push({ source: "native-json-empty", nativeJsonMetricsUrl: jsonUrl })
        }
        rawMetricsText.nativeJsonMetrics = text
      }
    } catch (error) {
      errors.push({
        source: "native-json-unavailable",
        nativeJsonMetricsUrl: jsonUrl,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
  if (perfUrl && perfUrl !== url && perfUrl !== jsonUrl) {
    try {
      const response = await fetchImpl(perfUrl, {
        method: "GET",
        headers: {
          accept: "application/json, text/plain",
          ...(config.engine.apiKey ? { authorization: `Bearer ${config.engine.apiKey}` } : {}),
        },
      })
      if (!response.ok) {
        errors.push({ source: "native-perf-json-unavailable", nativePerfMetricsUrl: perfUrl, status: response.status })
      } else {
        const text = await response.text()
        const parsed = parseNativeJsonMetrics(text)
        if (Object.keys(parsed).length) {
          Object.assign(perfJsonMetrics, parsed)
          perfMetricSeries.push(...parseNativeJsonMetricSeries(text))
          Object.assign(perfMetadata, parseNativePerfMetadata(text))
          sources.push("native-perf-json-snapshot")
        } else {
          errors.push({ source: "native-perf-json-empty", nativePerfMetricsUrl: perfUrl })
        }
        rawMetricsText.nativePerfMetrics = text
      }
    } catch (error) {
      errors.push({
        source: "native-perf-json-unavailable",
        nativePerfMetricsUrl: perfUrl,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
  if (!Object.keys(metrics).length && !Object.keys(jsonMetrics).length && !Object.keys(perfJsonMetrics).length) {
    return {
      available: false,
      source: typeof errors[0]?.source === "string" ? errors[0].source : "native-metrics-empty",
      metricsUrl: url,
      ...(jsonUrl ? { nativeJsonMetricsUrl: jsonUrl } : {}),
      ...(perfUrl ? { nativePerfMetricsUrl: perfUrl } : {}),
      ...(errors.length ? { errors } : {}),
    }
  }
  return {
    available: true,
    source: Array.from(new Set(sources)).join("+") || "native-metrics-snapshot",
    metricsUrl: url,
    ...(jsonUrl ? { nativeJsonMetricsUrl: jsonUrl } : {}),
    ...(perfUrl ? { nativePerfMetricsUrl: perfUrl } : {}),
    metrics,
    ...(metricSeries.length ? { metricSeries } : {}),
    ...(Object.keys(jsonMetrics).length ? { jsonMetrics } : {}),
    ...(jsonMetricSeries.length ? { jsonMetricSeries } : {}),
    ...(Object.keys(perfJsonMetrics).length ? { perfJsonMetrics } : {}),
    ...(perfMetricSeries.length ? { perfMetricSeries } : {}),
    ...(Object.keys(perfMetadata).length ? { perfMetadata } : {}),
    ...(Object.keys(rawMetricsText).length ? { rawMetricsText } : {}),
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
    const text = await response.text()
    const metrics = Object.fromEntries(
      Object.entries(parsePrometheusMetrics(text)).filter(([key]) => key.startsWith("DCGM_")),
    )
    const metricSeries = parsePrometheusMetricSeries(text).filter((row) => String(row.metricName).startsWith("DCGM_"))
    const invalidMetricSeries = parseInvalidDcgmMetricSeries(text)
    if (!Object.keys(metrics).length) return { available: false, source: "dcgm-prometheus-empty", metricsUrl: url }
    return {
      available: true,
      source: "dcgm-prometheus-snapshot",
      metricsUrl: url,
      metrics,
      metricSeries,
      ...(invalidMetricSeries.length ? { invalidMetricSeries, invalidMetricCount: invalidMetricSeries.length } : {}),
      rawMetricsText: text,
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

function metricSnapshotRows(
  requestId: string,
  source: string,
  snapshotPhase: "before" | "after",
  snapshot: Record<string, unknown>,
): ServingMetricSnapshot[] {
  const rows: ServingMetricSnapshot[] = []
  const rawMetricText = snapshot.rawMetricsText
  const rawTextByKind = rawMetricText && typeof rawMetricText === "object" && !Array.isArray(rawMetricText)
    ? rawMetricText as Record<string, unknown>
    : {}
  for (const [seriesKey, metricSource, rawTextKey] of [
    ["metricSeries", `${source}-prometheus`, "metrics"],
    ["jsonMetricSeries", `${source}-json`, "nativeJsonMetrics"],
    ["perfMetricSeries", `${source}-perf-json`, "nativePerfMetrics"],
  ] as const) {
    const series = snapshot[seriesKey]
    if (!Array.isArray(series)) continue
    const rawText = typeof rawMetricText === "string" && rawTextKey === "metrics"
      ? rawMetricText
      : rawTextByKind[rawTextKey]
    const rawMetricTextSha256 = typeof rawText === "string" ? sha256Text(rawText) : null
    series.forEach((item, metricSampleOrdinal) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return
      const row = item as Record<string, unknown>
      if (typeof row.metricName !== "string" || typeof row.value !== "number" || !Number.isFinite(row.value)) return
      rows.push({
        requestId,
        metricSource,
        snapshotPhase,
        metricName: row.metricName,
        metricLabelsSha256: typeof row.labelsSha256 === "string" ? row.labelsSha256 : sha256StableJson(row.labels ?? {}),
        metricValue: row.value,
        metricSampleOrdinal,
        capturedAtUtc: typeof snapshot.capturedAtUtc === "string" ? snapshot.capturedAtUtc : undefined,
        rawMetricTextSha256,
      })
    })
  }
  return rows
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

function metricPercentAverage(metrics: Record<string, number>, candidates: string[]): number | null {
  const value = metricAverage(metrics, candidates)
  return value == null ? null : value * 100
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

function elapsedMs(start: number | null, end: number | null): number | null {
  if (start == null || end == null) return null
  const elapsed = (end - start) * 1000
  return elapsed >= 0 ? elapsed : null
}

function counterDelta(before: Record<string, number>, after: Record<string, number>, candidates: string[]): number | null {
  let beforeValue = metricValue(before, candidates)
  const afterValue = metricValue(after, candidates)
  if (afterValue == null) return null
  if (beforeValue == null) beforeValue = 0
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
  const afterPerfMetrics = after.perfJsonMetrics && typeof after.perfJsonMetrics === "object" && !Array.isArray(after.perfJsonMetrics)
    ? after.perfJsonMetrics as Record<string, number>
    : {}
  const afterPerfMetadata = after.perfMetadata && typeof after.perfMetadata === "object" && !Array.isArray(after.perfMetadata)
    ? after.perfMetadata as Record<string, unknown>
    : {}
  const perfArrivalS = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.timing_metrics.arrival_time"])
  const perfFirstScheduledS = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.timing_metrics.first_scheduled_time"])
  const perfFirstTokenS = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.timing_metrics.first_token_time"])
  const perfLastTokenS = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.timing_metrics.last_token_time"])
  const perfQueueWaitMs = elapsedMs(perfArrivalS, perfFirstScheduledS)
  const perfPrefillMs = elapsedMs(perfFirstScheduledS, perfFirstTokenS)
  const perfTtftMs = elapsedMs(perfArrivalS, perfFirstTokenS)
  const perfDecodeMs = elapsedMs(perfFirstTokenS, perfLastTokenS)
  const perfE2eMs = elapsedMs(perfArrivalS, perfLastTokenS)
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
  const nativeE2eLatencyMs = firstNumber(histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:e2e_request_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang_e2e_request_latency_seconds",
    "trtllm:e2e_request_latency_seconds",
    "trtllm_e2e_request_latency_seconds",
  ]), perfE2eMs)
  const queueWaitMs = firstNumber(histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_queue_time_seconds",
    "sglang:queue_time_seconds",
    "sglang:request_queue_time_seconds",
    "sglang_request_queue_time_seconds",
    "trtllm:request_queue_time_seconds",
    "trtllm_request_queue_time_seconds",
    "trtllm_queue_time_seconds",
    "trtllm:queue_time_seconds",
  ]), perfQueueWaitMs, jsonMetricValue(afterJsonMetrics, ["avg.newActiveRequestsQueueLatencyMS", "latest.newActiveRequestsQueueLatencyMS"]))
  let prefillMs = firstNumber(histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_prefill_time_seconds",
    "sglang:per_stage_req_latency_seconds{stage=prefill_forward}",
    "sglang:per_stage_req_latency_seconds{mode=prefill_forward}",
    "sglang:request_prefill_time_seconds",
    "sglang_request_prefill_time_seconds",
    "trtllm:request_prefill_time_seconds",
    "trtllm_request_prefill_time_seconds",
    "trtllm:context_time_seconds",
    "trtllm_context_time_seconds",
  ]), perfPrefillMs)
  const nativeTtftMs = firstNumber(histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:time_to_first_token_seconds",
    "sglang:time_to_first_token_seconds",
    "sglang_time_to_first_token_seconds",
    "trtllm:time_to_first_token_seconds",
    "trtllm_time_to_first_token_seconds",
  ]), perfTtftMs)
  if (prefillMs == null && nativeTtftMs != null && queueWaitMs != null) {
    const derivedPrefillMs = nativeTtftMs - queueWaitMs
    if (derivedPrefillMs >= 0) prefillMs = derivedPrefillMs
  }
  let decodeMs = firstNumber(histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:request_decode_time_seconds",
    "sglang:per_stage_req_latency_seconds{stage=decode_forward}",
    "sglang:per_stage_req_latency_seconds{mode=decode}",
    "sglang:request_decode_time_seconds",
    "sglang_request_decode_time_seconds",
    "trtllm:request_decode_time_seconds",
    "trtllm_request_decode_time_seconds",
    "trtllm:generation_time_seconds",
    "trtllm_generation_time_seconds",
  ]), perfDecodeMs)
  if (decodeMs == null && nativeE2eLatencyMs != null && nativeTtftMs != null) {
    const derivedDecodeMs = nativeE2eLatencyMs - nativeTtftMs
    if (derivedDecodeMs >= 0) decodeMs = derivedDecodeMs
  }
  if (decodeMs == null && nativeE2eLatencyMs != null && queueWaitMs != null && prefillMs != null) {
    const derivedDecodeMs = nativeE2eLatencyMs - queueWaitMs - prefillMs
    if (derivedDecodeMs >= 0) decodeMs = derivedDecodeMs
  }
  const trtllmKvUsedBlocks = jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.usedNumBlocks", "max.kvCacheStats.usedNumBlocks"])
  const trtllmKvMaxBlocks = jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.maxNumBlocks", "max.kvCacheStats.maxNumBlocks"])
  const trtllmKvUsage = trtllmKvUsedBlocks != null && trtllmKvMaxBlocks != null && trtllmKvMaxBlocks > 0
    ? trtllmKvUsedBlocks / trtllmKvMaxBlocks
    : null
  const perfKvReusedBlocks = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.kv_cache_metrics.num_reused_blocks"])
  const perfKvMissedBlocks = jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.kv_cache_metrics.num_missed_blocks"])
  const perfCacheHitRate = perfKvReusedBlocks != null && perfKvMissedBlocks != null && perfKvReusedBlocks + perfKvMissedBlocks > 0
    ? perfKvReusedBlocks / (perfKvReusedBlocks + perfKvMissedBlocks)
    : null
  const prefixCacheHitRate = prefixHits != null && prefixQueries != null && prefixQueries > 0 ? prefixHits / prefixQueries : null
  const nativeInterTokenLatencyMs = histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
    "vllm:inter_token_latency_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang_inter_token_latency_seconds",
    "trtllm:inter_token_latency_seconds",
    "trtllm_inter_token_latency_seconds",
  ])
  const nativeTpotMs = firstNumber(
    histogramDeltaMeanMs(beforeMetrics, afterMetrics, [
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
    nativeInterTokenLatencyMs,
  )
  const promptTokensComputedDelta = counterDelta(beforeMetrics, afterMetrics, [
    "vllm:request_prefill_kv_computed_tokens_sum",
    "sglang:request_prefill_kv_computed_tokens_sum",
    "sglang_request_prefill_kv_computed_tokens_sum",
    "sglang:uncached_prompt_tokens_total",
    "sglang_uncached_prompt_tokens_total",
    "sglang:uncached_prompt_tokens_histogram_sum",
    "sglang_uncached_prompt_tokens_histogram_sum",
    "sglang:realtime_tokens_total{mode=prefill_compute}",
    "sglang_realtime_tokens_total{mode=prefill_compute}",
    "trtllm:request_prefill_kv_computed_tokens_sum",
    "trtllm_request_prefill_kv_computed_tokens_sum",
  ])
  let promptTokensCachedDelta = counterDelta(beforeMetrics, afterMetrics, [
    "vllm:prompt_tokens_cached_total",
    "sglang:prompt_tokens_cached_total",
    "sglang_prompt_tokens_cached_total",
    "sglang:cached_tokens_total",
    "sglang_cached_tokens_total",
    "sglang:realtime_tokens_total{mode=prefill_cache}",
    "sglang_realtime_tokens_total{mode=prefill_cache}",
    "trtllm:prompt_tokens_cached_total",
    "trtllm_prompt_tokens_cached_total",
  ])
  if (promptTokensCachedDelta == null && promptTokensComputedDelta != null) promptTokensCachedDelta = 0
  const values: Record<string, number | null> = {
    nativeTtftMs,
    nativeTpotMs,
    nativeInterTokenLatencyMs,
    nativeE2eLatencyMs,
    queueWaitMs,
    prefillMs,
    decodeMs,
    runningRequests: firstNumber(
      metricValue(afterMetrics, ["vllm:num_requests_running", "sglang:num_running_reqs", "sglang_num_running_reqs", "trtllm:num_requests_running", "trtllm_num_requests_running", "trtllm:num_active_requests", "trtllm_num_active_requests"]),
      jsonMetricValue(afterJsonMetrics, ["latest.numActiveRequests", "avg.numActiveRequests", "max.numActiveRequests"]),
    ),
    waitingRequests: firstNumber(
      metricValue(afterMetrics, ["vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang_num_queue_reqs", "trtllm:num_requests_waiting", "trtllm_num_requests_waiting", "trtllm:num_queued_requests", "trtllm_num_queued_requests"]),
      jsonMetricValue(afterJsonMetrics, ["latest.numQueuedRequests", "avg.numQueuedRequests", "max.numQueuedRequests"]),
    ),
    kvCacheUsagePct: firstNumber(
      metricValue(afterMetrics, ["vllm:kv_cache_usage_perc", "sglang:token_usage", "sglang_token_usage", "trtllm:kv_cache_usage_perc", "trtllm_kv_cache_usage_perc", "trtllm:kv_cache_utilization", "trtllm_kv_cache_utilization"]),
      trtllmKvUsage,
    ),
    trtllmIterationLatencyMs: jsonMetricValue(afterJsonMetrics, ["avg.iterLatencyMS", "latest.iterLatencyMS"]),
    trtllmGpuMemoryBytes: jsonMetricValue(afterJsonMetrics, ["latest.gpuMemUsage", "max.gpuMemUsage"]),
    trtllmKvCacheUsedBlocks: trtllmKvUsedBlocks,
    trtllmKvCacheMaxBlocks: trtllmKvMaxBlocks,
    trtllmPerfKvAllocatedBlocks: jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.kv_cache_metrics.num_total_allocated_blocks"]),
    trtllmPerfKvNewBlocks: jsonMetricValue(afterPerfMetrics, ["latest.perf_metrics.kv_cache_metrics.num_new_allocated_blocks"]),
    trtllmPerfKvReusedBlocks: perfKvReusedBlocks,
    trtllmPerfKvMissedBlocks: perfKvMissedBlocks,
    trtllmPerfRecordCount: numberFrom(afterPerfMetadata.recordCount),
    prefixCacheQueriesDelta: prefixQueries,
    prefixCacheHitsDelta: prefixHits,
    cacheHitRate: firstNumber(
      prefixCacheHitRate,
      metricValue(afterMetrics, ["sglang:cache_hit_rate", "sglang_cache_hit_rate", "trtllm:kv_cache_hit_rate", "trtllm_kv_cache_hit_rate"]),
      jsonMetricValue(afterJsonMetrics, ["latest.kvCacheStats.cacheHitRate", "avg.kvCacheStats.cacheHitRate"]),
      perfCacheHitRate,
    ),
    promptTokensCachedDelta,
    promptTokensComputedDelta,
  }
  const availableValues = Object.fromEntries(Object.entries(values).filter(([, value]) => value != null))
  const deltaSources = [
    ...(Object.keys(beforeMetrics).length && Object.keys(afterMetrics).length ? ["prometheus-delta"] : []),
    ...(Object.keys(afterJsonMetrics).length ? ["native-json-snapshot"] : []),
    ...(Object.keys(afterPerfMetrics).length ? ["native-perf-json-snapshot"] : []),
  ]
  return {
    available: Object.keys(availableValues).length > 0,
    source: deltaSources.join("+") || "native-metrics-delta",
    metricsUrl: after.metricsUrl ?? before.metricsUrl,
    nativeJsonMetricsUrl: after.nativeJsonMetricsUrl ?? before.nativeJsonMetricsUrl,
    nativePerfMetricsUrl: after.nativePerfMetricsUrl ?? before.nativePerfMetricsUrl,
    beforeCapturedAtUtc: before.capturedAtUtc,
    afterCapturedAtUtc: after.capturedAtUtc,
    ...(typeof afterPerfMetadata.requestIdSha256 === "string" ? { trtllmPerfRequestIdSha256: afterPerfMetadata.requestIdSha256 } : {}),
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
  const dcgmMetricNames = Object.keys(afterMetrics)
    .filter((name) => name.startsWith("DCGM_") && !name.endsWith("__sample_count"))
    .sort()
  const values: Record<string, number | null> = {
    powerWatts: metricValue(afterMetrics, ["DCGM_FI_DEV_POWER_USAGE"]),
    powerWattsPerGpu: metricAverage(afterMetrics, ["DCGM_FI_DEV_POWER_USAGE"]),
    gpuUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE"]),
    memoryCopyUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_PROF_DRAM_ACTIVE"]),
    smActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_SM_ACTIVE"]),
    dramActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_DRAM_ACTIVE"]),
    tensorActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"]),
    fp64ActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_PIPE_FP64_ACTIVE"]),
    fp32ActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_PIPE_FP32_ACTIVE"]),
    fp16ActivePct: metricPercentAverage(afterMetrics, ["DCGM_FI_PROF_PIPE_FP16_ACTIVE"]),
    pcieTxThroughputKiBps: metricAverage(afterMetrics, ["DCGM_FI_DEV_PCIE_TX_THROUGHPUT"]),
    pcieRxThroughputKiBps: metricAverage(afterMetrics, ["DCGM_FI_DEV_PCIE_RX_THROUGHPUT"]),
    pcieTxBytesDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_PROF_PCIE_TX_BYTES"]),
    pcieRxBytesDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_PROF_PCIE_RX_BYTES"]),
    pcieReplayDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_PCIE_REPLAY_COUNTER"]),
    nvlinkTxBytesDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_PROF_NVLINK_TX_BYTES"]),
    nvlinkRxBytesDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_PROF_NVLINK_RX_BYTES"]),
    nvlinkBandwidthTotalMBps: metricAverage(afterMetrics, ["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"]),
    encoderUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_ENC_UTIL"]),
    decoderUtilizationPct: metricAverage(afterMetrics, ["DCGM_FI_DEV_DEC_UTIL"]),
    gpuTemperatureC: metricAverage(afterMetrics, ["DCGM_FI_DEV_GPU_TEMP"]),
    smClockMHz: metricAverage(afterMetrics, ["DCGM_FI_DEV_SM_CLOCK"]),
    memoryClockMHz: metricAverage(afterMetrics, ["DCGM_FI_DEV_MEM_CLOCK"]),
    fbUsedMiB: metricValue(afterMetrics, ["DCGM_FI_DEV_FB_USED"]),
    fbFreeMiB: metricValue(afterMetrics, ["DCGM_FI_DEV_FB_FREE"]),
    energyJoules: energyMj == null ? null : energyMj / 1000,
    xidErrors: metricValue(afterMetrics, ["DCGM_FI_DEV_XID_ERRORS"]),
    xidErrorsDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_XID_ERRORS"]),
    eccSbeVolatileTotalDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_ECC_SBE_VOL_TOTAL"]),
    eccDbeVolatileTotalDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"]),
    powerViolationTimeUsDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_POWER_VIOLATION"]),
    thermalViolationTimeUsDelta: counterDelta(beforeMetrics, afterMetrics, ["DCGM_FI_DEV_THERMAL_VIOLATION"]),
    hardwareRawMetricCount: dcgmMetricNames.length,
  }
  const availableValues = Object.fromEntries(Object.entries(values).filter(([, value]) => value != null))
  return {
    available: Object.keys(availableValues).length > 0,
    source: "dcgm-prometheus-delta",
    metricsUrl: after.metricsUrl ?? before.metricsUrl,
    beforeCapturedAtUtc: before.capturedAtUtc,
    afterCapturedAtUtc: after.capturedAtUtc,
    ...availableValues,
    ...(dcgmMetricNames.length ? { hardwareRawMetricNamesSha256: sha256Text(dcgmMetricNames.join("\n")) } : {}),
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

function enrichNativeTokenTiming(telemetry: Record<string, unknown>, outputTokenCount: number): void {
  const decodeMs = numberFrom(telemetry.decodeMs)
  if (decodeMs == null || outputTokenCount <= 0) return
  const nativeTpotMs = decodeMs / Math.max(outputTokenCount - 1, 1)
  if (numberFrom(telemetry.nativeTpotMs) == null) telemetry.nativeTpotMs = nativeTpotMs
  if (numberFrom(telemetry.nativeInterTokenLatencyMs) == null) telemetry.nativeInterTokenLatencyMs = nativeTpotMs
}

function hardwareSampleFields(telemetry: Record<string, unknown>): Pick<
  ServingRequestSample,
  | "avgPowerWatts"
  | "avgPowerWattsPerGpu"
  | "gpuUtilizationPct"
  | "memoryCopyUtilizationPct"
  | "smActivePct"
  | "dramActivePct"
  | "tensorActivePct"
  | "fp64ActivePct"
  | "fp32ActivePct"
  | "fp16ActivePct"
  | "pcieTxThroughputKiBps"
  | "pcieRxThroughputKiBps"
  | "pcieTxBytesDelta"
  | "pcieRxBytesDelta"
  | "pcieReplayDelta"
  | "nvlinkTxBytesDelta"
  | "nvlinkRxBytesDelta"
  | "nvlinkBandwidthTotalMBps"
  | "encoderUtilizationPct"
  | "decoderUtilizationPct"
  | "gpuTemperatureC"
  | "smClockMHz"
  | "memoryClockMHz"
  | "fbUsedMiB"
  | "fbFreeMiB"
  | "energyJoules"
  | "xidErrors"
  | "xidErrorsDelta"
  | "eccSbeVolatileTotalDelta"
  | "eccDbeVolatileTotalDelta"
  | "powerViolationTimeUsDelta"
  | "thermalViolationTimeUsDelta"
  | "hardwareRawMetricCount"
  | "hardwareRawMetricNamesSha256"
> {
  return {
    avgPowerWatts: numberFrom(telemetry.powerWatts),
    avgPowerWattsPerGpu: numberFrom(telemetry.powerWattsPerGpu),
    gpuUtilizationPct: numberFrom(telemetry.gpuUtilizationPct),
    memoryCopyUtilizationPct: numberFrom(telemetry.memoryCopyUtilizationPct),
    smActivePct: numberFrom(telemetry.smActivePct),
    dramActivePct: numberFrom(telemetry.dramActivePct),
    tensorActivePct: numberFrom(telemetry.tensorActivePct),
    fp64ActivePct: numberFrom(telemetry.fp64ActivePct),
    fp32ActivePct: numberFrom(telemetry.fp32ActivePct),
    fp16ActivePct: numberFrom(telemetry.fp16ActivePct),
    pcieTxThroughputKiBps: numberFrom(telemetry.pcieTxThroughputKiBps),
    pcieRxThroughputKiBps: numberFrom(telemetry.pcieRxThroughputKiBps),
    pcieTxBytesDelta: numberFrom(telemetry.pcieTxBytesDelta),
    pcieRxBytesDelta: numberFrom(telemetry.pcieRxBytesDelta),
    pcieReplayDelta: numberFrom(telemetry.pcieReplayDelta),
    nvlinkTxBytesDelta: numberFrom(telemetry.nvlinkTxBytesDelta),
    nvlinkRxBytesDelta: numberFrom(telemetry.nvlinkRxBytesDelta),
    nvlinkBandwidthTotalMBps: numberFrom(telemetry.nvlinkBandwidthTotalMBps),
    encoderUtilizationPct: numberFrom(telemetry.encoderUtilizationPct),
    decoderUtilizationPct: numberFrom(telemetry.decoderUtilizationPct),
    gpuTemperatureC: numberFrom(telemetry.gpuTemperatureC),
    smClockMHz: numberFrom(telemetry.smClockMHz),
    memoryClockMHz: numberFrom(telemetry.memoryClockMHz),
    fbUsedMiB: numberFrom(telemetry.fbUsedMiB),
    fbFreeMiB: numberFrom(telemetry.fbFreeMiB),
    energyJoules: numberFrom(telemetry.energyJoules),
    xidErrors: numberFrom(telemetry.xidErrors),
    xidErrorsDelta: numberFrom(telemetry.xidErrorsDelta),
    eccSbeVolatileTotalDelta: numberFrom(telemetry.eccSbeVolatileTotalDelta),
    eccDbeVolatileTotalDelta: numberFrom(telemetry.eccDbeVolatileTotalDelta),
    powerViolationTimeUsDelta: numberFrom(telemetry.powerViolationTimeUsDelta),
    thermalViolationTimeUsDelta: numberFrom(telemetry.thermalViolationTimeUsDelta),
    hardwareRawMetricCount: numberFrom(telemetry.hardwareRawMetricCount),
    hardwareRawMetricNamesSha256: typeof telemetry.hardwareRawMetricNamesSha256 === "string" ? telemetry.hardwareRawMetricNamesSha256 : null,
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

  const model = tokenizerModel(config)
  if (model) {
    const tokenizerResolved = externalHfTokenId(model, token, config)
    if (tokenizerResolved != null) {
      return { tokenId: tokenizerResolved, tokenIdSource: "external-hf-tokenizer" }
    }
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
      tokenDetailsCapabilityStatus: requested ? "requested-not-exposed" : "not-requested",
      tokenDetailsUnsupportedReason: null,
    }
  }
  const tokenIdSources = Array.from(new Set(
    details
      .filter((detail) => detail.tokenId != null && typeof detail.tokenIdSource === "string")
      .map((detail) => String(detail.tokenIdSource)),
  ))
  const tokenDetailSources = Array.from(new Set(details.flatMap((detail) => {
    if (typeof detail.tokenDetailSource === "string" && detail.tokenDetailSource) return [detail.tokenDetailSource]
    return detail.logprob != null ? ["response-logprobs"] : []
  })))
  return {
    tokenDetailsAvailable: true,
    tokenIdsAvailable: details.some((detail) => detail.tokenId != null),
    logprobsAvailable: details.some((detail) => detail.logprob != null),
    tokenDetailCount: details.length,
    tokenDetailSource: tokenDetailSources.length ? tokenDetailSources.join("+") : "token-ids-only",
    tokenIdSource: tokenIdSources.length ? tokenIdSources.join("+") : null,
    tokenDetailsCapabilityStatus: "available",
    tokenDetailsUnsupportedReason: null,
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

function externalTokenizerTimeoutMs(config: ServingProducerConfig): number {
  const value = (config.engine as unknown as Record<string, unknown>).tokenizerResolveTimeoutSeconds
  const seconds = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 30
  return Math.max(1, Number.isFinite(seconds) ? seconds : 30) * 1000
}

function tokenizerTrustRemoteCode(config: ServingProducerConfig): boolean {
  return Boolean(config.engine.tokenizerTrustRemoteCode || config.engine.trustRemoteCode)
}

function externalHfTokenizer(
  config: ServingProducerConfig,
  mode: "token" | "prompt" | "text",
  model: string,
  payload: unknown,
): Record<string, unknown> | null {
  const pythonBin = tokenizerPythonBin(config)
  if (!pythonBin || !existsSync(pythonBin)) return null
  const code = `
import json
import sys

mode = sys.argv[1]
model = sys.argv[2]
payload = json.loads(sys.argv[3])
trust_remote_code = sys.argv[4] == "1"

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
    raise SystemExit(0)

def coerce_token_ids(value):
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            ids.append(item)
        elif isinstance(item, str) and item.isdigit():
            ids.append(int(item))
    return ids

def token_id(token):
    if not token:
        return None
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        unknown = getattr(tokenizer, "unk_token_id", None)
        if isinstance(value, int) and value >= 0 and value != unknown:
            return value
    except Exception:
        pass
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], int):
            return encoded[0]
    except Exception:
        pass
    try:
        encoded = tokenizer(token, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(input_ids, list) and len(input_ids) == 1 and isinstance(input_ids[0], int):
            return input_ids[0]
    except Exception:
        pass
    return None

if mode == "token":
    print(json.dumps({"ok": True, "tokenId": token_id(str(payload))}, sort_keys=True))
elif mode == "prompt":
    prompt_payload = payload if isinstance(payload, dict) else {}
    token_ids = []
    tokenization_mode = "tokenizer-empty"
    messages = prompt_payload.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
        for kwargs in ({"tokenize": True, "add_generation_prompt": True}, {"tokenize": True}):
            try:
                token_ids = coerce_token_ids(tokenizer.apply_chat_template(messages, **kwargs))
                if token_ids:
                    tokenization_mode = "chat-template"
                    break
            except Exception:
                pass
    if not token_ids:
        parts = []
        for message in prompt_payload.get("messages", []) or []:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        text = "\\n".join(parts) or str(prompt_payload.get("prompt") or "")
        if text:
            try:
                token_ids = coerce_token_ids(tokenizer.encode(text, add_special_tokens=False))
                if token_ids:
                    tokenization_mode = "prompt-text"
            except Exception:
                pass
    token_texts = []
    for token_id_value in token_ids:
        try:
            value = tokenizer.convert_ids_to_tokens(token_id_value)
            if isinstance(value, str):
                token_texts.append(value)
                continue
        except Exception:
            pass
        try:
            value = tokenizer.decode([token_id_value], skip_special_tokens=False)
            token_texts.append(value if isinstance(value, str) else None)
        except Exception:
            token_texts.append(None)
    print(json.dumps({"ok": True, "tokenIds": token_ids, "tokenTexts": token_texts, "mode": tokenization_mode}, sort_keys=True))
elif mode == "text":
    text = str(payload or "")
    token_ids = []
    tokenization_mode = "output-text"
    if text:
        try:
            token_ids = coerce_token_ids(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
        if not token_ids:
            try:
                encoded = tokenizer(text, add_special_tokens=False)
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
                token_ids = coerce_token_ids(input_ids)
            except Exception:
                pass
    token_texts = []
    for token_id_value in token_ids:
        try:
            value = tokenizer.convert_ids_to_tokens(token_id_value)
            if isinstance(value, str):
                token_texts.append(value)
                continue
        except Exception:
            pass
        try:
            value = tokenizer.decode([token_id_value], skip_special_tokens=False)
            token_texts.append(value if isinstance(value, str) else None)
        except Exception:
            token_texts.append(None)
    print(json.dumps({"ok": True, "tokenIds": token_ids, "tokenTexts": token_texts, "mode": tokenization_mode}, sort_keys=True))
else:
    print(json.dumps({"ok": False, "error": "unknown mode"}, sort_keys=True))
`.trim()
  const result = spawnSync(pythonBin, [
    "-c",
    code,
    mode,
    model,
    JSON.stringify(payload),
    tokenizerTrustRemoteCode(config) ? "1" : "0",
  ], {
    encoding: "utf8",
    timeout: externalTokenizerTimeoutMs(config),
    maxBuffer: 1024 * 1024,
  })
  const lines = String(result.stdout ?? "").split(/\r?\n/).reverse()
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line)
      if (parsed && typeof parsed === "object" && parsed.ok === true) return parsed as Record<string, unknown>
    } catch {
      continue
    }
  }
  return null
}

function externalHfTokenId(model: string, token: string, config: ServingProducerConfig): number | null {
  const pythonBin = tokenizerPythonBin(config)
  if (!pythonBin) return null
  const key = JSON.stringify([pythonBin, model, token, tokenizerTrustRemoteCode(config)])
  if (!EXTERNAL_TOKENIZER_CACHE.has(key)) {
    const result = externalHfTokenizer(config, "token", model, token)
    const tokenId = result?.tokenId
    EXTERNAL_TOKENIZER_CACHE.set(key, Number.isInteger(tokenId) ? Number(tokenId) : null)
  }
  return EXTERNAL_TOKENIZER_CACHE.get(key) ?? null
}

function externalPromptTokens(model: string, payload: Record<string, unknown>, config: ServingProducerConfig): {
  tokenIds: number[]
  tokenTexts: Array<string | null>
  mode: string
} | null {
  const pythonBin = tokenizerPythonBin(config)
  if (!pythonBin) return null
  const key = JSON.stringify([pythonBin, model, sha256Json(payload), tokenizerTrustRemoteCode(config)])
  if (!EXTERNAL_PROMPT_TOKENIZER_CACHE.has(key)) {
    const result = externalHfTokenizer(config, "prompt", model, payload)
    const tokenIds = coerceTokenIds(result?.tokenIds)
    EXTERNAL_PROMPT_TOKENIZER_CACHE.set(key, tokenIds.length ? {
      tokenIds,
      tokenTexts: Array.isArray(result?.tokenTexts)
        ? result.tokenTexts.map((item) => typeof item === "string" ? item : null)
        : [],
      mode: typeof result?.mode === "string" ? result.mode : "external",
    } : null)
  }
  return EXTERNAL_PROMPT_TOKENIZER_CACHE.get(key) ?? null
}

function externalTextTokens(model: string, text: string, config: ServingProducerConfig): {
  tokenIds: number[]
  tokenTexts: Array<string | null>
  mode: string
} | null {
  const pythonBin = tokenizerPythonBin(config)
  if (!pythonBin || !text) return null
  const key = JSON.stringify([pythonBin, model, sha256Text(text), tokenizerTrustRemoteCode(config)])
  if (!EXTERNAL_TEXT_TOKENIZER_CACHE.has(key)) {
    const result = externalHfTokenizer(config, "text", model, text)
    const tokenIds = coerceTokenIds(result?.tokenIds)
    EXTERNAL_TEXT_TOKENIZER_CACHE.set(key, tokenIds.length ? {
      tokenIds,
      tokenTexts: Array.isArray(result?.tokenTexts)
        ? result.tokenTexts.map((item) => typeof item === "string" ? item : null)
        : [],
      mode: typeof result?.mode === "string" ? result.mode : "output-text",
    } : null)
  }
  return EXTERNAL_TEXT_TOKENIZER_CACHE.get(key) ?? null
}

function outputTokenDetailsFromText(config: ServingProducerConfig, outputText: string): Array<Record<string, any>> {
  const model = tokenizerModel(config)
  if (!model || !outputText) return []
  const external = externalTextTokens(model, outputText, config)
  if (!external?.tokenIds.length) return []
  return external.tokenIds.map((tokenId, tokenIndex) => {
    const tokenText = external.tokenTexts[tokenIndex]
    return {
      tokenText,
      tokenSha256: typeof tokenText === "string" ? sha256Text(tokenText) : null,
      tokenBytes: typeof tokenText === "string" ? Buffer.byteLength(tokenText) : null,
      tokenId,
      tokenIdSource: "external-hf-tokenizer",
      logprob: null,
      topLogprobs: [],
      tokenDetailSource: `external-hf-tokenizer-${external.mode}`,
    }
  })
}

function promptTokenizerModel(config: ServingProducerConfig): string | null {
  return tokenizerModel(config)
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
    const external = tokenizerModel ? externalPromptTokens(tokenizerModel, payload, config) : null
    if (external?.tokenIds.length) {
      const source = "external-hf-tokenizer"
      return {
        summary: {
          promptTokenIdsAvailable: true,
          promptTokenDetailCount: external.tokenIds.length,
          promptTokenIdSource: source,
          promptTokenIdsSha256: sha256Json(external.tokenIds),
          promptTokenizationSource: external.mode,
          promptTokenizerModel: tokenizerModel,
        },
        details: external.tokenIds.map((tokenId, tokenIndex) => {
          const tokenText = external.tokenTexts[tokenIndex]
          return {
            tokenPhase: "prompt",
            tokenIndex,
            tokenId,
            tokenIdSource: source,
            tokenDetailSource: source,
            promptSha256: promptDigest,
            tokenLogprob: null,
            tokenTextSha256: typeof tokenText === "string" ? sha256Text(tokenText) : null,
            topLogprobsJson: null,
            ...tokenizerProvenance(config),
          }
        }),
      }
    }
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
      ...tokenizerProvenance(config),
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
    tokenizerModel: typeof detail.tokenizerModel === "string" ? detail.tokenizerModel : null,
    tokenizerPythonBinSha256: typeof detail.tokenizerPythonBinSha256 === "string" ? detail.tokenizerPythonBinSha256 : null,
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
      enrichNativeTokenTiming(telemetry, outputTokenCount)
      const tpotMs = firstOutput && lastOutput
        ? (lastOutput.receivedMs - firstOutput.receivedMs) / Math.max(outputTokenCount - 1, 1)
        : null
      const gaps = outputChunks.slice(1).map((chunk, index) => chunk.receivedMs - outputChunks[index].receivedMs)
      const fallbackTokenDetails = rawTokenDetails.length ? [] : outputTokenDetailsFromText(config, outputText)
      const tokenDetailsForSummary = rawTokenDetails.length ? rawTokenDetails : fallbackTokenDetails
      const tokenSummary = tokenDetailSummary(tokenDetailsForSummary, tokenDetailsRequested)
      const tokenTimeline: ServingTokenTimelineChunk[] = promptTokenTimeline(requestId, promptTokens.details, requestStartedAtUtc)
      let tokenIndex = 0
      if (fallbackTokenDetails.length) {
        const firstOutputMs = firstOutput?.receivedMs ?? 0
        const perTokenMs = typeof tpotMs === "number" && Number.isFinite(tpotMs) ? tpotMs : 0
        for (const detail of fallbackTokenDetails) {
          tokenTimeline.push({
            requestId,
            tokenPhase: "output",
            chunkIndex: null,
            tokenIndex,
            receivedAtUtc: tokenIndex === 0 ? String(firstOutput?.receivedAtUtc ?? requestCompletedAtUtc) : String(lastOutput?.receivedAtUtc ?? requestCompletedAtUtc),
            relativeMs: firstOutputMs + (perTokenMs * tokenIndex),
            contentBytes: numberFrom(detail.tokenBytes),
            contentSha256: typeof detail.tokenSha256 === "string" ? detail.tokenSha256 : sha256Text(outputText),
            isFirstOutput: tokenIndex === 0,
            tokenId: numberFrom(detail.tokenId),
            tokenIdSource: typeof detail.tokenIdSource === "string" ? detail.tokenIdSource : null,
            tokenLogprob: null,
            tokenTextSha256: typeof detail.tokenSha256 === "string" ? detail.tokenSha256 : null,
            topLogprobsJson: null,
            tokenDetailSource: typeof detail.tokenDetailSource === "string" ? detail.tokenDetailSource : String(tokenSummary.tokenDetailSource),
            ...tokenizerProvenance(config),
          })
          tokenIndex += 1
        }
      } else {
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
                ...tokenizerProvenance(config),
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
              ...tokenizerProvenance(config),
            })
          }
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
        trtllmPerfKvAllocatedBlocks: numberFrom(telemetry.trtllmPerfKvAllocatedBlocks),
        trtllmPerfKvNewBlocks: numberFrom(telemetry.trtllmPerfKvNewBlocks),
        trtllmPerfKvReusedBlocks: numberFrom(telemetry.trtllmPerfKvReusedBlocks),
        trtllmPerfKvMissedBlocks: numberFrom(telemetry.trtllmPerfKvMissedBlocks),
        trtllmPerfRecordCount: numberFrom(telemetry.trtllmPerfRecordCount),
        trtllmPerfRequestIdSha256: typeof telemetry.trtllmPerfRequestIdSha256 === "string" ? telemetry.trtllmPerfRequestIdSha256 : null,
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
        ...hardwareSampleFields(hwTelemetry),
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
        metricSnapshots: [
          ...metricSnapshotRows(requestId, "native", "before", nativeBefore),
          ...metricSnapshotRows(requestId, "native", "after", nativeAfter),
          ...metricSnapshotRows(requestId, "dcgm", "before", hardwareBefore),
          ...metricSnapshotRows(requestId, "dcgm", "after", hardwareAfter),
        ],
        error: response.ok ? undefined : JSON.stringify(lastBody),
        rawCapture: {
          requestId,
          endpoint,
          requestPayload: payload,
          responseEvents: events,
          outputText,
          tokenDetails: tokenDetailsForSummary,
          promptTokenDetails: promptTokens.details,
          nativeMetricsRaw: {
            before: nativeBefore,
            after: nativeAfter,
          },
          hardwareMetricsRaw: {
            before: hardwareBefore,
            after: hardwareAfter,
          },
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
    const completionTokens = Number(usage.completion_tokens ?? usage.completionTokens ?? 0)
    enrichNativeTokenTiming(telemetry, completionTokens)
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
        ...tokenizerProvenance(config),
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
      outputTokenCount: completionTokens,
      promptTokens: Number(usage.prompt_tokens ?? usage.promptTokens ?? 0),
      completionTokens,
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
      trtllmPerfKvAllocatedBlocks: numberFrom(telemetry.trtllmPerfKvAllocatedBlocks),
      trtllmPerfKvNewBlocks: numberFrom(telemetry.trtllmPerfKvNewBlocks),
      trtllmPerfKvReusedBlocks: numberFrom(telemetry.trtllmPerfKvReusedBlocks),
      trtllmPerfKvMissedBlocks: numberFrom(telemetry.trtllmPerfKvMissedBlocks),
      trtllmPerfRecordCount: numberFrom(telemetry.trtllmPerfRecordCount),
      trtllmPerfRequestIdSha256: typeof telemetry.trtllmPerfRequestIdSha256 === "string" ? telemetry.trtllmPerfRequestIdSha256 : null,
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
      ...hardwareSampleFields(hwTelemetry),
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
      metricSnapshots: [
        ...metricSnapshotRows(requestId, "native", "before", nativeBefore),
        ...metricSnapshotRows(requestId, "native", "after", nativeAfter),
        ...metricSnapshotRows(requestId, "dcgm", "before", hardwareBefore),
        ...metricSnapshotRows(requestId, "dcgm", "after", hardwareAfter),
      ],
      error: response.ok ? undefined : JSON.stringify(body),
      rawCapture: {
        requestId,
        endpoint,
        requestPayload: payload,
        responseBody: body,
        outputText,
        tokenDetails: rawTokenDetails,
        promptTokenDetails: promptTokens.details,
        nativeMetricsRaw: {
          before: nativeBefore,
          after: nativeAfter,
        },
        hardwareMetricsRaw: {
          before: hardwareBefore,
          after: hardwareAfter,
        },
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
  rawArtifactPath: string | null = null,
  eventLogPath: string | null = null,
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
    avgSmActivePct: avg(successful.map((sample) => sample.smActivePct ?? null).filter(finite)),
    avgDramActivePct: avg(successful.map((sample) => sample.dramActivePct ?? null).filter(finite)),
    avgTensorActivePct: avg(successful.map((sample) => sample.tensorActivePct ?? null).filter(finite)),
    avgFp64ActivePct: avg(successful.map((sample) => sample.fp64ActivePct ?? null).filter(finite)),
    avgFp32ActivePct: avg(successful.map((sample) => sample.fp32ActivePct ?? null).filter(finite)),
    avgFp16ActivePct: avg(successful.map((sample) => sample.fp16ActivePct ?? null).filter(finite)),
    avgPcieTxThroughputKiBps: avg(successful.map((sample) => sample.pcieTxThroughputKiBps ?? null).filter(finite)),
    avgPcieRxThroughputKiBps: avg(successful.map((sample) => sample.pcieRxThroughputKiBps ?? null).filter(finite)),
    avgPcieTxBytesDelta: avg(successful.map((sample) => sample.pcieTxBytesDelta ?? null).filter(finite)),
    avgPcieRxBytesDelta: avg(successful.map((sample) => sample.pcieRxBytesDelta ?? null).filter(finite)),
    avgPcieReplayDelta: avg(successful.map((sample) => sample.pcieReplayDelta ?? null).filter(finite)),
    avgNvlinkTxBytesDelta: avg(successful.map((sample) => sample.nvlinkTxBytesDelta ?? null).filter(finite)),
    avgNvlinkRxBytesDelta: avg(successful.map((sample) => sample.nvlinkRxBytesDelta ?? null).filter(finite)),
    avgNvlinkBandwidthTotalMBps: avg(successful.map((sample) => sample.nvlinkBandwidthTotalMBps ?? null).filter(finite)),
    avgEncoderUtilizationPct: avg(successful.map((sample) => sample.encoderUtilizationPct ?? null).filter(finite)),
    avgDecoderUtilizationPct: avg(successful.map((sample) => sample.decoderUtilizationPct ?? null).filter(finite)),
    avgXidErrorsDelta: avg(successful.map((sample) => sample.xidErrorsDelta ?? null).filter(finite)),
    avgEccSbeVolatileTotalDelta: avg(successful.map((sample) => sample.eccSbeVolatileTotalDelta ?? null).filter(finite)),
    avgEccDbeVolatileTotalDelta: avg(successful.map((sample) => sample.eccDbeVolatileTotalDelta ?? null).filter(finite)),
    avgPowerViolationTimeUsDelta: avg(successful.map((sample) => sample.powerViolationTimeUsDelta ?? null).filter(finite)),
    avgThermalViolationTimeUsDelta: avg(successful.map((sample) => sample.thermalViolationTimeUsDelta ?? null).filter(finite)),
    hardwareRawMetricCountMin: successful.map((sample) => sample.hardwareRawMetricCount ?? null).filter(finite).length
      ? Math.min(...successful.map((sample) => sample.hardwareRawMetricCount ?? null).filter(finite))
      : null,
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
    nativeTelemetryRequired: nativeTelemetryExpected(config),
    hardwareTelemetryAvailableCount: successful.filter((sample) => sample.hardwareTelemetryAvailable).length,
    hardwareTelemetryRequired: Boolean(config.engine.requireHardwareTelemetry || hardwareMetricsUrl(config)),
    tokenDetailsAvailableCount: successful.filter((sample) => sample.tokenDetailsAvailable).length,
    tokenIdsAvailableCount: successful.filter((sample) => sample.tokenIdsAvailable).length,
    logprobsAvailableCount: successful.filter((sample) => sample.logprobsAvailable).length,
    tokenDetailsRequired: Boolean(requestPayload(config.request, config.request.stream !== false).logprobs),
    promptTokenIdsAvailableCount: successful.filter((sample) => sample.promptTokenIdsAvailable).length,
    promptTokenDetailsRequired: promptTokenDetailsRequired(config),
    runtimeProvenanceAvailableCount: successful.filter(hasRuntimeProvenance).length,
    eventLogRequired: eventLogEnabled(config),
    eventLogWritten: Boolean(eventLogPath),
    eventLogPath,
    eventTopic: eventTopic(config),
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
    runtimeBackend: sample.runtimeBackend,
    operatingPoint: config.workload?.operatingPoint ?? "laptop-smoke",
    basis: "per_request",
    requestId: sample.requestId,
    requestIndex: sample.requestIndex,
    requestEndpoint: sample.endpoint,
    requestStartedAtUtc: sample.requestStartedAtUtc,
    requestCompletedAtUtc: sample.requestCompletedAtUtc,
    responseId: sample.responseId,
    responseModel: sample.responseModel,
    status: sample.status,
    ok: sample.ok,
    streaming: sample.streaming,
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
    outputBytes: sample.outputBytes,
    streamChunkCount: sample.streamChunkCount,
    firstChunkAtUtc: sample.firstChunkAtUtc,
    firstOutputAtUtc: sample.firstOutputAtUtc,
    lastOutputAtUtc: sample.lastOutputAtUtc,
    finishReason: sample.finishReason,
    ttftSource: sample.ttftSource,
    promptSha256: sample.promptSha256,
    requestPayloadSha256: sample.requestPayloadSha256,
    outputSha256: sample.outputSha256,
    errorSha256: sample.error ? sha256Text(sample.error) : null,
    nativeTelemetryAvailable: sample.nativeTelemetryAvailable,
    hardwareTelemetryAvailable: sample.hardwareTelemetryAvailable,
    nativeTelemetrySource: sample.nativeTelemetrySource,
    nativeMetricsUrl: sample.nativeMetricsUrl,
    nativeJsonMetricsUrl: sample.nativeTelemetry?.nativeJsonMetricsUrl,
    nativePerfMetricsUrl: sample.nativeTelemetry?.nativePerfMetricsUrl,
    nativeTtftMs: sample.nativeTtftMs,
    nativeTpotMs: sample.nativeTpotMs,
    nativeE2eLatencyMs: sample.nativeE2eLatencyMs,
    nativeInterTokenLatencyMs: sample.nativeInterTokenLatencyMs,
    nativeIterationLatencyMs: sample.nativeIterationLatencyMs,
    nativeGpuMemoryBytes: sample.nativeGpuMemoryBytes,
    nativeKvCacheUsedBlocks: sample.nativeKvCacheUsedBlocks,
    nativeKvCacheMaxBlocks: sample.nativeKvCacheMaxBlocks,
    trtllmPerfKvAllocatedBlocks: sample.trtllmPerfKvAllocatedBlocks,
    trtllmPerfKvNewBlocks: sample.trtllmPerfKvNewBlocks,
    trtllmPerfKvReusedBlocks: sample.trtllmPerfKvReusedBlocks,
    trtllmPerfKvMissedBlocks: sample.trtllmPerfKvMissedBlocks,
    trtllmPerfRecordCount: sample.trtllmPerfRecordCount,
    trtllmPerfRequestIdSha256: sample.trtllmPerfRequestIdSha256,
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
    smActivePct: sample.smActivePct,
    dramActivePct: sample.dramActivePct,
    tensorActivePct: sample.tensorActivePct,
    fp64ActivePct: sample.fp64ActivePct,
    fp32ActivePct: sample.fp32ActivePct,
    fp16ActivePct: sample.fp16ActivePct,
    pcieTxThroughputKiBps: sample.pcieTxThroughputKiBps,
    pcieRxThroughputKiBps: sample.pcieRxThroughputKiBps,
    pcieTxBytesDelta: sample.pcieTxBytesDelta,
    pcieRxBytesDelta: sample.pcieRxBytesDelta,
    pcieReplayDelta: sample.pcieReplayDelta,
    nvlinkTxBytesDelta: sample.nvlinkTxBytesDelta,
    nvlinkRxBytesDelta: sample.nvlinkRxBytesDelta,
    nvlinkBandwidthTotalMBps: sample.nvlinkBandwidthTotalMBps,
    encoderUtilizationPct: sample.encoderUtilizationPct,
    decoderUtilizationPct: sample.decoderUtilizationPct,
    gpuTemperatureC: sample.gpuTemperatureC,
    smClockMHz: sample.smClockMHz,
    memoryClockMHz: sample.memoryClockMHz,
    fbUsedMiB: sample.fbUsedMiB,
    fbFreeMiB: sample.fbFreeMiB,
    energyJoules: sample.energyJoules,
    xidErrors: sample.xidErrors,
    xidErrorsDelta: sample.xidErrorsDelta,
    eccSbeVolatileTotalDelta: sample.eccSbeVolatileTotalDelta,
    eccDbeVolatileTotalDelta: sample.eccDbeVolatileTotalDelta,
    powerViolationTimeUsDelta: sample.powerViolationTimeUsDelta,
    thermalViolationTimeUsDelta: sample.thermalViolationTimeUsDelta,
    hardwareRawMetricCount: sample.hardwareRawMetricCount,
    hardwareRawMetricNamesSha256: sample.hardwareRawMetricNamesSha256,
    tokenDetailsAvailable: sample.tokenDetailsAvailable,
    tokenIdsAvailable: sample.tokenIdsAvailable,
    logprobsAvailable: sample.logprobsAvailable,
    tokenDetailCount: sample.tokenDetailCount,
    tokenDetailSource: sample.tokenDetailSource,
    tokenIdSource: sample.tokenIdSource,
    tokenDetailsCapabilityStatus: sample.tokenDetailsCapabilityStatus,
    tokenDetailsUnsupportedReason: sample.tokenDetailsUnsupportedReason,
    tokenizerModel: sample.tokenizerModel,
    tokenizerPythonBinSha256: sample.tokenizerPythonBinSha256,
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
    hardwareInventorySha256: sample.hardwareInventorySha256,
    rawArtifactPath,
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
    tokenizerModel: chunk.tokenizerModel ?? sample.tokenizerModel,
    tokenizerPythonBinSha256: chunk.tokenizerPythonBinSha256 ?? sample.tokenizerPythonBinSha256,
    latestCapturedAtUtc: capturedAtUtc,
  })))
  const metricSnapshotRowsForRun = samples.flatMap((sample) => (sample.metricSnapshots ?? []).map((snapshot): Record<string, unknown> => ({
    surface: "serving_metric_snapshot",
    model: config.request.model,
    runtimeFramework: engineLabel,
    runtimeEngine: config.engine.engine,
    ...snapshot,
    rawArtifactPath,
    latestCapturedAtUtc: capturedAtUtc,
  })))
  return [row, ...sampleRows, ...timelineRows, ...metricSnapshotRowsForRun]
}

const PRODUCER_COVERAGE_DESCRIPTIONS: Record<string, string> = {
  clientStreamTiming: "Client stream=true timing for E2E, TTFB, TTFT, TTFOT, TPOT, and output token timeline rows.",
  nativeRuntimeTelemetry: "Native runtime timing/cache/concurrency fields exposed by vLLM, SGLang, or TensorRT-LLM metrics.",
  dcgmHardwareTelemetry: "DCGM hardware counters for power, utilization, profiling activity, PCIe/NVLink, clocks, memory, temperature, errors, violations, raw metric inventory, and energy.",
  promptTokenIds: "Tokenizer-exact prompt/input token IDs and prompt token provenance.",
  outputTokenIds: "Output token IDs and token provenance from runtime logprobs or tokenizer fallback.",
  outputTokenLogprobs: "Output token logprobs, top-logprobs, and token-logprob provenance.",
  operatorFullArtifacts: "Operator-full raw request/response artifacts retained outside customer-safe rows.",
  rawMetricSnapshots: "Operator-full before/after native and DCGM metric snapshots retained outside customer-safe rows.",
  metricSnapshots: "Queryable per-series native and DCGM before/after metric snapshots with label and raw-exposition provenance hashes.",
  runtimeProvenance: "Engine version, model revision, image, server args, process, container, pod, node, or host provenance.",
  kafkaEventLog: "Post-capture JSONL event log with request-sample, token-timeline, aggregate, and coverage events.",
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
    sample.runtimeBackend,
    sample.modelRevision,
    sample.imageTag,
    sample.imageDigest,
    sample.serverArgsSha256,
    sample.containerId,
    sample.hostName,
  ].every((value) => typeof value === "string" && value.length > 0)
}

function rawSnapshotAvailable(capture: Record<string, unknown>, key: string): boolean {
  const snapshot = capture[key]
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) return false
  const before = (snapshot as Record<string, unknown>).before
  const after = (snapshot as Record<string, unknown>).after
  if (!before || typeof before !== "object" || Array.isArray(before)) return false
  if (!after || typeof after !== "object" || Array.isArray(after)) return false
  const beforeRecord = before as Record<string, unknown>
  const afterRecord = after as Record<string, unknown>
  const beforeHasMetrics = Boolean(beforeRecord.available) && (
    beforeRecord.metrics != null && typeof beforeRecord.metrics === "object" && !Array.isArray(beforeRecord.metrics)
    || beforeRecord.jsonMetrics != null && typeof beforeRecord.jsonMetrics === "object" && !Array.isArray(beforeRecord.jsonMetrics)
  )
  const afterHasMetrics = Boolean(afterRecord.available) && (
    afterRecord.metrics != null && typeof afterRecord.metrics === "object" && !Array.isArray(afterRecord.metrics)
    || afterRecord.jsonMetrics != null && typeof afterRecord.jsonMetrics === "object" && !Array.isArray(afterRecord.jsonMetrics)
  )
  return beforeHasMetrics && afterHasMetrics
}

function producerCoverageRows(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  aggregateRow: Record<string, unknown>,
  rawArtifactPath: string,
  eventLogPath: string | null,
  rawCaptures: Record<string, unknown>[],
  capturedAtUtc: string,
): Record<string, unknown>[] {
  const successful = samples.filter((sample) => sample.ok)
  const expectedSamples = successful.length || samples.length || 1
  const tokenLogprobsRequired = Boolean(requestPayload(config.request, config.request.stream !== false).logprobs)
  const promptRequired = Boolean(aggregateRow.promptTokenDetailsRequired)
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
  const nativeExpected = aggregateRow.nativeTelemetryRequired || nativeProven > 0 ? expectedSamples : 0
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
    finite(sample.smActivePct) &&
    finite(sample.dramActivePct) &&
    finite(sample.tensorActivePct) &&
    finite(sample.pcieTxBytesDelta) &&
    finite(sample.pcieRxBytesDelta) &&
    finite(sample.nvlinkTxBytesDelta) &&
    finite(sample.nvlinkRxBytesDelta) &&
    finite(sample.hardwareRawMetricCount) &&
    typeof sample.hardwareRawMetricNamesSha256 === "string" &&
    finite(sample.energyJoules),
  ).length
  const hardwareExpected = aggregateRow.hardwareTelemetryRequired || hardwareProven > 0 ? expectedSamples : 0
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

  const outputIdsProven = successful.filter((sample) =>
    sample.tokenDetailsAvailable === true &&
    sample.tokenIdsAvailable === true &&
    typeof sample.tokenIdSource === "string",
  ).length
  const outputIdsExpected = tokenLogprobsRequired || outputIdsProven > 0 ? expectedSamples : 0
  specs.push([
    "outputTokenIds",
    outputIdsProven,
    outputIdsExpected,
    outputIdsExpected === 0 || outputIdsProven === expectedSamples ? [] : ["output token IDs missing"],
  ])

  const outputLogprobsProven = successful.filter((sample) =>
    sample.tokenDetailsAvailable === true &&
    sample.logprobsAvailable === true,
  ).length
  const outputLogprobsExpected = tokenLogprobsRequired || outputLogprobsProven > 0 ? expectedSamples : 0
  specs.push([
    "outputTokenLogprobs",
    outputLogprobsProven,
    outputLogprobsExpected,
    outputLogprobsExpected === 0 || outputLogprobsProven === expectedSamples ? [] : ["output token logprobs missing"],
  ])

  const rawPresent = rawArtifactPath && existsSync(rawArtifactPath) ? 1 : 0
  specs.push([
    "operatorFullArtifacts",
    rawPresent,
    1,
    rawPresent ? [] : ["operator-full raw artifact missing"],
  ])

  const rawSnapshotExpected = nativeExpected > 0 || hardwareExpected > 0 ? expectedSamples : 0
  const rawSnapshotProven = rawCaptures.filter((capture) =>
    (nativeExpected === 0 || rawSnapshotAvailable(capture, "nativeMetricsRaw")) &&
    (hardwareExpected === 0 || rawSnapshotAvailable(capture, "hardwareMetricsRaw")),
  ).length
  specs.push([
    "rawMetricSnapshots",
    rawSnapshotProven,
    rawSnapshotExpected,
    rawSnapshotExpected === 0 || rawSnapshotProven === rawSnapshotExpected ? [] : ["operator-full native/DCGM raw metric snapshots missing"],
  ])

  const metricSnapshotExpected = nativeExpected > 0 || hardwareExpected > 0 ? expectedSamples : 0
  const metricSnapshotProven = successful.filter((sample) =>
    (sample.metricSnapshots ?? []).some((snapshot) =>
      typeof snapshot.metricName === "string" &&
      finite(snapshot.metricValue) &&
      typeof snapshot.metricLabelsSha256 === "string" &&
      ["before", "after"].includes(snapshot.snapshotPhase),
    ),
  ).length
  specs.push([
    "metricSnapshots",
    metricSnapshotProven,
    metricSnapshotExpected,
    metricSnapshotExpected === 0 || metricSnapshotProven === metricSnapshotExpected ? [] : ["per-request native/DCGM metric snapshot rows missing"],
  ])

  const runtimeProven = successful.filter(hasRuntimeProvenance).length
  specs.push([
    "runtimeProvenance",
    runtimeProven,
    expectedSamples,
    runtimeProven === expectedSamples ? [] : ["runtime provenance missing or partial"],
  ])

  const eventLogExpected = eventLogEnabled(config) ? 1 : 0
  const eventLogProven = eventLogPath ? 1 : 0
  specs.push([
    "kafkaEventLog",
    eventLogProven,
    eventLogExpected,
    eventLogExpected === 0 || eventLogProven === eventLogExpected ? [] : ["Post-capture event log missing"],
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
    proofPath: category === "kafkaEventLog" ? eventLogPath : rawArtifactPath,
    latestCapturedAtUtc: capturedAtUtc,
  }))
}

function eventKey(...parts: unknown[]): string {
  return parts
    .filter((part) => part != null && String(part) !== "")
    .map((part) => String(part))
    .join("|")
}

function servingTelemetryEvent(args: {
  topic: string
  eventType: string
  partitionKey: string
  emittedAtUtc: string
  model: string
  campaignId: string
  runId: string
  engine: ServingEngineId
  runtimeFramework: string
  artifactPath: string
  rawArtifactPath: string
  manifestPath: string
  eventLogPath: string
  payload: Record<string, unknown>
}): Record<string, unknown> {
  const event = {
    schemaVersion: SERVING_EVENT_SCHEMA_VERSION,
    topic: args.topic,
    eventType: args.eventType,
    partitionKey: args.partitionKey,
    emittedAtUtc: args.emittedAtUtc,
    model: args.model,
    campaignId: args.campaignId,
    runId: args.runId,
    engine: args.engine,
    runtimeFramework: args.runtimeFramework,
    artifactPath: args.artifactPath,
    rawArtifactPath: args.rawArtifactPath,
    manifestPath: args.manifestPath,
    eventLogPath: args.eventLogPath,
    payload: args.payload,
  }
  return {
    ...event,
    eventId: sha256StableJson({
      schemaVersion: event.schemaVersion,
      eventType: event.eventType,
      partitionKey: event.partitionKey,
      payload: event.payload,
    }),
  }
}

function buildServingEventRecords(args: {
  config: ServingProducerConfig
  samples: ServingRequestSample[]
  measurements: Record<string, unknown>[]
  capturedAtUtc: string
  campaignId: string
  runId: string
  artifactPath: string
  rawArtifactPath: string
  manifestPath: string
  eventLogPath: string
}): Record<string, unknown>[] {
  const topic = eventTopic(args.config)
  const runtimeFramework = ENGINE_LABELS[args.config.engine.engine]
  const keyBase = eventKey(args.campaignId, args.runId, args.config.engine.engine)
  const common = {
    topic,
    emittedAtUtc: args.capturedAtUtc,
    model: args.config.request.model,
    campaignId: args.campaignId,
    runId: args.runId,
    engine: args.config.engine.engine,
    runtimeFramework,
    artifactPath: args.artifactPath,
    rawArtifactPath: args.rawArtifactPath,
    manifestPath: args.manifestPath,
    eventLogPath: args.eventLogPath,
  }
  const events = [
    servingTelemetryEvent({
      ...common,
      eventType: "serving.producer_run",
      partitionKey: keyBase,
      payload: {
        schemaVersion: "performance-iq.serving-producer-run-event.v1",
        campaignId: args.campaignId,
        runId: args.runId,
        engine: args.config.engine.engine,
        runtimeFramework,
        model: args.config.request.model,
        requestCount: args.samples.length,
        successCount: args.samples.filter((sample) => sample.ok).length,
        measurementCount: args.measurements.length,
        servingRequestSampleCount: args.measurements.filter((row) => row.surface === "serving_request_sample").length,
        servingTokenTimelineCount: args.measurements.filter((row) => row.surface === "serving_token_timeline").length,
        artifactPath: args.artifactPath,
        rawArtifactPath: args.rawArtifactPath,
        manifestPath: args.manifestPath,
        eventLogPath: args.eventLogPath,
        eventTopic: topic,
      },
    }),
  ]

  args.measurements.forEach((row, index) => {
    const surface = String(row.surface ?? "result")
    events.push(servingTelemetryEvent({
      ...common,
      eventType: `serving.measurement.${surface}`,
      partitionKey: eventKey(keyBase, surface, row.requestId, row.chunkIndex, row.tokenIndex, index),
      payload: {
        campaignId: args.campaignId,
        runId: args.runId,
        artifactPath: args.artifactPath,
        rawArtifactPath: args.rawArtifactPath,
        manifestPath: args.manifestPath,
        eventLogPath: args.eventLogPath,
        ...row,
      },
    }))
  })
  return events
}

async function writeServingEventLog(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  measurements: Record<string, unknown>[],
  capturedAtUtc: string,
  campaignId: string,
  runId: string,
  artifactPath: string,
  rawArtifactPath: string,
  manifestPath: string,
  eventLogPath: string,
): Promise<string> {
  await mkdir(path.dirname(eventLogPath), { recursive: true })
  const events = buildServingEventRecords({
    config,
    samples,
    measurements,
    capturedAtUtc,
    campaignId,
    runId,
    artifactPath,
    rawArtifactPath,
    manifestPath,
    eventLogPath,
  })
  await writeFile(eventLogPath, events.map((event) => stableJson(event)).join("\n") + "\n")
  return eventLogPath
}

async function writeSummaryArtifact(
  config: ServingProducerConfig,
  samples: ServingRequestSample[],
  measurements: Record<string, unknown>[],
  capturedAtUtc: string,
  rawArtifactPath: string,
  eventLogPath: string | null,
): Promise<string> {
  const artifactPath = summaryArtifactPath(config, capturedAtUtc)
  await mkdir(path.dirname(artifactPath), { recursive: true })
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
      eventLogPath,
      eventTopic: eventTopic(config),
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
  const rawPath = rawArtifactPath(config, capturedAtUtc)
  await mkdir(path.dirname(rawPath), { recursive: true })
  await writeFile(rawPath, JSON.stringify({
    schemaVersion: "performance-iq.serving-operator-full-raw.v1",
    confidentiality: "operator-full",
    capturedAtUtc,
    engine: config.engine.engine,
    engineLabel: ENGINE_LABELS[config.engine.engine],
    runtimeConfiguration: {
      frameworkVersion: config.engine.frameworkVersion,
      runtimeBackend: config.engine.runtimeBackend,
      modelRevision: config.engine.modelRevision,
      imageTag: config.engine.imageTag,
      imageDigest: config.engine.imageDigest,
      serverArgs: config.engine.serverArgs,
      tokenizerModel: tokenizerModel(config),
      processId: config.engine.processId ?? config.engine.pid,
      containerId: config.engine.containerId,
      podName: config.engine.podName,
      nodeName: config.engine.nodeName,
      hostName: config.engine.hostName ?? config.engine.hostname,
      hardwareInventoryPath: config.engine.hardwareInventoryPath,
      hardwareInventorySha256: config.engine.hardwareInventorySha256,
    },
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
  const manifestPath = manifestArtifactPath(config, capturedAtUtc)
  await mkdir(path.dirname(manifestPath), { recursive: true })
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

  const rawArtifactPath = await writeRawArtifact(config, rawCaptures, capturedAtUtc)
  const artifactPath = summaryArtifactPath(config, capturedAtUtc)
  const manifestPath = manifestArtifactPath(config, capturedAtUtc)
  const eventLogPath = servingEventLogPath(config, capturedAtUtc)
  const measurements = buildMeasurements(config, samples, capturedAtUtc, rawArtifactPath, eventLogPath)
  measurements.push(...producerCoverageRows(config, samples, measurements[0], rawArtifactPath, eventLogPath, rawCaptures, capturedAtUtc))
  if (eventLogPath) {
    await writeServingEventLog(
      config,
      samples,
      measurements,
      capturedAtUtc,
      campaignId,
      runId,
      artifactPath,
      rawArtifactPath,
      manifestPath,
      eventLogPath,
    )
  }
  await writeSummaryArtifact(config, samples, measurements, capturedAtUtc, rawArtifactPath, eventLogPath)
  const engineLabel = ENGINE_LABELS[config.engine.engine]
  const runInput: PerformanceIQRunInput = {
    sourceType: config.sourceType ?? "other-measured-producer",
    runClass: config.runClass ?? "measured",
    confidentiality: config.confidentiality ?? "operator-full",
    producer: {
      repo: "performance-iq-sdk",
      tool: `${config.engine.engine}-serving-producer`,
      commitSha: DEFAULT_PRODUCER_COMMIT,
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
      imageTag: config.engine.imageTag ?? (config.engine.imageDigest ? undefined : "uncontainerized-local"),
      framework: engineLabel,
    },
    artifacts: [
      { kind: "normalized-summary", path: artifactPath },
      { kind: "operator-full-serving-raw", path: rawArtifactPath },
      ...(eventLogPath ? [{ kind: "serving-telemetry-event-log", path: eventLogPath }] : []),
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
      "Post-capture serving events are written to a JSONL event log; experimental exporters may replay that artifact after durable capture.",
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
  await writeManifestArtifact(config, manifest, capturedAtUtc)
  const submission = config.performanceIq && config.submit !== false
    ? await config.performanceIq.submitRun(runInput, { idempotencyKey: manifest.campaign.runId })
    : undefined

  return {
    engine: config.engine.engine,
    manifest,
    runInput,
    artifactPath,
    manifestPath,
    ...(eventLogPath ? { eventLogPath } : {}),
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
