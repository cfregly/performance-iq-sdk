import { createHash } from "node:crypto"
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
  queueWaitMs?: number | null
  prefillMs?: number | null
  decodeMs?: number | null
  tokenTimeline?: ServingTokenTimelineChunk[]
  error?: string
}

export interface ServingTokenTimelineChunk {
  requestId: string
  chunkIndex: number
  receivedAtUtc: string
  relativeMs: number
  contentBytes: number
  contentSha256: string
  isFirstOutput: boolean
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

function metricCompleteness(row: Record<string, unknown>): number {
  const required = [
    row.outputTpm,
    row.totalTpm,
    row.usdPer1mOutputTokens,
    row.usdPer1mTotalTokens,
    row.tokensPerWatt,
    row.avgTtftMs,
    row.avgTpotMs,
    row.avgTtfotMs,
    row.requestCount === row.successCount ? row.requestCount : null,
    row.hardwareProvenance === "configured" ? 1 : null,
  ]
  if (row.nativeTelemetryRequired) {
    required.push(row.nativeTelemetryAvailableCount === row.successCount ? row.nativeTelemetryAvailableCount : null)
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

function numberFrom(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
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
  const started = performance.now()
  const requestStartedAtUtc = new Date().toISOString()
  const stream = config.request.stream !== false
  const payload = requestPayload(config.request, stream)
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
      const outputChunks: Array<{ chunkIndex: number; content: string; receivedMs: number; receivedAtUtc: string }> = []
      let responseId: string | undefined
      let responseModel: string | undefined
      let finishReason: string | undefined
      let usage: Record<string, any> = {}
      let lastBody: Record<string, any> = {}
      let telemetry = nativeTelemetry(config)
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
          outputChunks.push({
            chunkIndex: outputChunks.length,
            content,
            receivedMs: Number(event.receivedMs),
            receivedAtUtc: String(event.receivedAtUtc),
          })
        }
      }
      const firstChunk = events[0]
      const firstOutput = outputChunks[0]
      const lastOutput = outputChunks[outputChunks.length - 1]
      const outputText = outputChunks.map((chunk) => chunk.content).join("")
      const tokenCountSource = Object.keys(usage).length ? "response-usage" : "client-estimate"
      const promptTokens = Number(usage.prompt_tokens ?? usage.promptTokens ?? 0) || estimatedTokenCount(promptText(payload))
      const completionTokens = Number(usage.completion_tokens ?? usage.completionTokens ?? 0) || outputChunks.length
      const totalTokens = Number(usage.total_tokens ?? usage.totalTokens ?? 0) || (promptTokens + completionTokens)
      const outputTokenCount = completionTokens || outputChunks.length
      const tpotMs = firstOutput && lastOutput
        ? (lastOutput.receivedMs - firstOutput.receivedMs) / Math.max(outputTokenCount - 1, 1)
        : null
      const gaps = outputChunks.slice(1).map((chunk, index) => chunk.receivedMs - outputChunks[index].receivedMs)
      const tokenTimeline = outputChunks.map((chunk): ServingTokenTimelineChunk => ({
        requestId,
        chunkIndex: chunk.chunkIndex,
        receivedAtUtc: chunk.receivedAtUtc,
        relativeMs: chunk.receivedMs,
        contentBytes: Buffer.byteLength(chunk.content),
        contentSha256: sha256Text(chunk.content),
        isFirstOutput: chunk.chunkIndex === 0,
      }))
      const redacted = redactedRequest(payload)
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
        promptTokens,
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
        queueWaitMs: numberFrom(telemetry.queueWaitMs),
        prefillMs: numberFrom(telemetry.prefillMs),
        decodeMs: numberFrom(telemetry.decodeMs),
        tokenTimeline,
        error: response.ok ? undefined : JSON.stringify(lastBody),
        rawCapture: {
          requestId,
          endpoint,
          requestPayload: payload,
          responseEvents: events,
          outputText,
        },
      }
    }

    const latencyMs = performance.now() - started
    const requestCompletedAtUtc = new Date().toISOString()
    const body = await response.json().catch(() => ({})) as Record<string, any>
    const usage = body.usage ?? {}
    const outputText = choiceContent(body)
    const telemetry = nativeTelemetry(config, body)
    const redacted = redactedRequest(payload)
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
      queueWaitMs: numberFrom(telemetry.queueWaitMs),
      prefillMs: numberFrom(telemetry.prefillMs),
      decodeMs: numberFrom(telemetry.decodeMs),
      tokenTimeline: [],
      error: response.ok ? undefined : JSON.stringify(body),
      rawCapture: {
        requestId,
        endpoint,
        requestPayload: payload,
        responseBody: body,
        outputText,
      },
    }
  } catch (error) {
    const latencyMs = performance.now() - started
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
      nativeTelemetry: nativeTelemetry(config),
      nativeTelemetryAvailable: false,
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
  const gpuCount = config.pricing?.gpuCount ?? Number(config.workload?.parallelism ?? 1)
  const usdPerGpuHour = config.pricing?.usdPerGpuHour
  const powerWattsPerGpu = config.pricing?.powerWattsPerGpu
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
  const avgLatencyMs = successful.length ? sum(successful.map((sample) => sample.e2eLatencyMs)) / successful.length : null
  const avg = (values: number[]) => values.length ? sum(values) / values.length : null
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
    p95TimeToFirstByteMs: percentile(firstBytes, 95),
    avgTtftMs: avg(ttfts),
    p50TtftMs: percentile(ttfts, 50),
    p95TtftMs: percentile(ttfts, 95),
    p99TtftMs: percentile(ttfts, 99),
    avgTtfotMs: avg(ttfots),
    p95TtfotMs: percentile(ttfots, 95),
    avgTpotMs: avg(tpots),
    p95TpotMs: percentile(tpots, 95),
    avgInterTokenLatencyMs: avg(successful.map((sample) => sample.interTokenLatencyMs).filter(finite)),
    avgQueueWaitMs: avg(successful.map((sample) => sample.queueWaitMs ?? null).filter(finite)),
    avgPrefillMs: avg(successful.map((sample) => sample.prefillMs ?? null).filter(finite)),
    avgDecodeMs: avg(successful.map((sample) => sample.decodeMs ?? null).filter(finite)),
    usdPer1mOutputTokens,
    usdPer1mTotalTokens,
    avgPowerWattsPerGpu: powerWattsPerGpu ?? null,
    tokensPerWatt,
    campaignCount: Math.max(successful.length, 1),
    latestCapturedAtUtc: capturedAtUtc,
    experimentFamily: "serving-producer",
    experimentStatus: successful.length === samples.length ? "accepted" : "partial",
    verdictTier: successful.length === samples.length ? "request-captured" : "request-errors",
    solRigor: config.runClass === "measured" ? "l3" : "smoke",
    plotReadyPoints: 0,
    dcgmGrounded: false,
    streamingRequestCount: successful.filter((sample) => sample.streaming).length,
    nativeTelemetryAvailableCount: successful.filter((sample) => sample.nativeTelemetryAvailable).length,
    nativeTelemetryRequired: Boolean((config.engine as unknown as Record<string, unknown>).requireNativeTelemetry),
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
    queueWaitMs: sample.queueWaitMs,
    prefillMs: sample.prefillMs,
    decodeMs: sample.decodeMs,
    latestCapturedAtUtc: capturedAtUtc,
  }))
  const timelineRows = samples.flatMap((sample) => (sample.tokenTimeline ?? []).map((chunk): Record<string, unknown> => ({
    surface: "serving_token_timeline",
    model: config.request.model,
    runtimeFramework: engineLabel,
    runtimeEngine: config.engine.engine,
    requestId: sample.requestId,
    chunkIndex: chunk.chunkIndex,
    receivedAtUtc: chunk.receivedAtUtc,
    relativeMs: chunk.relativeMs,
    contentBytes: chunk.contentBytes,
    contentSha256: chunk.contentSha256,
    isFirstOutput: chunk.isFirstOutput,
    latestCapturedAtUtc: capturedAtUtc,
  })))
  return [row, ...sampleRows, ...timelineRows]
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
      "Metrics are derived from client-side streaming SSE timings, response usage fields, and native telemetry when exposed.",
    ].join(" "),
    limitations: [
      "Serving producer captures client stream timing, request-path, usage, latency, and provenance; hardware-level power/kernel counters require engine-side or cluster instrumentation.",
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
