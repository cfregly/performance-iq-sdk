import { createHash } from "node:crypto"
import { readFile, stat } from "node:fs/promises"

export const PRODUCER_MANIFEST_VERSION = "atlas-benchmark-evidence.producer-manifest.v1"
export const INGESTION_REQUEST_VERSION = "performance-iq.ingestion-request.v1"

export type SourceKind = "preserved-snapshot" | "fresh-gamble-run" | "other-measured-producer"
export type EvidenceClass = "measured" | "rehearsal" | "simulated"
export type Confidentiality = "internal-full" | "customer-safe" | "public-safe" | "redacted"
export type RunStatus = "accepted" | "rejected" | "processing" | "quote-ready"

export interface ProducerIdentity {
  repo: string
  tool: string
  version?: string
  commitSha: string
  operator?: string
}

export interface CampaignIdentity {
  campaignId: string
  runId: string
  slug?: string
  capturedAtUtc?: string
  completedAtUtc?: string
  publishedAtUtc?: string
}

export interface WorkloadIdentity {
  model: string
  hardware: string
  operatingPoint: string
  scenario?: string
  precision?: string
  parallelism?: string
  datasetOrPromptSet?: string
}

export interface RuntimeIdentity {
  imageDigest: string
  imageTag?: string
  cudaVersion?: string
  ncclVersion?: string
  driverVersion?: string
  framework?: string
}

export interface ArtifactMetadata {
  kind: string
  path: string
  sha256: string
  sizeBytes: number
}

export type ArtifactInput = string | {
  kind?: string
  path: string
  sha256?: string
  sizeBytes?: number
}

export interface RowProof {
  table: string
  campaignId?: string
  rowCount: number
  latestCapturedAtUtc?: string
}

export interface PerflakeProof {
  sourceTables: string[]
  modelTables: string[]
  rowProof: RowProof[]
}

export interface AtlasEvidenceReference {
  dashboardUrl?: string
  evidencePackPath: string
  exportGeneratedAtUtc?: string
  preflightPath?: string
}

export interface PerformanceIQRunInput {
  sourceKind: SourceKind
  evidenceClass?: EvidenceClass
  confidentiality: Confidentiality
  producer: ProducerIdentity
  campaign: CampaignIdentity
  workload: WorkloadIdentity
  runtime: RuntimeIdentity
  artifacts: ArtifactInput[]
  measurements?: Record<string, unknown>[]
  perflake?: PerflakeProof
  atlasEvidence?: Partial<AtlasEvidenceReference>
  methodology?: string
  limitations?: string[]
}

export interface ProducerEvidenceManifest {
  schemaVersion: typeof PRODUCER_MANIFEST_VERSION
  evidenceClass: EvidenceClass
  producerEvidenceSource: SourceKind
  generatedAtUtc: string
  producer: ProducerIdentity
  campaign: Required<Pick<CampaignIdentity, "campaignId" | "runId" | "capturedAtUtc" | "completedAtUtc">> &
    Omit<CampaignIdentity, "campaignId" | "runId" | "capturedAtUtc" | "completedAtUtc">
  workload: WorkloadIdentity
  runtime: RuntimeIdentity
  artifacts: ArtifactMetadata[]
  perflake: PerflakeProof
  atlasEvidence: AtlasEvidenceReference
  methodology: string
  limitations: string[]
  confidentiality: Confidentiality
}

export interface EvidenceSubmissionEnvelope {
  schemaVersion: typeof INGESTION_REQUEST_VERSION
  manifest: ProducerEvidenceManifest
  measurements?: Record<string, unknown>[]
}

export interface ValidationResult {
  ok: boolean
  liveProofReady: boolean
  sourceKind: SourceKind | undefined
  snapshotBacked: boolean
  freshProducer: boolean
  errors: string[]
  warnings: string[]
  manifest?: ProducerEvidenceManifest
}

export interface PerformanceIQClientOptions {
  baseUrl: string
  token?: string
  fetchImpl?: typeof fetch
}

export interface SubmitOptions {
  idempotencyKey?: string
  dryRun?: boolean
}

export interface EvidenceStatusRequest {
  consumer: "sales" | "support" | "agent"
  confidentialityMode: "internal_full" | "customer_safe" | "public_safe" | "redacted"
  question?: string
  filters?: {
    model?: string
    hardware?: string
    operatingPoint?: string
    basis?: string
  }
}

const IMAGE_DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/i
const SHA256_PATTERN = /^[0-9a-f]{64}$/i
const PLACEHOLDER_PATTERN = /\b(replace-with|example-only|do-not-quote|template only)\b/i
const DISALLOWED_REQUEST_KEYS = new Set(["sql", "queryName", "queries"])

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "")
}

function nowIso(): string {
  return new Date().toISOString()
}

function findDisallowedRequestKey(value: unknown): string | null {
  if (!value || typeof value !== "object") return null
  if (Array.isArray(value)) {
    for (const item of value) {
      const match = findDisallowedRequestKey(item)
      if (match) return match
    }
    return null
  }
  for (const [key, child] of Object.entries(value)) {
    if (DISALLOWED_REQUEST_KEYS.has(key)) return key
    const match = findDisallowedRequestKey(child)
    if (match) return match
  }
  return null
}

function isIsoDate(value: string | undefined): boolean {
  return typeof value === "string" && Number.isFinite(new Date(value).getTime())
}

function defaultArtifactKind(path: string): string {
  if (path.endsWith(".log") || path.endsWith(".txt")) return "raw-log"
  if (path.endsWith(".json") || path.endsWith(".jsonl")) return "normalized-summary"
  if (path.endsWith(".yaml") || path.endsWith(".yml")) return "config-snapshot"
  return "artifact"
}

async function hashFile(path: string): Promise<string> {
  const hash = createHash("sha256")
  hash.update(await readFile(path))
  return hash.digest("hex")
}

async function normalizeArtifact(input: ArtifactInput): Promise<ArtifactMetadata> {
  const raw = typeof input === "string" ? { path: input } : input
  const fileStat = await stat(raw.path)
  const sha256 = raw.sha256 ?? await hashFile(raw.path)
  return {
    kind: raw.kind ?? defaultArtifactKind(raw.path),
    path: raw.path,
    sha256,
    sizeBytes: raw.sizeBytes ?? fileStat.size,
  }
}

function defaultPerflakeProof(input: PerformanceIQRunInput): PerflakeProof {
  const rowCount = Math.max(input.measurements?.length ?? 0, 1)
  const table = "model_perflake.sdk_pending_ingest"
  return {
    sourceTables: ["performance_iq.sdk_submission"],
    modelTables: [table],
    rowProof: [
      {
        table,
        campaignId: input.campaign.campaignId,
        rowCount,
        latestCapturedAtUtc: input.campaign.capturedAtUtc ?? nowIso(),
      },
    ],
  }
}

export async function buildManifest(input: PerformanceIQRunInput): Promise<ProducerEvidenceManifest> {
  const generatedAtUtc = nowIso()
  const artifacts = await Promise.all(input.artifacts.map(normalizeArtifact))
  const capturedAtUtc = input.campaign.capturedAtUtc ?? generatedAtUtc
  const completedAtUtc = input.campaign.completedAtUtc ?? capturedAtUtc
  const perflake = input.perflake ?? defaultPerflakeProof(input)

  return {
    schemaVersion: PRODUCER_MANIFEST_VERSION,
    evidenceClass: input.evidenceClass ?? "measured",
    producerEvidenceSource: input.sourceKind,
    generatedAtUtc,
    producer: input.producer,
    campaign: {
      ...input.campaign,
      capturedAtUtc,
      completedAtUtc,
    },
    workload: input.workload,
    runtime: input.runtime,
    artifacts,
    perflake: {
      ...perflake,
      rowProof: perflake.rowProof.map((proof) => ({
        ...proof,
        campaignId: proof.campaignId ?? input.campaign.campaignId,
      })),
    },
    atlasEvidence: {
      evidencePackPath: input.atlasEvidence?.evidencePackPath ?? "performance-iq://pending/evidence-pack",
      dashboardUrl: input.atlasEvidence?.dashboardUrl,
      exportGeneratedAtUtc: input.atlasEvidence?.exportGeneratedAtUtc,
      preflightPath: input.atlasEvidence?.preflightPath,
    },
    methodology: input.methodology ?? "Submitted through the Performance IQ SDK.",
    limitations: input.limitations?.length ? input.limitations : ["No limitations were supplied by the producer."],
    confidentiality: input.confidentiality,
  }
}

export function buildEnvelope(
  manifest: ProducerEvidenceManifest,
  measurements?: Record<string, unknown>[],
): EvidenceSubmissionEnvelope {
  return {
    schemaVersion: INGESTION_REQUEST_VERSION,
    manifest,
    measurements,
  }
}

export function validateManifest(manifest: ProducerEvidenceManifest): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  const disallowed = findDisallowedRequestKey(manifest)
  if (disallowed) errors.push(`payload must not include ${disallowed}`)

  if (JSON.stringify(manifest).match(PLACEHOLDER_PATTERN)) {
    errors.push("manifest contains placeholder, template, or example-only markers")
  }
  if (manifest.schemaVersion !== PRODUCER_MANIFEST_VERSION) {
    errors.push(`schemaVersion must be ${PRODUCER_MANIFEST_VERSION}`)
  }
  if (!["measured", "rehearsal", "simulated"].includes(manifest.evidenceClass)) {
    errors.push("evidenceClass must be measured, rehearsal, or simulated")
  }
  if (manifest.evidenceClass !== "measured") {
    warnings.push("manifest is accepted as non-live evidence only; live proof requires evidenceClass=measured")
  }
  if (!["preserved-snapshot", "fresh-gamble-run", "other-measured-producer"].includes(manifest.producerEvidenceSource)) {
    errors.push("producerEvidenceSource is not supported")
  }
  if (manifest.confidentiality !== "internal-full") {
    errors.push("only internal-full submissions are enabled; customer-safe, public-safe, and redacted remain fail-closed")
  }
  if (!manifest.producer?.repo) errors.push("producer.repo is required")
  if (!manifest.producer?.tool) errors.push("producer.tool is required")
  if (!manifest.producer?.commitSha || manifest.producer.commitSha.length < 7) {
    errors.push("producer.commitSha must contain at least 7 characters")
  }
  if (!manifest.campaign?.campaignId) errors.push("campaign.campaignId is required")
  if (!manifest.campaign?.runId) errors.push("campaign.runId is required")
  if (!isIsoDate(manifest.campaign?.capturedAtUtc)) errors.push("campaign.capturedAtUtc must be a valid date-time")
  if (!isIsoDate(manifest.campaign?.completedAtUtc)) errors.push("campaign.completedAtUtc must be a valid date-time")
  if (!manifest.workload?.model) errors.push("workload.model is required")
  if (!manifest.workload?.hardware) errors.push("workload.hardware is required")
  if (!manifest.workload?.operatingPoint) errors.push("workload.operatingPoint is required")
  if (!IMAGE_DIGEST_PATTERN.test(manifest.runtime?.imageDigest ?? "")) {
    errors.push("runtime.imageDigest must match sha256:<64 hex chars>")
  }
  if (!manifest.artifacts?.length) errors.push("at least one artifact is required")
  for (const [index, artifact] of manifest.artifacts.entries()) {
    if (!artifact.kind) errors.push(`artifacts[${index}].kind is required`)
    if (!artifact.path) errors.push(`artifacts[${index}].path is required`)
    if (!SHA256_PATTERN.test(artifact.sha256)) errors.push(`artifacts[${index}].sha256 must be a 64-character SHA-256 hex digest`)
    if (!Number.isFinite(artifact.sizeBytes) || artifact.sizeBytes < 0) {
      errors.push(`artifacts[${index}].sizeBytes must be >= 0`)
    }
  }
  if (!manifest.perflake?.sourceTables?.length) errors.push("perflake.sourceTables must contain at least one table")
  if (!manifest.perflake?.modelTables?.length) errors.push("perflake.modelTables must contain at least one table")
  const modelTables = new Set(manifest.perflake?.modelTables ?? [])
  for (const [index, proof] of (manifest.perflake?.rowProof ?? []).entries()) {
    if (proof.campaignId !== manifest.campaign.campaignId) {
      errors.push(`perflake.rowProof[${index}].campaignId must match campaign.campaignId`)
    }
    if (!modelTables.has(proof.table)) {
      errors.push(`perflake.rowProof[${index}].table must be listed in perflake.modelTables`)
    }
    if (!Number.isFinite(proof.rowCount) || proof.rowCount < 1) {
      errors.push(`perflake.rowProof[${index}].rowCount must be >= 1`)
    }
  }
  if (!manifest.perflake?.rowProof?.length) errors.push("perflake.rowProof must contain at least one row proof")

  const liveProofReady = errors.length === 0 &&
    manifest.evidenceClass === "measured" &&
    manifest.producerEvidenceSource === "fresh-gamble-run"

  return {
    ok: errors.length === 0,
    liveProofReady,
    sourceKind: manifest.producerEvidenceSource,
    snapshotBacked: manifest.producerEvidenceSource === "preserved-snapshot",
    freshProducer: manifest.producerEvidenceSource === "fresh-gamble-run",
    errors,
    warnings,
    manifest,
  }
}

export async function validateRun(input: PerformanceIQRunInput): Promise<ValidationResult> {
  const errors: string[] = []
  const disallowed = findDisallowedRequestKey(input)
  if (disallowed) errors.push(`payload must not include ${disallowed}`)

  let manifest: ProducerEvidenceManifest | undefined
  try {
    manifest = await buildManifest(input)
  } catch (err) {
    errors.push(err instanceof Error ? err.message : String(err))
  }

  if (!manifest) {
    return {
      ok: false,
      liveProofReady: false,
      sourceKind: input.sourceKind,
      snapshotBacked: input.sourceKind === "preserved-snapshot",
      freshProducer: input.sourceKind === "fresh-gamble-run",
      errors,
      warnings: [],
    }
  }

  const result = validateManifest(manifest)
  return {
    ...result,
    errors: [...errors, ...result.errors],
    ok: errors.length === 0 && result.ok,
  }
}

async function readError(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? ""
  if (contentType.includes("application/json")) {
    const body = (await response.json().catch(() => null)) as { detail?: unknown; error?: unknown; message?: unknown } | null
    const detail = body?.detail ?? body?.error ?? body?.message
    if (typeof detail === "string") return detail
    if (detail) return JSON.stringify(detail)
  }
  const text = await response.text().catch(() => "")
  return text.trim() || response.statusText
}

export class PerformanceIQError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = "PerformanceIQError"
    this.status = status
  }
}

export class PerformanceIQ {
  private readonly baseUrl: string
  private readonly token?: string
  private readonly fetchImpl: typeof fetch

  constructor(options: PerformanceIQClientOptions) {
    this.baseUrl = normalizeBaseUrl(options.baseUrl)
    this.token = options.token
    this.fetchImpl = options.fetchImpl ?? fetch
  }

  async validateRun(input: PerformanceIQRunInput): Promise<ValidationResult> {
    return validateRun(input)
  }

  async submitRun(input: PerformanceIQRunInput, options: SubmitOptions = {}) {
    const local = await validateRun(input)
    if (!local.ok || !local.manifest) {
      throw new PerformanceIQError(`run failed local validation: ${local.errors.join("; ")}`)
    }
    if (options.dryRun) return local

    return this.postJson("/api/v1/evidence/runs", buildEnvelope(local.manifest, input.measurements), {
      "idempotency-key": options.idempotencyKey ?? local.manifest.campaign.runId,
    })
  }

  async submitManifest(manifestPath: string, options: SubmitOptions = {}) {
    const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as ProducerEvidenceManifest
    const local = validateManifest(manifest)
    if (!local.ok) {
      throw new PerformanceIQError(`manifest failed local validation: ${local.errors.join("; ")}`)
    }
    if (options.dryRun) return local
    return this.postJson("/api/v1/evidence/runs", buildEnvelope(manifest), {
      "idempotency-key": options.idempotencyKey ?? manifest.campaign.runId,
    })
  }

  async getRunStatus(runId: string) {
    return this.getJson(`/api/v1/evidence/runs/${encodeURIComponent(runId)}`)
  }

  async getEvidenceStatus(request: EvidenceStatusRequest) {
    const disallowed = findDisallowedRequestKey(request)
    if (disallowed) {
      throw new PerformanceIQError(`evidence status request must not include ${disallowed}`)
    }
    return this.postJson("/api/downstream/evidence-status", request)
  }

  private headers(extra: Record<string, string> = {}): HeadersInit {
    return {
      "content-type": "application/json",
      ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
      ...extra,
    }
  }

  private async postJson(path: string, body: unknown, headers: Record<string, string> = {}) {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers(headers),
      body: JSON.stringify(body),
    })
    if (!response.ok) throw new PerformanceIQError(await readError(response), response.status)
    return response.json()
  }

  private async getJson(path: string) {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: "GET",
      headers: this.headers(),
    })
    if (!response.ok) throw new PerformanceIQError(await readError(response), response.status)
    return response.json()
  }
}
