// Offline buyer-side verification of a Performance IQ deal-proof packet.
//
// This is the "don't trust us — replay it" primitive. A buyer's own engineers
// can run it with plain Node and nothing else: no Performance IQ server, no
// service token, no license. It checks the buyer-relevant invariants of the
// producer manifest contract
// (workbench/apps/data-platforms/performance-iq/contracts/producer-run-manifest.schema.json)
// — not the full producer schema, which is guarded on the producer side. The
// surface here is deliberately smaller and more stable than the producer
// schema so that a buyer can depend on it without tracking our releases.
//
// It is fail-closed by construction: example/template/rehearsal packets, stale
// evidence, mismatched artifact hashes, and redaction leaks in customer-facing
// tiers all fail. The same discipline that rejects our own placeholder
// manifests is what makes a passing packet quotable.

import crypto from "node:crypto"
import fs from "node:fs"
import path from "node:path"

export const PRODUCER_MANIFEST_VERSION = "performance-iq.producer-manifest.v1"
export const DEFAULT_FRESHNESS_MAX_DAYS = 30

const PLACEHOLDER_PATTERN = /\b(replace-with|example-only|do-not-quote|template only)\b/i
const SHA256_PATTERN = /^[0-9a-f]{64}$/i
const IMAGE_DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/i
const CONFIDENTIALITY = ["operator-full", "customer-safe", "public-safe", "redacted"]
const CUSTOMER_FACING = new Set(["customer-safe", "public-safe", "redacted"])
const ABSOLUTE_PATH_PATTERN = /(?:^|["'\s(])(?:\/[^"'\s)]+|[A-Za-z]:\\[^"'\s)]+)/
const PRIVATE_HOST_PATTERN = /\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)\b/i
const DAY_MS = 24 * 60 * 60 * 1000

function isNonEmptyString(value) {
  return typeof value === "string" && value.length > 0
}

function isDateTime(value) {
  return isNonEmptyString(value) && Number.isFinite(new Date(value).getTime())
}

function isLocalPath(value) {
  if (!isNonEmptyString(value)) return false
  return !/^[a-z][a-z0-9+.-]*:\/\//i.test(value)
}

function sha256File(target) {
  return crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex")
}

function looksLikeManifest(value) {
  if (value == null || typeof value !== "object") return false
  if (typeof value.schemaVersion === "string" && value.schemaVersion.startsWith("performance-iq.producer-manifest")) {
    return true
  }
  return value.runClass !== undefined && value.campaign !== undefined && value.producer !== undefined
}

// A packet may be a bare producer manifest or an envelope that carries one.
// A manifest-shaped object with a wrong/missing schemaVersion is still treated
// as a (malformed) manifest so the schema-version check reports a precise
// error rather than a bare "not a manifest".
export function extractManifest(packet) {
  const envelopeKeys = ["manifest", "producerManifest", "producer_manifest"]
  if (packet && packet.schemaVersion === PRODUCER_MANIFEST_VERSION) return packet
  for (const key of envelopeKeys) {
    const candidate = packet?.[key]
    if (candidate && candidate.schemaVersion === PRODUCER_MANIFEST_VERSION) return candidate
  }
  if (looksLikeManifest(packet)) return packet
  for (const key of envelopeKeys) {
    if (looksLikeManifest(packet?.[key])) return packet[key]
  }
  return null
}

/**
 * Verify a deal-proof packet offline.
 *
 * @param {object} packet - parsed manifest JSON, or an envelope containing one.
 * @param {object} [options]
 * @param {number} [options.now] - epoch ms treated as "now" (for testing).
 * @param {number} [options.freshnessMaxDays] - stale threshold, default 30.
 * @param {boolean} [options.requireMeasured] - reject non-measured runClass, default true.
 * @param {string|null} [options.artifactRoot] - dir to resolve relative artifact paths.
 * @returns {{ok: boolean, runClass: string|undefined, confidentiality: string|undefined,
 *   ageDays: number|null, freshnessMaxDays: number, checks: object[],
 *   errors: string[], warnings: string[], summary: string}}
 */
export function verifyPacket(packet, options = {}) {
  const {
    now = Date.now(),
    freshnessMaxDays = DEFAULT_FRESHNESS_MAX_DAYS,
    requireMeasured = true,
    artifactRoot = null,
  } = options

  const checks = []
  const record = (name, ok, detail, severity = "error") => {
    checks.push({ name, ok, detail, severity })
  }

  const manifest = extractManifest(packet)
  if (!manifest) {
    record(
      "structure",
      false,
      `packet is not a ${PRODUCER_MANIFEST_VERSION} manifest and carries no manifest envelope`,
    )
    return finalize(checks, { runClass: undefined, confidentiality: undefined, ageDays: null, freshnessMaxDays })
  }

  // 1. schema version
  record(
    "schema-version",
    manifest.schemaVersion === PRODUCER_MANIFEST_VERSION,
    manifest.schemaVersion === PRODUCER_MANIFEST_VERSION
      ? `schemaVersion is ${PRODUCER_MANIFEST_VERSION}`
      : `schemaVersion must be ${PRODUCER_MANIFEST_VERSION}, got ${JSON.stringify(manifest.schemaVersion)}`,
  )

  // 2. required buyer-relevant structure
  const missing = requiredFieldsMissing(manifest)
  record(
    "structure",
    missing.length === 0,
    missing.length === 0 ? "all buyer-relevant fields present" : `missing required fields: ${missing.join(", ")}`,
  )

  // 3. no placeholder / example / template markers
  const hasPlaceholder = PLACEHOLDER_PATTERN.test(JSON.stringify(manifest))
  record(
    "not-placeholder",
    !hasPlaceholder,
    hasPlaceholder
      ? "packet contains placeholder, template, or example-only markers"
      : "no placeholder markers",
  )

  // 4. evidence class (a quote-grade packet must be measured, never rehearsal/simulated)
  const runClass = manifest.runClass
  if (requireMeasured) {
    record(
      "evidence-class",
      runClass === "measured",
      runClass === "measured"
        ? "runClass is measured"
        : `runClass is ${JSON.stringify(runClass)}; only measured packets are quotable as live proof (pass --allow-rehearsal to inspect)`,
    )
  } else {
    record("evidence-class", true, `runClass is ${JSON.stringify(runClass)} (measured requirement waived)`, "info")
  }

  // 5. freshness — anchored on when the run completed
  const completedAt = manifest.campaign?.completedAtUtc ?? manifest.generatedAtUtc
  let ageDays = null
  if (!isDateTime(completedAt)) {
    record("freshness", false, "campaign.completedAtUtc is missing or not a valid date-time; freshness cannot be established")
  } else {
    ageDays = (now - new Date(completedAt).getTime()) / DAY_MS
    if (ageDays < 0) {
      record("freshness", true, `completedAt is ${Math.abs(ageDays).toFixed(1)}d in the future (clock skew?)`, "warning")
      ageDays = 0
    } else if (ageDays > freshnessMaxDays) {
      record(
        "freshness",
        false,
        `evidence is ${ageDays.toFixed(1)}d old, past the ${freshnessMaxDays}d freshness window — stale, not quote-ready`,
      )
    } else {
      record("freshness", true, `evidence is ${ageDays.toFixed(1)}d old, within the ${freshnessMaxDays}d window`)
    }
  }

  // 6. replay recipe — the fields a buyer needs to re-run the producer themselves
  const replayGaps = replayRecipeGaps(manifest)
  record(
    "replay-recipe",
    replayGaps.length === 0,
    replayGaps.length === 0
      ? "replay recipe complete (producer repo + commit, runtime image digest, methodology)"
      : `replay recipe incomplete: ${replayGaps.join(", ")}`,
  )

  // 7. artifact hashes — verify locally where the file is shipped; otherwise declared-only
  verifyArtifacts(manifest, artifactRoot, record)

  // 8. redaction — customer-facing tiers must not leak operator-only detail
  verifyRedaction(manifest, record)

  // 9. row-proof consistency (mirrors the producer-side gate)
  verifyRowProof(manifest, record)

  return finalize(checks, {
    runClass,
    confidentiality: manifest.confidentiality,
    ageDays,
    freshnessMaxDays,
  })
}

function requiredFieldsMissing(manifest) {
  const missing = []
  const need = (present, label) => {
    if (!present) missing.push(label)
  }
  need(isNonEmptyString(manifest.runClass), "runClass")
  need(isDateTime(manifest.generatedAtUtc), "generatedAtUtc")
  need(isNonEmptyString(manifest.producer?.repo), "producer.repo")
  need(isNonEmptyString(manifest.producer?.commitSha), "producer.commitSha")
  need(isNonEmptyString(manifest.campaign?.campaignId), "campaign.campaignId")
  need(isNonEmptyString(manifest.campaign?.runId), "campaign.runId")
  need(isNonEmptyString(manifest.workload?.model), "workload.model")
  need(isNonEmptyString(manifest.workload?.hardware), "workload.hardware")
  need(isNonEmptyString(manifest.runtime?.imageDigest), "runtime.imageDigest")
  need(Array.isArray(manifest.artifacts) && manifest.artifacts.length > 0, "artifacts[]")
  need(isNonEmptyString(manifest.methodology), "methodology")
  need(Array.isArray(manifest.limitations) && manifest.limitations.length > 0, "limitations[]")
  need(CONFIDENTIALITY.includes(manifest.confidentiality), "confidentiality")
  return missing
}

function replayRecipeGaps(manifest) {
  const gaps = []
  if (!isNonEmptyString(manifest.producer?.repo)) gaps.push("producer.repo")
  if (!(isNonEmptyString(manifest.producer?.commitSha) && manifest.producer.commitSha.length >= 7)) {
    gaps.push("producer.commitSha (>=7 chars)")
  }
  if (!IMAGE_DIGEST_PATTERN.test(String(manifest.runtime?.imageDigest ?? ""))) {
    gaps.push("runtime.imageDigest (sha256:<64 hex>)")
  }
  if (!isNonEmptyString(manifest.methodology)) gaps.push("methodology")
  return gaps
}

function verifyArtifacts(manifest, artifactRoot, record) {
  const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : []
  if (artifacts.length === 0) {
    record("artifact-hashes", false, "no artifacts declared")
    return
  }
  let declaredOnly = 0
  let verified = 0
  for (const [index, artifact] of artifacts.entries()) {
    const sha = String(artifact?.sha256 ?? "")
    if (!SHA256_PATTERN.test(sha)) {
      record("artifact-hashes", false, `artifacts[${index}].sha256 is not a 64-char SHA-256 hex digest`)
      return
    }
    const candidate = resolveArtifact(artifact?.path, artifactRoot, manifest)
    if (candidate && fs.existsSync(candidate)) {
      if (sha256File(candidate) !== sha) {
        record("artifact-hashes", false, `artifacts[${index}].sha256 mismatch for ${artifact.path} — content does not match the declared hash`)
        return
      }
      verified += 1
    } else {
      declaredOnly += 1
    }
  }
  const detail =
    verified > 0 && declaredOnly === 0
      ? `all ${verified} artifact hash(es) recomputed and matched`
      : declaredOnly > 0 && verified === 0
        ? `${declaredOnly} artifact hash(es) declared; raw files not shipped (expected for customer-safe — re-run the producer to reproduce)`
        : `${verified} verified, ${declaredOnly} declared-only`
  record("artifact-hashes", true, detail, declaredOnly > 0 && verified === 0 ? "info" : "error")
}

function resolveArtifact(artifactPath, artifactRoot, manifest) {
  if (!isLocalPath(artifactPath)) return null
  if (artifactRoot) {
    const byRoot = path.resolve(artifactRoot, path.basename(artifactPath))
    if (fs.existsSync(byRoot)) return byRoot
    return path.resolve(artifactRoot, artifactPath)
  }
  if (path.isAbsolute(artifactPath)) return artifactPath
  return path.resolve(process.cwd(), artifactPath)
}

function verifyRedaction(manifest, record) {
  const confidentiality = manifest.confidentiality
  const text = JSON.stringify(manifest)
  const leaks = []
  const absolute = text.match(ABSOLUTE_PATH_PATTERN)
  if (absolute) leaks.push(`absolute filesystem path (${absolute[0].trim().slice(0, 48)})`)
  const host = text.match(PRIVATE_HOST_PATTERN)
  if (host) leaks.push(`private host reference (${host[0]})`)

  if (CUSTOMER_FACING.has(confidentiality)) {
    record(
      "redaction",
      leaks.length === 0,
      leaks.length === 0
        ? `confidentiality is ${confidentiality} and no operator-only detail leaked`
        : `confidentiality is ${confidentiality} but packet leaks operator-only detail: ${leaks.join("; ")}`,
    )
  } else {
    record(
      "redaction",
      true,
      `confidentiality is ${confidentiality ?? "unset"}; redaction check applies only to customer-facing tiers`,
      "info",
    )
  }
}

function verifyRowProof(manifest, record) {
  const campaignId = manifest.campaign?.campaignId
  const modelTables = new Set(Array.isArray(manifest.store?.modelTables) ? manifest.store.modelTables : [])
  const rowProof = Array.isArray(manifest.store?.rowProof) ? manifest.store.rowProof : []
  if (rowProof.length === 0) {
    record("row-proof", true, "no row proof present (store-backed proof optional for a producer packet)", "info")
    return
  }
  for (const [index, proof] of rowProof.entries()) {
    if (proof?.campaignId !== campaignId) {
      record("row-proof", false, `store.rowProof[${index}].campaignId does not match campaign.campaignId`)
      return
    }
    if (modelTables.size > 0 && !modelTables.has(proof?.table)) {
      record("row-proof", false, `store.rowProof[${index}].table is not listed in store.modelTables`)
      return
    }
    if (!Number.isFinite(Number(proof?.rowCount)) || Number(proof.rowCount) < 1) {
      record("row-proof", false, `store.rowProof[${index}].rowCount must be >= 1`)
      return
    }
  }
  record("row-proof", true, `${rowProof.length} row-proof entr${rowProof.length === 1 ? "y" : "ies"} consistent with the campaign`)
}

function finalize(checks, meta) {
  const errors = checks.filter((c) => !c.ok && c.severity === "error").map((c) => c.detail)
  const warnings = checks.filter((c) => c.severity === "warning").map((c) => c.detail)
  const ok = errors.length === 0
  const summary = ok
    ? `PASS — packet is well-formed, ${meta.runClass ?? "unknown-class"}, within the ${meta.freshnessMaxDays}d freshness window${warnings.length ? ` (${warnings.length} warning${warnings.length === 1 ? "" : "s"})` : ""}`
    : `FAIL — ${errors.length} blocking issue${errors.length === 1 ? "" : "s"}: ${errors.join("; ")}`
  return { ok, ...meta, checks, errors, warnings, summary }
}
