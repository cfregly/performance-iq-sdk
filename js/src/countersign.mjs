import crypto from "node:crypto"
import fs from "node:fs"

export const DEAL_PACKET_VERSION = "performance-iq.deal-packet.v1"
export const COUNTERSIGN_REQUEST_VERSION = "performance-iq.countersign-request.v1"
export const COUNTERSIGN_LOG_VERSION = "performance-iq.transparency-log-entry.v1"
export const DIGEST_ALGORITHM = "sha256"
export const VENDOR_SIGNATURE_MODE = "vendor-ed25519"
export const DEMO_SIGNATURE_MODE = "demo-self-signed"

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

export function canonicalJson(value) {
  if (value === null) return "null"
  if (typeof value === "string") return JSON.stringify(value)
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("canonical JSON cannot encode non-finite numbers")
    return JSON.stringify(value)
  }
  if (typeof value === "boolean") return value ? "true" : "false"
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`
  if (isObject(value)) {
    const entries = Object.keys(value)
      .filter((key) => value[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    return `{${entries.join(",")}}`
  }
  throw new Error(`canonical JSON cannot encode ${typeof value}`)
}

export function canonicalPacketForDigest(packet) {
  if (!isObject(packet)) throw new Error("deal packet must be an object")
  const { countersignature: _countersignature, transparencyLog: _transparencyLog, ...digestPacket } = packet
  return digestPacket
}

export function canonicalPacketDigest(packet) {
  return crypto
    .createHash(DIGEST_ALGORITHM)
    .update(canonicalJson(canonicalPacketForDigest(packet)))
    .digest("hex")
}

export function countersignSigningPayload(packet, receipt = {}) {
  return {
    schemaVersion: COUNTERSIGN_REQUEST_VERSION,
    packetSchemaVersion: packet?.schemaVersion,
    packetDigest: canonicalPacketDigest(packet),
    digestAlgorithm: DIGEST_ALGORITHM,
    keyId: receipt.keyId,
  }
}

export function buildCountersignRequest(packet, options = {}) {
  return {
    schemaVersion: COUNTERSIGN_REQUEST_VERSION,
    packetSchemaVersion: packet?.schemaVersion,
    packetDigest: canonicalPacketDigest(packet),
    digestAlgorithm: DIGEST_ALGORITHM,
    keyId: options.keyId,
    tenantIdHash: options.tenantIdHash,
    requestedAtUtc: options.requestedAtUtc ?? new Date().toISOString(),
  }
}

function signatureBytes(signature) {
  if (Buffer.isBuffer(signature)) return signature
  if (typeof signature !== "string" || signature.length === 0) throw new Error("countersignature.signature is required")
  return Buffer.from(signature, "base64")
}

function keyInput(key) {
  if (!key) throw new Error("public key is required")
  if (typeof key !== "string") return key
  const trimmed = key.trim()
  if (trimmed.includes("BEGIN PUBLIC KEY")) return trimmed
  return crypto.createPublicKey({
    key: Buffer.from(trimmed, "base64"),
    format: "der",
    type: "spki",
  })
}

export function signCountersignature(packet, { privateKey, keyId, signedAtUtc, mode = VENDOR_SIGNATURE_MODE } = {}) {
  if (!privateKey) throw new Error("privateKey is required")
  if (!keyId) throw new Error("keyId is required")
  const receipt = {
    keyId,
    mode,
    signedAtUtc: signedAtUtc ?? new Date().toISOString(),
  }
  const payload = Buffer.from(canonicalJson(countersignSigningPayload(packet, receipt)))
  return {
    ...receipt,
    signature: crypto.sign(null, payload, privateKey).toString("base64"),
  }
}

export function attachCountersignature(packet, receipt) {
  if (!isObject(packet)) throw new Error("deal packet must be an object")
  if (!isObject(receipt)) throw new Error("receipt must be an object")
  const { transparencyLog, ...signatureFields } = receipt
  return {
    ...packet,
    countersignature: signatureFields,
    ...(transparencyLog ? { transparencyLog } : {}),
  }
}

export function computeLogEntryDigest(entry) {
  const { entryDigest: _entryDigest, ...digestEntry } = entry
  return crypto.createHash(DIGEST_ALGORITHM).update(canonicalJson(digestEntry)).digest("hex")
}

export function buildTransparencyLogEntry(packet, receipt, options = {}) {
  const entry = {
    schemaVersion: COUNTERSIGN_LOG_VERSION,
    seq: options.seq,
    prevEntryDigest: options.prevEntryDigest ?? null,
    packetDigest: canonicalPacketDigest(packet),
    digestAlgorithm: DIGEST_ALGORITHM,
    keyId: receipt.keyId,
    signedAtUtc: receipt.signedAtUtc,
    tenantIdHash: options.tenantIdHash,
    ...(options.dealIdHash ? { dealIdHash: options.dealIdHash } : {}),
  }
  return {
    ...entry,
    entryDigest: computeLogEntryDigest(entry),
  }
}

function loadLogEntries(logMirror) {
  if (!logMirror) return []
  const text = typeof logMirror === "string" && fs.existsSync(logMirror)
    ? fs.readFileSync(logMirror, "utf8")
    : String(logMirror)
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line))
}

export function verifyHashChain(logMirror, expected) {
  const entries = loadLogEntries(logMirror)
  if (entries.length === 0) {
    return { ok: false, errors: ["transparency log mirror is empty"], entry: null }
  }

  let prev = null
  let matched = null
  const errors = []
  for (const [index, entry] of entries.entries()) {
    const computed = computeLogEntryDigest(entry)
    if (entry.entryDigest !== computed) {
      errors.push(`log entry ${index} digest mismatch`)
      break
    }
    if ((entry.prevEntryDigest ?? null) !== prev) {
      errors.push(`log entry ${index} prevEntryDigest mismatch`)
      break
    }
    if (Number(entry.seq) !== index + 1) {
      errors.push(`log entry ${index} seq must be ${index + 1}`)
      break
    }
    if (
      entry.packetDigest === expected.packetDigest
      && (!expected.entryDigest || entry.entryDigest === expected.entryDigest)
      && (!expected.keyId || entry.keyId === expected.keyId)
    ) {
      matched = entry
    }
    prev = entry.entryDigest
  }

  if (errors.length > 0) return { ok: false, errors, entry: matched }
  if (!matched) return { ok: false, errors: ["packet digest not found in transparency log mirror"], entry: null }
  return { ok: true, errors: [], entry: matched }
}

export function verifyCountersignature(packet, options = {}) {
  const receipt = packet?.countersignature
  if (!receipt) return { ok: false, errors: ["countersignature is missing"], warnings: [], digest: canonicalPacketDigest(packet) }

  const digest = canonicalPacketDigest(packet)
  const errors = []
  const warnings = []

  if (receipt.mode === DEMO_SIGNATURE_MODE && !options.allowDemo) {
    errors.push("demo-self-signed receipt is not accepted without --allow-demo")
  }

  if (!receipt.keyId) errors.push("countersignature.keyId is required")
  if (!receipt.signedAtUtc) errors.push("countersignature.signedAtUtc is required")
  if (!receipt.signature) errors.push("countersignature.signature is required")

  if (errors.length === 0) {
    try {
      const payload = Buffer.from(canonicalJson(countersignSigningPayload(packet, receipt)))
      const verified = crypto.verify(null, payload, keyInput(options.publicKey), signatureBytes(receipt.signature))
      if (!verified) errors.push("countersignature signature verification failed")
    } catch (err) {
      errors.push(err instanceof Error ? err.message : String(err))
    }
  }

  const logRef = packet?.transparencyLog
  if (options.logMirror) {
    const chain = verifyHashChain(options.logMirror, {
      packetDigest: digest,
      entryDigest: logRef?.entryDigest,
      keyId: receipt.keyId,
    })
    if (!chain.ok) errors.push(...chain.errors)
  } else if (!logRef) {
    warnings.push("transparency log reference is missing")
  }

  if (logRef) {
    if (logRef.packetDigest && logRef.packetDigest !== digest) {
      errors.push("transparencyLog.packetDigest does not match canonical packet digest")
    }
    if (logRef.keyId && logRef.keyId !== receipt.keyId) {
      errors.push("transparencyLog.keyId does not match countersignature.keyId")
    }
  }

  return { ok: errors.length === 0, errors, warnings, digest }
}
