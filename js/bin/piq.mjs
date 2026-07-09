#!/usr/bin/env node

import crypto from "node:crypto"
import fs from "node:fs/promises"

import { attachCountersignature, buildCountersignRequest } from "../src/countersign.mjs"
import { verifyPacket, DEFAULT_FRESHNESS_MAX_DAYS } from "../src/verify-packet.mjs"

const [, , command, ...args] = process.argv

function usage() {
  console.error(`Usage:
  piq validate <manifest.json>          (server-side validation; needs PIQ_BASE_URL)
  piq submit-manifest <manifest.json>   (needs PIQ_BASE_URL + PIQ_TOKEN)
  piq status <run-id>                   (needs PIQ_BASE_URL)
  piq verify-packet <packet.json>       (OFFLINE buyer verification; no server, no token)
  piq countersign request <packet.json> (emit countersign-request.json to stdout)
  piq countersign attach <packet.json> <receipt.json> (emit packet with receipt attached)

verify-packet options:
  --max-age-days <n>          freshness window (default ${DEFAULT_FRESHNESS_MAX_DAYS})
  --allow-rehearsal           inspect non-measured packets instead of rejecting them
  --require-countersignature  require a valid vendor countersignature
  --allow-demo                accept demo-self-signed receipts for tests only
  --public-key <key>          Ed25519 public key as PEM or base64 DER (default PIQ_COUNTERSIGN_PUBLIC_KEY_B64)
  --log-mirror <path>         local JSONL transparency-log mirror
  --artifacts <dir>           directory holding raw artifacts, to recompute hashes
  --json                      print the full machine-readable result

countersign request options:
  --key-id <id>               requested signing key id
  --tenant-id-hash <sha256>   precomputed salted tenant hash

Environment:
  PIQ_BASE_URL   Performance IQ base URL (server commands only)
  PIQ_TOKEN      Service token for write APIs
  PIQ_COUNTERSIGN_PUBLIC_KEY_B64   verifier public key, never a private key
  PIQ_COUNTERSIGN_KEY_ID           default countersign key id`)
  process.exit(2)
}

function takeFlagValue(argv, name) {
  const index = argv.indexOf(name)
  if (index === -1) return undefined
  const value = argv[index + 1]
  argv.splice(index, 2)
  return value
}

function takeFlag(argv, name) {
  const index = argv.indexOf(name)
  if (index === -1) return false
  argv.splice(index, 1)
  return true
}

async function runVerifyPacket(argv) {
  const rest = [...argv]
  const maxAge = takeFlagValue(rest, "--max-age-days")
  const allowRehearsal = takeFlag(rest, "--allow-rehearsal")
  const requireCountersignature = takeFlag(rest, "--require-countersignature")
  const allowDemoSignature = takeFlag(rest, "--allow-demo")
  const countersignaturePublicKey = takeFlagValue(rest, "--public-key") ?? process.env.PIQ_COUNTERSIGN_PUBLIC_KEY_B64 ?? null
  const transparencyLogMirror = takeFlagValue(rest, "--log-mirror") ?? process.env.PIQ_TRANSPARENCY_LOG_PATH ?? null
  const artifactRoot = takeFlagValue(rest, "--artifacts")
  const asJson = takeFlag(rest, "--json")
  const [packetPath] = rest
  if (!packetPath) usage()

  const packet = JSON.parse(await fs.readFile(packetPath, "utf8"))
  const result = verifyPacket(packet, {
    freshnessMaxDays: maxAge !== undefined ? Number(maxAge) : DEFAULT_FRESHNESS_MAX_DAYS,
    requireMeasured: !allowRehearsal,
    artifactRoot: artifactRoot ?? null,
    requireCountersignature,
    allowDemoSignature,
    countersignaturePublicKey,
    transparencyLogMirror,
  })

  if (asJson) {
    console.log(JSON.stringify(result, null, 2))
  } else {
    for (const check of result.checks) {
      const mark = check.ok ? (check.severity === "warning" ? "!" : "✓") : "✗"
      console.log(`  ${mark} ${check.name}: ${check.detail}`)
    }
    console.log("")
    console.log(result.summary)
  }
  process.exit(result.ok ? 0 : 1)
}

async function runCountersign(argv) {
  const [subcommand, ...subArgs] = argv
  if (subcommand === "request") {
    const rest = [...subArgs]
    const keyId = takeFlagValue(rest, "--key-id") ?? process.env.PIQ_COUNTERSIGN_KEY_ID
    const tenantIdHash = takeFlagValue(rest, "--tenant-id-hash") ?? process.env.PIQ_TENANT_ID_HASH
    const [packetPath] = rest
    if (!packetPath) usage()
    const packet = JSON.parse(await fs.readFile(packetPath, "utf8"))
    console.log(JSON.stringify(buildCountersignRequest(packet, { keyId, tenantIdHash }), null, 2))
    return
  }
  if (subcommand === "attach") {
    const [packetPath, receiptPath] = subArgs
    if (!packetPath || !receiptPath) usage()
    const packet = JSON.parse(await fs.readFile(packetPath, "utf8"))
    const receipt = JSON.parse(await fs.readFile(receiptPath, "utf8"))
    console.log(JSON.stringify(attachCountersignature(packet, receipt), null, 2))
    return
  }
  usage()
}

function baseUrl() {
  const value = process.env.PIQ_BASE_URL
  if (!value) throw new Error("PIQ_BASE_URL is required")
  return value.replace(/\/+$/, "")
}

function headers(extra = {}) {
  return {
    "content-type": "application/json",
    ...(process.env.PIQ_TOKEN ? { authorization: `Bearer ${process.env.PIQ_TOKEN}` } : {}),
    ...extra,
  }
}

async function readError(response) {
  const text = await response.text().catch(() => "")
  try {
    const body = JSON.parse(text)
    return body.detail ?? body.error ?? body.message ?? response.statusText
  } catch {
    return text.trim() || response.statusText
  }
}

async function post(path, body, extraHeaders = {}) {
  const response = await fetch(`${baseUrl()}${path}`, {
    method: "POST",
    headers: headers(extraHeaders),
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

async function get(path) {
  const response = await fetch(`${baseUrl()}${path}`, { headers: headers() })
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

async function main() {
  if (!command) usage()
  if (command === "verify-packet") {
    await runVerifyPacket(args)
    return
  }
  if (command === "countersign") {
    await runCountersign(args)
    return
  }
  if (command === "validate") {
    const [manifestPath] = args
    if (!manifestPath) usage()
    const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"))
    const result = await post("/api/v1/runs/validate", { manifest })
    console.log(JSON.stringify(result, null, 2))
    process.exit(result.ok ? 0 : 1)
  }
  if (command === "submit-manifest") {
    const [manifestPath] = args
    if (!manifestPath) usage()
    const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"))
    const result = await post(
      "/api/v1/runs",
      { schemaVersion: "performance-iq.ingestion-request.v1", manifest },
      { "idempotency-key": manifest?.campaign?.runId ?? crypto.randomUUID() },
    )
    console.log(JSON.stringify(result, null, 2))
    return
  }
  if (command === "status") {
    const [runId] = args
    if (!runId) usage()
    console.log(JSON.stringify(await get(`/api/v1/runs/${encodeURIComponent(runId)}`), null, 2))
    return
  }
  usage()
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err))
  process.exit(1)
})
