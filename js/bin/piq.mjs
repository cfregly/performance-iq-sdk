#!/usr/bin/env node

import crypto from "node:crypto"
import fs from "node:fs/promises"

import { verifyPacket, DEFAULT_FRESHNESS_MAX_DAYS } from "../src/verify-packet.mjs"

const [, , command, ...args] = process.argv

function usage() {
  console.error(`Usage:
  piq validate <manifest.json>          (server-side validation; needs PIQ_BASE_URL)
  piq submit-manifest <manifest.json>   (needs PIQ_BASE_URL + PIQ_TOKEN)
  piq status <run-id>                   (needs PIQ_BASE_URL)
  piq verify-packet <packet.json>       (OFFLINE buyer verification; no server, no token)

verify-packet options:
  --max-age-days <n>   freshness window (default ${DEFAULT_FRESHNESS_MAX_DAYS})
  --allow-rehearsal    inspect non-measured packets instead of rejecting them
  --artifacts <dir>    directory holding raw artifacts, to recompute hashes
  --json               print the full machine-readable result

Environment:
  PIQ_BASE_URL   Performance IQ base URL (server commands only)
  PIQ_TOKEN      Service token for write APIs`)
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
  const artifactRoot = takeFlagValue(rest, "--artifacts")
  const asJson = takeFlag(rest, "--json")
  const [packetPath] = rest
  if (!packetPath) usage()

  const packet = JSON.parse(await fs.readFile(packetPath, "utf8"))
  const result = verifyPacket(packet, {
    freshnessMaxDays: maxAge !== undefined ? Number(maxAge) : DEFAULT_FRESHNESS_MAX_DAYS,
    requireMeasured: !allowRehearsal,
    artifactRoot: artifactRoot ?? null,
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
