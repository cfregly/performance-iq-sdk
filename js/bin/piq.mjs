#!/usr/bin/env node

import crypto from "node:crypto"
import fs from "node:fs/promises"

const [, , command, ...args] = process.argv

function usage() {
  console.error(`Usage:
  piq validate <manifest.json>
  piq submit-manifest <manifest.json>
  piq status <run-id>

Environment:
  PIQ_BASE_URL   Performance IQ base URL
  PIQ_TOKEN      Service token for write APIs`)
  process.exit(2)
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
