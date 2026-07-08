import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

import {
  buildManifest,
  PerformanceIQ,
  validateRun,
  type PerformanceIQRunInput,
} from "./index"

const tmpDirs: string[] = []

function tmpArtifact(contents = "{\"ok\":true}\n"): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-sdk-test-"))
  tmpDirs.push(dir)
  const artifactPath = path.join(dir, "summary.json")
  fs.writeFileSync(artifactPath, contents)
  return artifactPath
}

function runInput(overrides: Partial<PerformanceIQRunInput> = {}): PerformanceIQRunInput {
  return {
    sourceType: "fresh-run",
    confidentiality: "internal-full",
    producer: {
      repo: "producer-runner",
      tool: "runner",
      commitSha: "1234567890abcdef",
    },
    campaign: {
      campaignId: "campaign-sdk-test",
      runId: "run-sdk-test",
    },
    workload: {
      model: "llama-3.1-70b",
      hardware: "B200 SXM",
      operatingPoint: "peak",
    },
    runtime: {
      imageDigest: `sha256:${"a".repeat(64)}`,
    },
    artifacts: [tmpArtifact()],
    measurements: [{ outputTpm: 1234 }],
    ...overrides,
  }
}

afterEach(() => {
  for (const dir of tmpDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
  vi.restoreAllMocks()
})

describe("performance-iq-sdk js", () => {
  it("builds a manifest with artifact hashes and pending SDK row proof", async () => {
    const manifest = await buildManifest(runInput())

    expect(manifest.schemaVersion).toBe("performance-iq.producer-manifest.v1")
    expect(manifest.sourceType).toBe("fresh-run")
    expect(manifest.artifacts[0]).toMatchObject({
      kind: "normalized-summary",
      sha256: "e5f1eb4d806641698a35efe20e098efd20d7d57a9b90ee69079d5bb650920726",
      sizeBytes: 12,
    })
    expect(manifest.store).toMatchObject({
      sourceTables: ["performance_iq.sdk_submission"],
      modelTables: ["model_store.sdk_pending_ingest"],
      rowProof: [{ campaignId: "campaign-sdk-test", rowCount: 1 }],
    })
  })

  it("validates source kind, required fields, and live proof classification", async () => {
    const result = await validateRun(runInput())

    expect(result.ok).toBe(true)
    expect(result.liveProofReady).toBe(true)
    expect(result.freshRun).toBe(true)
    expect(result.snapshotBacked).toBe(false)
  })

  it("fails closed for customer-safe submissions until governance is implemented", async () => {
    const result = await validateRun(runInput({ confidentiality: "customer-safe" }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("internal-full")
  })

  it("rejects arbitrary SQL/query keys anywhere in the payload", async () => {
    const result = await validateRun(runInput({
      measurements: [{ sql: "SELECT * FROM model_store.secret" }],
    }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("sql")
  })

  it("reports missing local artifacts during local validation", async () => {
    const result = await validateRun(runInput({ artifacts: ["/missing/performance-iq-artifact.json"] }))

    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toContain("ENOENT")
  })

  it("submits validated runs with auth and idempotency headers", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ id: "run-sdk-test", status: "accepted" }), {
      status: 202,
      headers: { "content-type": "application/json" },
    }))
    const client = new PerformanceIQ({
      baseUrl: "https://performance-iq.example",
      token: "service-token",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })

    const result = await client.submitRun(runInput(), { idempotencyKey: "idem-1" })

    expect(result).toEqual({ id: "run-sdk-test", status: "accepted" })
    expect(fetchImpl).toHaveBeenCalledWith("https://performance-iq.example/api/v1/runs", expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({
        authorization: "Bearer service-token",
        "idempotency-key": "idem-1",
      }),
    }))
  })
})
