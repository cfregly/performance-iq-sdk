import crypto from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it } from "vitest"

import { DEAL_PACKET_VERSION } from "./countersign.mjs"
import { verifyPacket, extractManifest, PRODUCER_MANIFEST_VERSION } from "./verify-packet.mjs"

const tmpDirs = []

function tmpArtifact(contents = "{\"tokens_per_second\":842.5}\n") {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "piq-verify-test-"))
  tmpDirs.push(dir)
  const file = path.join(dir, "raw.log")
  fs.writeFileSync(file, contents)
  const sha256 = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex")
  return { dir, file, sha256, sizeBytes: fs.statSync(file).size, contents }
}

function fixtureSha256(label) {
  return crypto.createHash("sha256").update(label).digest("hex")
}

// A fresh, well-formed, measured, operator-full packet with a real local
// artifact whose declared hash matches — the "valid packet" fixture.
function validMeasuredPacket(nowIso = new Date(NOW).toISOString()) {
  const artifact = tmpArtifact()
  return {
    packet: {
      schemaVersion: PRODUCER_MANIFEST_VERSION,
      runClass: "measured",
      sourceType: "fresh-run",
      generatedAtUtc: nowIso,
      producer: {
        repo: "producer-runner",
        tool: "runner",
        version: "1.4.0",
        commitSha: "a1b2c3d4e5f6a7b8",
        operator: "founder",
      },
      campaign: {
        campaignId: "deal-acme-h200-2026-07",
        runId: "run-9f2c",
        slug: "acme-h200",
        capturedAtUtc: nowIso,
        completedAtUtc: nowIso,
        publishedAtUtc: nowIso,
      },
      workload: {
        model: "llama-3.1-70b",
        hardware: "NVIDIA H200 SXM",
        operatingPoint: "c=32, 1k in / 1k out",
      },
      runtime: {
        imageDigest: "sha256:" + fixtureSha256("runner-h200-1.4.0"),
        imageTag: "runner-h200-1.4.0",
        framework: "vLLM",
      },
      artifacts: [
        { kind: "raw-log", path: artifact.file, sha256: artifact.sha256, sizeBytes: artifact.sizeBytes },
      ],
      store: {
        modelTables: ["model_store.report_performance_iq_price_perf"],
        rowProof: [
          { table: "model_store.report_performance_iq_price_perf", campaignId: "deal-acme-h200-2026-07", rowCount: 12 },
        ],
      },
      platform: { decisionBriefPath: "deal-acme-h200/brief.md" },
      methodology: "Single measured Producer Runner run replayable from producer-runner@a1b2c3d.",
      limitations: ["Single operating point.", "One node."],
      confidentiality: "operator-full",
    },
    artifact,
  }
}

const NOW = new Date("2026-07-09T00:00:00Z").getTime()

afterEach(() => {
  while (tmpDirs.length) {
    fs.rmSync(tmpDirs.pop(), { recursive: true, force: true })
  }
})

describe("verifyPacket", () => {
  it("fixture 1 — valid packet passes with hashes recomputed", () => {
    const { packet } = validMeasuredPacket()
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(true)
    expect(result.errors).toEqual([])
    expect(result.checks.find((c) => c.name === "artifact-hashes").detail).toMatch(/recomputed and matched/)
    expect(result.summary).toMatch(/^PASS/)
  })

  it("fixture 2 — stale packet fails on freshness", () => {
    const staleIso = new Date(NOW - 45 * 24 * 60 * 60 * 1000).toISOString()
    const { packet } = validMeasuredPacket(staleIso)
    packet.campaign.completedAtUtc = staleIso
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/stale|freshness window/)
    // and it passes again with a wider window
    expect(verifyPacket(packet, { now: NOW, freshnessMaxDays: 60 }).ok).toBe(true)
  })

  it("fixture 3 — artifact hash mismatch fails", () => {
    const { packet } = validMeasuredPacket()
    packet.artifacts[0].sha256 = "0".repeat(64) // wrong hash for a file that exists
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/sha256 mismatch/)
  })

  it("fixture 4 — redaction failure in a customer-facing tier", () => {
    const { packet } = validMeasuredPacket()
    packet.confidentiality = "customer-safe"
    // customer-safe must not ship absolute paths or private hosts
    packet.platform.dashboardUrl = "http://localhost:3001"
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/leaks operator-only detail/)
  })

  it("fixture 5 — schema/structure failure", () => {
    const { packet } = validMeasuredPacket()
    delete packet.runtime // required
    packet.schemaVersion = "performance-iq.producer-manifest.v0"
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/schemaVersion must be|missing required fields/)
  })

  it("rejects rehearsal packets by default, inspects them with --allow-rehearsal", () => {
    const { packet } = validMeasuredPacket()
    packet.runClass = "rehearsal"
    expect(verifyPacket(packet, { now: NOW }).ok).toBe(false)
    const relaxed = verifyPacket(packet, { now: NOW, requireMeasured: false })
    expect(relaxed.checks.find((c) => c.name === "evidence-class").severity).toBe("info")
    expect(relaxed.ok).toBe(true)
  })

  it("rejects packets carrying placeholder/example markers", () => {
    const { packet } = validMeasuredPacket()
    packet.campaign.campaignId = "example-campaign-do-not-quote"
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/placeholder/)
  })

  it("treats a customer-safe packet with stripped raw artifacts as declared-only, not a failure", () => {
    const { packet } = validMeasuredPacket()
    packet.confidentiality = "customer-safe"
    // customer-safe ships the hash but not the raw file or its local path
    packet.artifacts = [{ kind: "raw-log", path: "artifacts/raw.log", sha256: fixtureSha256("redacted-raw-log"), sizeBytes: 4096 }]
    packet.platform = { decisionBriefPath: "brief.md" }
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(true)
    expect(result.checks.find((c) => c.name === "artifact-hashes").detail).toMatch(/declared/)
  })

  it("flags an incomplete replay recipe", () => {
    const { packet } = validMeasuredPacket()
    packet.runtime.imageDigest = "not-a-digest"
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/replay recipe incomplete/)
  })

  it("extractManifest unwraps an envelope carrying a manifest", () => {
    const { packet } = validMeasuredPacket()
    expect(extractManifest({ manifest: packet })).toBe(packet)
    expect(extractManifest({ nope: true })).toBeNull()
  })

  it("verifies a deal-packet envelope with scope and operator-attested census", () => {
    const { packet } = validMeasuredPacket()
    const envelope = {
      schemaVersion: DEAL_PACKET_VERSION,
      manifest: packet,
      packetScope: {
        testedConfigurationOnly: true,
        buyerQuestion: "Which H200 operating point should ACME quote?",
        workloadWindow: "2026-07-09T00:00:00Z/2026-07-09T12:00:00Z",
        notAServiceCommitment: true,
        limitationsEcho: packet.limitations,
      },
      campaignCensus: {
        campaignsRunForDeal: 1,
        exportedFromCampaign: packet.campaign.campaignId,
        attestedBy: "operator",
      },
    }
    const result = verifyPacket(envelope, { now: NOW })
    expect(result.ok).toBe(true)
    expect(result.checks.find((c) => c.name === "packet-scope").ok).toBe(true)
    expect(result.checks.find((c) => c.name === "campaign-census").detail).toMatch(/provider-attested, not system-enforced/)
  })

  it("requires a deal-packet receipt when --require-countersignature is set", () => {
    const { packet } = validMeasuredPacket()
    const result = verifyPacket(packet, { now: NOW, requireCountersignature: true })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/requires a performance-iq\.deal-packet\.v1 envelope/)
  })

  it("hard-fails placeholder runtime digests in quote-grade paths", () => {
    const { packet } = validMeasuredPacket()
    packet.runtime.imageDigest = "sha256:" + "f".repeat(64)
    const result = verifyPacket(packet, { now: NOW })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/placeholder evidence/)
  })

  it("only warns on placeholder evidence during legacy non-quote inspection", () => {
    const { packet } = validMeasuredPacket()
    packet.runClass = "rehearsal"
    packet.runtime.imageDigest = "sha256:" + "f".repeat(64)
    const result = verifyPacket(packet, { now: NOW, requireMeasured: false })
    expect(result.ok).toBe(true)
    expect(result.warnings.join(" ")).toMatch(/legacy inspection warning/)
  })
})
