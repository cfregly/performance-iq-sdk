import crypto from "node:crypto"

import { describe, expect, it } from "vitest"

import {
  DEMO_SIGNATURE_MODE,
  DEAL_PACKET_VERSION,
  attachCountersignature,
  buildCountersignRequest,
  buildTransparencyLogEntry,
  canonicalPacketDigest,
  signCountersignature,
  verifyCountersignature,
} from "./countersign.mjs"

const NOW = "2026-07-09T12:00:00.000Z"

function keypair() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519")
  return {
    publicKey,
    publicKeyB64: publicKey.export({ format: "der", type: "spki" }).toString("base64"),
    privateKey,
  }
}

function manifest() {
  return {
    schemaVersion: "performance-iq.producer-manifest.v1",
    runClass: "measured",
    sourceType: "fresh-run",
    generatedAtUtc: NOW,
    producer: { repo: "producer-runner", tool: "runner", commitSha: "abc123def456" },
    campaign: {
      campaignId: "deal-acme-h200-2026-07",
      runId: "run-1",
      capturedAtUtc: NOW,
      completedAtUtc: NOW,
    },
    workload: { model: "llama-3.1-70b", hardware: "H200", operatingPoint: "c=32" },
    runtime: { imageDigest: `sha256:${"0123456789abcdef".repeat(4)}`, imageTag: "runner-h200-1.4.0" },
    artifacts: [{ kind: "raw-log", path: "raw.log", sha256: "1234567890abcdef".repeat(4), sizeBytes: 10 }],
    store: {
      sourceTables: ["platform_store_nessie.performance_iq_events"],
      modelTables: ["model_store.report_performance_iq_price_perf"],
      rowProof: [{ table: "model_store.report_performance_iq_price_perf", campaignId: "deal-acme-h200-2026-07", rowCount: 1 }],
    },
    platform: { decisionBriefPath: "performance-iq-pack.md" },
    methodology: "Measured run.",
    limitations: ["One operating point."],
    confidentiality: "operator-full",
  }
}

function packet() {
  return {
    schemaVersion: DEAL_PACKET_VERSION,
    manifest: manifest(),
    packetScope: {
      testedConfigurationOnly: true,
      buyerQuestion: "Which H200 operating point should ACME quote?",
      workloadWindow: "2026-07-09T00:00:00Z/2026-07-09T12:00:00Z",
      notAServiceCommitment: true,
      limitationsEcho: ["One operating point."],
    },
    campaignCensus: {
      campaignsRunForDeal: 1,
      exportedFromCampaign: "deal-acme-h200-2026-07",
      attestedBy: "operator",
    },
    buyerWorkloadAttestation: {
      specFileName: "acme-workload.json",
      specSha256: "abcdef1234567890".repeat(4),
    },
  }
}

function signedPacket(mode = "vendor-ed25519") {
  const keys = keypair()
  const unsigned = packet()
  const receipt = signCountersignature(unsigned, {
    privateKey: keys.privateKey,
    keyId: "piq-test-key-1",
    signedAtUtc: NOW,
    mode,
  })
  const logEntry = buildTransparencyLogEntry(unsigned, receipt, {
    seq: 1,
    tenantIdHash: "1111222233334444".repeat(4),
  })
  return {
    unsigned,
    signed: attachCountersignature(unsigned, { ...receipt, transparencyLog: logEntry }),
    receipt,
    logMirror: `${JSON.stringify(logEntry)}\n`,
    ...keys,
  }
}

describe("countersignature helpers", () => {
  it("builds a request from the canonical packet digest", () => {
    const unsigned = packet()
    const request = buildCountersignRequest(unsigned, {
      keyId: "piq-test-key-1",
      tenantIdHash: "1111222233334444".repeat(4),
      requestedAtUtc: NOW,
    })
    expect(request.schemaVersion).toBe("performance-iq.countersign-request.v1")
    expect(request.packetDigest).toBe(canonicalPacketDigest(unsigned))
    expect(request.digestAlgorithm).toBe("sha256")
  })

  it("validates a vendor receipt offline against a cloned hash-chain mirror", () => {
    const { signed, publicKeyB64, logMirror } = signedPacket()
    const result = verifyCountersignature(signed, { publicKey: publicKeyB64, logMirror })
    expect(result.ok).toBe(true)
    expect(result.errors).toEqual([])
  })

  it("fails when the manifest or envelope is tampered after signing", () => {
    const { signed, publicKeyB64, logMirror } = signedPacket()
    signed.manifest.workload.hardware = "GB200"
    const result = verifyCountersignature(signed, { publicKey: publicKeyB64, logMirror })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/verification failed|not found/)
  })

  it("fails with the wrong public key", () => {
    const { signed, logMirror } = signedPacket()
    const wrong = keypair()
    const result = verifyCountersignature(signed, { publicKey: wrong.publicKeyB64, logMirror })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/verification failed/)
  })

  it("rejects demo-self-signed receipts unless explicitly allowed", () => {
    const { signed, publicKeyB64, logMirror } = signedPacket(DEMO_SIGNATURE_MODE)
    expect(verifyCountersignature(signed, { publicKey: publicKeyB64, logMirror }).ok).toBe(false)
    expect(verifyCountersignature(signed, { publicKey: publicKeyB64, logMirror, allowDemo: true }).ok).toBe(true)
  })

  it("fails closed when the hash-chain log entry is tampered", () => {
    const { signed, publicKeyB64, logMirror } = signedPacket()
    const tampered = JSON.parse(logMirror)
    tampered.packetDigest = "0".repeat(64)
    const result = verifyCountersignature(signed, { publicKey: publicKeyB64, logMirror: `${JSON.stringify(tampered)}\n` })
    expect(result.ok).toBe(false)
    expect(result.errors.join(" ")).toMatch(/digest mismatch|not found/)
  })
})
