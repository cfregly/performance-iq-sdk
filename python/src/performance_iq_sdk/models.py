from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
from typing import Any, Literal, NotRequired, TypedDict

PRODUCER_MANIFEST_VERSION = "performance-iq.producer-manifest.v1"
INGESTION_REQUEST_VERSION = "performance-iq.ingestion-request.v1"

SourceType = Literal["preserved-snapshot", "fresh-run", "other-measured-producer"]
RunClass = Literal["measured", "rehearsal", "simulated"]
Confidentiality = Literal["internal-full", "customer-safe", "public-safe", "redacted"]


class ProducerIdentity(TypedDict):
    repo: str
    tool: str
    commitSha: str
    version: NotRequired[str]
    operator: NotRequired[str]


class CampaignIdentity(TypedDict):
    campaignId: str
    runId: str
    slug: NotRequired[str]
    capturedAtUtc: NotRequired[str]
    completedAtUtc: NotRequired[str]
    publishedAtUtc: NotRequired[str]


class WorkloadIdentity(TypedDict):
    model: str
    hardware: str
    acceleratorVendor: NotRequired[str]
    acceleratorModel: NotRequired[str]
    acceleratorArchitecture: NotRequired[str]
    interconnect: NotRequired[str]
    operatingPoint: str
    scenario: NotRequired[str]
    precision: NotRequired[str]
    parallelism: NotRequired[str]
    datasetOrPromptSet: NotRequired[str]


class RuntimeIdentity(TypedDict):
    imageDigest: str
    imageTag: NotRequired[str]
    cudaVersion: NotRequired[str]
    ncclVersion: NotRequired[str]
    driverVersion: NotRequired[str]
    framework: NotRequired[str]


class ArtifactMetadata(TypedDict):
    kind: str
    path: str
    sha256: str
    sizeBytes: int


class RowProof(TypedDict):
    table: str
    rowCount: int | float
    campaignId: NotRequired[str]
    latestCapturedAtUtc: NotRequired[str]


class StoreProof(TypedDict):
    sourceTables: list[str]
    modelTables: list[str]
    rowProof: list[RowProof]


class PlatformReference(TypedDict):
    decisionBriefPath: str
    dashboardUrl: NotRequired[str]
    exportGeneratedAtUtc: NotRequired[str]
    preflightPath: NotRequired[str]


class PerformanceIQRunInput(TypedDict):
    sourceType: SourceType
    confidentiality: Confidentiality
    producer: ProducerIdentity
    campaign: CampaignIdentity
    workload: WorkloadIdentity
    runtime: RuntimeIdentity
    artifacts: list[str | dict[str, Any]]
    runClass: NotRequired[RunClass]
    measurements: NotRequired[list[dict[str, Any]]]
    store: NotRequired[StoreProof]
    platform: NotRequired[dict[str, Any]]
    methodology: NotRequired[str]
    limitations: NotRequired[list[str]]


IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r"\b(replace-with|example-only|do-not-quote|template only)\b", re.IGNORECASE)
DISALLOWED_REQUEST_KEYS = {"sql", "queryName", "queries"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _is_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _find_disallowed_key(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            match = _find_disallowed_key(item)
            if match:
                return match
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            if key in DISALLOWED_REQUEST_KEYS:
                return key
            match = _find_disallowed_key(child)
            if match:
                return match
    return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_artifact_kind(path: str) -> str:
    if path.endswith((".log", ".txt")):
        return "raw-log"
    if path.endswith((".json", ".jsonl")):
        return "normalized-summary"
    if path.endswith((".yaml", ".yml")):
        return "config-snapshot"
    return "artifact"


def _normalize_artifact(artifact: str | dict[str, Any]) -> ArtifactMetadata:
    raw: dict[str, Any] = {"path": artifact} if isinstance(artifact, str) else dict(artifact)
    path = raw["path"]
    stat = os.stat(path)
    return {
        "kind": raw.get("kind") or _default_artifact_kind(path),
        "path": path,
        "sha256": raw.get("sha256") or _sha256_file(path),
        "sizeBytes": raw.get("sizeBytes") if raw.get("sizeBytes") is not None else stat.st_size,
    }


def _default_store(input: PerformanceIQRunInput) -> StoreProof:
    row_count = max(len(input.get("measurements", [])), 1)
    table = "model_store.sdk_pending_ingest"
    return {
        "sourceTables": ["performance_iq.sdk_submission"],
        "modelTables": [table],
        "rowProof": [
            {
                "table": table,
                "campaignId": input["campaign"]["campaignId"],
                "rowCount": row_count,
                "latestCapturedAtUtc": input["campaign"].get("capturedAtUtc") or _now_iso(),
            }
        ],
    }


def build_manifest(input: PerformanceIQRunInput) -> dict[str, Any]:
    generated_at = _now_iso()
    campaign = dict(input["campaign"])
    campaign.setdefault("capturedAtUtc", generated_at)
    campaign.setdefault("completedAtUtc", campaign["capturedAtUtc"])

    store = input.get("store") or _default_store(input)
    row_proof = []
    for proof in store["rowProof"]:
        next_proof = dict(proof)
        next_proof.setdefault("campaignId", campaign["campaignId"])
        row_proof.append(next_proof)

    platform = dict(input.get("platform", {}))
    platform.setdefault("decisionBriefPath", "performance-iq://pending/decision-brief")

    return {
        "schemaVersion": PRODUCER_MANIFEST_VERSION,
        "runClass": input.get("runClass", "measured"),
        "sourceType": input["sourceType"],
        "generatedAtUtc": generated_at,
        "producer": dict(input["producer"]),
        "campaign": campaign,
        "workload": dict(input["workload"]),
        "runtime": dict(input["runtime"]),
        "artifacts": [_normalize_artifact(artifact) for artifact in input["artifacts"]],
        "store": {**store, "rowProof": row_proof},
        "platform": platform,
        "methodology": input.get("methodology") or "Submitted through the Performance IQ SDK.",
        "limitations": input.get("limitations") or ["No limitations were supplied by the producer."],
        "confidentiality": input["confidentiality"],
    }


def build_envelope(manifest: dict[str, Any], measurements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    envelope = {
        "schemaVersion": INGESTION_REQUEST_VERSION,
        "manifest": manifest,
    }
    if measurements is not None:
        envelope["measurements"] = measurements
    return envelope


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    disallowed = _find_disallowed_key(manifest)
    if disallowed:
        errors.append(f"payload must not include {disallowed}")

    if PLACEHOLDER_PATTERN.search(str(manifest)):
        errors.append("manifest contains placeholder, template, or example-only markers")
    if manifest.get("schemaVersion") != PRODUCER_MANIFEST_VERSION:
        errors.append(f"schemaVersion must be {PRODUCER_MANIFEST_VERSION}")
    if manifest.get("runClass") not in {"measured", "rehearsal", "simulated"}:
        errors.append("runClass must be measured, rehearsal, or simulated")
    if manifest.get("runClass") != "measured":
        warnings.append("manifest is accepted as non-live results only; live proof requires runClass=measured")
    if manifest.get("sourceType") not in {"preserved-snapshot", "fresh-run", "other-measured-producer"}:
        errors.append("sourceType is not supported")
    if manifest.get("confidentiality") != "internal-full":
        errors.append("only internal-full submissions are enabled; customer-safe, public-safe, and redacted remain fail-closed")

    producer = manifest.get("producer") or {}
    campaign = manifest.get("campaign") or {}
    workload = manifest.get("workload") or {}
    runtime = manifest.get("runtime") or {}
    store = manifest.get("store") or {}

    if not producer.get("repo"):
        errors.append("producer.repo is required")
    if not producer.get("tool"):
        errors.append("producer.tool is required")
    if len(producer.get("commitSha") or "") < 7:
        errors.append("producer.commitSha must contain at least 7 characters")
    if not campaign.get("campaignId"):
        errors.append("campaign.campaignId is required")
    if not campaign.get("runId"):
        errors.append("campaign.runId is required")
    if not _is_datetime(campaign.get("capturedAtUtc")):
        errors.append("campaign.capturedAtUtc must be a valid date-time")
    if not _is_datetime(campaign.get("completedAtUtc")):
        errors.append("campaign.completedAtUtc must be a valid date-time")
    if not workload.get("model"):
        errors.append("workload.model is required")
    if not workload.get("hardware"):
        errors.append("workload.hardware is required")
    if not workload.get("operatingPoint"):
        errors.append("workload.operatingPoint is required")
    if not IMAGE_DIGEST_PATTERN.match(runtime.get("imageDigest") or ""):
        errors.append("runtime.imageDigest must match sha256:<64 hex chars>")

    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    if not artifacts:
        errors.append("at least one artifact is required")
    for index, artifact in enumerate(artifacts):
        if not artifact.get("kind"):
            errors.append(f"artifacts[{index}].kind is required")
        if not artifact.get("path"):
            errors.append(f"artifacts[{index}].path is required")
        if not SHA256_PATTERN.match(artifact.get("sha256") or ""):
            errors.append(f"artifacts[{index}].sha256 must be a 64-character SHA-256 hex digest")
        if not isinstance(artifact.get("sizeBytes"), (int, float)) or artifact["sizeBytes"] < 0:
            errors.append(f"artifacts[{index}].sizeBytes must be >= 0")

    source_tables = store.get("sourceTables") if isinstance(store.get("sourceTables"), list) else []
    model_tables = store.get("modelTables") if isinstance(store.get("modelTables"), list) else []
    row_proof = store.get("rowProof") if isinstance(store.get("rowProof"), list) else []
    if not source_tables:
        errors.append("store.sourceTables must contain at least one table")
    if not model_tables:
        errors.append("store.modelTables must contain at least one table")
    if not row_proof:
        errors.append("store.rowProof must contain at least one row proof")
    for index, proof in enumerate(row_proof):
        if proof.get("campaignId") != campaign.get("campaignId"):
            errors.append(f"store.rowProof[{index}].campaignId must match campaign.campaignId")
        if proof.get("table") not in model_tables:
            errors.append(f"store.rowProof[{index}].table must be listed in store.modelTables")
        if not isinstance(proof.get("rowCount"), (int, float)) or proof["rowCount"] < 1:
            errors.append(f"store.rowProof[{index}].rowCount must be >= 1")

    source_kind = manifest.get("sourceType")
    live_proof_ready = not errors and manifest.get("runClass") == "measured" and source_kind == "fresh-run"
    return {
        "ok": not errors,
        "liveProofReady": live_proof_ready,
        "sourceType": source_kind,
        "snapshotBacked": source_kind == "preserved-snapshot",
        "freshRun": source_kind == "fresh-run",
        "errors": errors,
        "warnings": warnings,
        "manifest": manifest,
    }


def validate_run(input: PerformanceIQRunInput) -> dict[str, Any]:
    errors: list[str] = []
    disallowed = _find_disallowed_key(input)
    if disallowed:
        errors.append(f"payload must not include {disallowed}")
    try:
        manifest = build_manifest(input)
    except Exception as exc:  # local artifact errors should be surfaced cleanly
        return {
            "ok": False,
            "liveProofReady": False,
            "sourceType": input.get("sourceType"),
            "snapshotBacked": input.get("sourceType") == "preserved-snapshot",
            "freshRun": input.get("sourceType") == "fresh-run",
            "errors": [*errors, str(exc)],
            "warnings": [],
        }
    result = validate_manifest(manifest)
    result["errors"] = [*errors, *result["errors"]]
    result["ok"] = not result["errors"]
    return result
