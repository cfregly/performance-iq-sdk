from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request
from typing import Any, Callable, Literal, TypedDict

from performance_iq_sdk.client import PerformanceIQ
from performance_iq_sdk.models import PerformanceIQRunInput, build_manifest

ServingEngineId = Literal["vllm", "sglang", "tensorrt-llm"]

SERVING_ENGINE_LABELS: dict[str, str] = {
    "vllm": "vLLM",
    "sglang": "SGLang",
    "tensorrt-llm": "TensorRT-LLM",
}

DEFAULT_IMAGE_DIGEST = "sha256:" + "0" * 64


class ServingPostResult(TypedDict):
    status: int
    body: dict[str, Any]


HttpPostJson = Callable[[str, dict[str, str], dict[str, Any]], ServingPostResult]


def laptop_smoke_model() -> str:
    return "Qwen/Qwen2.5-0.5B-Instruct"


def serving_engine_label(engine: ServingEngineId | str) -> str:
    return SERVING_ENGINE_LABELS[str(engine)]


def _now_iso(now: Callable[[], dt.datetime] | None = None) -> str:
    value = now() if now else dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> ServingPostResult:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
        return {
            "status": response.status,
            "body": json.loads(body) if body else {},
        }


def _request_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": request["model"],
        "messages": request["messages"],
        "max_tokens": request.get("maxTokens", request.get("max_tokens", 64)),
        "temperature": request.get("temperature", 0),
    }
    top_p = request.get("topP", request.get("top_p"))
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


def _send_chat_completion(
    engine: dict[str, Any],
    request: dict[str, Any],
    request_index: int,
    http_post_json: HttpPostJson | None,
) -> dict[str, Any]:
    endpoint = f"{_normalize_base_url(engine['baseUrl'])}{engine.get('requestPath', '/v1/chat/completions')}"
    headers = {"content-type": "application/json"}
    if engine.get("apiKey"):
        headers["authorization"] = f"Bearer {engine['apiKey']}"
    started = time.perf_counter()
    try:
        result = (http_post_json or _post_json)(endpoint, headers, _request_payload(request))
        latency_ms = (time.perf_counter() - started) * 1000
        body = result.get("body", {})
        usage = body.get("usage") or {}
        status = int(result.get("status", 0))
        return {
            "requestIndex": request_index,
            "status": status,
            "ok": 200 <= status < 300,
            "latencyMs": latency_ms,
            "promptTokens": int(usage.get("prompt_tokens", usage.get("promptTokens", 0)) or 0),
            "completionTokens": int(usage.get("completion_tokens", usage.get("completionTokens", 0)) or 0),
            "totalTokens": int(usage.get("total_tokens", usage.get("totalTokens", 0)) or 0),
            "responseId": body.get("id"),
            "responseModel": body.get("model"),
            "finishReason": ((body.get("choices") or [{}])[0] or {}).get("finish_reason"),
            "error": None if 200 <= status < 300 else json.dumps(body),
        }
    except Exception as exc:
        return {
            "requestIndex": request_index,
            "status": 0,
            "ok": False,
            "latencyMs": (time.perf_counter() - started) * 1000,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "error": str(exc),
        }


def _sum(samples: list[dict[str, Any]], key: str) -> float:
    return sum(float(sample.get(key) or 0) for sample in samples)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((percentile / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _build_measurements(
    engine: dict[str, Any],
    request: dict[str, Any],
    workload: dict[str, Any],
    pricing: dict[str, Any] | None,
    samples: list[dict[str, Any]],
    captured_at_utc: str,
) -> list[dict[str, Any]]:
    engine_id = str(engine["engine"])
    engine_label = serving_engine_label(engine_id)
    successful = [sample for sample in samples if sample.get("ok")]
    duration_seconds = max(_sum(successful, "latencyMs") / 1000, 0.001)
    output_tokens = _sum(successful, "completionTokens")
    total_tokens = _sum(successful, "totalTokens")
    prompt_tokens = _sum(successful, "promptTokens")
    usd_per_gpu_hour = (pricing or {}).get("usdPerGpuHour")
    power_watts_per_gpu = (pricing or {}).get("powerWattsPerGpu")
    gpu_count = float((pricing or {}).get("gpuCount") or workload.get("parallelism") or 1)
    cost_usd = (
        (duration_seconds / 3600) * float(usd_per_gpu_hour) * gpu_count
        if isinstance(usd_per_gpu_hour, (int, float))
        else None
    )
    row: dict[str, Any] = {
        "surface": "result",
        "model": request["model"],
        "runtimeFramework": engine_label,
        "runtimeEngine": engine_id,
        "operatingPoint": workload.get("operatingPoint", "laptop-smoke"),
        "basis": "per_engine",
        "requestCount": len(samples),
        "successCount": len(successful),
        "errorCount": len(samples) - len(successful),
        "promptTokens": prompt_tokens,
        "completionTokens": output_tokens,
        "totalTokens": total_tokens,
        "outputTpm": output_tokens / (duration_seconds / 60),
        "totalTpm": total_tokens / (duration_seconds / 60),
        "avgLatencyMs": _sum(successful, "latencyMs") / len(successful) if successful else None,
        "p95LatencyMs": _percentile([float(sample["latencyMs"]) for sample in successful], 95),
        "usdPer1mOutputTokens": cost_usd / (output_tokens / 1_000_000) if cost_usd and output_tokens else None,
        "usdPer1mTotalTokens": cost_usd / (total_tokens / 1_000_000) if cost_usd and total_tokens else None,
        "avgPowerWattsPerGpu": power_watts_per_gpu if isinstance(power_watts_per_gpu, (int, float)) else None,
        "tokensPerWatt": (
            (total_tokens / duration_seconds) / (float(power_watts_per_gpu) * gpu_count)
            if isinstance(power_watts_per_gpu, (int, float)) and power_watts_per_gpu > 0
            else None
        ),
        "campaignCount": max(len(successful), 1),
        "latestCapturedAtUtc": captured_at_utc,
        "experimentFamily": "serving-producer",
        "experimentStatus": "accepted" if len(successful) == len(samples) else "partial",
        "verdictTier": "request-captured" if len(successful) == len(samples) else "request-errors",
        "solRigor": "smoke",
        "plotReadyPoints": 0,
        "dcgmGrounded": False,
        "tags": ",".join(["serving-producer", engine_id, engine_label, request["model"]]),
    }
    required = [
        row["outputTpm"],
        row["totalTpm"],
        row["usdPer1mOutputTokens"],
        row["usdPer1mTotalTokens"],
        row["tokensPerWatt"],
    ]
    row["metricCompleteness"] = sum(isinstance(value, (int, float)) for value in required) / len(required)
    return [row]


def _write_summary_artifact(
    engine: dict[str, Any],
    request: dict[str, Any],
    artifact_dir: str,
    captured_at_utc: str,
    samples: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in request["model"]).strip("-")
    artifact_path = os.path.join(
        artifact_dir,
        f"{engine['engine']}-{safe_model}-{captured_at_utc.replace(':', '-').replace('.', '-')}.json",
    )
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schemaVersion": "performance-iq.serving-producer-summary.v1",
                "capturedAtUtc": captured_at_utc,
                "engine": engine["engine"],
                "engineLabel": serving_engine_label(engine["engine"]),
                "baseUrl": engine["baseUrl"],
                "requestPath": engine.get("requestPath", "/v1/chat/completions"),
                "endpointPreflight": engine.get("endpointPreflight"),
                "model": request["model"],
                "request": _request_payload(request),
                "samples": samples,
                "measurements": measurements,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return artifact_path


def run_serving_producer(
    *,
    engine: dict[str, Any],
    request: dict[str, Any],
    performance_iq: PerformanceIQ | None = None,
    submit: bool = True,
    artifact_dir: str | None = None,
    producer: dict[str, Any] | None = None,
    campaign: dict[str, Any] | None = None,
    workload: dict[str, Any] | None = None,
    source_type: str = "other-measured-producer",
    run_class: str = "measured",
    confidentiality: str = "operator-full",
    pricing: dict[str, Any] | None = None,
    http_post_json: HttpPostJson | None = None,
    now: Callable[[], dt.datetime] | None = None,
) -> dict[str, Any]:
    if engine.get("engine") not in SERVING_ENGINE_LABELS:
        raise ValueError(f"unsupported serving engine: {engine.get('engine')}")
    captured_at_utc = _now_iso(now)
    repetitions = max(1, int(request.get("repetitions", 1)))
    samples = [
        _send_chat_completion(engine, request, index, http_post_json)
        for index in range(repetitions)
    ]
    workload = {
        "model": request["model"],
        "hardware": "unknown",
        "operatingPoint": "serving-smoke",
        "scenario": f"OpenAI-compatible chat completions through {serving_engine_label(engine['engine'])}",
        **(workload or {}),
    }
    measurements = _build_measurements(engine, request, workload, pricing, samples, captured_at_utc)
    artifact_path = _write_summary_artifact(
        engine,
        request,
        artifact_dir or os.path.join(os.getcwd(), ".performance-iq", "serving-producers"),
        captured_at_utc,
        samples,
        measurements,
    )
    campaign_id = (campaign or {}).get("campaignId") or f"serving-{engine['engine']}-{request['model']}"
    run_id = (campaign or {}).get("runId") or f"{campaign_id}-{captured_at_utc.replace(':', '-')}"
    run_input: PerformanceIQRunInput = {
        "sourceType": source_type,  # type: ignore[typeddict-item]
        "runClass": run_class,  # type: ignore[typeddict-item]
        "confidentiality": confidentiality,  # type: ignore[typeddict-item]
        "producer": {
            "repo": "performance-iq-sdk",
            "tool": f"{engine['engine']}-serving-producer",
            "commitSha": "localservingproducer",
            **(producer or {}),
        },
        "campaign": {
            "campaignId": campaign_id,
            "runId": run_id,
            "capturedAtUtc": captured_at_utc,
            "completedAtUtc": _now_iso(now),
            **(campaign or {}),
        },
        "workload": workload,  # type: ignore[typeddict-item]
        "runtime": {
            "imageDigest": engine.get("imageDigest") or DEFAULT_IMAGE_DIGEST,
            "imageTag": engine.get("imageTag") or (
                f"{engine['engine']}:{engine['frameworkVersion']}" if engine.get("frameworkVersion") else ""
            ),
            "framework": serving_engine_label(engine["engine"]),
        },
        "artifacts": [{"kind": "normalized-summary", "path": artifact_path}],
        "measurements": measurements,
        "platform": {"decisionBriefPath": "performance-iq://serving-producer"},
        "methodology": (
            f"{serving_engine_label(engine['engine'])} producer sent {repetitions} "
            "OpenAI-compatible chat completion request(s); metrics come from response "
            "usage fields and wall-clock request latency."
        ),
        "limitations": [
            "Serving producer captures request-path, usage, latency, and provenance; hardware-level power/kernel counters require engine-side or cluster instrumentation."
        ],
    }
    if not run_input["runtime"].get("imageTag"):
        del run_input["runtime"]["imageTag"]

    manifest = build_manifest(run_input)
    submission = performance_iq.submit_run(run_input, idempotency_key=manifest["campaign"]["runId"]) if performance_iq and submit else None
    return {
        "engine": engine["engine"],
        "manifest": manifest,
        "runInput": run_input,
        "artifactPath": artifact_path,
        "samples": samples,
        "measurements": measurements,
        "submission": submission,
    }
