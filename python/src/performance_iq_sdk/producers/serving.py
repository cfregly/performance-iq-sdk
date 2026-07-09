from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
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
HttpStreamJson = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]
HttpGetText = Callable[[str, dict[str, str]], str]

PROMETHEUS_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)")
PROMETHEUS_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
_TOKENIZER_CACHE: dict[str, Any] = {}


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


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "value"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_stable_json(value))


def _sha256_optional_json(value: Any) -> str | None:
    return _sha256_json(value) if value is not None else None


def _nested_value(source: dict[str, Any], parent: str, *keys: str) -> Any:
    candidate = source.get(parent)
    if not isinstance(candidate, dict):
        return None
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return value
    return None


def _engine_provenance_value(engine: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = engine.get(key)
        if value is not None:
            return value
    return None


def _runtime_provenance(engine: dict[str, Any], native_telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    telemetry = native_telemetry if isinstance(native_telemetry, dict) else {}
    server_args = telemetry.get("serverArgs", engine.get("serverArgs"))
    return {
        "engineVersion": telemetry.get("engineVersion", engine.get("frameworkVersion")),
        "modelRevision": telemetry.get("modelRevision", engine.get("modelRevision")),
        "imageTag": _engine_provenance_value(engine, "imageTag", "containerImageTag"),
        "imageDigest": _engine_provenance_value(engine, "imageDigest", "containerImageDigest"),
        "serverArgsSha256": _sha256_optional_json(server_args),
        "processId": _engine_provenance_value(engine, "processId", "pid") or _nested_value(engine, "process", "pid", "processId"),
        "containerId": _engine_provenance_value(engine, "containerId") or _nested_value(engine, "container", "id", "containerId"),
        "podName": _engine_provenance_value(engine, "podName") or _nested_value(engine, "container", "podName"),
        "nodeName": _engine_provenance_value(engine, "nodeName") or _nested_value(engine, "container", "nodeName"),
        "hostName": _engine_provenance_value(engine, "hostName", "hostname") or _nested_value(engine, "process", "hostName", "hostname"),
    }


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


def _get_text(url: str, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode("utf-8", errors="replace")


def _post_json_stream(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(http_request) as response:
        for raw_line in response:
            received_ms = (time.perf_counter() - started) * 1000
            received_at_utc = _now_iso()
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                events.append({
                    "done": True,
                    "raw": data,
                    "receivedMs": received_ms,
                    "receivedAtUtc": received_at_utc,
                })
                continue
            try:
                body = json.loads(data)
            except json.JSONDecodeError:
                body = {"_parseError": data}
            events.append({
                "body": body,
                "raw": data,
                "receivedMs": received_ms,
                "receivedAtUtc": received_at_utc,
            })
    return {"status": response.status, "events": events, "headers": dict(response.headers)}


def _request_payload(request: dict[str, Any], *, stream: bool | None = None) -> dict[str, Any]:
    payload = {
        "model": request["model"],
        "messages": request["messages"],
        "max_tokens": request.get("maxTokens", request.get("max_tokens", 64)),
        "temperature": request.get("temperature", 0),
    }
    top_p = request.get("topP", request.get("top_p"))
    if top_p is not None:
        payload["top_p"] = top_p
    stream_enabled = request.get("stream") if stream is None else stream
    if stream_enabled:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if request.get("captureTokenDetails") or request.get("logprobs") or request.get("topLogprobs") is not None or request.get("top_logprobs") is not None:
        payload["logprobs"] = bool(request.get("logprobs", True))
        top_logprobs = request.get("topLogprobs", request.get("top_logprobs"))
        if top_logprobs is not None:
            payload["top_logprobs"] = int(top_logprobs)
    return payload


def _redacted_request(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    prompt_text = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
    return {
        "model": payload.get("model"),
        "messageCount": len(messages),
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "stream": payload.get("stream", False),
        "promptBytes": len(prompt_text.encode("utf-8")),
        "promptSha256": _sha256_text(prompt_text),
        "requestPayloadSha256": _sha256_json(payload),
    }


def _prompt_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    return "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))


def _estimated_token_count(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4) if text else 0


def _request_trace_id(engine: dict[str, Any], run_id: str, request_index: int) -> str:
    return "-".join([
        "piq",
        str(engine["engine"]),
        _safe_slug(run_id),
        f"request-{request_index + 1}",
    ])


def _trace_headers(engine: dict[str, Any], campaign_id: str, run_id: str, request_id: str) -> dict[str, str]:
    return {
        "x-performance-iq-engine": str(engine["engine"]),
        "x-performance-iq-campaign-id": campaign_id,
        "x-performance-iq-run-id": run_id,
        "x-performance-iq-request-id": request_id,
    }


def _choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _choice_content(body: dict[str, Any]) -> str:
    choice = _choice(body)
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _usage_value(usage: dict[str, Any], snake_key: str, camel_key: str) -> int:
    return int(usage.get(snake_key, usage.get(camel_key, 0)) or 0)


def _native_telemetry(engine: dict[str, Any], body: dict[str, Any] | None = None) -> dict[str, Any]:
    configured = engine.get("nativeTelemetry")
    if isinstance(configured, dict):
        return {"available": True, "source": "engine-config", **configured}
    body = body or {}
    for key in ("nativeTelemetry", "native_telemetry", "metrics", "timings"):
        candidate = body.get(key)
        if isinstance(candidate, dict):
            return {"available": True, "source": f"response.{key}", **candidate}
    return {
        "available": False,
        "source": "not-exposed-by-openai-compatible-response",
        "engineVersion": engine.get("frameworkVersion"),
        "modelRevision": engine.get("modelRevision"),
        "serverArgs": engine.get("serverArgs"),
        "queueWaitMs": None,
        "prefillMs": None,
        "decodeMs": None,
        "batchSize": None,
        "concurrency": None,
        "kvCacheUsagePct": None,
        "cacheHitRate": None,
    }


def _default_native_metrics_url(engine: dict[str, Any]) -> str:
    base_url = _normalize_base_url(str(engine["baseUrl"]))
    if engine.get("engine") == "tensorrt-llm":
        return f"{base_url}/prometheus/metrics"
    return f"{base_url}/metrics"


def _default_native_json_metrics_url(engine: dict[str, Any]) -> str | None:
    if engine.get("engine") != "tensorrt-llm":
        return None
    return f"{_normalize_base_url(str(engine['baseUrl']))}/metrics"


def _metrics_url(engine: dict[str, Any]) -> str | None:
    configured = engine.get("metricsUrl")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if engine.get("collectNativeMetrics") is True:
        return _default_native_metrics_url(engine)
    return None


def _native_json_metrics_url(engine: dict[str, Any]) -> str | None:
    for key in ("nativeJsonMetricsUrl", "jsonMetricsUrl"):
        configured = engine.get(key)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    if engine.get("collectNativeMetrics") is True:
        return _default_native_json_metrics_url(engine)
    return None


def _metrics_headers(engine: dict[str, Any]) -> dict[str, str]:
    return {
        "accept": "text/plain",
        **({"authorization": f"Bearer {engine['apiKey']}"} if engine.get("apiKey") else {}),
    }


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        metric_name = match.group(1)
        metrics[metric_name] = metrics.get(metric_name, 0.0) + value
        metrics[f"{metric_name}__sample_count"] = metrics.get(f"{metric_name}__sample_count", 0.0) + 1
        labels = dict(PROMETHEUS_LABEL_RE.findall(match.group(2) or ""))
        for label_name in ("stage", "mode", "source", "reason", "finished_reason"):
            label_value = labels.get(label_name)
            if label_value:
                labelled_name = f"{metric_name}{{{label_name}={label_value}}}"
                metrics[labelled_name] = metrics.get(labelled_name, 0.0) + value
                metrics[f"{labelled_name}__sample_count"] = metrics.get(f"{labelled_name}__sample_count", 0.0) + 1
                for suffix in ("_sum", "_count"):
                    if metric_name.endswith(suffix):
                        labelled_histogram_name = f"{metric_name[:-len(suffix)]}{{{label_name}={label_value}}}{suffix}"
                        metrics[labelled_histogram_name] = metrics.get(labelled_histogram_name, 0.0) + value
                        metrics[f"{labelled_histogram_name}__sample_count"] = metrics.get(f"{labelled_histogram_name}__sample_count", 0.0) + 1
    return metrics


def _flatten_numeric_json_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_json_metrics(nested, child_prefix))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_numeric_json_metrics(nested, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def _parse_native_json_metrics(text: str) -> dict[str, float]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    records = parsed if isinstance(parsed, list) else [parsed]
    numeric_records = [
        _flatten_numeric_json_metrics(record)
        for record in records
        if isinstance(record, dict)
    ]
    numeric_records = [record for record in numeric_records if record]
    if not numeric_records:
        return {}
    latest = numeric_records[-1]
    summary: dict[str, float] = {f"latest.{key}": value for key, value in latest.items()}
    for key in sorted({metric_key for record in numeric_records for metric_key in record}):
        values = [record[key] for record in numeric_records if key in record]
        if values:
            summary[f"avg.{key}"] = sum(values) / len(values)
            summary[f"max.{key}"] = max(values)
    return summary


def _hardware_metrics_url(engine: dict[str, Any]) -> str | None:
    configured = engine.get("hardwareMetricsUrl") or engine.get("dcgmMetricsUrl")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if engine.get("collectHardwareMetrics") is True:
        return f"{_normalize_base_url(str(engine['baseUrl']))}/metrics"
    return None


def _hardware_telemetry(engine: dict[str, Any]) -> dict[str, Any]:
    configured = engine.get("hardwareTelemetry")
    if isinstance(configured, dict):
        return {"available": True, "source": "engine-config", **configured}
    return {"available": False, "source": "not-configured"}


def _read_hardware_metrics(engine: dict[str, Any], http_get_text: HttpGetText | None = None) -> dict[str, Any]:
    url = _hardware_metrics_url(engine)
    if not url:
        return {"available": False, "source": "hardware-metrics-url-not-configured"}
    try:
        text = (http_get_text or _get_text)(url, _metrics_headers(engine))
    except Exception as exc:
        return {"available": False, "source": "hardware-prometheus-unavailable", "metricsUrl": url, "error": str(exc)}
    metrics = _parse_prometheus_metrics(text)
    dcgm_metrics = {key: value for key, value in metrics.items() if key.startswith("DCGM_FI_")}
    if not dcgm_metrics:
        return {"available": False, "source": "dcgm-prometheus-empty", "metricsUrl": url}
    return {
        "available": True,
        "source": "dcgm-prometheus-snapshot",
        "metricsUrl": url,
        "metrics": dcgm_metrics,
        "capturedAtUtc": _now_iso(),
    }


def _read_native_metrics(engine: dict[str, Any], http_get_text: HttpGetText | None = None) -> dict[str, Any]:
    url = _metrics_url(engine)
    json_url = _native_json_metrics_url(engine)
    if not url and not json_url:
        return {"available": False, "source": "metrics-url-not-configured"}
    get_text = http_get_text or _get_text
    metrics: dict[str, float] = {}
    json_metrics: dict[str, float] = {}
    sources: list[str] = []
    errors: list[dict[str, str]] = []
    captured_at = _now_iso()
    if url:
        try:
            text = get_text(url, _metrics_headers(engine))
            metrics.update(_parse_prometheus_metrics(text))
            parsed_json = _parse_native_json_metrics(text)
            if metrics:
                sources.append("prometheus-snapshot")
            if parsed_json:
                json_metrics.update(parsed_json)
                sources.append("native-json-snapshot")
        except Exception as exc:
            errors.append({"url": url, "source": "prometheus-unavailable", "error": str(exc)})
    if json_url and json_url != url:
        try:
            text = get_text(json_url, {**_metrics_headers(engine), "accept": "application/json, text/plain"})
            parsed_json = _parse_native_json_metrics(text)
            if parsed_json:
                json_metrics.update(parsed_json)
                sources.append("native-json-snapshot")
            elif not metrics:
                errors.append({"url": json_url, "source": "native-json-empty", "error": "no numeric JSON metrics"})
        except Exception as exc:
            errors.append({"url": json_url, "source": "native-json-unavailable", "error": str(exc)})
    if not metrics and not json_metrics:
        return {
            "available": False,
            "source": errors[0]["source"] if errors else "native-metrics-empty",
            "metricsUrl": url,
            **({"nativeJsonMetricsUrl": json_url} if json_url else {}),
            **({"errors": errors} if errors else {}),
        }
    return {
        "available": True,
        "source": "+".join(dict.fromkeys(sources)) if sources else "native-metrics-snapshot",
        "metricsUrl": url,
        **({"nativeJsonMetricsUrl": json_url} if json_url else {}),
        "metrics": metrics,
        **({"jsonMetrics": json_metrics} if json_metrics else {}),
        **({"errors": errors} if errors else {}),
        "capturedAtUtc": captured_at,
    }


def _metric_value(metrics: dict[str, float], candidates: list[str]) -> float | None:
    for name in candidates:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metric_average(metrics: dict[str, float], candidates: list[str]) -> float | None:
    for name in candidates:
        value = metrics.get(name)
        if not isinstance(value, (int, float)):
            continue
        count = metrics.get(f"{name}__sample_count")
        if isinstance(count, (int, float)) and count > 0:
            return float(value) / float(count)
        return float(value)
    return None


def _json_metric_value(metrics: dict[str, float], candidates: list[str]) -> float | None:
    return _metric_value(metrics, candidates)


def _first_number(*values: float | None) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _counter_delta(before: dict[str, float], after: dict[str, float], candidates: list[str]) -> float | None:
    before_value = _metric_value(before, candidates)
    after_value = _metric_value(after, candidates)
    if before_value is None or after_value is None:
        return None
    delta = after_value - before_value
    return delta if delta >= 0 else None


def _histogram_delta_mean_ms(before: dict[str, float], after: dict[str, float], bases: list[str]) -> float | None:
    for base in bases:
        sum_delta = _counter_delta(before, after, [f"{base}_sum"])
        count_delta = _counter_delta(before, after, [f"{base}_count"])
        if sum_delta is not None and count_delta and count_delta > 0:
            return (sum_delta / count_delta) * 1000
    return None


def _native_metrics_delta(engine: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return {
            "available": False,
            "source": "prometheus-delta-unavailable",
            "metricsUrl": before.get("metricsUrl") or after.get("metricsUrl") or _metrics_url(engine),
            "before": before,
            "after": after,
        }
    before_metrics = before.get("metrics") if isinstance(before.get("metrics"), dict) else {}
    after_metrics = after.get("metrics") if isinstance(after.get("metrics"), dict) else {}
    after_json_metrics = after.get("jsonMetrics") if isinstance(after.get("jsonMetrics"), dict) else {}
    if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
        return {"available": False, "source": "prometheus-delta-invalid"}

    native_ttft_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:time_to_first_token_seconds",
        "sglang:time_to_first_token_seconds",
        "sglang_time_to_first_token_seconds",
        "trtllm:time_to_first_token_seconds",
        "trtllm_time_to_first_token_seconds",
    ])
    native_tpot_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_time_per_output_token_seconds",
        "vllm:time_per_output_token_seconds",
        "sglang:request_time_per_output_token_seconds",
        "sglang:time_per_output_token_seconds",
        "sglang_request_time_per_output_token_seconds",
        "sglang_time_per_output_token_seconds",
        "trtllm:request_time_per_output_token_seconds",
        "trtllm:time_per_output_token_seconds",
        "trtllm_request_time_per_output_token_seconds",
        "trtllm_time_per_output_token_seconds",
    ])
    native_inter_token_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:inter_token_latency_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang_inter_token_latency_seconds",
        "trtllm:inter_token_latency_seconds",
        "trtllm_inter_token_latency_seconds",
    ])
    native_e2e_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:e2e_request_latency_seconds",
        "sglang:e2e_request_latency_seconds",
        "sglang_e2e_request_latency_seconds",
        "trtllm:e2e_request_latency_seconds",
        "trtllm_e2e_request_latency_seconds",
    ])
    queue_wait_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_queue_time_seconds",
        "sglang:queue_time_seconds",
        "sglang:request_queue_time_seconds",
        "sglang_request_queue_time_seconds",
        "trtllm:request_queue_time_seconds",
        "trtllm_request_queue_time_seconds",
        "trtllm_queue_time_seconds",
        "trtllm:queue_time_seconds",
        "trtllm_request_queue_time_seconds",
    ])
    prefill_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_prefill_time_seconds",
        "sglang:per_stage_req_latency_seconds{stage=prefill_forward}",
        "sglang:per_stage_req_latency_seconds{mode=prefill_forward}",
        "sglang:request_prefill_time_seconds",
        "sglang_request_prefill_time_seconds",
        "trtllm:request_prefill_time_seconds",
        "trtllm_request_prefill_time_seconds",
        "trtllm:context_time_seconds",
        "trtllm_context_time_seconds",
    ])
    decode_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_decode_time_seconds",
        "sglang:per_stage_req_latency_seconds{stage=decode_forward}",
        "sglang:per_stage_req_latency_seconds{mode=decode}",
        "sglang:request_decode_time_seconds",
        "sglang_request_decode_time_seconds",
        "trtllm:request_decode_time_seconds",
        "trtllm_request_decode_time_seconds",
        "trtllm:generation_time_seconds",
        "trtllm_generation_time_seconds",
    ])
    if decode_ms is None and native_e2e_ms is not None and queue_wait_ms is not None and prefill_ms is not None:
        derived_decode_ms = native_e2e_ms - queue_wait_ms - prefill_ms
        if derived_decode_ms >= 0:
            decode_ms = derived_decode_ms
    prefix_queries = _counter_delta(before_metrics, after_metrics, [
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
        "sglang:prefix_cache_queries_total",
        "sglang:prefix_cache_queries",
        "sglang_prefix_cache_queries_total",
        "sglang_prefix_cache_queries",
        "trtllm_prefix_cache_queries",
        "trtllm:prefix_cache_queries_total",
        "trtllm_prefix_cache_queries_total",
    ])
    prefix_hits = _counter_delta(before_metrics, after_metrics, [
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
        "sglang:prefix_cache_hits_total",
        "sglang:prefix_cache_hits",
        "sglang_prefix_cache_hits_total",
        "sglang_prefix_cache_hits",
        "trtllm_prefix_cache_hits",
        "trtllm:prefix_cache_hits_total",
        "trtllm_prefix_cache_hits_total",
    ])
    cache_hit_rate = (prefix_hits / prefix_queries) if prefix_hits is not None and prefix_queries and prefix_queries > 0 else None
    trtllm_kv_used_blocks = _json_metric_value(after_json_metrics, ["latest.kvCacheStats.usedNumBlocks", "max.kvCacheStats.usedNumBlocks"])
    trtllm_kv_max_blocks = _json_metric_value(after_json_metrics, ["latest.kvCacheStats.maxNumBlocks", "max.kvCacheStats.maxNumBlocks"])
    trtllm_kv_usage = (
        trtllm_kv_used_blocks / trtllm_kv_max_blocks
        if trtllm_kv_used_blocks is not None and trtllm_kv_max_blocks and trtllm_kv_max_blocks > 0
        else None
    )
    values = {
        "nativeTtftMs": native_ttft_ms,
        "nativeTpotMs": native_tpot_ms,
        "nativeInterTokenLatencyMs": native_inter_token_ms,
        "nativeE2eLatencyMs": native_e2e_ms,
        "queueWaitMs": queue_wait_ms,
        "prefillMs": prefill_ms,
        "decodeMs": decode_ms,
        "runningRequests": _first_number(
            _metric_value(after_metrics, ["vllm:num_requests_running", "sglang:num_running_reqs", "sglang_num_running_reqs", "trtllm:num_requests_running", "trtllm_num_requests_running", "trtllm:num_active_requests", "trtllm_num_active_requests"]),
            _json_metric_value(after_json_metrics, ["latest.numActiveRequests", "avg.numActiveRequests", "max.numActiveRequests"]),
        ),
        "waitingRequests": _metric_value(after_metrics, ["vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang_num_queue_reqs", "trtllm:num_requests_waiting", "trtllm_num_requests_waiting", "trtllm:num_queued_requests", "trtllm_num_queued_requests"]),
        "kvCacheUsagePct": _first_number(
            _metric_value(after_metrics, ["vllm:kv_cache_usage_perc", "sglang:token_usage", "sglang_token_usage", "trtllm:kv_cache_usage_perc", "trtllm_kv_cache_usage_perc", "trtllm:kv_cache_utilization", "trtllm_kv_cache_utilization"]),
            trtllm_kv_usage,
        ),
        "trtllmIterationLatencyMs": _json_metric_value(after_json_metrics, ["avg.iterLatencyMS", "latest.iterLatencyMS"]),
        "trtllmGpuMemoryBytes": _json_metric_value(after_json_metrics, ["latest.gpuMemUsage", "max.gpuMemUsage"]),
        "trtllmKvCacheUsedBlocks": trtllm_kv_used_blocks,
        "trtllmKvCacheMaxBlocks": trtllm_kv_max_blocks,
        "prefixCacheQueriesDelta": prefix_queries,
        "prefixCacheHitsDelta": prefix_hits,
        "cacheHitRate": _first_number(
            cache_hit_rate,
            _metric_value(after_metrics, ["sglang:cache_hit_rate", "sglang_cache_hit_rate", "trtllm:kv_cache_hit_rate", "trtllm_kv_cache_hit_rate"]),
            _json_metric_value(after_json_metrics, ["latest.kvCacheStats.cacheHitRate", "avg.kvCacheStats.cacheHitRate"]),
        ),
        "promptTokensCachedDelta": _counter_delta(before_metrics, after_metrics, [
            "vllm:prompt_tokens_cached_total",
            "sglang:prompt_tokens_cached_total",
            "sglang_prompt_tokens_cached_total",
            "trtllm:prompt_tokens_cached_total",
            "trtllm_prompt_tokens_cached_total",
        ]),
        "promptTokensComputedDelta": _counter_delta(before_metrics, after_metrics, [
            "vllm:request_prefill_kv_computed_tokens_sum",
            "sglang:request_prefill_kv_computed_tokens_sum",
            "sglang_request_prefill_kv_computed_tokens_sum",
            "trtllm:request_prefill_kv_computed_tokens_sum",
            "trtllm_request_prefill_kv_computed_tokens_sum",
        ]),
    }
    available_values = {key: value for key, value in values.items() if value is not None}
    delta_sources = []
    if before_metrics and after_metrics:
        delta_sources.append("prometheus-delta")
    if after_json_metrics:
        delta_sources.append("native-json-snapshot")
    return {
        "available": bool(available_values),
        "source": "+".join(delta_sources) if delta_sources else "native-metrics-delta",
        "metricsUrl": after.get("metricsUrl") or before.get("metricsUrl"),
        "nativeJsonMetricsUrl": after.get("nativeJsonMetricsUrl") or before.get("nativeJsonMetricsUrl"),
        "beforeCapturedAtUtc": before.get("capturedAtUtc"),
        "afterCapturedAtUtc": after.get("capturedAtUtc"),
        **available_values,
    }


def _hardware_metrics_delta(engine: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    configured = _hardware_telemetry(engine)
    if configured.get("available"):
        return configured
    if not before.get("available") or not after.get("available"):
        return {
            "available": False,
            "source": "dcgm-delta-unavailable",
            "metricsUrl": before.get("metricsUrl") or after.get("metricsUrl") or _hardware_metrics_url(engine),
            "before": before,
            "after": after,
        }
    before_metrics = before.get("metrics") if isinstance(before.get("metrics"), dict) else {}
    after_metrics = after.get("metrics") if isinstance(after.get("metrics"), dict) else {}
    if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
        return {"available": False, "source": "dcgm-delta-invalid"}
    energy_mj = _counter_delta(before_metrics, after_metrics, ["DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"])
    values = {
        "powerWatts": _metric_value(after_metrics, ["DCGM_FI_DEV_POWER_USAGE"]),
        "powerWattsPerGpu": _metric_average(after_metrics, ["DCGM_FI_DEV_POWER_USAGE"]),
        "gpuUtilizationPct": _metric_average(after_metrics, ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE"]),
        "memoryCopyUtilizationPct": _metric_average(after_metrics, ["DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_PROF_DRAM_ACTIVE"]),
        "gpuTemperatureC": _metric_average(after_metrics, ["DCGM_FI_DEV_GPU_TEMP"]),
        "smClockMHz": _metric_average(after_metrics, ["DCGM_FI_DEV_SM_CLOCK"]),
        "memoryClockMHz": _metric_average(after_metrics, ["DCGM_FI_DEV_MEM_CLOCK"]),
        "fbUsedMiB": _metric_value(after_metrics, ["DCGM_FI_DEV_FB_USED"]),
        "fbFreeMiB": _metric_value(after_metrics, ["DCGM_FI_DEV_FB_FREE"]),
        "energyJoules": (energy_mj / 1000) if energy_mj is not None else None,
    }
    available_values = {key: value for key, value in values.items() if value is not None}
    return {
        "available": bool(available_values),
        "source": "dcgm-prometheus-delta",
        "metricsUrl": after.get("metricsUrl") or before.get("metricsUrl"),
        "beforeCapturedAtUtc": before.get("capturedAtUtc"),
        "afterCapturedAtUtc": after.get("capturedAtUtc"),
        **available_values,
    }


def _combine_native_telemetry(*items: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    available_sources: list[str] = []
    fallback_sources: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("source"):
            if item.get("available"):
                available_sources.append(str(item["source"]))
            else:
                fallback_sources.append(str(item["source"]))
        combined.update(item)
    combined["available"] = any(bool(item.get("available")) for item in items if isinstance(item, dict))
    sources = available_sources or fallback_sources
    if sources:
        combined["source"] = "+".join(dict.fromkeys(sources))
    return combined


def _native_iteration_fields(native_telemetry: dict[str, Any]) -> dict[str, Any]:
    return {
        "nativeIterationLatencyMs": native_telemetry.get("nativeIterationLatencyMs", native_telemetry.get("trtllmIterationLatencyMs")),
        "nativeGpuMemoryBytes": native_telemetry.get("nativeGpuMemoryBytes", native_telemetry.get("trtllmGpuMemoryBytes")),
        "nativeKvCacheUsedBlocks": native_telemetry.get("nativeKvCacheUsedBlocks", native_telemetry.get("trtllmKvCacheUsedBlocks")),
        "nativeKvCacheMaxBlocks": native_telemetry.get("nativeKvCacheMaxBlocks", native_telemetry.get("trtllmKvCacheMaxBlocks")),
    }


def _extract_token_id(item: dict[str, Any]) -> int | None:
    for key in ("token_id", "tokenId", "id"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _extract_token_id_with_source(item: dict[str, Any], token: str, engine: dict[str, Any], request: dict[str, Any]) -> tuple[int | None, str | None]:
    token_id = _extract_token_id(item)
    if token_id is not None:
        return token_id, "response-logprobs"

    token_id_map = engine.get("tokenIdMap") or engine.get("token_id_map")
    if isinstance(token_id_map, dict) and token in token_id_map:
        mapped = token_id_map[token]
        if isinstance(mapped, int):
            return mapped, "configured-token-id-map"
        if isinstance(mapped, str) and mapped.isdigit():
            return int(mapped), "configured-token-id-map"

    resolver = engine.get("tokenIdResolver") or engine.get("token_id_resolver")
    if callable(resolver):
        resolved = resolver(token, item, engine, request)
        if isinstance(resolved, int):
            return resolved, "configured-token-id-resolver"
        if isinstance(resolved, str) and resolved.isdigit():
            return int(resolved), "configured-token-id-resolver"

    tokenizer = engine.get("tokenizer")
    if tokenizer is not None:
        resolved = _token_id_from_tokenizer(tokenizer, token)
        if resolved is not None:
            return resolved, "configured-tokenizer"

    tokenizer_model = engine.get("tokenizerModel") or engine.get("tokenizer_model") or request.get("tokenizerModel")
    if not tokenizer_model and (engine.get("resolveTokenIdsWithTokenizer") or request.get("resolveTokenIdsWithTokenizer")):
        tokenizer_model = request.get("model")
    if isinstance(tokenizer_model, str) and tokenizer_model:
        tokenizer = _load_hf_tokenizer(tokenizer_model, engine)
        if tokenizer is not None:
            resolved = _token_id_from_tokenizer(tokenizer, token)
            if resolved is not None:
                return resolved, "hf-tokenizer"
    return None, None


def _load_hf_tokenizer(model: str, engine: dict[str, Any]) -> Any | None:
    if model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model]
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=bool(engine.get("tokenizerTrustRemoteCode") or engine.get("trustRemoteCode")),
        )
    except Exception:
        return None
    _TOKENIZER_CACHE[model] = tokenizer
    return tokenizer


def _token_id_from_tokenizer(tokenizer: Any, token: str) -> int | None:
    if not token:
        return None
    try:
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            value = tokenizer.convert_tokens_to_ids(token)
            unknown_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(value, int) and value >= 0 and value != unknown_id:
                return value
    except Exception:
        pass
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], int):
            return encoded[0]
    except Exception:
        pass
    try:
        encoded = tokenizer(token, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(input_ids, list) and len(input_ids) == 1 and isinstance(input_ids[0], int):
            return input_ids[0]
    except Exception:
        pass
    return None


def _token_text_from_id(tokenizer: Any, token_id: int) -> str | None:
    try:
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            value = tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(value, str):
                return value
    except Exception:
        pass
    try:
        if hasattr(tokenizer, "decode"):
            value = tokenizer.decode([token_id], skip_special_tokens=False)
            if isinstance(value, str):
                return value
    except Exception:
        pass
    return None


def _coerce_token_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    token_ids: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            token_ids.append(item)
        elif isinstance(item, str) and item.isdigit():
            token_ids.append(int(item))
    return token_ids


def _prompt_tokenizer(engine: dict[str, Any], request: dict[str, Any]) -> tuple[Any | None, str | None, str | None]:
    tokenizer = engine.get("tokenizer")
    if tokenizer is not None:
        model = engine.get("tokenizerModel") or engine.get("tokenizer_model") or request.get("tokenizerModel") or request.get("model")
        return tokenizer, "configured-tokenizer", str(model) if model else None
    tokenizer_model = engine.get("tokenizerModel") or engine.get("tokenizer_model") or request.get("tokenizerModel")
    if not tokenizer_model and (engine.get("resolveTokenIdsWithTokenizer") or request.get("resolveTokenIdsWithTokenizer")):
        tokenizer_model = request.get("model")
    if isinstance(tokenizer_model, str) and tokenizer_model:
        tokenizer = _load_hf_tokenizer(tokenizer_model, engine)
        if tokenizer is not None:
            return tokenizer, "hf-tokenizer", tokenizer_model
    return None, None, None


def _encode_prompt_token_ids(tokenizer: Any, payload: dict[str, Any]) -> tuple[list[int], str]:
    messages = payload.get("messages")
    if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
        for kwargs in (
            {"tokenize": True, "add_generation_prompt": True},
            {"tokenize": True},
        ):
            try:
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
                token_ids = _coerce_token_ids(encoded)
                if token_ids:
                    return token_ids, "chat-template"
            except Exception:
                pass
    prompt_text = _prompt_text(payload)
    if not prompt_text:
        return [], "prompt-text-empty"
    try:
        encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
        token_ids = _coerce_token_ids(encoded)
        if token_ids:
            return token_ids, "prompt-text"
    except Exception:
        pass
    try:
        encoded = tokenizer(prompt_text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        token_ids = _coerce_token_ids(input_ids)
        if token_ids:
            return token_ids, "prompt-text"
    except Exception:
        pass
    return [], "tokenizer-empty"


def _prompt_token_details(engine: dict[str, Any], request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    explicit_token_ids = _coerce_token_ids(request.get("promptTokenIds") or engine.get("promptTokenIds"))
    tokenizer = None
    tokenizer_model = request.get("tokenizerModel") or engine.get("tokenizerModel") or engine.get("tokenizer_model")
    if explicit_token_ids:
        token_ids = explicit_token_ids
        source = "configured-prompt-token-ids"
        tokenization_source = "configured-prompt-token-ids"
    else:
        tokenizer, source, tokenizer_model = _prompt_tokenizer(engine, request)
        if tokenizer is None or source is None:
            return {
                "summary": {
                    "promptTokenIdsAvailable": False,
                    "promptTokenDetailCount": 0,
                    "promptTokenIdSource": None,
                    "promptTokenIdsSha256": None,
                    "promptTokenizationSource": "tokenizer-not-configured",
                    "promptTokenizerModel": tokenizer_model,
                },
                "details": [],
            }
        token_ids, tokenization_mode = _encode_prompt_token_ids(tokenizer, payload)
        tokenization_source = f"{source}-{tokenization_mode}"
    details = []
    for index, token_id in enumerate(token_ids):
        token_text = _token_text_from_id(tokenizer, token_id) if tokenizer is not None else None
        details.append({
            "tokenPhase": "prompt",
            "tokenIndex": index,
            "tokenId": token_id,
            "tokenIdSource": source,
            "tokenText": token_text,
            "tokenTextSha256": _sha256_text(token_text) if token_text else None,
            "tokenBytes": len(token_text.encode("utf-8")) if token_text else None,
            "tokenLogprob": None,
            "tokenDetailSource": tokenization_source,
        })
    return {
        "summary": {
            "promptTokenIdsAvailable": bool(details),
            "promptTokenDetailCount": len(details),
            "promptTokenIdSource": source if details else None,
            "promptTokenIdsSha256": _sha256_json(token_ids) if details else None,
            "promptTokenizationSource": tokenization_source,
            "promptTokenizerModel": tokenizer_model,
        },
        "details": details,
    }


def _redacted_prompt_token_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in detail.items() if key != "tokenText"}
        for detail in details
    ]


def _prompt_token_timeline(request_id: str, details: list[dict[str, Any]], received_at_utc: str) -> list[dict[str, Any]]:
    return [
        {
            "requestId": request_id,
            "tokenPhase": "prompt",
            "chunkIndex": None,
            "tokenIndex": detail.get("tokenIndex"),
            "receivedAtUtc": received_at_utc,
            "relativeMs": 0,
            "contentBytes": detail.get("tokenBytes"),
            "contentSha256": detail.get("tokenTextSha256"),
            "isFirstOutput": False,
            "tokenId": detail.get("tokenId"),
            "tokenIdSource": detail.get("tokenIdSource"),
            "tokenLogprob": None,
            "tokenTextSha256": detail.get("tokenTextSha256"),
            "topLogprobsJson": None,
            "tokenDetailSource": detail.get("tokenDetailSource"),
        }
        for detail in details
    ]


def _sanitize_top_logprobs(value: Any, engine: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        token = item.get("token") if isinstance(item.get("token"), str) else ""
        token_id, token_id_source = _extract_token_id_with_source(item, token, engine, request)
        sanitized.append({
            "tokenSha256": _sha256_text(token) if token else None,
            "tokenBytes": len(token.encode("utf-8")) if token else None,
            "tokenId": token_id,
            "tokenIdSource": token_id_source,
            "logprob": item.get("logprob") if isinstance(item.get("logprob"), (int, float)) else None,
        })
    return sanitized


def _choice_token_details(body: dict[str, Any], engine: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    logprobs = _choice(body).get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    details = []
    for item in content:
        if not isinstance(item, dict):
            continue
        token = item.get("token") if isinstance(item.get("token"), str) else ""
        token_bytes = item.get("bytes") if isinstance(item.get("bytes"), list) else None
        token_id, token_id_source = _extract_token_id_with_source(item, token, engine, request)
        details.append({
            "tokenText": token,
            "tokenSha256": _sha256_text(token) if token else None,
            "tokenBytes": len(token.encode("utf-8")) if token else (len(token_bytes) if token_bytes else None),
            "tokenId": token_id,
            "tokenIdSource": token_id_source,
            "logprob": item.get("logprob") if isinstance(item.get("logprob"), (int, float)) else None,
            "topLogprobs": _sanitize_top_logprobs(item.get("top_logprobs", item.get("topLogprobs")), engine, request),
        })
    return details


def _redacted_token_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in detail.items()
            if key != "tokenText"
        }
        for detail in details
    ]


def _token_detail_summary(details: list[dict[str, Any]], requested: bool) -> dict[str, Any]:
    if not details:
        return {
            "tokenDetailsAvailable": False,
            "tokenIdsAvailable": False,
            "logprobsAvailable": False,
            "tokenDetailCount": 0,
            "tokenDetailSource": "requested-not-exposed" if requested else "not-requested",
            "tokenIdSource": None,
        }
    return {
        "tokenDetailsAvailable": True,
        "tokenIdsAvailable": any(detail.get("tokenId") is not None for detail in details),
        "logprobsAvailable": any(detail.get("logprob") is not None for detail in details),
        "tokenDetailCount": len(details),
        "tokenDetailSource": "response-logprobs",
        "tokenIdSource": "+".join(dict.fromkeys(
            str(detail.get("tokenIdSource"))
            for detail in details
            if detail.get("tokenId") is not None and detail.get("tokenIdSource")
        )) or None,
    }


def _avg_numbers(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _normalize_stream_result(result: dict[str, Any], started: float) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for raw_event in result.get("events", []):
        if not isinstance(raw_event, dict):
            continue
        received_ms = raw_event.get("receivedMs")
        event = {
            **raw_event,
            "receivedMs": float(received_ms) if isinstance(received_ms, (int, float)) else (time.perf_counter() - started) * 1000,
            "receivedAtUtc": raw_event.get("receivedAtUtc") if isinstance(raw_event.get("receivedAtUtc"), str) else _now_iso(),
        }
        events.append(event)
    return {
        "status": int(result.get("status", 0)),
        "events": events,
        "headers": result.get("headers") if isinstance(result.get("headers"), dict) else {},
    }


def _send_streaming_chat_completion(
    engine: dict[str, Any],
    request: dict[str, Any],
    request_index: int,
    campaign_id: str,
    run_id: str,
    http_stream_json: HttpStreamJson | None,
    http_get_text: HttpGetText | None,
) -> dict[str, Any]:
    endpoint = f"{_normalize_base_url(engine['baseUrl'])}{engine.get('requestPath', '/v1/chat/completions')}"
    request_id = _request_trace_id(engine, run_id, request_index)
    headers = {
        "content-type": "application/json",
        **_trace_headers(engine, campaign_id, run_id, request_id),
    }
    if engine.get("apiKey"):
        headers["authorization"] = f"Bearer {engine['apiKey']}"
    payload = _request_payload(request, stream=True)
    prompt_tokens_capture = _prompt_token_details(engine, request, payload)
    prompt_token_summary = prompt_tokens_capture["summary"]
    prompt_token_details = prompt_tokens_capture["details"]
    native_before = _read_native_metrics(engine, http_get_text)
    hardware_before = _read_hardware_metrics(engine, http_get_text)
    token_details_requested = bool(payload.get("logprobs"))
    started = time.perf_counter()
    request_started_at_utc = _now_iso()
    try:
        raw_result = (http_stream_json or (lambda url, hdrs, body: _post_json_stream(url, hdrs, body, started)))(
            endpoint,
            headers,
            payload,
        )
        result = _normalize_stream_result(raw_result, started)
        e2e_latency_ms = (time.perf_counter() - started) * 1000
        request_completed_at_utc = _now_iso()
        status = int(result.get("status", 0))
        events = result.get("events") or []
        output_chunks: list[dict[str, Any]] = []
        response_id = None
        response_model = None
        finish_reason = None
        usage: dict[str, Any] = {}
        last_body: dict[str, Any] = {}
        native_telemetry = _native_telemetry(engine)
        for event in events:
            body = event.get("body") if isinstance(event.get("body"), dict) else {}
            if body:
                last_body = body
                response_id = response_id or body.get("id")
                response_model = response_model or body.get("model")
                finish_reason = finish_reason or _choice(body).get("finish_reason")
                if isinstance(body.get("usage"), dict):
                    usage = body["usage"]
                telemetry = _native_telemetry(engine, body)
                if telemetry.get("available"):
                    native_telemetry = telemetry
            content = _choice_content(body)
            if content:
                token_details = _choice_token_details(body, engine, request)
                output_chunks.append({
                    "chunkIndex": len(output_chunks),
                    "content": content,
                    "contentBytes": len(content.encode("utf-8")),
                    "contentSha256": _sha256_text(content),
                    "receivedMs": event["receivedMs"],
                    "receivedAtUtc": event["receivedAtUtc"],
                    "tokenDetails": token_details,
                })
        native_after = _read_native_metrics(engine, http_get_text)
        hardware_after = _read_hardware_metrics(engine, http_get_text)
        native_telemetry = _combine_native_telemetry(
            native_telemetry,
            _native_metrics_delta(engine, native_before, native_after),
        )
        native_iteration_fields = _native_iteration_fields(native_telemetry)
        hardware_telemetry = _hardware_metrics_delta(engine, hardware_before, hardware_after)
        runtime_provenance = _runtime_provenance(engine, native_telemetry)
        first_chunk = events[0] if events else None
        first_output = output_chunks[0] if output_chunks else None
        last_output = output_chunks[-1] if output_chunks else None
        output_text = "".join(chunk["content"] for chunk in output_chunks)
        token_count_source = "response-usage" if usage else "client-estimate"
        prompt_tokens = _usage_value(usage, "prompt_tokens", "promptTokens") or _estimated_token_count(_prompt_text(payload))
        completion_tokens = _usage_value(usage, "completion_tokens", "completionTokens") or len(output_chunks)
        total_tokens = _usage_value(usage, "total_tokens", "totalTokens") or (prompt_tokens + completion_tokens)
        output_token_count = completion_tokens or len(output_chunks)
        tpot_ms = (
            (float(last_output["receivedMs"]) - float(first_output["receivedMs"])) / max(output_token_count - 1, 1)
            if first_output and last_output
            else None
        )
        chunk_gaps = [
            float(output_chunks[index]["receivedMs"]) - float(output_chunks[index - 1]["receivedMs"])
            for index in range(1, len(output_chunks))
        ]
        raw_token_details = [
            {**detail, "chunkIndex": chunk["chunkIndex"]}
            for chunk in output_chunks
            for detail in (chunk.get("tokenDetails") or [])
        ]
        token_summary = _token_detail_summary(raw_token_details, token_details_requested)
        token_timeline = _prompt_token_timeline(request_id, prompt_token_details, request_started_at_utc)
        token_index = 0
        for chunk in output_chunks:
            chunk_details = chunk.get("tokenDetails") or []
            if chunk_details:
                for detail in chunk_details:
                    top_logprobs = detail.get("topLogprobs") if isinstance(detail.get("topLogprobs"), list) else []
                    token_timeline.append({
                        "requestId": request_id,
                        "tokenPhase": "output",
                        "chunkIndex": chunk["chunkIndex"],
                        "tokenIndex": token_index,
                        "receivedAtUtc": chunk["receivedAtUtc"],
                        "relativeMs": chunk["receivedMs"],
                        "contentBytes": detail.get("tokenBytes") or chunk["contentBytes"],
                        "contentSha256": detail.get("tokenSha256") or chunk["contentSha256"],
                        "isFirstOutput": token_index == 0,
                        "tokenId": detail.get("tokenId"),
                        "tokenIdSource": detail.get("tokenIdSource"),
                        "tokenLogprob": detail.get("logprob"),
                        "tokenTextSha256": detail.get("tokenSha256"),
                        "topLogprobsJson": _stable_json(top_logprobs) if top_logprobs else None,
                        "tokenDetailSource": token_summary["tokenDetailSource"],
                    })
                    token_index += 1
            else:
                token_timeline.append({
                    "requestId": request_id,
                    "tokenPhase": "output",
                    "chunkIndex": chunk["chunkIndex"],
                    "tokenIndex": None,
                    "receivedAtUtc": chunk["receivedAtUtc"],
                    "relativeMs": chunk["receivedMs"],
                    "contentBytes": chunk["contentBytes"],
                    "contentSha256": chunk["contentSha256"],
                    "isFirstOutput": chunk["chunkIndex"] == 0,
                    "tokenId": None,
                    "tokenIdSource": None,
                    "tokenLogprob": None,
                    "tokenTextSha256": None,
                    "topLogprobsJson": None,
                    "tokenDetailSource": token_summary["tokenDetailSource"],
                })
        return {
            "requestId": request_id,
            "requestIndex": request_index,
            "endpoint": endpoint,
            "requestStartedAtUtc": request_started_at_utc,
            "requestCompletedAtUtc": request_completed_at_utc,
            "status": status,
            "ok": 200 <= status < 300,
            "latencyMs": e2e_latency_ms,
            "e2eLatencyMs": e2e_latency_ms,
            "timeToFirstByteMs": first_chunk.get("receivedMs") if first_chunk else None,
            "ttftMs": first_output.get("receivedMs") if first_output else None,
            "ttfotMs": first_output.get("receivedMs") if first_output else None,
            "tpotMs": tpot_ms,
            "interTokenLatencyMs": (sum(chunk_gaps) / len(chunk_gaps)) if chunk_gaps else tpot_ms,
            "firstChunkAtUtc": first_chunk.get("receivedAtUtc") if first_chunk else None,
            "firstOutputAtUtc": first_output.get("receivedAtUtc") if first_output else None,
            "lastOutputAtUtc": last_output.get("receivedAtUtc") if last_output else None,
            "streamChunkCount": len(events),
            "outputTokenCount": output_token_count,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "tokenCountSource": token_count_source,
            "responseId": response_id,
            "responseModel": response_model,
            "finishReason": finish_reason,
            "ttftSource": "client-stream-content",
            "streaming": True,
            "promptSha256": _redacted_request(payload)["promptSha256"],
            "requestPayloadSha256": _sha256_json(payload),
            "outputSha256": _sha256_text(output_text),
            "outputBytes": len(output_text.encode("utf-8")),
            "nativeTelemetry": native_telemetry,
            "nativeTelemetryAvailable": bool(native_telemetry.get("available")),
            "hardwareTelemetry": hardware_telemetry,
            "hardwareTelemetryAvailable": bool(hardware_telemetry.get("available")),
            "nativeTelemetrySource": native_telemetry.get("source"),
            "nativeMetricsUrl": native_telemetry.get("metricsUrl"),
            "nativeTtftMs": native_telemetry.get("nativeTtftMs"),
            "nativeTpotMs": native_telemetry.get("nativeTpotMs"),
            "nativeE2eLatencyMs": native_telemetry.get("nativeE2eLatencyMs"),
            "nativeInterTokenLatencyMs": native_telemetry.get("nativeInterTokenLatencyMs"),
            **native_iteration_fields,
            "runningRequests": native_telemetry.get("runningRequests"),
            "waitingRequests": native_telemetry.get("waitingRequests"),
            "kvCacheUsagePct": native_telemetry.get("kvCacheUsagePct"),
            "cacheHitRate": native_telemetry.get("cacheHitRate"),
            "prefixCacheQueriesDelta": native_telemetry.get("prefixCacheQueriesDelta"),
            "prefixCacheHitsDelta": native_telemetry.get("prefixCacheHitsDelta"),
            "promptTokensCachedDelta": native_telemetry.get("promptTokensCachedDelta"),
            "promptTokensComputedDelta": native_telemetry.get("promptTokensComputedDelta"),
            "hardwareTelemetrySource": hardware_telemetry.get("source"),
            "hardwareMetricsUrl": hardware_telemetry.get("metricsUrl"),
            "avgPowerWatts": hardware_telemetry.get("powerWatts"),
            "avgPowerWattsPerGpu": hardware_telemetry.get("powerWattsPerGpu"),
            "gpuUtilizationPct": hardware_telemetry.get("gpuUtilizationPct"),
            "memoryCopyUtilizationPct": hardware_telemetry.get("memoryCopyUtilizationPct"),
            "gpuTemperatureC": hardware_telemetry.get("gpuTemperatureC"),
            "smClockMHz": hardware_telemetry.get("smClockMHz"),
            "memoryClockMHz": hardware_telemetry.get("memoryClockMHz"),
            "fbUsedMiB": hardware_telemetry.get("fbUsedMiB"),
            "fbFreeMiB": hardware_telemetry.get("fbFreeMiB"),
            "energyJoules": hardware_telemetry.get("energyJoules"),
            "queueWaitMs": native_telemetry.get("queueWaitMs"),
            "prefillMs": native_telemetry.get("prefillMs"),
            "decodeMs": native_telemetry.get("decodeMs"),
            **runtime_provenance,
            **prompt_token_summary,
            **token_summary,
            "tokenTimeline": token_timeline,
            "error": None if 200 <= status < 300 else json.dumps(last_body),
            "_rawCapture": {
                "requestId": request_id,
                "endpoint": endpoint,
                "requestPayload": payload,
                "responseEvents": events,
                "outputText": output_text,
                "promptTokenDetails": prompt_token_details,
                "tokenDetails": raw_token_details,
                "nativeTelemetry": native_telemetry,
                "hardwareTelemetry": hardware_telemetry,
                "runtimeProvenance": runtime_provenance,
            },
        }
    except Exception as exc:
        e2e_latency_ms = (time.perf_counter() - started) * 1000
        native_telemetry = _native_telemetry(engine)
        hardware_telemetry = _hardware_telemetry(engine)
        return {
            "requestId": request_id,
            "requestIndex": request_index,
            "endpoint": endpoint,
            "requestStartedAtUtc": request_started_at_utc,
            "requestCompletedAtUtc": _now_iso(),
            "status": 0,
            "ok": False,
            "latencyMs": e2e_latency_ms,
            "e2eLatencyMs": e2e_latency_ms,
            "timeToFirstByteMs": None,
            "ttftMs": None,
            "ttfotMs": None,
            "tpotMs": None,
            "interTokenLatencyMs": None,
            "streamChunkCount": 0,
            "outputTokenCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "ttftSource": "client-stream-content",
            "streaming": True,
            "nativeTelemetry": native_telemetry,
            "nativeTelemetryAvailable": False,
            "hardwareTelemetry": hardware_telemetry,
            "hardwareTelemetryAvailable": False,
            "nativeTelemetrySource": native_telemetry.get("source"),
            "hardwareTelemetrySource": hardware_telemetry.get("source"),
            **_runtime_provenance(engine, native_telemetry),
            "tokenDetailsAvailable": False,
            "tokenIdsAvailable": False,
            "logprobsAvailable": False,
            "tokenDetailCount": 0,
            "tokenDetailSource": "error",
            "tokenIdSource": None,
            "promptTokenIdsAvailable": False,
            "promptTokenDetailCount": 0,
            "promptTokenIdSource": None,
            "promptTokenIdsSha256": None,
            "promptTokenizationSource": "error",
            "promptTokenizerModel": None,
            "error": str(exc),
        }


def _send_non_streaming_chat_completion(
    engine: dict[str, Any],
    request: dict[str, Any],
    request_index: int,
    campaign_id: str,
    run_id: str,
    http_post_json: HttpPostJson | None,
    http_get_text: HttpGetText | None,
) -> dict[str, Any]:
    endpoint = f"{_normalize_base_url(engine['baseUrl'])}{engine.get('requestPath', '/v1/chat/completions')}"
    request_id = _request_trace_id(engine, run_id, request_index)
    headers = {
        "content-type": "application/json",
        **_trace_headers(engine, campaign_id, run_id, request_id),
    }
    if engine.get("apiKey"):
        headers["authorization"] = f"Bearer {engine['apiKey']}"
    payload = _request_payload(request, stream=False)
    prompt_tokens_capture = _prompt_token_details(engine, request, payload)
    prompt_token_summary = prompt_tokens_capture["summary"]
    prompt_token_details = prompt_tokens_capture["details"]
    native_before = _read_native_metrics(engine, http_get_text)
    hardware_before = _read_hardware_metrics(engine, http_get_text)
    token_details_requested = bool(payload.get("logprobs"))
    started = time.perf_counter()
    request_started_at_utc = _now_iso()
    try:
        result = (http_post_json or _post_json)(endpoint, headers, payload)
        latency_ms = (time.perf_counter() - started) * 1000
        request_completed_at_utc = _now_iso()
        body = result.get("body", {})
        usage = body.get("usage") or {}
        status = int(result.get("status", 0))
        output_text = _choice_content(body)
        native_after = _read_native_metrics(engine, http_get_text)
        hardware_after = _read_hardware_metrics(engine, http_get_text)
        native_telemetry = _combine_native_telemetry(
            _native_telemetry(engine, body),
            _native_metrics_delta(engine, native_before, native_after),
        )
        native_iteration_fields = _native_iteration_fields(native_telemetry)
        hardware_telemetry = _hardware_metrics_delta(engine, hardware_before, hardware_after)
        runtime_provenance = _runtime_provenance(engine, native_telemetry)
        raw_token_details = _choice_token_details(body, engine, request)
        token_summary = _token_detail_summary(raw_token_details, token_details_requested)
        token_timeline = _prompt_token_timeline(request_id, prompt_token_details, request_started_at_utc) + [
            {
                "requestId": request_id,
                "tokenPhase": "output",
                "chunkIndex": 0,
                "tokenIndex": index,
                "receivedAtUtc": request_completed_at_utc,
                "relativeMs": latency_ms,
                "contentBytes": detail.get("tokenBytes"),
                "contentSha256": detail.get("tokenSha256"),
                "isFirstOutput": index == 0,
                "tokenId": detail.get("tokenId"),
                "tokenIdSource": detail.get("tokenIdSource"),
                "tokenLogprob": detail.get("logprob"),
                "tokenTextSha256": detail.get("tokenSha256"),
                "topLogprobsJson": _stable_json(detail.get("topLogprobs")) if detail.get("topLogprobs") else None,
                "tokenDetailSource": token_summary["tokenDetailSource"],
            }
            for index, detail in enumerate(raw_token_details)
        ]
        return {
            "requestId": request_id,
            "requestIndex": request_index,
            "endpoint": endpoint,
            "requestStartedAtUtc": request_started_at_utc,
            "requestCompletedAtUtc": request_completed_at_utc,
            "status": status,
            "ok": 200 <= status < 300,
            "latencyMs": latency_ms,
            "e2eLatencyMs": latency_ms,
            "timeToFirstByteMs": None,
            "ttftMs": None,
            "ttfotMs": None,
            "tpotMs": None,
            "interTokenLatencyMs": None,
            "streamChunkCount": 0,
            "outputTokenCount": _usage_value(usage, "completion_tokens", "completionTokens"),
            "promptTokens": _usage_value(usage, "prompt_tokens", "promptTokens"),
            "completionTokens": _usage_value(usage, "completion_tokens", "completionTokens"),
            "totalTokens": _usage_value(usage, "total_tokens", "totalTokens"),
            "tokenCountSource": "response-usage" if usage else "none",
            "responseId": body.get("id"),
            "responseModel": body.get("model"),
            "finishReason": _choice(body).get("finish_reason"),
            "ttftSource": "not-streamed",
            "streaming": False,
            "promptSha256": _redacted_request(payload)["promptSha256"],
            "requestPayloadSha256": _sha256_json(payload),
            "outputSha256": _sha256_text(output_text),
            "outputBytes": len(output_text.encode("utf-8")),
            "nativeTelemetry": native_telemetry,
            "nativeTelemetryAvailable": bool(native_telemetry.get("available")),
            "hardwareTelemetry": hardware_telemetry,
            "hardwareTelemetryAvailable": bool(hardware_telemetry.get("available")),
            "nativeTelemetrySource": native_telemetry.get("source"),
            "nativeMetricsUrl": native_telemetry.get("metricsUrl"),
            "nativeTtftMs": native_telemetry.get("nativeTtftMs"),
            "nativeTpotMs": native_telemetry.get("nativeTpotMs"),
            "nativeE2eLatencyMs": native_telemetry.get("nativeE2eLatencyMs"),
            "nativeInterTokenLatencyMs": native_telemetry.get("nativeInterTokenLatencyMs"),
            **native_iteration_fields,
            "runningRequests": native_telemetry.get("runningRequests"),
            "waitingRequests": native_telemetry.get("waitingRequests"),
            "kvCacheUsagePct": native_telemetry.get("kvCacheUsagePct"),
            "cacheHitRate": native_telemetry.get("cacheHitRate"),
            "prefixCacheQueriesDelta": native_telemetry.get("prefixCacheQueriesDelta"),
            "prefixCacheHitsDelta": native_telemetry.get("prefixCacheHitsDelta"),
            "promptTokensCachedDelta": native_telemetry.get("promptTokensCachedDelta"),
            "promptTokensComputedDelta": native_telemetry.get("promptTokensComputedDelta"),
            "hardwareTelemetrySource": hardware_telemetry.get("source"),
            "hardwareMetricsUrl": hardware_telemetry.get("metricsUrl"),
            "avgPowerWatts": hardware_telemetry.get("powerWatts"),
            "avgPowerWattsPerGpu": hardware_telemetry.get("powerWattsPerGpu"),
            "gpuUtilizationPct": hardware_telemetry.get("gpuUtilizationPct"),
            "memoryCopyUtilizationPct": hardware_telemetry.get("memoryCopyUtilizationPct"),
            "gpuTemperatureC": hardware_telemetry.get("gpuTemperatureC"),
            "smClockMHz": hardware_telemetry.get("smClockMHz"),
            "memoryClockMHz": hardware_telemetry.get("memoryClockMHz"),
            "fbUsedMiB": hardware_telemetry.get("fbUsedMiB"),
            "fbFreeMiB": hardware_telemetry.get("fbFreeMiB"),
            "energyJoules": hardware_telemetry.get("energyJoules"),
            "queueWaitMs": native_telemetry.get("queueWaitMs"),
            "prefillMs": native_telemetry.get("prefillMs"),
            "decodeMs": native_telemetry.get("decodeMs"),
            **runtime_provenance,
            **prompt_token_summary,
            **token_summary,
            "tokenTimeline": token_timeline,
            "error": None if 200 <= status < 300 else json.dumps(body),
            "_rawCapture": {
                "requestId": request_id,
                "endpoint": endpoint,
                "requestPayload": payload,
                "responseBody": body,
                "outputText": output_text,
                "promptTokenDetails": prompt_token_details,
                "tokenDetails": raw_token_details,
                "nativeTelemetry": native_telemetry,
                "hardwareTelemetry": hardware_telemetry,
                "runtimeProvenance": runtime_provenance,
            },
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        native_telemetry = _native_telemetry(engine)
        hardware_telemetry = _hardware_telemetry(engine)
        return {
            "requestId": request_id,
            "requestIndex": request_index,
            "endpoint": endpoint,
            "requestStartedAtUtc": request_started_at_utc,
            "requestCompletedAtUtc": _now_iso(),
            "status": 0,
            "ok": False,
            "latencyMs": latency_ms,
            "e2eLatencyMs": latency_ms,
            "timeToFirstByteMs": None,
            "ttftMs": None,
            "ttfotMs": None,
            "tpotMs": None,
            "interTokenLatencyMs": None,
            "streamChunkCount": 0,
            "outputTokenCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "tokenCountSource": "none",
            "ttftSource": "not-streamed",
            "streaming": False,
            "nativeTelemetry": native_telemetry,
            "nativeTelemetryAvailable": False,
            "hardwareTelemetry": hardware_telemetry,
            "hardwareTelemetryAvailable": False,
            "nativeTelemetrySource": native_telemetry.get("source"),
            "hardwareTelemetrySource": hardware_telemetry.get("source"),
            **_runtime_provenance(engine, native_telemetry),
            "tokenDetailsAvailable": False,
            "tokenIdsAvailable": False,
            "logprobsAvailable": False,
            "tokenDetailCount": 0,
            "tokenDetailSource": "error",
            "tokenIdSource": None,
            "promptTokenIdsAvailable": False,
            "promptTokenDetailCount": 0,
            "promptTokenIdSource": None,
            "promptTokenIdsSha256": None,
            "promptTokenizationSource": "error",
            "promptTokenizerModel": None,
            "error": str(exc),
        }


def _send_chat_completion(
    engine: dict[str, Any],
    request: dict[str, Any],
    request_index: int,
    campaign_id: str,
    run_id: str,
    http_post_json: HttpPostJson | None,
    http_stream_json: HttpStreamJson | None,
    http_get_text: HttpGetText | None,
) -> dict[str, Any]:
    if bool(request.get("stream", True)) and http_post_json is None:
        return _send_streaming_chat_completion(engine, request, request_index, campaign_id, run_id, http_stream_json, http_get_text)
    return _send_non_streaming_chat_completion(engine, request, request_index, campaign_id, run_id, http_post_json, http_get_text)


def _sum(samples: list[dict[str, Any]], key: str) -> float:
    return sum(float(sample.get(key) or 0) for sample in samples)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((percentile / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _number_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]


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
    duration_seconds = max(_sum(successful, "e2eLatencyMs") / 1000, 0.001)
    output_tokens = _sum(successful, "completionTokens")
    total_tokens = _sum(successful, "totalTokens")
    prompt_tokens = _sum(successful, "promptTokens")
    latencies = _number_values(successful, "e2eLatencyMs")
    first_bytes = _number_values(successful, "timeToFirstByteMs")
    ttfts = _number_values(successful, "ttftMs")
    ttfots = _number_values(successful, "ttfotMs")
    tpots = _number_values(successful, "tpotMs")
    inter_token_latencies = _number_values(successful, "interTokenLatencyMs")
    queue_waits = _number_values(successful, "queueWaitMs")
    prefills = _number_values(successful, "prefillMs")
    decodes = _number_values(successful, "decodeMs")
    usd_per_gpu_hour = (pricing or {}).get("usdPerGpuHour")
    configured_power_watts_per_gpu = (pricing or {}).get("powerWattsPerGpu")
    observed_power_watts_per_gpu = _avg_numbers(successful, "avgPowerWattsPerGpu")
    power_watts_per_gpu = (
        configured_power_watts_per_gpu
        if isinstance(configured_power_watts_per_gpu, (int, float))
        else observed_power_watts_per_gpu
    )
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
        "avgLatencyMs": _sum(successful, "e2eLatencyMs") / len(successful) if successful else None,
        "p50LatencyMs": _percentile(latencies, 50),
        "p95LatencyMs": _percentile(latencies, 95),
        "p99LatencyMs": _percentile(latencies, 99),
        "avgTimeToFirstByteMs": _avg_numbers(successful, "timeToFirstByteMs"),
        "p50TimeToFirstByteMs": _percentile(first_bytes, 50),
        "p95TimeToFirstByteMs": _percentile(first_bytes, 95),
        "p99TimeToFirstByteMs": _percentile(first_bytes, 99),
        "avgTtftMs": _avg_numbers(successful, "ttftMs"),
        "p50TtftMs": _percentile(ttfts, 50),
        "p95TtftMs": _percentile(ttfts, 95),
        "p99TtftMs": _percentile(ttfts, 99),
        "avgTtfotMs": _avg_numbers(successful, "ttfotMs"),
        "p50TtfotMs": _percentile(ttfots, 50),
        "p95TtfotMs": _percentile(ttfots, 95),
        "p99TtfotMs": _percentile(ttfots, 99),
        "avgTpotMs": _avg_numbers(successful, "tpotMs"),
        "p50TpotMs": _percentile(tpots, 50),
        "p95TpotMs": _percentile(tpots, 95),
        "p99TpotMs": _percentile(tpots, 99),
        "avgInterTokenLatencyMs": _avg_numbers(successful, "interTokenLatencyMs"),
        "p50InterTokenLatencyMs": _percentile(inter_token_latencies, 50),
        "p95InterTokenLatencyMs": _percentile(inter_token_latencies, 95),
        "p99InterTokenLatencyMs": _percentile(inter_token_latencies, 99),
        "avgQueueWaitMs": _avg_numbers(successful, "queueWaitMs"),
        "p50QueueWaitMs": _percentile(queue_waits, 50),
        "p95QueueWaitMs": _percentile(queue_waits, 95),
        "p99QueueWaitMs": _percentile(queue_waits, 99),
        "avgPrefillMs": _avg_numbers(successful, "prefillMs"),
        "p50PrefillMs": _percentile(prefills, 50),
        "p95PrefillMs": _percentile(prefills, 95),
        "p99PrefillMs": _percentile(prefills, 99),
        "avgDecodeMs": _avg_numbers(successful, "decodeMs"),
        "p50DecodeMs": _percentile(decodes, 50),
        "p95DecodeMs": _percentile(decodes, 95),
        "p99DecodeMs": _percentile(decodes, 99),
        "avgNativeIterationLatencyMs": _avg_numbers(successful, "nativeIterationLatencyMs"),
        "avgNativeGpuMemoryBytes": _avg_numbers(successful, "nativeGpuMemoryBytes"),
        "avgNativeKvCacheUsedBlocks": _avg_numbers(successful, "nativeKvCacheUsedBlocks"),
        "avgNativeKvCacheMaxBlocks": _avg_numbers(successful, "nativeKvCacheMaxBlocks"),
        "usdPer1mOutputTokens": cost_usd / (output_tokens / 1_000_000) if cost_usd and output_tokens else None,
        "usdPer1mTotalTokens": cost_usd / (total_tokens / 1_000_000) if cost_usd and total_tokens else None,
        "avgPowerWatts": _avg_numbers(successful, "avgPowerWatts"),
        "avgPowerWattsPerGpu": power_watts_per_gpu if isinstance(power_watts_per_gpu, (int, float)) else None,
        "powerSource": "pricing-config" if isinstance(configured_power_watts_per_gpu, (int, float)) else ("dcgm" if observed_power_watts_per_gpu is not None else "unknown"),
        "avgGpuUtilizationPct": _avg_numbers(successful, "gpuUtilizationPct"),
        "avgMemoryCopyUtilizationPct": _avg_numbers(successful, "memoryCopyUtilizationPct"),
        "totalEnergyJoules": _sum(successful, "energyJoules"),
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
        "dcgmGrounded": bool(successful) and all(sample.get("hardwareTelemetryAvailable") for sample in successful),
        "streamingRequestCount": sum(1 for sample in successful if sample.get("streaming")),
        "nativeTelemetryAvailableCount": sum(1 for sample in successful if sample.get("nativeTelemetryAvailable")),
        "nativeTelemetryRequired": bool(engine.get("requireNativeTelemetry")),
        "hardwareTelemetryAvailableCount": sum(1 for sample in successful if sample.get("hardwareTelemetryAvailable")),
        "hardwareTelemetryRequired": bool(engine.get("requireHardwareTelemetry") or _hardware_metrics_url(engine)),
        "tokenDetailsAvailableCount": sum(1 for sample in successful if sample.get("tokenDetailsAvailable")),
        "tokenIdsAvailableCount": sum(1 for sample in successful if sample.get("tokenIdsAvailable")),
        "logprobsAvailableCount": sum(1 for sample in successful if sample.get("logprobsAvailable")),
        "tokenDetailsRequired": bool(_request_payload(request, stream=bool(request.get("stream", True))).get("logprobs")),
        "promptTokenIdsAvailableCount": sum(1 for sample in successful if sample.get("promptTokenIdsAvailable")),
        "promptTokenDetailsRequired": bool(
            request.get("promptTokenIds")
            or engine.get("promptTokenIds")
            or request.get("tokenizerModel")
            or engine.get("tokenizerModel")
            or engine.get("tokenizer_model")
            or engine.get("tokenizer")
            or request.get("resolveTokenIdsWithTokenizer")
            or engine.get("resolveTokenIdsWithTokenizer")
        ),
        "hardwareProvenance": "configured" if workload.get("hardware") and workload.get("hardware") != "unknown" else "unknown",
        "tags": ",".join(["serving-producer", engine_id, engine_label, request["model"]]),
    }
    required = [
        row["outputTpm"],
        row["totalTpm"],
        row["usdPer1mOutputTokens"],
        row["usdPer1mTotalTokens"],
        row["tokensPerWatt"],
        row["p50LatencyMs"],
        row["p95LatencyMs"],
        row["p99LatencyMs"],
        row["avgTimeToFirstByteMs"],
        row["avgTtftMs"],
        row["p50TtftMs"],
        row["p95TtftMs"],
        row["p99TtftMs"],
        row["avgTpotMs"],
        row["p50TpotMs"],
        row["p95TpotMs"],
        row["p99TpotMs"],
        row["avgTtfotMs"],
        row["p50TtfotMs"],
        row["p95TtfotMs"],
        row["p99TtfotMs"],
        row["requestCount"] if row["requestCount"] == row["successCount"] else None,
        row["streamingRequestCount"] if row["streamingRequestCount"] == row["successCount"] else None,
        1 if row["hardwareProvenance"] == "configured" else None,
    ]
    if row["nativeTelemetryRequired"]:
        required.append(
            row["nativeTelemetryAvailableCount"]
            if row["nativeTelemetryAvailableCount"] == row["successCount"]
            else None
        )
        required.extend([
            row["avgQueueWaitMs"],
            row["p50QueueWaitMs"],
            row["p95QueueWaitMs"],
            row["p99QueueWaitMs"],
            row["avgPrefillMs"],
            row["p50PrefillMs"],
            row["p95PrefillMs"],
            row["p99PrefillMs"],
            row["avgDecodeMs"],
            row["p50DecodeMs"],
            row["p95DecodeMs"],
            row["p99DecodeMs"],
        ])
    if row["hardwareTelemetryRequired"]:
        required.append(
            row["hardwareTelemetryAvailableCount"]
            if row["hardwareTelemetryAvailableCount"] == row["successCount"]
            else None
        )
        required.extend([
            row["avgPowerWatts"],
            row["avgPowerWattsPerGpu"],
            row["avgGpuUtilizationPct"],
            row["avgMemoryCopyUtilizationPct"],
            row["totalEnergyJoules"],
        ])
    if row["tokenDetailsRequired"]:
        required.append(
            row["logprobsAvailableCount"]
            if row["logprobsAvailableCount"] == row["successCount"]
            else None
        )
    if row["promptTokenDetailsRequired"]:
        required.append(
            row["promptTokenIdsAvailableCount"]
            if row["promptTokenIdsAvailableCount"] == row["successCount"]
            else None
        )
    row["metricCompleteness"] = sum(isinstance(value, (int, float)) for value in required) / len(required)
    sample_rows = []
    timeline_rows = []
    for sample in samples:
        sample_rows.append({
            "surface": "serving_request_sample",
            "model": request["model"],
            "hardware": workload.get("hardware"),
            "runtimeFramework": engine_label,
            "runtimeEngine": engine_id,
            "operatingPoint": workload.get("operatingPoint", "laptop-smoke"),
            "basis": "per_request",
            "requestId": sample.get("requestId"),
            "requestIndex": sample.get("requestIndex"),
            "status": sample.get("status"),
            "ok": sample.get("ok"),
            "e2eLatencyMs": sample.get("e2eLatencyMs"),
            "latencyMs": sample.get("latencyMs"),
            "timeToFirstByteMs": sample.get("timeToFirstByteMs"),
            "ttftMs": sample.get("ttftMs"),
            "ttfotMs": sample.get("ttfotMs"),
            "tpotMs": sample.get("tpotMs"),
            "interTokenLatencyMs": sample.get("interTokenLatencyMs"),
            "promptTokens": sample.get("promptTokens"),
            "completionTokens": sample.get("completionTokens"),
            "totalTokens": sample.get("totalTokens"),
            "outputTokenCount": sample.get("outputTokenCount"),
            "tokenCountSource": sample.get("tokenCountSource"),
            "streamChunkCount": sample.get("streamChunkCount"),
            "finishReason": sample.get("finishReason"),
            "ttftSource": sample.get("ttftSource"),
            "promptSha256": sample.get("promptSha256"),
            "requestPayloadSha256": sample.get("requestPayloadSha256"),
            "outputSha256": sample.get("outputSha256"),
            "nativeTelemetryAvailable": sample.get("nativeTelemetryAvailable"),
            "hardwareTelemetryAvailable": sample.get("hardwareTelemetryAvailable"),
            "nativeTelemetrySource": sample.get("nativeTelemetrySource"),
            "nativeMetricsUrl": sample.get("nativeMetricsUrl"),
            "nativeTtftMs": sample.get("nativeTtftMs"),
            "nativeTpotMs": sample.get("nativeTpotMs"),
            "nativeE2eLatencyMs": sample.get("nativeE2eLatencyMs"),
            "nativeInterTokenLatencyMs": sample.get("nativeInterTokenLatencyMs"),
            "nativeIterationLatencyMs": sample.get("nativeIterationLatencyMs"),
            "nativeGpuMemoryBytes": sample.get("nativeGpuMemoryBytes"),
            "nativeKvCacheUsedBlocks": sample.get("nativeKvCacheUsedBlocks"),
            "nativeKvCacheMaxBlocks": sample.get("nativeKvCacheMaxBlocks"),
            "runningRequests": sample.get("runningRequests"),
            "waitingRequests": sample.get("waitingRequests"),
            "kvCacheUsagePct": sample.get("kvCacheUsagePct"),
            "cacheHitRate": sample.get("cacheHitRate"),
            "prefixCacheQueriesDelta": sample.get("prefixCacheQueriesDelta"),
            "prefixCacheHitsDelta": sample.get("prefixCacheHitsDelta"),
            "promptTokensCachedDelta": sample.get("promptTokensCachedDelta"),
            "promptTokensComputedDelta": sample.get("promptTokensComputedDelta"),
            "hardwareTelemetrySource": sample.get("hardwareTelemetrySource"),
            "hardwareMetricsUrl": sample.get("hardwareMetricsUrl"),
            "avgPowerWatts": sample.get("avgPowerWatts"),
            "avgPowerWattsPerGpu": sample.get("avgPowerWattsPerGpu"),
            "gpuUtilizationPct": sample.get("gpuUtilizationPct"),
            "memoryCopyUtilizationPct": sample.get("memoryCopyUtilizationPct"),
            "gpuTemperatureC": sample.get("gpuTemperatureC"),
            "smClockMHz": sample.get("smClockMHz"),
            "memoryClockMHz": sample.get("memoryClockMHz"),
            "fbUsedMiB": sample.get("fbUsedMiB"),
            "fbFreeMiB": sample.get("fbFreeMiB"),
            "energyJoules": sample.get("energyJoules"),
            "tokenDetailsAvailable": sample.get("tokenDetailsAvailable"),
            "tokenIdsAvailable": sample.get("tokenIdsAvailable"),
            "logprobsAvailable": sample.get("logprobsAvailable"),
            "tokenDetailCount": sample.get("tokenDetailCount"),
            "tokenDetailSource": sample.get("tokenDetailSource"),
            "tokenIdSource": sample.get("tokenIdSource"),
            "promptTokenIdsAvailable": sample.get("promptTokenIdsAvailable"),
            "promptTokenDetailCount": sample.get("promptTokenDetailCount"),
            "promptTokenIdSource": sample.get("promptTokenIdSource"),
            "promptTokenIdsSha256": sample.get("promptTokenIdsSha256"),
            "promptTokenizationSource": sample.get("promptTokenizationSource"),
            "promptTokenizerModel": sample.get("promptTokenizerModel"),
            "queueWaitMs": sample.get("queueWaitMs"),
            "prefillMs": sample.get("prefillMs"),
            "decodeMs": sample.get("decodeMs"),
            "engineVersion": sample.get("engineVersion"),
            "modelRevision": sample.get("modelRevision"),
            "imageTag": sample.get("imageTag"),
            "imageDigest": sample.get("imageDigest"),
            "serverArgsSha256": sample.get("serverArgsSha256"),
            "processId": sample.get("processId"),
            "containerId": sample.get("containerId"),
            "podName": sample.get("podName"),
            "nodeName": sample.get("nodeName"),
            "hostName": sample.get("hostName"),
            "latestCapturedAtUtc": captured_at_utc,
        })
        for chunk in sample.get("tokenTimeline") or []:
            if not isinstance(chunk, dict):
                continue
            timeline_rows.append({
                "surface": "serving_token_timeline",
                "model": request["model"],
                "runtimeFramework": engine_label,
                "runtimeEngine": engine_id,
                "requestId": sample.get("requestId"),
                "tokenPhase": chunk.get("tokenPhase", "output"),
                "chunkIndex": chunk.get("chunkIndex"),
                "receivedAtUtc": chunk.get("receivedAtUtc"),
                "relativeMs": chunk.get("relativeMs"),
                "contentBytes": chunk.get("contentBytes"),
                "contentSha256": chunk.get("contentSha256"),
                "isFirstOutput": chunk.get("isFirstOutput"),
                "tokenIndex": chunk.get("tokenIndex"),
                "tokenId": chunk.get("tokenId"),
                "tokenIdSource": chunk.get("tokenIdSource"),
                "tokenLogprob": chunk.get("tokenLogprob"),
                "tokenTextSha256": chunk.get("tokenTextSha256"),
                "topLogprobsJson": chunk.get("topLogprobsJson"),
                "tokenDetailSource": chunk.get("tokenDetailSource"),
                "latestCapturedAtUtc": captured_at_utc,
            })
    return [row, *sample_rows, *timeline_rows]


PRODUCER_COVERAGE_DESCRIPTIONS: dict[str, str] = {
    "clientStreamTiming": "Client stream=true timing for E2E, TTFB, TTFT, TTFOT, TPOT, and output token timeline rows.",
    "nativeRuntimeTelemetry": "Native runtime timing/cache/concurrency fields exposed by vLLM, SGLang, or TensorRT-LLM metrics.",
    "dcgmHardwareTelemetry": "DCGM hardware counters for power, utilization, clocks, memory, temperature, and energy.",
    "promptTokenIds": "Tokenizer-exact prompt/input token IDs and prompt token provenance.",
    "outputTokenIdsLogprobs": "Output token IDs, token logprobs, top-logprobs, and token provenance.",
    "operatorFullArtifacts": "Operator-full raw request/response artifacts retained outside customer-safe rows.",
    "runtimeProvenance": "Engine version, model revision, image, server args, process, container, pod, node, or host provenance.",
}


def _coverage_status(proven: int, expected: int) -> str:
    if expected <= 0:
        return "not_configured"
    if proven >= expected:
        return "proven"
    if proven > 0:
        return "partial"
    return "missing"


def _has_runtime_provenance(sample: dict[str, Any]) -> bool:
    return any(
        sample.get(key) not in (None, "", [])
        for key in [
            "engineVersion",
            "modelRevision",
            "imageDigest",
            "serverArgsSha256",
            "processId",
            "containerId",
            "podName",
            "nodeName",
            "hostName",
        ]
    )


def _coverage_row(
    *,
    request: dict[str, Any],
    workload: dict[str, Any],
    engine_id: str,
    engine_label: str,
    captured_at_utc: str,
    category: str,
    proven: int,
    expected: int,
    missing: list[str],
    all_proven: bool,
) -> dict[str, Any]:
    return {
        "surface": "serving_telemetry_coverage",
        "model": request["model"],
        "hardware": workload.get("hardware"),
        "runtimeFramework": engine_label,
        "runtimeEngine": engine_id,
        "coverageSource": "producer-submit",
        "coverageCategory": category,
        "coverageStatus": _coverage_status(proven, expected),
        "provenCount": proven,
        "expectedCount": expected,
        "missingJson": json.dumps(missing, sort_keys=True, separators=(",", ":")),
        "description": PRODUCER_COVERAGE_DESCRIPTIONS[category],
        "allProven": all_proven,
        "latestCapturedAtUtc": captured_at_utc,
    }


def _build_producer_coverage_rows(
    engine: dict[str, Any],
    request: dict[str, Any],
    workload: dict[str, Any],
    samples: list[dict[str, Any]],
    aggregate_row: dict[str, Any],
    raw_artifact_path: str,
    captured_at_utc: str,
) -> list[dict[str, Any]]:
    engine_id = str(engine["engine"])
    engine_label = serving_engine_label(engine_id)
    successful = [sample for sample in samples if sample.get("ok")]
    expected_samples = len(successful) or len(samples) or 1
    token_required = bool(_request_payload(request, stream=bool(request.get("stream", True))).get("logprobs"))
    prompt_required = bool(aggregate_row.get("promptTokenDetailsRequired"))
    native_expected = expected_samples if aggregate_row.get("nativeTelemetryRequired") else 0
    hardware_expected = expected_samples if aggregate_row.get("hardwareTelemetryRequired") else 0

    coverage_specs: list[tuple[str, int, int, list[str]]] = []
    stream_proven = sum(
        1 for sample in successful
        if sample.get("streaming") is True
        and all(isinstance(sample.get(key), (int, float)) for key in ["e2eLatencyMs", "timeToFirstByteMs", "ttftMs", "ttfotMs", "tpotMs"])
        and any(isinstance(row, dict) and row.get("tokenPhase", "output") == "output" for row in (sample.get("tokenTimeline") or []))
    )
    coverage_specs.append((
        "clientStreamTiming",
        stream_proven,
        expected_samples,
        [] if stream_proven == expected_samples else ["stream timing or output token timeline rows missing"],
    ))

    native_proven = sum(
        1 for sample in successful
        if sample.get("nativeTelemetryAvailable") is True
        and all(isinstance(sample.get(key), (int, float)) for key in ["nativeTtftMs", "nativeTpotMs", "nativeE2eLatencyMs", "queueWaitMs", "prefillMs", "decodeMs"])
    )
    coverage_specs.append((
        "nativeRuntimeTelemetry",
        native_proven,
        native_expected,
        [] if native_expected == 0 or native_proven == native_expected else ["native runtime metrics missing"],
    ))

    hardware_proven = sum(
        1 for sample in successful
        if sample.get("hardwareTelemetryAvailable") is True
        and all(isinstance(sample.get(key), (int, float)) for key in ["avgPowerWatts", "gpuUtilizationPct", "gpuTemperatureC", "energyJoules"])
    )
    coverage_specs.append((
        "dcgmHardwareTelemetry",
        hardware_proven,
        hardware_expected,
        [] if hardware_expected == 0 or hardware_proven == hardware_expected else ["DCGM hardware counters missing"],
    ))

    prompt_proven = sum(
        1 for sample in successful
        if sample.get("promptTokenIdsAvailable") is True
        and isinstance(sample.get("promptTokenIdsSha256"), str)
        and isinstance(sample.get("promptTokenIdSource"), str)
    )
    coverage_specs.append((
        "promptTokenIds",
        prompt_proven,
        expected_samples if prompt_required else 0,
        [] if not prompt_required or prompt_proven == expected_samples else ["prompt token IDs missing"],
    ))

    output_proven = sum(
        1 for sample in successful
        if sample.get("tokenDetailsAvailable") is True
        and sample.get("tokenIdsAvailable") is True
        and sample.get("logprobsAvailable") is True
        and isinstance(sample.get("tokenIdSource"), str)
    )
    coverage_specs.append((
        "outputTokenIdsLogprobs",
        output_proven,
        expected_samples if token_required else 0,
        [] if not token_required or output_proven == expected_samples else ["output token IDs/logprobs missing"],
    ))

    raw_present = 1 if raw_artifact_path and os.path.exists(raw_artifact_path) else 0
    coverage_specs.append((
        "operatorFullArtifacts",
        raw_present,
        1,
        [] if raw_present else ["operator-full raw artifact missing"],
    ))

    runtime_proven = sum(1 for sample in successful if _has_runtime_provenance(sample))
    coverage_specs.append((
        "runtimeProvenance",
        runtime_proven,
        expected_samples,
        [] if runtime_proven == expected_samples else ["runtime provenance missing or partial"],
    ))

    all_proven = all(_coverage_status(proven, expected) in {"proven", "not_configured"} for _, proven, expected, _ in coverage_specs)
    return [
        _coverage_row(
            request=request,
            workload=workload,
            engine_id=engine_id,
            engine_label=engine_label,
            captured_at_utc=captured_at_utc,
            category=category,
            proven=proven,
            expected=expected,
            missing=missing,
            all_proven=all_proven,
        )
        for category, proven, expected, missing in coverage_specs
    ]


def _sanitized_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in sample.items() if not key.startswith("_")} for sample in samples]


def _write_summary_artifact(
    engine: dict[str, Any],
    request: dict[str, Any],
    artifact_dir: str,
    captured_at_utc: str,
    samples: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    raw_artifact_path: str | None,
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    safe_model = _safe_slug(request["model"])
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
                "capturePolicy": {
                    "mode": "operator-full",
                    "rawArtifactPath": raw_artifact_path,
                    "summaryPayload": "redacted-hashes-and-derived-fields",
                },
                "request": _redacted_request(_request_payload(request, stream=bool(request.get("stream", True)))),
                "requestTrace": [
                    {
                        "requestId": sample.get("requestId"),
                        "requestIndex": sample.get("requestIndex"),
                        "endpoint": sample.get("endpoint"),
                        "requestStartedAtUtc": sample.get("requestStartedAtUtc"),
                        "requestCompletedAtUtc": sample.get("requestCompletedAtUtc"),
                        "firstChunkAtUtc": sample.get("firstChunkAtUtc"),
                        "firstOutputAtUtc": sample.get("firstOutputAtUtc"),
                        "lastOutputAtUtc": sample.get("lastOutputAtUtc"),
                        "responseId": sample.get("responseId"),
                        "responseModel": sample.get("responseModel"),
                    }
                    for sample in samples
                ],
                "samples": _sanitized_samples(samples),
                "tokenTimeline": [
                    chunk
                    for sample in samples
                    for chunk in (sample.get("tokenTimeline") or [])
                    if isinstance(chunk, dict)
                ],
                "nativeTelemetry": [
                    {
                        "requestId": sample.get("requestId"),
                        **(sample.get("nativeTelemetry") if isinstance(sample.get("nativeTelemetry"), dict) else {}),
                    }
                    for sample in samples
                ],
                "hardwareTelemetry": [
                    {
                        "requestId": sample.get("requestId"),
                        **(sample.get("hardwareTelemetry") if isinstance(sample.get("hardwareTelemetry"), dict) else {}),
                    }
                    for sample in samples
                ],
                "tokenDetails": [
                    {
                        "requestId": sample.get("requestId"),
                        "tokenDetailsAvailable": sample.get("tokenDetailsAvailable"),
                        "tokenIdsAvailable": sample.get("tokenIdsAvailable"),
                        "logprobsAvailable": sample.get("logprobsAvailable"),
                        "tokenDetailCount": sample.get("tokenDetailCount"),
                        "tokenDetailSource": sample.get("tokenDetailSource"),
                        "tokenIdSource": sample.get("tokenIdSource"),
                    }
                    for sample in samples
                ],
                "promptTokenDetails": [
                    {
                        "requestId": sample.get("requestId"),
                        "promptTokenIdsAvailable": sample.get("promptTokenIdsAvailable"),
                        "promptTokenDetailCount": sample.get("promptTokenDetailCount"),
                        "promptTokenIdSource": sample.get("promptTokenIdSource"),
                        "promptTokenIdsSha256": sample.get("promptTokenIdsSha256"),
                        "promptTokenizationSource": sample.get("promptTokenizationSource"),
                        "promptTokenizerModel": sample.get("promptTokenizerModel"),
                    }
                    for sample in samples
                ],
                "measurements": measurements,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return artifact_path


def _write_raw_artifact(
    engine: dict[str, Any],
    request: dict[str, Any],
    artifact_dir: str,
    captured_at_utc: str,
    raw_captures: list[dict[str, Any]],
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    safe_model = _safe_slug(request["model"])
    raw_path = os.path.join(
        artifact_dir,
        f"{engine['engine']}-{safe_model}-{captured_at_utc.replace(':', '-').replace('.', '-')}-operator-full.json",
    )
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schemaVersion": "performance-iq.serving-operator-full-raw.v1",
                "confidentiality": "operator-full",
                "capturedAtUtc": captured_at_utc,
                "engine": engine["engine"],
                "engineLabel": serving_engine_label(engine["engine"]),
                "requestPayload": _request_payload(request, stream=bool(request.get("stream", True))),
                "captures": raw_captures,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return raw_path


def _write_manifest_artifact(
    engine: dict[str, Any],
    request: dict[str, Any],
    artifact_dir: str,
    captured_at_utc: str,
    manifest: dict[str, Any],
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    safe_model = _safe_slug(request["model"])
    manifest_path = os.path.join(
        artifact_dir,
        f"{engine['engine']}-{safe_model}-{captured_at_utc.replace(':', '-').replace('.', '-')}-manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


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
    http_stream_json: HttpStreamJson | None = None,
    http_get_text: HttpGetText | None = None,
    now: Callable[[], dt.datetime] | None = None,
) -> dict[str, Any]:
    if engine.get("engine") not in SERVING_ENGINE_LABELS:
        raise ValueError(f"unsupported serving engine: {engine.get('engine')}")
    captured_at_utc = _now_iso(now)
    repetitions = max(1, int(request.get("repetitions", 1)))
    campaign_id = (campaign or {}).get("campaignId") or f"serving-{engine['engine']}-{request['model']}"
    run_id = (campaign or {}).get("runId") or f"{campaign_id}-{captured_at_utc.replace(':', '-')}"
    samples = [
        _send_chat_completion(engine, request, index, campaign_id, run_id, http_post_json, http_stream_json, http_get_text)
        for index in range(repetitions)
    ]
    raw_captures = [
        sample.pop("_rawCapture")
        for sample in samples
        if isinstance(sample.get("_rawCapture"), dict)
    ]
    workload = {
        "model": request["model"],
        "hardware": "unknown",
        "operatingPoint": "serving-smoke",
        "scenario": f"OpenAI-compatible chat completions through {serving_engine_label(engine['engine'])}",
        **(workload or {}),
    }
    measurements = _build_measurements(engine, request, workload, pricing, samples, captured_at_utc)
    artifact_root = artifact_dir or os.path.join(os.getcwd(), ".performance-iq", "serving-producers")
    raw_artifact_path = _write_raw_artifact(engine, request, artifact_root, captured_at_utc, raw_captures)
    measurements.extend(_build_producer_coverage_rows(
        engine,
        request,
        workload,
        samples,
        measurements[0],
        raw_artifact_path,
        captured_at_utc,
    ))
    artifact_path = _write_summary_artifact(
        engine,
        request,
        artifact_root,
        captured_at_utc,
        samples,
        measurements,
        raw_artifact_path,
    )
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
        "artifacts": [
            {"kind": "normalized-summary", "path": artifact_path},
            {"kind": "operator-full-serving-raw", "path": raw_artifact_path},
        ],
        "measurements": measurements,
        "platform": {
            "decisionBriefPath": "performance-iq://serving-producer",
            "requestTraceIds": [sample["requestId"] for sample in samples],
        },
        "methodology": (
            f"{serving_engine_label(engine['engine'])} producer sent {repetitions} "
            "OpenAI-compatible chat completion request(s) with x-performance-iq-* "
            "trace headers; default metrics come from client-side streaming SSE "
            "timings, response usage fields, response token logprobs/IDs when "
            "exposed, native telemetry when exposed, and DCGM hardware metrics "
            "when a hardware metrics endpoint is configured."
        ),
        "limitations": [
            "Serving producer captures client stream timing, request-path, usage, latency, token logprobs/IDs when exposed, and provenance; hardware-level DCGM counters require a reachable DCGM/Prometheus metrics endpoint or configured hardware telemetry."
        ],
    }
    if not run_input["runtime"].get("imageTag"):
        del run_input["runtime"]["imageTag"]

    manifest = build_manifest(run_input)
    manifest_path = _write_manifest_artifact(engine, request, artifact_root, captured_at_utc, manifest)
    submission = performance_iq.submit_run(run_input, idempotency_key=manifest["campaign"]["runId"]) if performance_iq and submit else None
    return {
        "engine": engine["engine"],
        "manifest": manifest,
        "manifestPath": manifest_path,
        "runInput": run_input,
        "artifactPath": artifact_path,
        "samples": samples,
        "measurements": measurements,
        "submission": submission,
    }
