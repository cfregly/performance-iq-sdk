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

PROMETHEUS_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)")


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


def _metrics_url(engine: dict[str, Any]) -> str | None:
    configured = engine.get("metricsUrl")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if engine.get("collectNativeMetrics") is True:
        return f"{_normalize_base_url(str(engine['baseUrl']))}/metrics"
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
            value = float(match.group(2))
        except ValueError:
            continue
        metrics[match.group(1)] = metrics.get(match.group(1), 0.0) + value
    return metrics


def _read_native_metrics(engine: dict[str, Any], http_get_text: HttpGetText | None = None) -> dict[str, Any]:
    url = _metrics_url(engine)
    if not url:
        return {"available": False, "source": "metrics-url-not-configured"}
    try:
        text = (http_get_text or _get_text)(url, _metrics_headers(engine))
    except Exception as exc:
        return {"available": False, "source": "prometheus-unavailable", "metricsUrl": url, "error": str(exc)}
    metrics = _parse_prometheus_metrics(text)
    if not metrics:
        return {"available": False, "source": "prometheus-empty", "metricsUrl": url}
    return {
        "available": True,
        "source": "prometheus-snapshot",
        "metricsUrl": url,
        "metrics": metrics,
        "capturedAtUtc": _now_iso(),
    }


def _metric_value(metrics: dict[str, float], candidates: list[str]) -> float | None:
    for name in candidates:
        value = metrics.get(name)
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
        "sglang:request_time_per_output_token_seconds",
        "sglang_request_time_per_output_token_seconds",
        "trtllm:request_time_per_output_token_seconds",
        "trtllm_request_time_per_output_token_seconds",
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
        "sglang:request_queue_time_seconds",
        "sglang_request_queue_time_seconds",
        "trtllm:request_queue_time_seconds",
        "trtllm_request_queue_time_seconds",
    ])
    prefill_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_prefill_time_seconds",
        "sglang:request_prefill_time_seconds",
        "sglang_request_prefill_time_seconds",
        "trtllm:request_prefill_time_seconds",
        "trtllm_request_prefill_time_seconds",
    ])
    decode_ms = _histogram_delta_mean_ms(before_metrics, after_metrics, [
        "vllm:request_decode_time_seconds",
        "sglang:request_decode_time_seconds",
        "sglang_request_decode_time_seconds",
        "trtllm:request_decode_time_seconds",
        "trtllm_request_decode_time_seconds",
    ])
    prefix_queries = _counter_delta(before_metrics, after_metrics, [
        "vllm:prefix_cache_queries_total",
        "sglang:prefix_cache_queries_total",
        "sglang_prefix_cache_queries_total",
        "trtllm:prefix_cache_queries_total",
        "trtllm_prefix_cache_queries_total",
    ])
    prefix_hits = _counter_delta(before_metrics, after_metrics, [
        "vllm:prefix_cache_hits_total",
        "sglang:prefix_cache_hits_total",
        "sglang_prefix_cache_hits_total",
        "trtllm:prefix_cache_hits_total",
        "trtllm_prefix_cache_hits_total",
    ])
    cache_hit_rate = (prefix_hits / prefix_queries) if prefix_hits is not None and prefix_queries and prefix_queries > 0 else None
    values = {
        "nativeTtftMs": native_ttft_ms,
        "nativeTpotMs": native_tpot_ms,
        "nativeInterTokenLatencyMs": native_inter_token_ms,
        "nativeE2eLatencyMs": native_e2e_ms,
        "queueWaitMs": queue_wait_ms,
        "prefillMs": prefill_ms,
        "decodeMs": decode_ms,
        "runningRequests": _metric_value(after_metrics, ["vllm:num_requests_running", "sglang:num_running_reqs", "sglang_num_running_reqs"]),
        "waitingRequests": _metric_value(after_metrics, ["vllm:num_requests_waiting", "sglang:num_queue_reqs", "sglang_num_queue_reqs"]),
        "kvCacheUsagePct": _metric_value(after_metrics, ["vllm:kv_cache_usage_perc", "sglang:token_usage", "sglang_token_usage"]),
        "prefixCacheQueriesDelta": prefix_queries,
        "prefixCacheHitsDelta": prefix_hits,
        "cacheHitRate": cache_hit_rate,
        "promptTokensCachedDelta": _counter_delta(before_metrics, after_metrics, [
            "vllm:prompt_tokens_cached_total",
            "sglang:prompt_tokens_cached_total",
            "sglang_prompt_tokens_cached_total",
        ]),
        "promptTokensComputedDelta": _counter_delta(before_metrics, after_metrics, [
            "vllm:request_prefill_kv_computed_tokens_sum",
            "sglang:request_prefill_kv_computed_tokens_sum",
            "sglang_request_prefill_kv_computed_tokens_sum",
        ]),
    }
    available_values = {key: value for key, value in values.items() if value is not None}
    return {
        "available": bool(available_values),
        "source": "prometheus-delta",
        "metricsUrl": after.get("metricsUrl") or before.get("metricsUrl"),
        "beforeCapturedAtUtc": before.get("capturedAtUtc"),
        "afterCapturedAtUtc": after.get("capturedAtUtc"),
        **available_values,
    }


def _combine_native_telemetry(*items: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    sources: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("source"):
            sources.append(str(item["source"]))
        combined.update(item)
    combined["available"] = any(bool(item.get("available")) for item in items if isinstance(item, dict))
    if sources:
        combined["source"] = "+".join(dict.fromkeys(sources))
    return combined


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
    native_before = _read_native_metrics(engine, http_get_text)
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
                output_chunks.append({
                    "chunkIndex": len(output_chunks),
                    "content": content,
                    "contentBytes": len(content.encode("utf-8")),
                    "contentSha256": _sha256_text(content),
                    "receivedMs": event["receivedMs"],
                    "receivedAtUtc": event["receivedAtUtc"],
                })
        native_after = _read_native_metrics(engine, http_get_text)
        native_telemetry = _combine_native_telemetry(
            native_telemetry,
            _native_metrics_delta(engine, native_before, native_after),
        )
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
            "queueWaitMs": native_telemetry.get("queueWaitMs"),
            "prefillMs": native_telemetry.get("prefillMs"),
            "decodeMs": native_telemetry.get("decodeMs"),
            "tokenTimeline": [
                {
                    "requestId": request_id,
                    "chunkIndex": chunk["chunkIndex"],
                    "receivedAtUtc": chunk["receivedAtUtc"],
                    "relativeMs": chunk["receivedMs"],
                    "contentBytes": chunk["contentBytes"],
                    "contentSha256": chunk["contentSha256"],
                    "isFirstOutput": chunk["chunkIndex"] == 0,
                }
                for chunk in output_chunks
            ],
            "error": None if 200 <= status < 300 else json.dumps(last_body),
            "_rawCapture": {
                "requestId": request_id,
                "endpoint": endpoint,
                "requestPayload": payload,
                "responseEvents": events,
                "outputText": output_text,
            },
        }
    except Exception as exc:
        e2e_latency_ms = (time.perf_counter() - started) * 1000
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
            "nativeTelemetry": _native_telemetry(engine),
            "nativeTelemetryAvailable": False,
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
    native_before = _read_native_metrics(engine, http_get_text)
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
        native_telemetry = _combine_native_telemetry(
            _native_telemetry(engine, body),
            _native_metrics_delta(engine, native_before, native_after),
        )
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
            "queueWaitMs": native_telemetry.get("queueWaitMs"),
            "prefillMs": native_telemetry.get("prefillMs"),
            "decodeMs": native_telemetry.get("decodeMs"),
            "tokenTimeline": [],
            "error": None if 200 <= status < 300 else json.dumps(body),
            "_rawCapture": {
                "requestId": request_id,
                "endpoint": endpoint,
                "requestPayload": payload,
                "responseBody": body,
                "outputText": output_text,
            },
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
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
            "nativeTelemetry": _native_telemetry(engine),
            "nativeTelemetryAvailable": False,
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
    latencies = [float(sample["e2eLatencyMs"]) for sample in successful if isinstance(sample.get("e2eLatencyMs"), (int, float))]
    ttfts = [float(sample["ttftMs"]) for sample in successful if isinstance(sample.get("ttftMs"), (int, float))]
    ttfots = [float(sample["ttfotMs"]) for sample in successful if isinstance(sample.get("ttfotMs"), (int, float))]
    tpots = [float(sample["tpotMs"]) for sample in successful if isinstance(sample.get("tpotMs"), (int, float))]
    first_bytes = [
        float(sample["timeToFirstByteMs"])
        for sample in successful
        if isinstance(sample.get("timeToFirstByteMs"), (int, float))
    ]
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
        "avgLatencyMs": _sum(successful, "e2eLatencyMs") / len(successful) if successful else None,
        "p50LatencyMs": _percentile(latencies, 50),
        "p95LatencyMs": _percentile(latencies, 95),
        "p99LatencyMs": _percentile(latencies, 99),
        "avgTimeToFirstByteMs": _avg_numbers(successful, "timeToFirstByteMs"),
        "avgTtftMs": _avg_numbers(successful, "ttftMs"),
        "p50TtftMs": _percentile(ttfts, 50),
        "p95TtftMs": _percentile(ttfts, 95),
        "p99TtftMs": _percentile(ttfts, 99),
        "avgTtfotMs": _avg_numbers(successful, "ttfotMs"),
        "p95TtfotMs": _percentile(ttfots, 95),
        "avgTpotMs": _avg_numbers(successful, "tpotMs"),
        "p95TpotMs": _percentile(tpots, 95),
        "avgInterTokenLatencyMs": _avg_numbers(successful, "interTokenLatencyMs"),
        "p95TimeToFirstByteMs": _percentile(first_bytes, 95),
        "avgQueueWaitMs": _avg_numbers(successful, "queueWaitMs"),
        "avgPrefillMs": _avg_numbers(successful, "prefillMs"),
        "avgDecodeMs": _avg_numbers(successful, "decodeMs"),
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
        "streamingRequestCount": sum(1 for sample in successful if sample.get("streaming")),
        "nativeTelemetryAvailableCount": sum(1 for sample in successful if sample.get("nativeTelemetryAvailable")),
        "nativeTelemetryRequired": bool(engine.get("requireNativeTelemetry")),
        "hardwareProvenance": "configured" if workload.get("hardware") and workload.get("hardware") != "unknown" else "unknown",
        "tags": ",".join(["serving-producer", engine_id, engine_label, request["model"]]),
    }
    required = [
        row["outputTpm"],
        row["totalTpm"],
        row["usdPer1mOutputTokens"],
        row["usdPer1mTotalTokens"],
        row["tokensPerWatt"],
        row["avgTtftMs"],
        row["avgTpotMs"],
        row["avgTtfotMs"],
        row["requestCount"] if row["requestCount"] == row["successCount"] else None,
        1 if row["hardwareProvenance"] == "configured" else None,
    ]
    if row["nativeTelemetryRequired"]:
        required.append(
            row["nativeTelemetryAvailableCount"]
            if row["nativeTelemetryAvailableCount"] == row["successCount"]
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
            "queueWaitMs": sample.get("queueWaitMs"),
            "prefillMs": sample.get("prefillMs"),
            "decodeMs": sample.get("decodeMs"),
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
                "chunkIndex": chunk.get("chunkIndex"),
                "receivedAtUtc": chunk.get("receivedAtUtc"),
                "relativeMs": chunk.get("relativeMs"),
                "contentBytes": chunk.get("contentBytes"),
                "contentSha256": chunk.get("contentSha256"),
                "isFirstOutput": chunk.get("isFirstOutput"),
                "latestCapturedAtUtc": captured_at_utc,
            })
    return [row, *sample_rows, *timeline_rows]


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
            "timings, response usage fields, and native telemetry when exposed."
        ),
        "limitations": [
            "Serving producer captures client stream timing, request-path, usage, latency, and provenance; hardware-level power/kernel counters require engine-side or cluster instrumentation."
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
