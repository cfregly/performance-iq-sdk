from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

from performance_iq_sdk.client import PerformanceIQ
from performance_iq_sdk.producers.serving import (
    HttpGetText,
    HttpPostJson,
    HttpStreamJson,
    ServingEngineId,
    laptop_smoke_model,
    run_serving_producer,
    serving_engine_label,
)
from performance_iq_sdk.serving_receipts import (
    REQUEST_RECEIPT_SCHEMA_VERSION,
    load_receipts,
    recording_proxy_server,
)

ENGINE_IDS: tuple[ServingEngineId, ...] = ("vllm", "sglang", "tensorrt-llm")
ENGINE_URL_ENV = {
    "vllm": "PIQ_VLLM_URL",
    "sglang": "PIQ_SGLANG_URL",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_URL",
}
ENGINE_API_KEY_ENV = {
    "vllm": "PIQ_VLLM_API_KEY",
    "sglang": "PIQ_SGLANG_API_KEY",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_API_KEY",
}
ENGINE_METRICS_URL_ENV = {
    "vllm": "PIQ_VLLM_METRICS_URL",
    "sglang": "PIQ_SGLANG_METRICS_URL",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_METRICS_URL",
}
ENGINE_JSON_METRICS_URL_ENV = {
    "vllm": "PIQ_VLLM_JSON_METRICS_URL",
    "sglang": "PIQ_SGLANG_JSON_METRICS_URL",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_JSON_METRICS_URL",
}
ENGINE_HARDWARE_METRICS_URL_ENV = {
    "vllm": "PIQ_VLLM_HARDWARE_METRICS_URL",
    "sglang": "PIQ_SGLANG_HARDWARE_METRICS_URL",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_HARDWARE_METRICS_URL",
}
QUERY_NAMES = (
    "price_performance",
    "capacity_best",
    "campaign_provenance",
    "run_details",
    "serving_request_samples",
    "serving_token_timeline",
    "serving_telemetry_coverage",
)
CAMPAIGN_ID_QUERY_COLUMN = {
    "campaign_provenance": 0,
    "run_details": 0,
    "serving_request_samples": 0,
    "serving_token_timeline": 0,
    "serving_telemetry_coverage": 0,
}
ENGINE_DEFAULT_PORT = {
    "vllm": 8000,
    "sglang": 30000,
    "tensorrt-llm": 8001,
}
ENGINE_RUNTIME_PYTHON_ENV = {
    "vllm": "PIQ_VLLM_PYTHON_BIN",
    "sglang": "PIQ_SGLANG_PYTHON_BIN",
    "tensorrt-llm": "PIQ_TENSORRT_LLM_PYTHON_BIN",
}
PROOF_SCHEMA_VERSION = "performance-iq.serving-smoke-proof.v1"
PROOF_VERIFICATION_SCHEMA_VERSION = "performance-iq.serving-smoke-proof-verification.v1"
EVIDENCE_INDEX_SCHEMA_VERSION = "performance-iq.serving-evidence-index.v1"
SERVING_SUMMARY_SCHEMA_VERSION = "performance-iq.serving-producer-summary.v1"
PRODUCER_MANIFEST_SCHEMA_VERSION = "performance-iq.producer-manifest.v1"
SERVING_EVENT_SCHEMA_VERSION = "performance-iq.serving-telemetry-event.v1"
SERVING_EVENT_DEFAULT_TOPIC = "performance-iq.serving.telemetry.v1"
SERVING_KAFKA_PUBLICATION_SCHEMA_VERSION = "performance-iq.serving-kafka-publication.v1"
LOW_FREE_SPACE_BYTES = 30 * 1024 * 1024 * 1024
SIZE_TIMEOUT_SECONDS = 15
COMMAND_PROBE_TIMEOUT_SECONDS = 30
REQUIRED_HARDWARE_SAMPLE_COUNTERS = (
    "avgPowerWatts",
    "avgPowerWattsPerGpu",
    "gpuUtilizationPct",
    "memoryCopyUtilizationPct",
    "gpuTemperatureC",
    "smClockMHz",
    "memoryClockMHz",
    "fbUsedMiB",
    "fbFreeMiB",
    "energyJoules",
)
REQUIRED_NATIVE_SAMPLE_FIELDS = (
    "nativeTtftMs",
    "nativeTpotMs",
    "nativeE2eLatencyMs",
    "queueWaitMs",
    "prefillMs",
    "decodeMs",
    "runningRequests",
    "waitingRequests",
    "kvCacheUsagePct",
    "cacheHitRate",
    "promptTokensCachedDelta",
    "promptTokensComputedDelta",
)
STRICT_PRODUCT_TELEMETRY_CATEGORIES = (
    "clientStreamTiming",
    "requestReceipts",
    "dashboardFineGrainRows",
    "nativeRuntimeTelemetry",
    "dcgmHardwareTelemetry",
    "promptTokenIds",
    "outputTokenIdsLogprobs",
    "operatorFullArtifacts",
    "rawMetricSnapshots",
    "runtimeProvenance",
    "kafkaEventLog",
)
REAL_RUNTIME_PROOF_BLOCKING_MARKERS = (
    "fake",
    "synthetic",
    "fixture",
    "mock",
    "not real runtime",
    "not real-runtime",
    "contract proof only",
)


def default_native_metrics_url(engine: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if engine == "tensorrt-llm":
        return f"{base}/prometheus/metrics"
    return f"{base}/metrics"


def default_native_json_metrics_url(engine: str, base_url: str) -> str | None:
    if engine != "tensorrt-llm":
        return None
    return f"{base_url.rstrip('/')}/metrics"


def _utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "run"


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return fallback
    return value.strip()


def _engine_env_name(engine: str, suffix: str) -> str:
    return f"PIQ_{engine.upper().replace('-', '_')}_{suffix}"


def _json_or_text_env(name: str) -> Any:
    value = _env(name)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _float_env(name: str, fallback: float | None = None) -> float | None:
    value = _env(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _int_env(name: str, fallback: int | None = None) -> int | None:
    value = _env(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _bool_env(name: str, fallback: bool | None = None) -> bool | None:
    value = _env(name)
    if value is None:
        return fallback
    return value.lower() in {"1", "true", "yes", "on"}


def command_probe_timeout_seconds() -> float:
    value = _env("PIQ_COMMAND_PROBE_TIMEOUT_SECONDS")
    if value is None:
        return COMMAND_PROBE_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return COMMAND_PROBE_TIMEOUT_SECONDS
    return max(1.0, timeout)


def preferred_runtime_python(engine: str) -> str | None:
    explicit = _env(ENGINE_RUNTIME_PYTHON_ENV.get(engine, ""))
    if explicit:
        return explicit
    if engine not in {"vllm", "sglang"}:
        return None
    for python_bin in runtime_python_candidates(engine):
        candidate = _runtime_candidate(engine, python_bin)
        if candidate.get("usable") and isinstance(candidate.get("python"), str):
            return str(candidate["python"])
    return None


def engine_configs_from_env(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    urls = {
        "vllm": args.vllm_url or _env("PIQ_VLLM_URL"),
        "sglang": args.sglang_url or _env("PIQ_SGLANG_URL"),
        "tensorrt-llm": args.tensorrt_llm_url or _env("PIQ_TENSORRT_LLM_URL"),
    }
    missing = [f"{engine} ({ENGINE_URL_ENV[engine]})" for engine in ENGINE_IDS if not urls[engine]]
    configs: list[dict[str, Any]] = []
    for engine in ENGINE_IDS:
        url = urls[engine]
        if not url:
            continue
        api_key = _env(ENGINE_API_KEY_ENV[engine])
        metrics_url = _env(ENGINE_METRICS_URL_ENV[engine], default_native_metrics_url(engine, url))
        json_metrics_url = _env(ENGINE_JSON_METRICS_URL_ENV[engine], default_native_json_metrics_url(engine, url))
        hardware_metrics_url = _env(ENGINE_HARDWARE_METRICS_URL_ENV[engine])
        if getattr(args, "collect_hardware_metrics", False) and not hardware_metrics_url:
            hardware_metrics_url = metrics_url
        framework_version = _env(_engine_env_name(engine, "FRAMEWORK_VERSION"), args.framework_version)
        model_revision = _env(_engine_env_name(engine, "MODEL_REVISION"), getattr(args, "model_revision", None))
        image_digest = _env(_engine_env_name(engine, "IMAGE_DIGEST"), args.image_digest)
        image_tag = _env(_engine_env_name(engine, "IMAGE_TAG"), args.image_tag)
        server_args = _json_or_text_env(_engine_env_name(engine, "SERVER_ARGS"))
        if server_args is None:
            server_args = getattr(args, "server_args", None)
        tokenizer_model = _env(_engine_env_name(engine, "TOKENIZER_MODEL"), getattr(args, "tokenizer_model", None))
        resolve_token_ids = bool(getattr(args, "resolve_token_ids_with_tokenizer", False))
        tokenizer_python_bin = preferred_runtime_python(engine) if resolve_token_ids else None
        capture_token_details = _bool_env(_engine_env_name(engine, "CAPTURE_TOKEN_DETAILS"))
        top_logprobs = _int_env(_engine_env_name(engine, "TOP_LOGPROBS"))
        process_id = _env(_engine_env_name(engine, "PROCESS_ID"), getattr(args, "process_id", None))
        container_id = _env(_engine_env_name(engine, "CONTAINER_ID"), getattr(args, "container_id", None))
        pod_name = _env(_engine_env_name(engine, "POD_NAME"), getattr(args, "pod_name", None))
        node_name = _env(_engine_env_name(engine, "NODE_NAME"), getattr(args, "node_name", None))
        host_name = _env(_engine_env_name(engine, "HOST_NAME"), getattr(args, "host_name", None))
        configs.append({
            "engine": engine,
            "baseUrl": url,
            "metricsUrl": metrics_url,
            **({"nativeJsonMetricsUrl": json_metrics_url} if json_metrics_url else {}),
            **({"hardwareMetricsUrl": hardware_metrics_url} if hardware_metrics_url else {}),
            **({"requireNativeTelemetry": True} if getattr(args, "require_native_telemetry", False) else {}),
            **({"requireHardwareTelemetry": True} if getattr(args, "require_hardware_telemetry", False) else {}),
            **({"apiKey": api_key} if api_key else {}),
            **({"frameworkVersion": framework_version} if framework_version else {}),
            **({"modelRevision": model_revision} if model_revision else {}),
            **({"imageDigest": image_digest} if image_digest else {}),
            **({"imageTag": image_tag} if image_tag else {}),
            **({"serverArgs": server_args} if server_args is not None else {}),
            **({"tokenizerModel": tokenizer_model} if tokenizer_model else {}),
            **({"resolveTokenIdsWithTokenizer": True} if resolve_token_ids else {}),
            **({"tokenizerPythonBin": tokenizer_python_bin} if tokenizer_python_bin else {}),
            **({"captureTokenDetails": capture_token_details} if capture_token_details is not None else {}),
            **({"topLogprobs": top_logprobs} if top_logprobs is not None else {}),
            **({"processId": process_id} if process_id else {}),
            **({"containerId": container_id} if container_id else {}),
            **({"podName": pod_name} if pod_name else {}),
            **({"nodeName": node_name} if node_name else {}),
            **({"hostName": host_name} if host_name else {}),
        })
    return configs, missing


def command_probe(command: str, *args: str) -> dict[str, Any]:
    if os.path.sep in command:
        path = os.path.abspath(command)
        if not os.path.exists(path):
            return {"command": command, "available": False, "status": "missing"}
    else:
        path = shutil.which(command)
    if not path:
        return {"command": command, "available": False, "status": "missing"}
    try:
        result = subprocess.run(
            [path, *args],
            text=True,
            capture_output=True,
            timeout=command_probe_timeout_seconds(),
        )
    except Exception as exc:
        return {
            "command": command,
            "path": path,
            "available": True,
            "status": "error",
            "detail": str(exc),
        }
    output = (result.stdout or result.stderr).strip()
    return {
        "command": command,
        "path": path,
        "available": True,
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "output": output.splitlines()[:8],
    }


def module_probe(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    return {
        "module": module,
        "available": spec is not None,
        "origin": spec.origin if spec else None,
    }


def vllm_extension_probe() -> dict[str, Any]:
    find_error = None
    try:
        spec = importlib.util.find_spec("vllm._C")
    except ModuleNotFoundError as exc:
        spec = None
        find_error = str(exc)
    result: dict[str, Any] = {
        "module": "vllm._C",
        "available": spec is not None,
        "origin": spec.origin if spec else None,
    }
    if find_error:
        result["findError"] = find_error
    if spec is None:
        return result
    try:
        importlib.import_module("vllm._C")
        result["imported"] = True
    except Exception as exc:
        result["imported"] = False
        result["importError"] = str(exc)
        return result
    try:
        import torch  # type: ignore[import-not-found]

        result["torchCInitCpuMemoryEnv"] = hasattr(torch.ops._C, "init_cpu_memory_env")
    except Exception as exc:
        result["torchProbeError"] = str(exc)
    return result


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _workspace_root() -> str:
    return os.path.abspath(os.path.join(_repo_root(), ".."))


def _unique_paths(paths: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if not path:
            continue
        absolute = os.path.abspath(os.path.expanduser(path))
        if absolute not in seen:
            unique.append(absolute)
            seen.add(absolute)
    return unique


def _source_roots(source_path: str | None) -> list[str]:
    if not source_path:
        return []
    absolute = os.path.abspath(os.path.expanduser(source_path))
    roots = [absolute]
    if os.path.basename(absolute) == "python":
        roots.append(os.path.dirname(absolute))
    return _unique_paths(roots)


def runtime_python_candidates(engine: str) -> list[str]:
    workspace_root = _workspace_root()
    explicit = _env(ENGINE_RUNTIME_PYTHON_ENV.get(engine, ""))
    candidates: list[str | None] = [explicit]
    if engine == "vllm":
        source_roots = _source_roots(_env("PIQ_VLLM_SOURCE_PATH", "/Users/admin/vllm"))
        for root in source_roots:
            candidates.extend([
                os.path.join(root, ".venv-piq/bin/python"),
                os.path.join(root, ".venv/bin/python"),
            ])
        candidates.append(os.path.join(workspace_root, ".runtime/vllm-macos/bin/python"))
    elif engine == "sglang":
        source_roots = _source_roots(_env("PIQ_SGLANG_SOURCE_PATH", os.path.join(workspace_root, ".runtime/sglang/python")))
        for root in source_roots:
            candidates.extend([
                os.path.join(root, "sglang-metal/bin/python"),
                os.path.join(root, ".venv/bin/python"),
                os.path.join(root, ".venv-piq/bin/python"),
            ])
        candidates.extend([
            os.path.join(workspace_root, ".runtime/sglang/sglang-metal/bin/python"),
            os.path.join(workspace_root, ".runtime/sglang/.venv/bin/python"),
        ])
    return _unique_paths(candidates)


def external_python_module_probe(python_bin: str, module: str, *, import_module: bool = False) -> dict[str, Any]:
    if not os.path.exists(python_bin):
        return {
            "python": python_bin,
            "module": module,
            "available": False,
            "status": "missing-python",
        }
    code = """
import importlib
import importlib.util
import json
import sys

module = sys.argv[1]
should_import = sys.argv[2] == "1"
result = {"python": sys.executable, "module": module}
try:
    spec = importlib.util.find_spec(module)
except Exception as exc:
    result.update({"available": False, "origin": None, "findError": str(exc)})
else:
    result.update({"available": spec is not None, "origin": spec.origin if spec else None})

if result.get("available") and should_import:
    try:
        importlib.import_module(module)
        result["imported"] = True
    except Exception as exc:
        result["imported"] = False
        result["importError"] = str(exc)
    if module == "vllm._C":
        try:
            import torch
            result["torchCInitCpuMemoryEnv"] = hasattr(torch.ops._C, "init_cpu_memory_env")
        except Exception as exc:
            result["torchProbeError"] = str(exc)

print(json.dumps(result, sort_keys=True))
""".strip()
    try:
        completed = subprocess.run(
            [python_bin, "-c", code, module, "1" if import_module else "0"],
            text=True,
            capture_output=True,
            timeout=command_probe_timeout_seconds(),
        )
    except Exception as exc:
        return {
            "python": python_bin,
            "module": module,
            "available": False,
            "status": "error",
            "detail": str(exc),
        }
    output = (completed.stdout or "").splitlines()
    parsed: dict[str, Any] | None = None
    for line in reversed(output):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        return {
            "python": python_bin,
            "module": module,
            "available": False,
            "status": "error",
            "returncode": completed.returncode,
            "output": ((completed.stdout or completed.stderr).strip()).splitlines()[:8],
        }
    parsed["status"] = "ok" if completed.returncode == 0 else "error"
    parsed["returncode"] = completed.returncode
    if completed.stderr.strip():
        parsed["stderrPreview"] = completed.stderr.strip().splitlines()[:8]
    return parsed


def _runtime_candidate(engine: str, python_bin: str) -> dict[str, Any]:
    module_name = "vllm" if engine == "vllm" else "sglang"
    candidate: dict[str, Any] = {
        "python": python_bin,
        "module": external_python_module_probe(python_bin, module_name),
    }
    if engine == "vllm":
        vllm_command = os.path.join(os.path.dirname(python_bin), "vllm")
        candidate["command"] = command_probe(vllm_command, "--version")
        candidate["extension"] = external_python_module_probe(python_bin, "vllm._C", import_module=True)
        candidate["usable"] = bool(
            candidate["module"].get("available")
            and candidate["extension"].get("available")
            and candidate["extension"].get("imported") is not False
        )
    elif engine == "sglang":
        candidate["launchModule"] = external_python_module_probe(python_bin, "sglang.launch_server")
        candidate["usable"] = bool(
            candidate["module"].get("available")
            and candidate["launchModule"].get("available")
        )
    else:
        candidate["usable"] = False
    return candidate


def local_runtime_discovery() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    for engine in ("vllm", "sglang"):
        candidates = [_runtime_candidate(engine, path) for path in runtime_python_candidates(engine)]
        usable_candidates = [candidate for candidate in candidates if candidate.get("usable")]
        discovered[engine] = {
            "pythonEnv": ENGINE_RUNTIME_PYTHON_ENV[engine],
            "candidates": candidates,
            "usable": bool(usable_candidates),
            "preferred": usable_candidates[0] if usable_candidates else None,
        }
    return discovered


def storage_probe(path: str | None = None) -> dict[str, Any]:
    target = path or os.getcwd()
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return {"path": target, "available": False, "error": str(exc)}
    return {
        "path": target,
        "available": True,
        "totalBytes": usage.total,
        "freeBytes": usage.free,
        "freeGiB": round(usage.free / (1024 ** 3), 2),
        "lowFreeSpace": usage.free < LOW_FREE_SPACE_BYTES,
    }


def directory_size(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    if not os.path.isdir(path):
        return {"path": path, "exists": True, "directory": False}
    try:
        result = subprocess.run(
            ["du", "-sk", path],
            text=True,
            capture_output=True,
            timeout=SIZE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return {"path": path, "exists": True, "directory": True, "error": str(exc)}
    if result.returncode != 0:
        return {
            "path": path,
            "exists": True,
            "directory": True,
            "error": (result.stderr or result.stdout).strip(),
        }
    size_kib = int((result.stdout.strip().split() or ["0"])[0])
    return {
        "path": path,
        "exists": True,
        "directory": True,
        "sizeBytes": size_kib * 1024,
        "sizeGiB": round(size_kib / (1024 ** 2), 2),
    }


def directory_children(path: str, limit: int = 12) -> list[dict[str, Any]]:
    if not os.path.isdir(path):
        return []
    children: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return []
    for name in names[:limit]:
        child_path = os.path.join(path, name)
        children.append(directory_size(child_path))
    return children


def huggingface_cache_roots() -> list[str]:
    roots: list[str] = []
    hf_home = _env("HF_HOME")
    if hf_home:
        roots.append(os.path.join(os.path.expanduser(hf_home), "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen: set[str] = set()
    unique_roots: list[str] = []
    for root in roots:
        absolute = os.path.abspath(root)
        if absolute not in seen:
            unique_roots.append(absolute)
            seen.add(absolute)
    return unique_roots


def huggingface_model_cache_name(model: str) -> str:
    return "models--" + model.replace("/", "--")


def model_cache_diagnostics(model: str) -> dict[str, Any]:
    cache_name = huggingface_model_cache_name(model)
    candidates = [os.path.join(root, cache_name) for root in huggingface_cache_roots()]
    return {
        "model": model,
        "huggingFaceCacheName": cache_name,
        "candidates": [directory_size(path) for path in candidates],
    }


def cache_diagnostics(model: str) -> dict[str, Any]:
    cache_roots = huggingface_cache_roots()
    roots = [
        *cache_roots,
        os.path.expanduser("~/.cache/pip"),
        os.path.expanduser("~/.cache/uv"),
    ]
    return {
        "modelCache": model_cache_diagnostics(model),
        "roots": [
            {
                **directory_size(root),
                "children": directory_children(root, limit=8),
            }
            for root in roots
        ],
    }


def port_diagnostics(port: int) -> dict[str, Any]:
    connected = False
    error: str | None = None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            connected = True
    except OSError as exc:
        error = str(exc)
    owner: dict[str, Any] | None = None
    lsof = shutil.which("lsof")
    if lsof:
        try:
            result = subprocess.run(
                [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                text=True,
                capture_output=True,
                timeout=5,
            )
            output = (result.stdout or result.stderr).strip()
            owner = {
                "command": "lsof",
                "status": "ok" if result.returncode == 0 else "not-listening",
                "output": output.splitlines()[:8],
            }
        except Exception as exc:
            owner = {"command": "lsof", "status": "error", "error": str(exc)}
    return {
        "port": port,
        "connects": connected,
        **({"error": error} if error else {}),
        **({"owner": owner} if owner is not None else {}),
    }


def engine_port_diagnostics() -> dict[str, Any]:
    return {
        engine: port_diagnostics(port)
        for engine, port in ENGINE_DEFAULT_PORT.items()
    }


def environment_diagnostics() -> dict[str, Any]:
    names = [
        "PIQ_BASE_URL",
        "PIQ_TOKEN",
        "PIQ_SERVING_MODEL",
        "PIQ_VLLM_URL",
        "PIQ_VLLM_METRICS_URL",
        "PIQ_SGLANG_URL",
        "PIQ_SGLANG_METRICS_URL",
        "PIQ_TENSORRT_LLM_URL",
        "PIQ_TENSORRT_LLM_METRICS_URL",
        "PIQ_TENSORRT_LLM_JSON_METRICS_URL",
        "PIQ_TENSORRT_LLM_IMAGE",
        "PIQ_PYTHON_BIN",
        "PIQ_SERVING_BIN_DIR",
        "PIQ_VLLM_PYTHON_BIN",
        "PIQ_SGLANG_PYTHON_BIN",
        "PIQ_TENSORRT_LLM_PYTHON_BIN",
        "PIQ_VLLM_SOURCE_PATH",
        "PIQ_SGLANG_SOURCE_PATH",
        "PIQ_VLLM_CAPTURE_TOKEN_DETAILS",
        "PIQ_SGLANG_CAPTURE_TOKEN_DETAILS",
        "PIQ_TENSORRT_LLM_CAPTURE_TOKEN_DETAILS",
        "PIQ_VLLM_TOP_LOGPROBS",
        "PIQ_SGLANG_TOP_LOGPROBS",
        "PIQ_TENSORRT_LLM_TOP_LOGPROBS",
        "PIQ_COMMAND_PROBE_TIMEOUT_SECONDS",
        "HF_HOME",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    return {
        name: {"set": bool(_env(name))}
        for name in names
    }


def real_engine_blockers(preflight: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    storage = preflight.get("storage", {})
    if storage.get("lowFreeSpace"):
        blockers.append(
            f"Only {storage.get('freeGiB')} GiB free under {storage.get('path')}; source builds and model downloads are likely to fail."
        )
    for missing in preflight.get("missingEngineUrls", []):
        blockers.append(f"Missing configured endpoint URL for {missing}.")
    local = preflight.get("localRuntime", {})
    runtime_candidates = local.get("runtimeCandidates", {})
    vllm_candidate = runtime_candidates.get("vllm", {}) if isinstance(runtime_candidates, dict) else {}
    sglang_candidate = runtime_candidates.get("sglang", {}) if isinstance(runtime_candidates, dict) else {}
    if not local.get("vllmModule", {}).get("available") and not vllm_candidate.get("usable"):
        blockers.append("Python module 'vllm' is not importable in the smoke-runner Python environment.")
    elif local.get("vllmModule", {}).get("available") and not local.get("vllmExtension", {}).get("available") and not vllm_candidate.get("usable"):
        blockers.append("Python module 'vllm' is importable, but compiled extension 'vllm._C' is missing.")
    elif local.get("vllmExtension", {}).get("imported") is False and not vllm_candidate.get("usable"):
        blockers.append("Python module 'vllm._C' is present, but failed to import in the smoke-runner Python environment.")
    if not local.get("sglangModule", {}).get("available") and not sglang_candidate.get("usable"):
        blockers.append("Python module 'sglang' is not importable in the smoke-runner Python environment.")
    if not local.get("tensorrtLlmServeCommand", {}).get("available"):
        blockers.append("'trtllm-serve' is not available on PATH.")
    if not local.get("nvidiaSmiCommand", {}).get("available"):
        blockers.append("'nvidia-smi' is not available; this host cannot prove local TensorRT-LLM on NVIDIA hardware.")
    for item in preflight.get("endpoints", []):
        if not item.get("ok"):
            blockers.append(
                f"{item.get('engine')} endpoint is not ready at {item.get('url')}: "
                f"{item.get('status', item.get('error', 'unknown error'))}."
            )
    for engine, detail in diagnostics.get("ports", {}).items():
        if detail.get("connects") and engine in {"vllm", "tensorrt-llm"}:
            owner_output = detail.get("owner", {}).get("output", [])
            if owner_output:
                blockers.append(f"Default {engine} port {detail.get('port')} is already occupied: {owner_output[-1]}")
    return blockers


def runtime_diagnostics(engines: list[dict[str, Any]], missing_urls: list[str], model: str) -> dict[str, Any]:
    preflight = runtime_preflight(engines, missing_urls, model=model)
    diagnostics = {
        "environment": environment_diagnostics(),
        "ports": engine_port_diagnostics(),
        "caches": cache_diagnostics(model),
    }
    return {
        "preflight": preflight,
        "diagnostics": diagnostics,
        "blockers": real_engine_blockers(preflight, diagnostics),
    }


def _model_ids(body: dict[str, Any]) -> list[str]:
    items = body.get("data")
    if not isinstance(items, list):
        return []
    model_ids: list[str] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])
    return model_ids


def _with_model_check(result: dict[str, Any], body: dict[str, Any] | None, model: str | None) -> dict[str, Any]:
    if not model or body is None:
        return result
    served_models = _model_ids(body)
    if not served_models:
        return {
            **result,
            "ok": False,
            "modelChecked": False,
            "servedModels": [],
            "modelAvailable": None,
            "error": "GET /v1/models did not return standard data[].id model entries.",
        }
    model_available = model in served_models
    return {
        **result,
        "ok": bool(result.get("ok")) and model_available,
        "modelChecked": True,
        "servedModels": served_models[:20],
        "modelAvailable": model_available,
    }


def endpoint_probe(engine: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    base_url = str(engine["baseUrl"]).rstrip("/")
    url = f"{base_url}/v1/models"
    headers = {
        "accept": "application/json",
        **({"authorization": f"Bearer {engine['apiKey']}"} if engine.get("apiKey") else {}),
    }
    request = urllib.request.Request(
        url,
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_body = response.read(4096).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw_body) if raw_body.strip().startswith("{") else None
            except json.JSONDecodeError:
                body = None
            result = _with_model_check({
                "engine": engine["engine"],
                "url": url,
                "reachable": True,
                "status": response.status,
                "ok": 200 <= response.status < 300,
                "bodyPreview": raw_body[:300],
            }, body, model)
            if engine.get("engine") == "sglang":
                result.update(_sglang_runtime_endpoint_info(base_url, headers))
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read(2048).decode("utf-8", errors="replace")
        return {
            "engine": engine["engine"],
            "url": url,
            "reachable": True,
            "status": exc.code,
            "ok": False,
            "authFailed": exc.code in {401, 403},
            "bodyPreview": body[:300],
        }
    except Exception as exc:
        return {
            "engine": engine["engine"],
            "url": url,
            "reachable": False,
            "ok": False,
            "error": str(exc),
        }


SGLANG_SERVER_INFO_KEYS = (
    "device",
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "weight_version",
    "context_length",
    "max_total_tokens",
    "disable_overlap_schedule",
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "sampling_backend",
    "enable_metrics",
    "enable_cache_report",
)


def _json_endpoint(base_url: str, path: str, headers: dict[str, str], *, max_bytes: int = 65536) -> dict[str, Any] | None:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        raw_body = response.read(max_bytes).decode("utf-8", errors="replace")
    body = json.loads(raw_body) if raw_body.strip().startswith("{") else None
    return body if isinstance(body, dict) else None


def _sglang_runtime_endpoint_info(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        model_info = _json_endpoint(base_url, "/model_info", headers)
    except Exception as exc:
        result["modelInfoError"] = str(exc)
    else:
        if model_info:
            result["modelInfo"] = {
                key: value
                for key, value in model_info.items()
                if key in {
                    "model_path",
                    "tokenizer_path",
                    "is_generation",
                    "weight_version",
                    "model_type",
                    "architectures",
                }
            }
            result["modelInfoSha256"] = _sha256_json(model_info)
    try:
        server_info = _json_endpoint(base_url, "/server_info", headers)
    except Exception as exc:
        result["serverInfoError"] = str(exc)
    else:
        if server_info:
            result["serverInfo"] = {
                key: server_info.get(key)
                for key in SGLANG_SERVER_INFO_KEYS
                if key in server_info
            }
            result["serverInfoSha256"] = _sha256_json(server_info)
    return result


def _sglang_mps_logprobs_unsafe(engine: dict[str, Any]) -> bool:
    if engine.get("engine") != "sglang":
        return False
    endpoint_preflight = engine.get("endpointPreflight") if isinstance(engine.get("endpointPreflight"), dict) else {}
    server_info = endpoint_preflight.get("serverInfo") if isinstance(endpoint_preflight.get("serverInfo"), dict) else {}
    if str(server_info.get("device") or "").lower() == "mps":
        return True
    if _env("PIQ_SGLANG_BACKEND", "").lower() in {"mlx", "mps", "metal"}:
        return True
    return False


def token_detail_capability_policy(engine: dict[str, Any], requested: bool) -> dict[str, Any] | None:
    if not requested:
        return None
    if _sglang_mps_logprobs_unsafe(engine) and _bool_env("PIQ_SGLANG_ALLOW_UNSAFE_TOKEN_DETAILS") is not True:
        return {
            "requested": True,
            "supported": False,
            "safeToRequest": False,
            "status": "unsupported-runtime",
            "reason": "sglang-mps-mlx-logprobs-crash",
            "evidence": (
                "Local SGLang MPS/MLX returns no decode next_token_logprobs; "
                "chat logprobs=true crashed scheduler _normalize_decode_outputs with NoneType.tolist."
            ),
            "operatorOverrideEnv": "PIQ_SGLANG_ALLOW_UNSAFE_TOKEN_DETAILS=true",
        }
    return {
        "requested": True,
        "supported": True,
        "safeToRequest": True,
        "status": "requested",
    }


def _runtime_discovery_summary(runtime_candidates: dict[str, Any], engine: str) -> dict[str, Any]:
    engine_candidates = runtime_candidates.get(engine) if isinstance(runtime_candidates.get(engine), dict) else {}
    preferred = engine_candidates.get("preferred") if isinstance(engine_candidates.get("preferred"), dict) else {}
    command = preferred.get("command") if isinstance(preferred.get("command"), dict) else {}
    return {
        "pythonEnv": ENGINE_RUNTIME_PYTHON_ENV[engine],
        "usable": bool(engine_candidates.get("usable")),
        "preferredPython": preferred.get("python"),
        "preferredCommand": command.get("command") or command.get("path"),
    }


def runtime_launch_plan(model: str, runtime_candidates: dict[str, Any] | None = None) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    apple_silicon = system == "Darwin" and machine == "arm64"
    linux_nvidia = system == "Linux" and shutil.which("nvidia-smi") is not None
    quoted_model = shlex.quote(model)
    storage = storage_probe()
    warnings = []
    if storage.get("lowFreeSpace"):
        warnings.append(
            f"{storage.get('freeGiB')} GiB free under {storage.get('path')}; source builds and model downloads may fail."
        )
    runtime_candidates = runtime_candidates if runtime_candidates is not None else local_runtime_discovery()
    runtime_discovery = {
        "vllm": _runtime_discovery_summary(runtime_candidates, "vllm"),
        "sglang": _runtime_discovery_summary(runtime_candidates, "sglang"),
    }
    vllm_command = runtime_discovery["vllm"].get("preferredCommand") or "vllm"
    sglang_python = runtime_discovery["sglang"].get("preferredPython") or "python"
    return {
        "model": model,
        "host": {
            "system": system,
            "machine": machine,
        },
        "storage": storage,
        "warnings": warnings,
        "endpointEnv": {
            "PIQ_VLLM_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['vllm']}",
            "PIQ_SGLANG_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['sglang']}",
            "PIQ_TENSORRT_LLM_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['tensorrt-llm']}",
            "PIQ_VLLM_METRICS_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['vllm']}/metrics",
            "PIQ_SGLANG_METRICS_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['sglang']}/metrics",
            "PIQ_TENSORRT_LLM_METRICS_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['tensorrt-llm']}/prometheus/metrics",
            "PIQ_TENSORRT_LLM_JSON_METRICS_URL": f"http://127.0.0.1:{ENGINE_DEFAULT_PORT['tensorrt-llm']}/metrics",
        },
        "runtimeDiscovery": runtime_discovery,
        "telemetryModel": {
            "streamingCollection": (
                "The producer sends OpenAI-compatible stream=true requests, reads SSE data frames as they arrive, "
                "and timestamps first byte, first output token/content, each chunk/token row, and completion on the measuring client."
            ),
            "kafkaBoundary": (
                "Kafka is a post-capture ingestion/export boundary for already timestamped serving events; "
                "do not place Kafka between the producer and serving engine when measuring TTFT/TPOT."
            ),
            "eventLog": "PIQ_SERVING_EVENT_LOG writes Kafka-ready JSONL events after capture.",
            "requestSurfaces": [
                "serving_request_samples",
                "serving_token_timeline",
                "serving_telemetry_coverage",
            ],
            "strictTelemetry": [
                "client stream timing",
                "native engine timing/cache/concurrency metrics",
                "DCGM hardware counters",
                "tokenizer-exact prompt token IDs",
                "response or tokenizer-resolved output token IDs",
                "response logprobs/top-logprobs",
                "operator-full raw artifacts",
                "raw native/DCGM metric snapshots",
                "request receipts",
                "runtime provenance",
            ],
        },
        "strictProof": {
            "mode": "strict-recorded-smoke",
            "requires": [
                "OpenAI-compatible streaming chat completions for all configured engines",
                "OpenAI-compatible token logprobs plus response or tokenizer-resolved token IDs",
                "native engine Prometheus metrics",
                "DCGM exporter Prometheus metrics",
                "configured per-GPU hourly cost",
                "Performance IQ dashboard query access",
            ],
            "dashboardSurfaces": list(QUERY_NAMES),
            "command": (
                f"PIQ_SERVING_MODEL={quoted_model} "
                "PIQ_SERVING_USD_PER_GPU_HOUR=<actual-blended-gpu-hourly-cost> "
                "PIQ_SERVING_CAPTURE_TOKEN_DETAILS=true "
                "PIQ_SERVING_TOP_LOGPROBS=5 "
                "PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER=true "
                "PIQ_SERVING_COLLECT_HARDWARE_METRICS=true "
                "PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY=true "
                "PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY=true "
                "PIQ_SERVING_VERIFY_AFTER_CAPTURE=true "
                "PIQ_SERVING_REQUIRE_TELEMETRY_COVERAGE=true "
                "PIQ_SERVING_REQUIRE_REAL_RUNTIME_PROOF=true "
                "bash ops/serving-producers/run-smoke.sh strict-recorded-smoke"
            ),
            "verify": (
                "bash ops/serving-producers/run-smoke.sh verify-proof "
                "$PIQ_ARTIFACT_DIR/serving-smoke-proof-<suffix>.json "
                "--require-telemetry-coverage --require-real-runtime-proof"
            ),
        },
        "engines": {
            "vllm": {
                "localSupport": "apple-silicon-source-build-required" if apple_silicon else "install-required",
                "install": [
                    "git clone https://github.com/vllm-project/vllm.git",
                    "cd vllm",
                    "uv venv --python 3.12 --seed --managed-python",
                    "source .venv/bin/activate",
                    "uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match",
                    "uv pip install -e .",
                ] if apple_silicon else [
                    "Follow the vLLM CPU/GPU install path for this host, then run the serve command below.",
                ],
                "serve": (
                    f"{shlex.quote(str(vllm_command))} serve {quoted_model} --host 127.0.0.1 "
                    f"--port {ENGINE_DEFAULT_PORT['vllm']} --served-model-name {quoted_model}"
                ),
                "verify": f"curl -fsS http://127.0.0.1:{ENGINE_DEFAULT_PORT['vllm']}/v1/models",
            },
            "sglang": {
                "localSupport": "apple-metal-source-build-required" if apple_silicon else "install-required",
                "install": [
                    "git clone https://github.com/sgl-project/sglang.git",
                    "cd sglang",
                    "uv venv -p 3.12 sglang-metal",
                    "source sglang-metal/bin/activate",
                    "uv pip install --upgrade pip",
                    "uv run sgl-kernel/setup_metal.py install",
                    "rm -f python/pyproject.toml",
                    "mv python/pyproject_other.toml python/pyproject.toml",
                    'uv pip install -e "python[all_mps]"',
                ] if apple_silicon else [
                    "Follow the SGLang platform install path for this host, then run the serve command below.",
                ],
                "serve": (
                    f"SGLANG_USE_MLX=1 {shlex.quote(str(sglang_python))} -m sglang.launch_server --model {quoted_model} "
                    f"--disable-cuda-graph --host 127.0.0.1 --port {ENGINE_DEFAULT_PORT['sglang']} "
                    f"--served-model-name {quoted_model} --enable-metrics"
                ) if apple_silicon else (
                    f"{shlex.quote(str(sglang_python))} -m sglang.launch_server --model-path {quoted_model} --host 127.0.0.1 "
                    f"--port {ENGINE_DEFAULT_PORT['sglang']} --served-model-name {quoted_model} --enable-metrics"
                ),
                "verify": f"curl -fsS http://127.0.0.1:{ENGINE_DEFAULT_PORT['sglang']}/v1/models",
            },
            "tensorrt-llm": {
                "localSupport": "ready-on-this-host" if linux_nvidia else "requires-linux-nvidia-target-or-remote-endpoint",
                "install": [
                    "Run TensorRT-LLM on a Linux x86_64/aarch64 host with a supported NVIDIA GPU.",
                    "Expose its OpenAI-compatible server to this smoke runner and set PIQ_TENSORRT_LLM_URL.",
                ] if not linux_nvidia else [
                    "Install TensorRT-LLM for this NVIDIA Linux host, then run the serve command below.",
                ],
                "serve": (
                    f"trtllm-serve {quoted_model} --host 127.0.0.1 "
                    f"--port {ENGINE_DEFAULT_PORT['tensorrt-llm']}"
                ),
                "verify": f"curl -fsS http://127.0.0.1:{ENGINE_DEFAULT_PORT['tensorrt-llm']}/v1/models",
            },
        },
    }


def runtime_preflight(engines: list[dict[str, Any]], missing_urls: list[str], model: str | None = None) -> dict[str, Any]:
    endpoint_results = [endpoint_probe(engine, model=model) for engine in engines]
    runtime_candidates = local_runtime_discovery()
    return {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.executable,
        },
        "localRuntime": {
            "commandProbeTimeoutSeconds": command_probe_timeout_seconds(),
            "vllmCommand": command_probe("vllm", "--version"),
            "vllmModule": module_probe("vllm"),
            "vllmExtension": vllm_extension_probe(),
            "sglangModule": module_probe("sglang"),
            "runtimeCandidates": runtime_candidates,
            "tensorrtLlmServeCommand": command_probe("trtllm-serve", "--help"),
            "nvidiaSmiCommand": command_probe("nvidia-smi"),
        },
        "storage": storage_probe(),
        "missingEngineUrls": missing_urls,
        "endpoints": endpoint_results,
        "configuredEngineCount": len(endpoint_results),
        "launchPlan": runtime_launch_plan(model or laptop_smoke_model(), runtime_candidates=runtime_candidates),
        "ready": bool(endpoint_results) and not missing_urls and all(item.get("ok") for item in endpoint_results),
    }


def attach_endpoint_preflight(engines: list[dict[str, Any]], preflight: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not preflight:
        return engines
    by_engine = {
        item.get("engine"): item
        for item in preflight.get("endpoints", [])
        if isinstance(item, dict)
    }
    return [
        {
            **engine,
            **({"endpointPreflight": by_engine[engine["engine"]]} if engine.get("engine") in by_engine else {}),
        }
        for engine in engines
    ]


def _dashboard_rows(body: dict[str, Any], name: str) -> list[Any]:
    rows = body.get(name, {}).get("rows", [])
    return rows if isinstance(rows, list) else []


def query_dashboard(base_url: str, token: str | None = None, campaign_ids: list[str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/store/queries",
        data=json.dumps({"queries": list(QUERY_NAMES)}).encode("utf-8"),
        headers={
            "content-type": "application/json",
            **({"authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
        rows = {name: _dashboard_rows(body, name) for name in QUERY_NAMES if name in body}
        surface_campaign_ids = {
            name: sorted({
                row[index]
                for row in rows.get(name, [])
                if len(row) > index and isinstance(row[index], str)
            })
            for name, index in CAMPAIGN_ID_QUERY_COLUMN.items()
        }
        submitted_campaign_ids = set(campaign_ids or [])
        submitted_campaign_rows = {
            name: [
                row for row in rows.get(name, [])
                if len(row) > index and row[index] in submitted_campaign_ids
            ]
            for name, index in CAMPAIGN_ID_QUERY_COLUMN.items()
        } if submitted_campaign_ids else {}
        return {
            "storeProvider": response.headers.get("x-piq-store-provider"),
            "rowCounts": {name: body[name]["rowCount"] for name in QUERY_NAMES if name in body},
            "rows": rows,
            "campaignIds": surface_campaign_ids.get("campaign_provenance", []),
            "surfaceCampaignIds": surface_campaign_ids,
            "submittedCampaignRows": submitted_campaign_rows,
            "runtimeFrameworks": sorted({
                row[2]
                for row in rows.get("price_performance", [])
                if len(row) > 2
            }),
        }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _unique_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        absolute = os.path.abspath(candidate)
        if absolute not in seen:
            unique.append(absolute)
            seen.add(absolute)
    return unique


def _resolve_proof_member_path(path: Any, proof_dir: str) -> str:
    if not isinstance(path, str) or not path:
        return ""
    if os.path.isabs(path):
        return path
    candidates = _unique_candidates([
        path,
        os.path.join(proof_dir, path),
        os.path.join(proof_dir, os.path.basename(path)),
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _serving_event_validation_errors(event: dict[str, Any], line_number: int) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if event.get("schemaVersion") != SERVING_EVENT_SCHEMA_VERSION:
        errors.append(f"event log line {line_number} schemaVersion must be {SERVING_EVENT_SCHEMA_VERSION}.")
    if not isinstance(event.get("topic"), str) or not event.get("topic"):
        errors.append(f"event log line {line_number} topic is required.")
    event_type = event.get("eventType")
    if not isinstance(event_type, str) or not event_type:
        errors.append(f"event log line {line_number} eventType is required.")
        return None, errors
    event_id = event.get("eventId")
    partition_key = event.get("partitionKey")
    payload = event.get("payload")
    if not isinstance(event_id, str) or len(event_id) != 64:
        errors.append(f"event log line {line_number} eventId must be a 64-character digest.")
    if not isinstance(partition_key, str) or not partition_key:
        errors.append(f"event log line {line_number} partitionKey is required.")
    if not isinstance(payload, dict):
        errors.append(f"event log line {line_number} payload must be an object.")
    if isinstance(event_id, str) and len(event_id) == 64 and isinstance(partition_key, str) and isinstance(payload, dict):
        expected_event_id = _sha256_json({
            "schemaVersion": event.get("schemaVersion"),
            "eventType": event_type,
            "partitionKey": partition_key,
            "payload": payload,
        })
        if event_id != expected_event_id:
            errors.append(f"event log line {line_number} eventId digest does not match event payload.")
    return event_type, errors


def load_serving_event_log(event_log_path: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    with open(event_log_path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                validation_errors.append(f"event log line {line_number} is not valid JSON: {exc}")
                continue
            if not isinstance(event, dict):
                validation_errors.append(f"event log line {line_number} must be a JSON object.")
                continue
            _event_type, event_errors = _serving_event_validation_errors(event, line_number)
            validation_errors.extend(event_errors)
            events.append(event)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return events


def _load_json_file(path: str, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path:
        errors.append(f"{label} path is missing.")
        return None
    if not os.path.exists(path):
        errors.append(f"{label} does not exist: {path}")
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is not readable JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object.")
        return None
    return value


def _read_json_object(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _successful_status(value: Any) -> bool:
    return isinstance(value, int) and 200 <= value < 300


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and value > 0


def _receipts_by_engine_and_request_id(receipts: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_engine: dict[str, dict[str, dict[str, Any]]] = {}
    for receipt in receipts:
        if receipt.get("schemaVersion") != REQUEST_RECEIPT_SCHEMA_VERSION:
            continue
        engine = receipt.get("engine")
        request_id = receipt.get("requestId")
        if isinstance(engine, str) and isinstance(request_id, str) and request_id:
            by_engine.setdefault(engine, {})[request_id] = receipt
    return by_engine


def _validate_measurement_row(
    *,
    engine: str,
    row: dict[str, Any],
    expected_model: str | None,
    expected_framework: str,
    request_count: int | None,
    success_count: int | None,
    errors: list[str],
) -> None:
    expected_values = {
        "surface": "result",
        "model": expected_model,
        "runtimeFramework": expected_framework,
        "runtimeEngine": engine,
        "basis": "per_engine",
        "experimentFamily": "serving-producer",
        "experimentStatus": "accepted",
        "verdictTier": "request-captured",
    }
    for key, expected in expected_values.items():
        if expected is not None and row.get(key) != expected:
            errors.append(f"{engine} measurement {key} must be {expected}.")
    if isinstance(request_count, int) and row.get("requestCount") != request_count:
        errors.append(f"{engine} measurement requestCount must match proof submission.")
    if isinstance(success_count, int) and row.get("successCount") != success_count:
        errors.append(f"{engine} measurement successCount must match proof submission.")
    if row.get("errorCount") not in (0, 0.0):
        errors.append(f"{engine} measurement errorCount must be zero.")

    for key in ["promptTokens", "completionTokens", "totalTokens", "outputTpm", "totalTpm"]:
        if not _is_positive_number(row.get(key)):
            errors.append(f"{engine} measurement {key} must be a positive number.")
    for key in [
        "avgLatencyMs", "p50LatencyMs", "p95LatencyMs", "p99LatencyMs",
        "avgTimeToFirstByteMs", "p50TimeToFirstByteMs", "p95TimeToFirstByteMs", "p99TimeToFirstByteMs",
        "avgTtftMs", "p50TtftMs", "p95TtftMs", "p99TtftMs",
        "avgTpotMs", "p50TpotMs", "p95TpotMs", "p99TpotMs",
        "avgTtfotMs", "p50TtfotMs", "p95TtfotMs", "p99TtfotMs",
    ]:
        if not _is_number(row.get(key)) or row.get(key) < 0:
            errors.append(f"{engine} measurement {key} must be a non-negative number.")
    if not _is_number(row.get("metricCompleteness")) or not 0 <= row.get("metricCompleteness") <= 1:
        errors.append(f"{engine} measurement metricCompleteness must be between 0 and 1.")
    if (
        row.get("nativeTelemetryRequired") is True
        and isinstance(success_count, int)
        and row.get("nativeTelemetryAvailableCount") != success_count
    ):
        errors.append(f"{engine} measurement nativeTelemetryAvailableCount must equal successCount when native telemetry is required.")
    if row.get("hardwareTelemetryRequired") is True and isinstance(success_count, int):
        if row.get("hardwareTelemetryAvailableCount") != success_count:
            errors.append(f"{engine} measurement hardwareTelemetryAvailableCount must equal successCount when hardware telemetry is required.")
        if row.get("dcgmGrounded") is not True:
            errors.append(f"{engine} measurement dcgmGrounded must be true when hardware telemetry is required.")
    if row.get("tokenDetailsRequired") is True and isinstance(success_count, int):
        for key in ["tokenDetailsAvailableCount", "tokenIdsAvailableCount", "logprobsAvailableCount"]:
            if row.get(key) != success_count:
                errors.append(f"{engine} measurement {key} must equal successCount when token details are required.")
    if row.get("promptTokenDetailsRequired") is True and isinstance(success_count, int):
        if row.get("promptTokenIdsAvailableCount") != success_count:
            errors.append(f"{engine} measurement promptTokenIdsAvailableCount must equal successCount when prompt token details are required.")
    if (
        (
            row.get("nativeTelemetryRequired") is True
            or row.get("hardwareTelemetryRequired") is True
            or row.get("tokenDetailsRequired") is True
            or row.get("promptTokenDetailsRequired") is True
        )
        and row.get("metricCompleteness") != 1
    ):
        errors.append(f"{engine} measurement metricCompleteness must be 1 when required telemetry is missing-sensitive.")

    for key in ["operatingPoint", "latestCapturedAtUtc", "solRigor", "tags"]:
        if not isinstance(row.get(key), str) or not row.get(key):
            errors.append(f"{engine} measurement {key} is required.")
    tags = str(row.get("tags") or "")
    for fragment in ["serving-producer", engine, expected_framework, str(expected_model or "")]:
        if fragment and fragment not in tags:
            errors.append(f"{engine} measurement tags must include {fragment}.")


def _coverage_status(proven: int, expected: int) -> str:
    if expected <= 0:
        return "missing"
    if proven == expected:
        return "proven"
    if proven > 0:
        return "partial"
    return "missing"


def _coverage_item(status: str, proven: int, expected: int, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "provenCount": proven,
        "expectedCount": expected,
        "missing": missing or [],
    }


def _engine_runtime_provenance_present(sample: dict[str, Any]) -> bool:
    keys = [
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
    return any(sample.get(key) not in (None, "", []) for key in keys)


def _telemetry_coverage_from_proof(
    *,
    proof: dict[str, Any],
    proof_dir: str,
    expected_engines: list[str],
    receipt_counts: dict[str, int],
    event_counts_by_engine: dict[str, dict[str, int]],
) -> dict[str, Any]:
    categories = {
        "clientStreamTiming": {
            "description": "Client-side stream=true timing for E2E, TTFB, TTFT, TTFOT, TPOT, and output timeline rows.",
        },
        "requestReceipts": {
            "description": "Engine-side request receipts proving trace IDs reached POST /v1/chat/completions.",
        },
        "dashboardFineGrainRows": {
            "description": "Queryable serving_request_samples and serving_token_timeline dashboard rows.",
        },
        "nativeRuntimeTelemetry": {
            "description": "Native engine timing/cache/concurrency metrics such as queue, prefill, decode, KV, and cache state.",
        },
        "dcgmHardwareTelemetry": {
            "description": "DCGM hardware counters for power, utilization, clocks, memory, temperature, and energy.",
        },
        "promptTokenIds": {
            "description": "Tokenizer-exact prompt/input token IDs with prompt token provenance.",
        },
        "outputTokenIdsLogprobs": {
            "description": "Output token IDs, logprobs, top-logprobs, and token-detail provenance.",
        },
        "operatorFullArtifacts": {
            "description": "Operator-full raw request/response, token, native, hardware, and runtime artifacts.",
        },
        "rawMetricSnapshots": {
            "description": "Operator-full before/after native and DCGM metric snapshots for all configured telemetry endpoints.",
        },
        "runtimeProvenance": {
            "description": "Runtime version/revision/image/process/container/pod/node/host/server-arg provenance when configured.",
        },
        "kafkaEventLog": {
            "description": "Post-capture Kafka-ready event log with request-sample and token-timeline events.",
        },
    }
    coverage = {
        "schemaVersion": "performance-iq.serving-telemetry-coverage.v1",
        "model": proof.get("model"),
        "expectedEngines": expected_engines,
        "categories": categories,
        "engines": {},
        "categorySummary": {},
        "allProven": False,
    }
    dashboard = proof.get("dashboard") if isinstance(proof.get("dashboard"), dict) else {}
    submitted_dashboard_rows = dashboard.get("submittedCampaignRows") if isinstance(dashboard.get("submittedCampaignRows"), dict) else {}
    submissions = {
        item.get("engine"): item
        for item in proof.get("submissions", [])
        if isinstance(item, dict) and item.get("engine") in ENGINE_IDS
    }

    def submitted_dashboard_row_count(surface: str, campaign_id: str | None) -> int:
        if not campaign_id:
            return 0
        index = CAMPAIGN_ID_QUERY_COLUMN.get(surface)
        rows = submitted_dashboard_rows.get(surface)
        if index is None or not isinstance(rows, list):
            return 0
        return sum(
            1 for row in rows
            if isinstance(row, list) and len(row) > index and row[index] == campaign_id
        )

    for engine in expected_engines:
        submission = submissions.get(engine)
        engine_coverage: dict[str, Any] = {}
        if not isinstance(submission, dict):
            coverage["engines"][engine] = {
                category: _coverage_item("missing", 0, 1, [f"{engine} submission missing"])
                for category in categories
            }
            continue
        artifact_path = _resolve_proof_member_path(submission.get("artifactPath"), proof_dir)
        artifact = _read_json_object(artifact_path or "")
        samples = artifact.get("samples") if isinstance(artifact.get("samples"), list) else []
        measurements = artifact.get("measurements") if isinstance(artifact.get("measurements"), list) else []
        first_measurement = measurements[0] if measurements and isinstance(measurements[0], dict) else {}
        token_details_capability = artifact.get("tokenDetailsCapability") if isinstance(artifact.get("tokenDetailsCapability"), dict) else {}
        token_timeline = artifact.get("tokenTimeline") if isinstance(artifact.get("tokenTimeline"), list) else []
        native_telemetry = artifact.get("nativeTelemetry") if isinstance(artifact.get("nativeTelemetry"), list) else []
        hardware_telemetry = artifact.get("hardwareTelemetry") if isinstance(artifact.get("hardwareTelemetry"), list) else []
        capture_policy = artifact.get("capturePolicy") if isinstance(artifact.get("capturePolicy"), dict) else {}
        expected_samples = len(samples) or int(submission.get("requestCount") or 0) or 1
        sample_request_ids = [
            sample.get("requestId")
            for sample in samples
            if isinstance(sample, dict) and isinstance(sample.get("requestId"), str)
        ]

        output_rows = [
            row for row in token_timeline
            if isinstance(row, dict) and row.get("tokenPhase", "output") == "output"
        ]
        output_row_request_ids = {
            row.get("requestId") for row in output_rows
            if isinstance(row.get("requestId"), str)
        }
        stream_proven = sum(
            1 for sample in samples
            if isinstance(sample, dict)
            and sample.get("streaming") is True
            and sample.get("requestId") in output_row_request_ids
            and all(_is_number(sample.get(key)) for key in ["e2eLatencyMs", "timeToFirstByteMs", "ttftMs", "ttfotMs", "tpotMs"])
        )
        stream_missing = []
        if stream_proven != expected_samples:
            stream_missing.append("one or more request samples are missing stream timing fields or request-level output token timeline rows")
        engine_coverage["clientStreamTiming"] = _coverage_item(
            _coverage_status(stream_proven, expected_samples),
            stream_proven,
            expected_samples,
            stream_missing,
        )

        receipts = receipt_counts.get(engine, 0)
        engine_coverage["requestReceipts"] = _coverage_item(
            _coverage_status(receipts, expected_samples),
            receipts,
            expected_samples,
            [] if receipts >= expected_samples else ["request receipt count is below request sample count"],
        )

        campaign_id = submission.get("campaignId") if isinstance(submission.get("campaignId"), str) else None
        request_dashboard_rows = submitted_dashboard_row_count("serving_request_samples", campaign_id)
        timeline_dashboard_rows = submitted_dashboard_row_count("serving_token_timeline", campaign_id)
        fine_grain_proven = int(request_dashboard_rows > 0) + int(timeline_dashboard_rows > 0)
        fine_grain_missing = []
        if request_dashboard_rows <= 0:
            fine_grain_missing.append("dashboard serving_request_samples rows are missing for this engine campaign")
        if timeline_dashboard_rows <= 0:
            fine_grain_missing.append("dashboard serving_token_timeline rows are missing for this engine campaign")
        engine_coverage["dashboardFineGrainRows"] = _coverage_item(
            _coverage_status(fine_grain_proven, 2),
            fine_grain_proven,
            2,
            fine_grain_missing,
        )

        native_sample_request_ids = {
            sample.get("requestId") for sample in samples
            if isinstance(sample, dict)
            and sample.get("nativeTelemetryAvailable") is True
            and isinstance(sample.get("requestId"), str)
            and all(_is_number(sample.get(key)) for key in REQUIRED_NATIVE_SAMPLE_FIELDS)
        }
        native_row_request_ids = {
            row.get("requestId") for row in native_telemetry
            if isinstance(row, dict)
            and row.get("available") is True
            and isinstance(row.get("requestId"), str)
            and all(_is_number(row.get(key)) for key in REQUIRED_NATIVE_SAMPLE_FIELDS)
        }
        native_proven = sum(
            1 for request_id in sample_request_ids
            if request_id in native_sample_request_ids and request_id in native_row_request_ids
        )
        native_required = first_measurement.get("nativeTelemetryRequired") is True
        engine_coverage["nativeRuntimeTelemetry"] = _coverage_item(
            _coverage_status(native_proven, expected_samples),
            native_proven,
            expected_samples,
            [] if native_proven == expected_samples else [
                "native telemetry fields are missing"
                + (" even though native telemetry is required" if native_required else "")
            ],
        )

        hardware_sample_request_ids = {
            sample.get("requestId") for sample in samples
            if isinstance(sample, dict)
            and sample.get("hardwareTelemetryAvailable") is True
            and isinstance(sample.get("requestId"), str)
            and all(_is_number(sample.get(key)) for key in REQUIRED_HARDWARE_SAMPLE_COUNTERS)
        }
        hardware_row_request_ids = {
            row.get("requestId") for row in hardware_telemetry
            if isinstance(row, dict)
            and row.get("available") is True
            and isinstance(row.get("requestId"), str)
            and all(_is_number(row.get(key)) for key in ["powerWatts", "powerWattsPerGpu", "gpuUtilizationPct", "memoryCopyUtilizationPct", "gpuTemperatureC", "smClockMHz", "memoryClockMHz", "fbUsedMiB", "fbFreeMiB", "energyJoules"])
        }
        hardware_required = first_measurement.get("hardwareTelemetryRequired") is True
        hardware_ok_count = sum(
            1 for request_id in sample_request_ids
            if request_id in hardware_sample_request_ids and request_id in hardware_row_request_ids
        )
        engine_coverage["dcgmHardwareTelemetry"] = _coverage_item(
            _coverage_status(hardware_ok_count, expected_samples),
            hardware_ok_count,
            expected_samples,
            [] if hardware_ok_count == expected_samples else [
                "DCGM hardware sample/artifact counters are missing"
                + (" even though hardware telemetry is required" if hardware_required else "")
            ],
        )

        prompt_sample_proven = sum(
            1 for sample in samples
            if isinstance(sample, dict)
            and sample.get("promptTokenIdsAvailable") is True
            and isinstance(sample.get("promptTokenIdSource"), str)
            and isinstance(sample.get("promptTokenIdsSha256"), str)
        )
        prompt_rows = [
            row for row in token_timeline
            if isinstance(row, dict) and row.get("tokenPhase") == "prompt" and isinstance(row.get("tokenId"), int)
        ]
        prompt_row_request_ids = {
            row.get("requestId") for row in prompt_rows
            if isinstance(row.get("requestId"), str)
        }
        prompt_rows_proven = sum(1 for request_id in sample_request_ids if request_id in prompt_row_request_ids)
        prompt_required = first_measurement.get("promptTokenDetailsRequired") is True
        prompt_ok = prompt_sample_proven == expected_samples and prompt_rows_proven == expected_samples
        prompt_expected = expected_samples if prompt_required else 0
        prompt_proven = min(prompt_sample_proven, prompt_rows_proven)
        engine_coverage["promptTokenIds"] = _coverage_item(
            "proven" if prompt_ok else _coverage_status(prompt_proven, prompt_expected),
            prompt_proven,
            prompt_expected,
            [] if prompt_ok else [
                "prompt token IDs or prompt token timeline rows are missing"
                + (" even though prompt token details are required" if prompt_required else "")
            ],
        )

        output_sample_proven = sum(
            1 for sample in samples
            if isinstance(sample, dict)
            and sample.get("tokenDetailsAvailable") is True
            and sample.get("tokenIdsAvailable") is True
            and sample.get("logprobsAvailable") is True
            and isinstance(sample.get("tokenIdSource"), str)
        )
        valid_output_row_request_ids = set()
        for request_id in sample_request_ids:
            rows_for_request = [
                row for row in output_rows
                if isinstance(row, dict) and row.get("requestId") == request_id
            ]
            if rows_for_request and all(
                isinstance(row.get("tokenId"), int)
                and isinstance(row.get("tokenIdSource"), str)
                and _is_number(row.get("tokenLogprob"))
                for row in rows_for_request
            ):
                valid_output_row_request_ids.add(request_id)
        output_rows_proven = len(valid_output_row_request_ids)
        token_requested_unsupported = (
            token_details_capability.get("requested") is True
            and token_details_capability.get("supported") is False
        )
        token_required = first_measurement.get("tokenDetailsRequired") is True or token_requested_unsupported
        output_ok = output_sample_proven == expected_samples and output_rows_proven == expected_samples
        output_expected = expected_samples if token_required else 0
        output_proven = min(output_sample_proven, output_rows_proven)
        output_missing = []
        if not output_ok:
            if token_requested_unsupported:
                reason = token_details_capability.get("reason") or "runtime-token-details-unsupported"
                output_missing.append(f"output token IDs/logprobs were requested but unsupported by runtime: {reason}")
            else:
                output_missing.append(
                    "output token IDs/logprobs or token timeline details are missing"
                    + (" even though output token details are required" if token_required else "")
                )
        engine_coverage["outputTokenIdsLogprobs"] = _coverage_item(
            "proven" if output_ok else _coverage_status(output_proven, output_expected),
            output_proven,
            output_expected,
            [] if output_ok else output_missing,
        )

        raw_path = _resolve_proof_member_path(capture_policy.get("rawArtifactPath"), proof_dir)
        raw_present = capture_policy.get("mode") == "operator-full" and bool(raw_path) and os.path.exists(raw_path or "")
        raw_artifact = _read_json_object(raw_path or "") if raw_present else {}
        raw_captures = raw_artifact.get("captures") if isinstance(raw_artifact.get("captures"), list) else []
        engine_coverage["operatorFullArtifacts"] = _coverage_item(
            "proven" if raw_present else "missing",
            1 if raw_present else 0,
            1,
            [] if raw_present else ["operator-full raw artifact is missing"],
        )

        def raw_snapshot_available(capture: dict[str, Any], key: str) -> bool:
            snapshot = capture.get(key)
            if not isinstance(snapshot, dict):
                return False
            before = snapshot.get("before")
            after = snapshot.get("after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                return False
            before_has_metrics = bool(before.get("available")) and (
                isinstance(before.get("metrics"), dict) or isinstance(before.get("jsonMetrics"), dict)
            )
            after_has_metrics = bool(after.get("available")) and (
                isinstance(after.get("metrics"), dict) or isinstance(after.get("jsonMetrics"), dict)
            )
            return before_has_metrics and after_has_metrics

        raw_metric_request_ids = {
            capture.get("requestId") for capture in raw_captures
            if isinstance(capture, dict)
            and isinstance(capture.get("requestId"), str)
            and (not native_required or raw_snapshot_available(capture, "nativeMetricsRaw"))
            and (not hardware_required or raw_snapshot_available(capture, "hardwareMetricsRaw"))
        }
        raw_metric_proven = sum(1 for request_id in sample_request_ids if request_id in raw_metric_request_ids)
        raw_metric_expected = expected_samples if native_required or hardware_required else 0
        engine_coverage["rawMetricSnapshots"] = _coverage_item(
            _coverage_status(raw_metric_proven, raw_metric_expected),
            raw_metric_proven,
            raw_metric_expected,
            [] if raw_metric_expected == 0 or raw_metric_proven == raw_metric_expected else ["operator-full raw native/DCGM metric snapshots are missing"],
        )

        runtime_proven = sum(
            1 for sample in samples
            if isinstance(sample, dict) and _engine_runtime_provenance_present(sample)
        )
        engine_coverage["runtimeProvenance"] = _coverage_item(
            _coverage_status(runtime_proven, expected_samples),
            runtime_proven,
            expected_samples,
            [] if runtime_proven == expected_samples else ["runtime version/image/process/container/node provenance is absent or partial"],
        )

        engine_event_counts = event_counts_by_engine.get(engine, {})
        request_sample_events = engine_event_counts.get("serving.measurement.serving_request_sample", 0)
        token_timeline_events = engine_event_counts.get("serving.measurement.serving_token_timeline", 0)
        event_log_proven = int(request_sample_events >= expected_samples) + int(token_timeline_events >= expected_samples)
        event_log_missing = []
        if request_sample_events < expected_samples:
            event_log_missing.append("Kafka-ready event log is missing request-sample events for this engine")
        if token_timeline_events < expected_samples:
            event_log_missing.append("Kafka-ready event log is missing token-timeline events for this engine")
        engine_coverage["kafkaEventLog"] = _coverage_item(
            _coverage_status(event_log_proven, 2),
            event_log_proven,
            2,
            event_log_missing,
        )
        coverage["engines"][engine] = engine_coverage

    for category in categories:
        items = [
            engine_coverage.get(category, {})
            for engine_coverage in coverage["engines"].values()
            if isinstance(engine_coverage, dict)
        ]
        expected_items = [
            item for item in items
            if isinstance(item, dict) and int(item.get("expectedCount") or 0) > 0
        ]
        expected_count = len(expected_items)
        proven = sum(1 for item in expected_items if item.get("status") == "proven")
        status = "proven" if expected_count == 0 or proven == expected_count else "partial" if proven else "missing"
        coverage["categorySummary"][category] = {
            "status": status,
            "provenEngines": proven,
            "expectedEngines": expected_count,
        }
    coverage["allProven"] = all(
        summary.get("status") == "proven"
        for summary in coverage["categorySummary"].values()
    ) if coverage["categorySummary"] else False
    return coverage


def build_evidence_index(summary: dict[str, Any]) -> dict[str, Any]:
    receipt_ids = receipt_ids_by_engine_safe(str(summary.get("receiptLogPath") or ""))
    dashboard = summary.get("dashboard") if isinstance(summary.get("dashboard"), dict) else {}
    submitted_rows = dashboard.get("submittedCampaignRows") if isinstance(dashboard.get("submittedCampaignRows"), dict) else {}
    row_counts = dashboard.get("rowCounts") if isinstance(dashboard.get("rowCounts"), dict) else {}
    engines: dict[str, Any] = {}
    for submission in summary.get("submissions", []):
        if not isinstance(submission, dict) or submission.get("engine") not in ENGINE_IDS:
            continue
        engine = str(submission["engine"])
        artifact = _read_json_object(str(submission.get("artifactPath") or ""))
        manifest = _read_json_object(str(submission.get("manifestPath") or ""))
        samples = artifact.get("samples") if isinstance(artifact.get("samples"), list) else []
        measurements = artifact.get("measurements") if isinstance(artifact.get("measurements"), list) else []
        measurement = measurements[0] if measurements and isinstance(measurements[0], dict) else {}
        request_trace_ids = [
            sample["requestId"]
            for sample in samples
            if isinstance(sample, dict) and isinstance(sample.get("requestId"), str)
        ]
        campaign_id = str(submission.get("campaignId") or "")
        dashboard_surfaces = {}
        for name, index in CAMPAIGN_ID_QUERY_COLUMN.items():
            rows = [
                row for row in submitted_rows.get(name, [])
                if isinstance(row, list) and len(row) > index and row[index] == campaign_id
            ]
            dashboard_surfaces[name] = {
                "rowCount": len(rows),
                "submittedRows": rows,
            }
        engines[engine] = {
            "runtimeFramework": submission.get("runtimeFramework"),
            "campaignId": submission.get("campaignId"),
            "runId": submission.get("runId"),
            "requestCount": submission.get("requestCount"),
            "successCount": submission.get("successCount"),
            "artifact": {
                "path": submission.get("artifactPath"),
                "sha256": submission.get("artifactSha256"),
                "schemaVersion": artifact.get("schemaVersion"),
            },
            "manifest": {
                "path": submission.get("manifestPath"),
                "schemaVersion": manifest.get("schemaVersion"),
            },
            "requestTraceIds": request_trace_ids,
            "receiptIds": sorted(receipt_ids.get(engine, set())),
            "measurement": {
                key: measurement.get(key)
                for key in [
                    "model",
                    "runtimeFramework",
                    "runtimeEngine",
                    "requestCount",
                    "successCount",
                    "promptTokens",
                    "completionTokens",
                    "totalTokens",
                    "outputTpm",
                    "totalTpm",
                    "avgLatencyMs",
                    "p50LatencyMs",
                    "p95LatencyMs",
                    "p99LatencyMs",
                    "avgTimeToFirstByteMs",
                    "p50TimeToFirstByteMs",
                    "p95TimeToFirstByteMs",
                    "p99TimeToFirstByteMs",
                    "avgTtftMs",
                    "p50TtftMs",
                    "p95TtftMs",
                    "p99TtftMs",
                    "avgTpotMs",
                    "p50TpotMs",
                    "p95TpotMs",
                    "p99TpotMs",
                    "avgTtfotMs",
                    "p50TtfotMs",
                    "p95TtfotMs",
                    "p99TtfotMs",
                    "metricCompleteness",
                    "tags",
                ]
                if key in measurement
            },
            "dashboard": {
                "rowCounts": dict(row_counts),
                "surfaces": dashboard_surfaces,
            },
        }
    return {
        "schemaVersion": EVIDENCE_INDEX_SCHEMA_VERSION,
        "model": summary.get("model"),
        "engines": engines,
    }


def receipt_ids_by_engine_safe(receipt_log_path: str) -> dict[str, set[str]]:
    if not receipt_log_path or not os.path.exists(receipt_log_path):
        return {}
    try:
        receipts = load_receipts(receipt_log_path)
    except ValueError:
        return {}
    return {
        engine: set(receipts_by_request.keys())
        for engine, receipts_by_request in _receipts_by_engine_and_request_id(receipts).items()
    }


def verify_proof_summary(proof_path: str, *, require_all_engines: bool = True) -> dict[str, Any]:
    proof_abs = os.path.abspath(proof_path)
    proof_dir = os.path.dirname(proof_abs)
    errors: list[str] = []
    warnings: list[str] = []
    artifact_hashes: dict[str, str] = {}
    campaign_ids: list[str] = []
    receipt_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    event_counts_by_engine: dict[str, dict[str, int]] = {engine: {} for engine in ENGINE_IDS}

    proof = _load_json_file(proof_abs, errors, "proof summary")
    if not proof:
        return {
            "schemaVersion": PROOF_VERIFICATION_SCHEMA_VERSION,
            "proofPath": proof_abs,
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "engineCount": 0,
            "campaignIds": campaign_ids,
            "artifactHashes": artifact_hashes,
        }

    if proof.get("schemaVersion") != PROOF_SCHEMA_VERSION:
        errors.append(f"proof schemaVersion must be {PROOF_SCHEMA_VERSION}.")

    expected_model = proof.get("model")
    if not isinstance(expected_model, str) or not expected_model:
        errors.append("proof model is required.")

    submissions = proof.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        errors.append("proof submissions must contain at least one item.")
        submissions = []

    submissions_by_engine: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(submissions):
        if not isinstance(item, dict):
            errors.append(f"submissions[{index}] must be an object.")
            continue
        engine = item.get("engine")
        if engine not in ENGINE_IDS:
            errors.append(f"submissions[{index}].engine must be one of {', '.join(ENGINE_IDS)}.")
            continue
        if engine in submissions_by_engine:
            errors.append(f"duplicate submission for engine {engine}.")
            continue
        submissions_by_engine[engine] = item

    expected_engines = list(ENGINE_IDS) if require_all_engines else list(submissions_by_engine.keys())
    if require_all_engines:
        missing = [engine for engine in ENGINE_IDS if engine not in submissions_by_engine]
        if missing:
            errors.append("proof is missing required engine submissions: " + ", ".join(missing) + ".")

    evidence_index = proof.get("evidenceIndex")
    evidence_engines: dict[str, Any] = {}
    if not isinstance(evidence_index, dict):
        errors.append("proof evidenceIndex is required.")
    else:
        if evidence_index.get("schemaVersion") != EVIDENCE_INDEX_SCHEMA_VERSION:
            errors.append(f"proof evidenceIndex schemaVersion must be {EVIDENCE_INDEX_SCHEMA_VERSION}.")
        if evidence_index.get("model") != expected_model:
            errors.append("proof evidenceIndex model must match proof model.")
        raw_engines = evidence_index.get("engines")
        if isinstance(raw_engines, dict):
            evidence_engines = raw_engines
        else:
            errors.append("proof evidenceIndex.engines must be an object.")

    receipt_log_path = _resolve_proof_member_path(proof.get("receiptLogPath"), proof_dir)
    receipt_by_engine: dict[str, dict[str, dict[str, Any]]] = {}
    if not receipt_log_path:
        errors.append("proof receiptLogPath is required for request receipt verification.")
    elif not os.path.exists(receipt_log_path):
        errors.append(f"receipt log does not exist: {receipt_log_path}")
    else:
        try:
            receipts = load_receipts(receipt_log_path)
            receipt_by_engine = _receipts_by_engine_and_request_id(receipts)
            receipt_counts = {
                engine: len(receipt_by_engine.get(engine, {}))
                for engine in ENGINE_IDS
            }
        except ValueError as exc:
            errors.append(str(exc))

    event_log_path = _resolve_proof_member_path(proof.get("eventLogPath"), proof_dir)
    if event_log_path:
        if not os.path.exists(event_log_path):
            errors.append(f"event log does not exist: {event_log_path}")
        else:
            try:
                for event in load_serving_event_log(event_log_path):
                    event_type = event.get("eventType")
                    if isinstance(event_type, str):
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                        event_engine = event.get("engine")
                        if event_engine in ENGINE_IDS:
                            engine_counts = event_counts_by_engine.setdefault(str(event_engine), {})
                            engine_counts[event_type] = engine_counts.get(event_type, 0) + 1
            except ValueError as exc:
                errors.append(str(exc))
            except OSError as exc:
                errors.append(f"event log is not readable: {exc}")

    preflight = proof.get("preflight")
    endpoint_preflight_by_engine: dict[str, dict[str, Any]] = {}
    if not isinstance(preflight, dict):
        errors.append("proof preflight object is required.")
    else:
        if preflight.get("ready") is not True:
            errors.append("proof preflight.ready must be true.")
        endpoints = preflight.get("endpoints")
        if not isinstance(endpoints, list):
            errors.append("proof preflight.endpoints must be a list.")
        else:
            for index, item in enumerate(endpoints):
                if not isinstance(item, dict):
                    errors.append(f"preflight.endpoints[{index}] must be an object.")
                    continue
                engine = item.get("engine")
                if engine in ENGINE_IDS:
                    endpoint_preflight_by_engine[str(engine)] = item
            for engine in expected_engines:
                item = endpoint_preflight_by_engine.get(engine)
                if not item:
                    errors.append(f"preflight is missing endpoint evidence for {engine}.")
                    continue
                if item.get("ok") is not True:
                    errors.append(f"{engine} preflight endpoint is not ok.")
                if item.get("modelAvailable") is not True:
                    errors.append(f"{engine} preflight did not prove the smoke model is available.")
                served_models = item.get("servedModels")
                if isinstance(served_models, list) and expected_model and expected_model not in served_models:
                    errors.append(f"{engine} preflight servedModels does not include {expected_model}.")

    for engine, submission in submissions_by_engine.items():
        expected_framework = serving_engine_label(engine)
        evidence = evidence_engines.get(engine) if isinstance(evidence_engines.get(engine), dict) else None
        if evidence is None:
            errors.append(f"{engine} evidenceIndex entry is required.")
        request_count = submission.get("requestCount")
        success_count = submission.get("successCount")
        if not isinstance(request_count, int) or request_count < 1:
            errors.append(f"{engine} requestCount must be a positive integer.")
        if success_count != request_count:
            errors.append(f"{engine} successCount must equal requestCount.")
        if submission.get("errorCount") not in (0, 0.0):
            errors.append(f"{engine} errorCount must be zero.")
        if submission.get("errors"):
            errors.append(f"{engine} submission includes request errors.")
        if submission.get("status") != "accepted":
            errors.append(f"{engine} submission status must be accepted.")
        if submission.get("runtimeFramework") != expected_framework:
            errors.append(f"{engine} runtimeFramework must be {expected_framework}.")
        if evidence:
            for key in ["runtimeFramework", "campaignId", "runId", "requestCount", "successCount"]:
                if evidence.get(key) != submission.get(key):
                    errors.append(f"{engine} evidenceIndex {key} must match proof submission.")
            evidence_artifact = evidence.get("artifact") if isinstance(evidence.get("artifact"), dict) else {}
            if evidence_artifact.get("path") != submission.get("artifactPath"):
                errors.append(f"{engine} evidenceIndex artifact.path must match proof submission.")
            if evidence_artifact.get("sha256") != submission.get("artifactSha256"):
                errors.append(f"{engine} evidenceIndex artifact.sha256 must match proof submission.")
            evidence_manifest = evidence.get("manifest") if isinstance(evidence.get("manifest"), dict) else {}
            if evidence_manifest.get("path") != submission.get("manifestPath"):
                errors.append(f"{engine} evidenceIndex manifest.path must match proof submission.")

        campaign_id = submission.get("campaignId")
        run_id = submission.get("runId")
        if isinstance(campaign_id, str) and campaign_id:
            campaign_ids.append(campaign_id)
        else:
            errors.append(f"{engine} campaignId is required.")
        if not isinstance(run_id, str) or not run_id:
            errors.append(f"{engine} runId is required.")

        artifact_path = _resolve_proof_member_path(submission.get("artifactPath"), proof_dir)
        manifest_path = _resolve_proof_member_path(submission.get("manifestPath"), proof_dir)
        artifact_sha = submission.get("artifactSha256")
        manifest_artifacts: list[Any] = []
        if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
            errors.append(f"{engine} artifactSha256 must be a 64-character hex digest.")

        artifact = _load_json_file(artifact_path, errors, f"{engine} summary artifact")
        if artifact_path and os.path.exists(artifact_path):
            computed_sha = sha256_file(artifact_path)
            artifact_hashes[engine] = computed_sha
            if artifact_sha != computed_sha:
                errors.append(f"{engine} artifactSha256 does not match artifact contents.")

        manifest = _load_json_file(manifest_path, errors, f"{engine} manifest")
        if manifest:
            if manifest.get("schemaVersion") != PRODUCER_MANIFEST_SCHEMA_VERSION:
                errors.append(f"{engine} manifest schemaVersion must be {PRODUCER_MANIFEST_SCHEMA_VERSION}.")
            manifest_campaign = manifest.get("campaign") if isinstance(manifest.get("campaign"), dict) else {}
            if manifest_campaign.get("campaignId") != campaign_id:
                errors.append(f"{engine} manifest campaignId does not match proof submission.")
            if manifest_campaign.get("runId") != run_id:
                errors.append(f"{engine} manifest runId does not match proof submission.")
            manifest_workload = manifest.get("workload") if isinstance(manifest.get("workload"), dict) else {}
            if expected_model and manifest_workload.get("model") != expected_model:
                errors.append(f"{engine} manifest workload.model does not match proof model.")
            manifest_runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
            if manifest_runtime.get("framework") != expected_framework:
                errors.append(f"{engine} manifest runtime.framework must be {expected_framework}.")
            manifest_platform = manifest.get("platform") if isinstance(manifest.get("platform"), dict) else {}
            trace_ids = manifest_platform.get("requestTraceIds")
            if not isinstance(trace_ids, list) or len(trace_ids) != request_count:
                errors.append(f"{engine} manifest platform.requestTraceIds must match requestCount.")
            manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
            first_artifact = manifest_artifacts[0] if isinstance(manifest_artifacts, list) and manifest_artifacts else {}
            if not isinstance(first_artifact, dict):
                errors.append(f"{engine} manifest must include an artifact entry.")
            else:
                if first_artifact.get("path") != submission.get("artifactPath"):
                    errors.append(f"{engine} manifest artifact path does not match proof submission.")
                if first_artifact.get("sha256") != artifact_sha:
                    errors.append(f"{engine} manifest artifact sha256 does not match proof submission.")

        if artifact:
            if artifact.get("schemaVersion") != SERVING_SUMMARY_SCHEMA_VERSION:
                errors.append(f"{engine} summary artifact schemaVersion must be {SERVING_SUMMARY_SCHEMA_VERSION}.")
            if artifact.get("engine") != engine:
                errors.append(f"{engine} summary artifact engine does not match proof submission.")
            if expected_model and artifact.get("model") != expected_model:
                errors.append(f"{engine} summary artifact model does not match proof model.")
            artifact_preflight = artifact.get("endpointPreflight")
            if not isinstance(artifact_preflight, dict):
                errors.append(f"{engine} summary artifact is missing endpointPreflight.")
            else:
                if artifact_preflight.get("ok") is not True:
                    errors.append(f"{engine} summary artifact endpointPreflight.ok must be true.")
                if artifact_preflight.get("modelAvailable") is not True:
                    errors.append(f"{engine} summary artifact endpointPreflight.modelAvailable must be true.")
            samples = artifact.get("samples")
            request_path = artifact.get("requestPath") or "/v1/chat/completions"
            sample_trace_ids: list[str] = []
            if not isinstance(samples, list):
                errors.append(f"{engine} summary artifact samples must be a list.")
            else:
                if isinstance(request_count, int) and len(samples) != request_count:
                    errors.append(f"{engine} sample count does not match requestCount.")
                if sum(1 for sample in samples if isinstance(sample, dict) and sample.get("ok")) != success_count:
                    errors.append(f"{engine} successful sample count does not match successCount.")
                for index, sample in enumerate(samples):
                    if not isinstance(sample, dict):
                        errors.append(f"{engine} samples[{index}] must be an object.")
                        continue
                    request_id = sample.get("requestId")
                    if not isinstance(request_id, str) or not request_id:
                        errors.append(f"{engine} samples[{index}].requestId is required.")
                    else:
                        sample_trace_ids.append(request_id)
                    if not isinstance(sample.get("endpoint"), str) or not sample.get("endpoint"):
                        errors.append(f"{engine} samples[{index}].endpoint is required.")
                    if sample.get("streaming") is not True:
                        errors.append(f"{engine} samples[{index}].streaming must be true for serving telemetry proof.")
                    for key in ["e2eLatencyMs", "timeToFirstByteMs", "ttftMs", "ttfotMs", "tpotMs"]:
                        if not _is_number(sample.get(key)) or sample.get(key) < 0:
                            errors.append(f"{engine} samples[{index}].{key} must be a non-negative number.")
                    if not isinstance(sample.get("ttftSource"), str) or not sample.get("ttftSource"):
                        errors.append(f"{engine} samples[{index}].ttftSource is required.")
                    if sample.get("tokenCountSource") not in {"response-usage", "client-estimate"}:
                        errors.append(f"{engine} samples[{index}].tokenCountSource must describe token count provenance.")
                    if not isinstance(sample.get("tokenDetailSource"), str) or not sample.get("tokenDetailSource"):
                        errors.append(f"{engine} samples[{index}].tokenDetailSource must describe token detail provenance.")
                    if not isinstance(sample.get("hardwareTelemetryAvailable"), bool):
                        errors.append(f"{engine} samples[{index}].hardwareTelemetryAvailable must be a boolean.")
                    if sample.get("logprobsAvailable") and not sample.get("tokenDetailsAvailable"):
                        errors.append(f"{engine} samples[{index}].logprobsAvailable requires tokenDetailsAvailable.")
                    if sample.get("tokenIdsAvailable") and not sample.get("tokenDetailsAvailable"):
                        errors.append(f"{engine} samples[{index}].tokenIdsAvailable requires tokenDetailsAvailable.")
                    if sample.get("tokenIdsAvailable") and not isinstance(sample.get("tokenIdSource"), str):
                        errors.append(f"{engine} samples[{index}].tokenIdSource is required when token IDs are available.")
                    if not _is_positive_number(sample.get("outputTokenCount")):
                        errors.append(f"{engine} samples[{index}].outputTokenCount must be a positive number.")
                    if not isinstance(sample.get("promptSha256"), str) or len(sample.get("promptSha256")) != 64:
                        errors.append(f"{engine} samples[{index}].promptSha256 must be a 64-character hex digest.")
                    if not isinstance(sample.get("outputSha256"), str) or len(sample.get("outputSha256")) != 64:
                        errors.append(f"{engine} samples[{index}].outputSha256 must be a 64-character hex digest.")
                    receipt = receipt_by_engine.get(engine, {}).get(str(request_id))
                    if not receipt:
                        continue
                    if receipt.get("method") != "POST":
                        errors.append(f"{engine} receipt {request_id} must be a POST request.")
                    if receipt.get("path") != request_path:
                        errors.append(f"{engine} receipt {request_id} path must be {request_path}.")
                    if not _successful_status(receipt.get("status")):
                        errors.append(f"{engine} receipt {request_id} must have a successful 2xx status.")
                    if receipt.get("campaignId") != campaign_id:
                        errors.append(f"{engine} receipt {request_id} campaignId does not match proof submission.")
                    if receipt.get("runId") != run_id:
                        errors.append(f"{engine} receipt {request_id} runId does not match proof submission.")
                if manifest and isinstance(trace_ids, list) and sample_trace_ids != trace_ids:
                    errors.append(f"{engine} sample requestIds must match manifest platform.requestTraceIds.")
                missing_receipts = sorted(set(sample_trace_ids) - set(receipt_by_engine.get(engine, {}).keys()))
                if missing_receipts:
                    errors.append(f"{engine} receipt log is missing requestIds: {', '.join(missing_receipts)}.")
                if evidence:
                    if evidence.get("requestTraceIds") != sample_trace_ids:
                        errors.append(f"{engine} evidenceIndex requestTraceIds must match sample requestIds.")
                    if sorted(evidence.get("receiptIds") or []) != sorted(sample_trace_ids):
                        errors.append(f"{engine} evidenceIndex receiptIds must match sample requestIds.")
            request_trace = artifact.get("requestTrace")
            if not isinstance(request_trace, list) or len(request_trace) != request_count:
                errors.append(f"{engine} summary artifact requestTrace must match requestCount.")
            token_timeline = artifact.get("tokenTimeline")
            if not isinstance(token_timeline, list) or not token_timeline:
                errors.append(f"{engine} summary artifact tokenTimeline must include streaming chunk rows.")
            else:
                for token_index, token_row in enumerate(token_timeline):
                    if not isinstance(token_row, dict):
                        errors.append(f"{engine} tokenTimeline[{token_index}] must be an object.")
                        continue
                    if not isinstance(token_row.get("tokenDetailSource"), str) or not token_row.get("tokenDetailSource"):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenDetailSource is required.")
                    if token_row.get("tokenPhase") not in (None, "prompt", "output"):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenPhase must be prompt or output when present.")
                    if token_row.get("tokenLogprob") is not None and not _is_number(token_row.get("tokenLogprob")):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenLogprob must be numeric when present.")
                    if token_row.get("tokenId") is not None and not isinstance(token_row.get("tokenId"), int):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenId must be an integer when present.")
                    if token_row.get("tokenId") is not None and not isinstance(token_row.get("tokenIdSource"), str):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenIdSource is required when tokenId is present.")
                output_timeline_request_ids = {
                    token_row.get("requestId")
                    for token_row in token_timeline
                    if isinstance(token_row, dict)
                    and token_row.get("tokenPhase", "output") == "output"
                    and isinstance(token_row.get("requestId"), str)
                }
                missing_output_timeline = sorted(set(sample_trace_ids) - output_timeline_request_ids)
                if missing_output_timeline:
                    errors.append(f"{engine} tokenTimeline is missing output rows for requestIds: {', '.join(missing_output_timeline)}.")
            native_telemetry = artifact.get("nativeTelemetry")
            if not isinstance(native_telemetry, list) or len(native_telemetry) != request_count:
                errors.append(f"{engine} summary artifact nativeTelemetry must match requestCount.")
            elif sample_trace_ids:
                native_telemetry_request_ids = {
                    row.get("requestId")
                    for row in native_telemetry
                    if isinstance(row, dict) and isinstance(row.get("requestId"), str)
                }
                missing_native_telemetry = sorted(set(sample_trace_ids) - native_telemetry_request_ids)
                if missing_native_telemetry:
                    errors.append(f"{engine} nativeTelemetry is missing rows for requestIds: {', '.join(missing_native_telemetry)}.")
            hardware_telemetry = artifact.get("hardwareTelemetry")
            if not isinstance(hardware_telemetry, list) or len(hardware_telemetry) != request_count:
                errors.append(f"{engine} summary artifact hardwareTelemetry must match requestCount.")
            elif sample_trace_ids:
                hardware_telemetry_request_ids = {
                    row.get("requestId")
                    for row in hardware_telemetry
                    if isinstance(row, dict) and isinstance(row.get("requestId"), str)
                }
                missing_hardware_telemetry = sorted(set(sample_trace_ids) - hardware_telemetry_request_ids)
                if missing_hardware_telemetry:
                    errors.append(f"{engine} hardwareTelemetry is missing rows for requestIds: {', '.join(missing_hardware_telemetry)}.")
            capture_policy = artifact.get("capturePolicy") if isinstance(artifact.get("capturePolicy"), dict) else {}
            if capture_policy.get("mode") != "operator-full":
                errors.append(f"{engine} summary artifact capturePolicy.mode must be operator-full.")
            raw_artifact_path = _resolve_proof_member_path(capture_policy.get("rawArtifactPath"), proof_dir)
            raw_captures: list[Any] = []
            if not raw_artifact_path or not os.path.exists(raw_artifact_path):
                errors.append(f"{engine} operator-full raw artifact is required.")
            else:
                raw_artifact = _load_json_file(raw_artifact_path, errors, f"{engine} operator-full raw artifact")
                raw_captures = raw_artifact.get("captures") if isinstance(raw_artifact, dict) and isinstance(raw_artifact.get("captures"), list) else []
                if not raw_captures or len(raw_captures) != request_count:
                    errors.append(f"{engine} operator-full raw artifact captures must match requestCount.")
                elif sample_trace_ids:
                    raw_capture_request_ids = {
                        capture.get("requestId")
                        for capture in raw_captures
                        if isinstance(capture, dict) and isinstance(capture.get("requestId"), str)
                    }
                    missing_raw_captures = sorted(set(sample_trace_ids) - raw_capture_request_ids)
                    if missing_raw_captures:
                        errors.append(f"{engine} operator-full raw artifact captures are missing requestIds: {', '.join(missing_raw_captures)}.")
            raw_manifest_artifacts = [
                item for item in manifest_artifacts
                if isinstance(item, dict) and item.get("kind") == "operator-full-serving-raw"
            ]
            if not raw_manifest_artifacts:
                errors.append(f"{engine} manifest must include an operator-full-serving-raw artifact.")
            elif raw_manifest_artifacts[0].get("path") != capture_policy.get("rawArtifactPath"):
                errors.append(f"{engine} manifest raw artifact path must match summary capturePolicy.")
            measurements = artifact.get("measurements")
            if not isinstance(measurements, list) or not measurements:
                errors.append(f"{engine} summary artifact measurements must be non-empty.")
            else:
                first_measurement = measurements[0]
                if not isinstance(first_measurement, dict):
                    errors.append(f"{engine} summary artifact measurements[0] must be an object.")
                else:
                    _validate_measurement_row(
                        engine=engine,
                        row=first_measurement,
                        expected_model=expected_model if isinstance(expected_model, str) else None,
                        expected_framework=expected_framework,
                        request_count=request_count if isinstance(request_count, int) else None,
                        success_count=success_count if isinstance(success_count, int) else None,
                        errors=errors,
                    )
                    if first_measurement.get("nativeTelemetryRequired") is True:
                        for index, sample in enumerate(samples):
                            if not isinstance(sample, dict):
                                continue
                            if sample.get("nativeTelemetryAvailable") is not True:
                                errors.append(f"{engine} samples[{index}].nativeTelemetryAvailable must be true when native telemetry is required.")
                            for key in REQUIRED_NATIVE_SAMPLE_FIELDS:
                                if not _is_number(sample.get(key)):
                                    errors.append(f"{engine} samples[{index}].{key} must be numeric when native telemetry is required.")
                        for index, telemetry in enumerate(native_telemetry if isinstance(native_telemetry, list) else []):
                            if isinstance(telemetry, dict) and telemetry.get("available") is not True:
                                errors.append(f"{engine} nativeTelemetry[{index}].available must be true when native telemetry is required.")
                            if isinstance(telemetry, dict):
                                for key in REQUIRED_NATIVE_SAMPLE_FIELDS:
                                    if not _is_number(telemetry.get(key)):
                                        errors.append(f"{engine} nativeTelemetry[{index}].{key} must be numeric when native telemetry is required.")
                    if first_measurement.get("hardwareTelemetryRequired") is True:
                        for index, sample in enumerate(samples):
                            if isinstance(sample, dict) and sample.get("hardwareTelemetryAvailable") is not True:
                                errors.append(f"{engine} samples[{index}].hardwareTelemetryAvailable must be true when hardware telemetry is required.")
                            if isinstance(sample, dict):
                                for key in REQUIRED_HARDWARE_SAMPLE_COUNTERS:
                                    if not _is_number(sample.get(key)):
                                        errors.append(f"{engine} samples[{index}].{key} must be numeric when hardware telemetry is required.")
                        for index, telemetry in enumerate(hardware_telemetry if isinstance(hardware_telemetry, list) else []):
                            if isinstance(telemetry, dict) and telemetry.get("available") is not True:
                                errors.append(f"{engine} hardwareTelemetry[{index}].available must be true when hardware telemetry is required.")
                            if isinstance(telemetry, dict):
                                for key in ["powerWatts", "powerWattsPerGpu", "gpuUtilizationPct", "memoryCopyUtilizationPct", "gpuTemperatureC", "smClockMHz", "memoryClockMHz", "fbUsedMiB", "fbFreeMiB", "energyJoules"]:
                                    if not _is_number(telemetry.get(key)):
                                        errors.append(f"{engine} hardwareTelemetry[{index}].{key} must be numeric when hardware telemetry is required.")
                    if first_measurement.get("nativeTelemetryRequired") is True or first_measurement.get("hardwareTelemetryRequired") is True:
                        def raw_snapshot_available_for_validation(capture: dict[str, Any], key: str) -> bool:
                            snapshot = capture.get(key)
                            if not isinstance(snapshot, dict):
                                return False
                            before = snapshot.get("before")
                            after = snapshot.get("after")
                            if not isinstance(before, dict) or not isinstance(after, dict):
                                return False
                            before_has_metrics = bool(before.get("available")) and (
                                isinstance(before.get("metrics"), dict) or isinstance(before.get("jsonMetrics"), dict)
                            )
                            after_has_metrics = bool(after.get("available")) and (
                                isinstance(after.get("metrics"), dict) or isinstance(after.get("jsonMetrics"), dict)
                            )
                            return before_has_metrics and after_has_metrics

                        for index, capture in enumerate(raw_captures):
                            if not isinstance(capture, dict):
                                errors.append(f"{engine} operator-full raw captures[{index}] must be an object.")
                                continue
                            if first_measurement.get("nativeTelemetryRequired") is True and not raw_snapshot_available_for_validation(capture, "nativeMetricsRaw"):
                                errors.append(f"{engine} operator-full raw captures[{index}].nativeMetricsRaw must include before/after native metrics when native telemetry is required.")
                            if first_measurement.get("hardwareTelemetryRequired") is True and not raw_snapshot_available_for_validation(capture, "hardwareMetricsRaw"):
                                errors.append(f"{engine} operator-full raw captures[{index}].hardwareMetricsRaw must include before/after DCGM metrics when hardware telemetry is required.")
                    if first_measurement.get("tokenDetailsRequired") is True:
                        for index, sample in enumerate(samples):
                            if not isinstance(sample, dict):
                                continue
                            for key in ["tokenDetailsAvailable", "tokenIdsAvailable", "logprobsAvailable"]:
                                if sample.get(key) is not True:
                                    errors.append(f"{engine} samples[{index}].{key} must be true when token details are required.")
                            if sample.get("tokenDetailSource") != "response-logprobs":
                                errors.append(f"{engine} samples[{index}].tokenDetailSource must be response-logprobs when token details are required.")
                            if not isinstance(sample.get("tokenIdSource"), str):
                                errors.append(f"{engine} samples[{index}].tokenIdSource must describe tokenizer/response provenance when token details are required.")
                        for token_index, token_row in enumerate(token_timeline if isinstance(token_timeline, list) else []):
                            if not isinstance(token_row, dict):
                                continue
                            if token_row.get("tokenPhase", "output") != "output":
                                continue
                            if not isinstance(token_row.get("tokenIndex"), int):
                                errors.append(f"{engine} tokenTimeline[{token_index}].tokenIndex must be an integer when token details are required.")
                            if not isinstance(token_row.get("tokenId"), int):
                                errors.append(f"{engine} tokenTimeline[{token_index}].tokenId must be an integer when token details are required.")
                            if not isinstance(token_row.get("tokenIdSource"), str):
                                errors.append(f"{engine} tokenTimeline[{token_index}].tokenIdSource must describe tokenizer/response provenance when token details are required.")
                            if not _is_number(token_row.get("tokenLogprob")):
                                errors.append(f"{engine} tokenTimeline[{token_index}].tokenLogprob must be numeric when token details are required.")
                            if token_row.get("tokenDetailSource") != "response-logprobs":
                                errors.append(f"{engine} tokenTimeline[{token_index}].tokenDetailSource must be response-logprobs when token details are required.")
                    if first_measurement.get("promptTokenDetailsRequired") is True:
                        for index, sample in enumerate(samples):
                            if not isinstance(sample, dict):
                                continue
                            if sample.get("promptTokenIdsAvailable") is not True:
                                errors.append(f"{engine} samples[{index}].promptTokenIdsAvailable must be true when prompt token details are required.")
                            if not isinstance(sample.get("promptTokenIdSource"), str):
                                errors.append(f"{engine} samples[{index}].promptTokenIdSource must describe prompt token provenance when prompt token details are required.")
                            if not isinstance(sample.get("promptTokenIdsSha256"), str) or len(sample.get("promptTokenIdsSha256")) != 64:
                                errors.append(f"{engine} samples[{index}].promptTokenIdsSha256 must be a 64-character digest when prompt token details are required.")
                        prompt_rows = [
                            token_row for token_row in (token_timeline if isinstance(token_timeline, list) else [])
                            if isinstance(token_row, dict) and token_row.get("tokenPhase") == "prompt"
                        ]
                        if not prompt_rows:
                            errors.append(f"{engine} tokenTimeline must include prompt token rows when prompt token details are required.")
                        prompt_timeline_request_ids = {
                            token_row.get("requestId")
                            for token_row in prompt_rows
                            if isinstance(token_row.get("requestId"), str)
                        }
                        missing_prompt_timeline = sorted(set(sample_trace_ids) - prompt_timeline_request_ids)
                        if missing_prompt_timeline:
                            errors.append(f"{engine} tokenTimeline is missing prompt rows for requestIds: {', '.join(missing_prompt_timeline)}.")
                        for token_index, token_row in enumerate(prompt_rows):
                            if not isinstance(token_row.get("tokenIndex"), int):
                                errors.append(f"{engine} prompt tokenTimeline[{token_index}].tokenIndex must be an integer.")
                            if not isinstance(token_row.get("tokenId"), int):
                                errors.append(f"{engine} prompt tokenTimeline[{token_index}].tokenId must be an integer.")
                            if not isinstance(token_row.get("tokenIdSource"), str):
                                errors.append(f"{engine} prompt tokenTimeline[{token_index}].tokenIdSource must describe tokenizer provenance.")
                    if evidence:
                        evidence_measurement = evidence.get("measurement") if isinstance(evidence.get("measurement"), dict) else {}
                        for key in [
                            "outputTpm", "totalTpm", "avgLatencyMs",
                            "p50LatencyMs", "p95LatencyMs", "p99LatencyMs",
                            "avgTimeToFirstByteMs", "p50TimeToFirstByteMs", "p95TimeToFirstByteMs", "p99TimeToFirstByteMs",
                            "avgTtftMs", "p50TtftMs", "p95TtftMs", "p99TtftMs",
                            "avgTpotMs", "p50TpotMs", "p95TpotMs", "p99TpotMs",
                            "avgTtfotMs", "p50TtfotMs", "p95TtfotMs", "p99TtfotMs",
                            "metricCompleteness",
                        ]:
                            if evidence_measurement.get(key) != first_measurement.get(key):
                                errors.append(f"{engine} evidenceIndex measurement {key} must match summary artifact.")

    dashboard = proof.get("dashboard")
    if not isinstance(dashboard, dict):
        errors.append("proof dashboard object is required.")
    else:
        row_counts = dashboard.get("rowCounts") if isinstance(dashboard.get("rowCounts"), dict) else {}
        dashboard_rows = dashboard.get("rows") if isinstance(dashboard.get("rows"), dict) else {}
        expected_count = len(expected_engines)
        for name in QUERY_NAMES:
            if _number(row_counts.get(name)) < expected_count:
                errors.append(f"dashboard {name} rowCount must be at least {expected_count}.")
            rows = dashboard_rows.get(name) if isinstance(dashboard_rows, dict) else None
            if not isinstance(rows, list) or not rows:
                errors.append(f"dashboard {name} rows must include inspectable row data.")
        surface_campaign_ids = dashboard.get("surfaceCampaignIds")
        if not isinstance(surface_campaign_ids, dict):
            errors.append("dashboard surfaceCampaignIds is required.")
        else:
            required_campaign_ids = set(campaign_ids)
            for name in CAMPAIGN_ID_QUERY_COLUMN:
                ids = surface_campaign_ids.get(name)
                if not isinstance(ids, list) or not required_campaign_ids.issubset(set(ids)):
                    errors.append(f"dashboard {name} is missing submitted campaign IDs.")
        submitted_campaign_rows = dashboard.get("submittedCampaignRows")
        if not isinstance(submitted_campaign_rows, dict):
            errors.append("dashboard submittedCampaignRows is required.")
        else:
            required_campaign_ids = set(campaign_ids)
            for name, index in CAMPAIGN_ID_QUERY_COLUMN.items():
                rows = submitted_campaign_rows.get(name)
                if not isinstance(rows, list):
                    errors.append(f"dashboard {name} submittedCampaignRows must be a list.")
                    continue
                row_campaign_ids = {
                    row[index]
                    for row in rows
                    if isinstance(row, list) and len(row) > index and isinstance(row[index], str)
                }
                if not required_campaign_ids.issubset(row_campaign_ids):
                    errors.append(f"dashboard {name} submittedCampaignRows is missing submitted campaign rows.")
        for engine, submission in submissions_by_engine.items():
            evidence = evidence_engines.get(engine) if isinstance(evidence_engines.get(engine), dict) else None
            if not evidence:
                continue
            evidence_dashboard = evidence.get("dashboard") if isinstance(evidence.get("dashboard"), dict) else {}
            evidence_surfaces = evidence_dashboard.get("surfaces") if isinstance(evidence_dashboard.get("surfaces"), dict) else {}
            for name in CAMPAIGN_ID_QUERY_COLUMN:
                surface = evidence_surfaces.get(name) if isinstance(evidence_surfaces.get(name), dict) else {}
                rows = surface.get("submittedRows")
                if not isinstance(rows, list) or not rows:
                    errors.append(f"{engine} evidenceIndex dashboard {name} must include submitted rows.")
        runtime_frameworks = dashboard.get("runtimeFrameworks")
        expected_frameworks = {serving_engine_label(engine) for engine in expected_engines}
        if not isinstance(runtime_frameworks, list) or not expected_frameworks.issubset(set(runtime_frameworks)):
            errors.append("dashboard runtimeFrameworks is missing submitted serving frameworks.")

    if event_log_path and os.path.exists(event_log_path):
        for engine in expected_engines:
            submission = submissions_by_engine.get(engine, {})
            expected_request_events = int(submission.get("requestCount") or 1)
            expected_timeline_events = int(submission.get("successCount") or expected_request_events)
            engine_event_counts = event_counts_by_engine.get(engine, {})
            request_sample_events = engine_event_counts.get("serving.measurement.serving_request_sample", 0)
            token_timeline_events = engine_event_counts.get("serving.measurement.serving_token_timeline", 0)
            if request_sample_events < expected_request_events:
                errors.append(
                    f"{engine} event log must include at least {expected_request_events} "
                    "serving.measurement.serving_request_sample events."
                )
            if token_timeline_events < expected_timeline_events:
                errors.append(
                    f"{engine} event log must include at least {expected_timeline_events} "
                    "serving.measurement.serving_token_timeline events."
                )

    kafka_publication = proof.get("kafkaPublication")
    if isinstance(kafka_publication, dict):
        if kafka_publication.get("schemaVersion") != SERVING_KAFKA_PUBLICATION_SCHEMA_VERSION:
            errors.append(f"kafkaPublication schemaVersion must be {SERVING_KAFKA_PUBLICATION_SCHEMA_VERSION}.")
        published_count = kafka_publication.get("publishedCount")
        if not isinstance(published_count, int) or published_count < 0:
            errors.append("kafkaPublication publishedCount must be a non-negative integer.")
        elif event_counts and published_count != sum(event_counts.values()):
            errors.append("kafkaPublication publishedCount must equal event log event count.")

    telemetry_coverage = _telemetry_coverage_from_proof(
        proof=proof,
        proof_dir=proof_dir,
        expected_engines=expected_engines,
        receipt_counts=receipt_counts,
        event_counts_by_engine=event_counts_by_engine,
    )

    return {
        "schemaVersion": PROOF_VERIFICATION_SCHEMA_VERSION,
        "proofPath": proof_abs,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "engineCount": len(submissions_by_engine),
        "requiredEngines": expected_engines,
        "proofBoundary": proof.get("proofBoundary") if isinstance(proof.get("proofBoundary"), str) else None,
        "campaignIds": sorted(campaign_ids),
        "artifactHashes": artifact_hashes,
        "receiptLogPath": receipt_log_path or None,
        "receiptCounts": receipt_counts,
        "eventLogPath": event_log_path or None,
        "eventCounts": event_counts,
        "eventCountsByEngine": event_counts_by_engine,
        "dashboard": proof.get("dashboard") if isinstance(proof.get("dashboard"), dict) else None,
        "telemetryCoverage": telemetry_coverage,
    }


def _load_json_object(path: str, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return parsed


def _enrich_artifact_rows(rows: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            enriched.append({**context, "rowIndex": index, **row})
    return enriched


def _telemetry_coverage_rows_from_verification(
    proof: dict[str, Any],
    verification: dict[str, Any],
    proof_path: str,
) -> list[dict[str, Any]]:
    coverage = verification.get("telemetryCoverage") if isinstance(verification, dict) else None
    if not isinstance(coverage, dict):
        return []
    categories = coverage.get("categories") if isinstance(coverage.get("categories"), dict) else {}
    engines = coverage.get("engines") if isinstance(coverage.get("engines"), dict) else {}
    submissions = {
        item.get("engine"): item
        for item in proof.get("submissions", [])
        if isinstance(item, dict) and isinstance(item.get("engine"), str)
    }
    rows: list[dict[str, Any]] = []
    for engine, engine_coverage in engines.items():
        if not isinstance(engine, str) or not isinstance(engine_coverage, dict):
            continue
        submission = submissions.get(engine) if isinstance(submissions.get(engine), dict) else {}
        for category, item in engine_coverage.items():
            if not isinstance(category, str) or not isinstance(item, dict):
                continue
            category_meta = categories.get(category) if isinstance(categories.get(category), dict) else {}
            rows.append({
                "engine": engine,
                "runtimeFramework": submission.get("runtimeFramework") or serving_engine_label(engine),
                "campaignId": submission.get("campaignId"),
                "runId": submission.get("runId"),
                "model": coverage.get("model") or proof.get("model"),
                "coverageSource": "proof-verifier",
                "coverageCategory": category,
                "coverageStatus": item.get("status"),
                "provenCount": item.get("provenCount"),
                "expectedCount": item.get("expectedCount"),
                "missingJson": json.dumps(item.get("missing") or [], sort_keys=True, separators=(",", ":")),
                "description": category_meta.get("description"),
                "allProven": coverage.get("allProven"),
                "proofPath": proof_path,
            })
    return rows


def extract_proof_rows(
    proof_path: str,
    *,
    require_all_engines: bool = True,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proof_abs = os.path.abspath(proof_path)
    proof_dir = os.path.dirname(proof_abs)
    proof = _load_json_object(proof_abs, "proof summary")
    verification = verification or verify_proof_summary(proof_abs, require_all_engines=require_all_engines)

    rows: dict[str, Any] = {
        "schemaVersion": "performance-iq.serving-proof-rows.v1",
        "proofPath": proof_abs,
        "generatedAtUtc": _utc_now_iso(),
        "verification": verification,
        "submissions": [],
        "dashboardRows": proof.get("dashboard", {}).get("rows", {}) if isinstance(proof.get("dashboard"), dict) else {},
        "servingRequestSamples": [],
        "servingTokenTimeline": [],
        "nativeTelemetry": [],
        "hardwareTelemetry": [],
        "requestTrace": [],
        "measurements": [],
        "requestReceipts": [],
        "eventLogRows": [],
        "telemetryCoverage": verification.get("telemetryCoverage") if isinstance(verification, dict) else None,
        "telemetryCoverageRows": _telemetry_coverage_rows_from_verification(proof, verification, proof_abs),
    }

    submissions = proof.get("submissions")
    if isinstance(submissions, list):
        for submission in submissions:
            if not isinstance(submission, dict):
                continue
            context = {
                "engine": submission.get("engine"),
                "runtimeEngine": submission.get("engine"),
                "runtimeFramework": submission.get("runtimeFramework"),
                "campaignId": submission.get("campaignId"),
                "runId": submission.get("runId"),
                "artifactPath": submission.get("artifactPath"),
                "manifestPath": submission.get("manifestPath"),
            }
            rows["submissions"].append({**context, **submission})
            artifact_path = _resolve_proof_member_path(submission.get("artifactPath"), proof_dir)
            if not artifact_path or not os.path.exists(artifact_path):
                continue
            artifact = _load_json_object(artifact_path, f"{submission.get('engine', 'engine')} summary artifact")
            capture_policy = artifact.get("capturePolicy") if isinstance(artifact.get("capturePolicy"), dict) else {}
            artifact_context = {
                **context,
                "rawArtifactPath": capture_policy.get("rawArtifactPath"),
            }
            rows["servingRequestSamples"].extend(_enrich_artifact_rows(artifact.get("samples"), artifact_context))
            rows["servingTokenTimeline"].extend(_enrich_artifact_rows(artifact.get("tokenTimeline"), artifact_context))
            rows["nativeTelemetry"].extend(_enrich_artifact_rows(artifact.get("nativeTelemetry"), artifact_context))
            rows["hardwareTelemetry"].extend(_enrich_artifact_rows(artifact.get("hardwareTelemetry"), artifact_context))
            rows["requestTrace"].extend(_enrich_artifact_rows(artifact.get("requestTrace"), artifact_context))
            rows["measurements"].extend(_enrich_artifact_rows(artifact.get("measurements"), artifact_context))

    receipt_log_path = _resolve_proof_member_path(proof.get("receiptLogPath"), proof_dir)
    if receipt_log_path and os.path.exists(receipt_log_path):
        rows["receiptLogPath"] = receipt_log_path
        rows["requestReceipts"] = load_receipts(receipt_log_path)

    event_log_path = _resolve_proof_member_path(proof.get("eventLogPath"), proof_dir)
    if event_log_path and os.path.exists(event_log_path):
        rows["eventLogPath"] = event_log_path
        rows["eventLogRows"] = load_serving_event_log(event_log_path)

    rows["rowCounts"] = {
        key: len(value)
        for key, value in rows.items()
        if isinstance(value, list)
    }
    return rows


def write_proof_rows(
    proof_path: str,
    output_path: str,
    *,
    require_all_engines: bool = True,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = extract_proof_rows(
        proof_path,
        require_all_engines=require_all_engines,
        verification=verification,
    )
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return rows


def default_proof_summary_path(artifact_dir: str, run_suffix: str) -> str:
    return os.path.join(artifact_dir, f"serving-smoke-proof-{_safe_slug(run_suffix)}.json")


def default_receipt_log_path(artifact_dir: str, run_suffix: str) -> str:
    return os.path.join(artifact_dir, f"serving-request-receipts-{_safe_slug(run_suffix)}.jsonl")


def start_recording_proxies(
    engines: list[dict[str, Any]],
    receipt_log: str,
    listen_host: str = "127.0.0.1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proxied_engines: list[dict[str, Any]] = []
    proxies: list[dict[str, Any]] = []
    for engine in engines:
        server = recording_proxy_server(
            engine=engine["engine"],
            target_base_url=engine["baseUrl"],
            receipt_log=receipt_log,
            listen_host=listen_host,
            listen_port=0,
        )
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxy_url = f"http://{host}:{port}"
        proxies.append({
            "engine": engine["engine"],
            "listenUrl": proxy_url,
            "targetBaseUrl": engine["baseUrl"],
            "receiptLogPath": receipt_log,
            "server": server,
            "thread": thread,
        })
        proxied_engines.append({
            **engine,
            "upstreamBaseUrl": engine["baseUrl"],
            "baseUrl": proxy_url,
            "receiptProxy": {
                "listenUrl": proxy_url,
                "targetBaseUrl": engine["baseUrl"],
                "receiptLogPath": receipt_log,
            },
        })
    return proxied_engines, proxies


def stop_recording_proxies(proxies: list[dict[str, Any]]) -> None:
    for proxy in proxies:
        proxy["server"].shutdown()
    for proxy in proxies:
        proxy["server"].server_close()
        proxy["thread"].join(timeout=2)


def proxy_summary(proxies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "engine": proxy["engine"],
            "listenUrl": proxy["listenUrl"],
            "targetBaseUrl": proxy["targetBaseUrl"],
            "receiptLogPath": proxy["receiptLogPath"],
        }
        for proxy in proxies
    ]


def write_proof_summary(summary: dict[str, Any], artifact_dir: str, summary_out: str | None = None) -> str:
    proof_path = summary_out or default_proof_summary_path(artifact_dir, str(summary.get("runSuffix") or _utc_slug()))
    parent = os.path.dirname(proof_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    summary["proofSummaryPath"] = proof_path
    summary.setdefault("evidenceIndex", build_evidence_index(summary))
    body = {
        **summary,
        "schemaVersion": PROOF_SCHEMA_VERSION,
        "smokeSummarySchemaVersion": summary.get("schemaVersion"),
        "writtenAtUtc": _utc_now_iso(),
    }
    with open(proof_path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2)
        handle.write("\n")
    return proof_path


def strict_telemetry_gate(verification: dict[str, Any]) -> dict[str, Any]:
    coverage = verification.get("telemetryCoverage") if isinstance(verification.get("telemetryCoverage"), dict) else {}
    category_summary = coverage.get("categorySummary") if isinstance(coverage.get("categorySummary"), dict) else {}
    engines = coverage.get("engines") if isinstance(coverage.get("engines"), dict) else {}
    required_engines = coverage.get("expectedEngines") if isinstance(coverage.get("expectedEngines"), list) else verification.get("requiredEngines")
    required_engine_count = len(required_engines) if isinstance(required_engines, list) else 0
    missing = {
        category
        for category, item in sorted(category_summary.items())
        if isinstance(item, dict) and item.get("status") != "proven"
    }
    not_configured = set()
    for category in STRICT_PRODUCT_TELEMETRY_CATEGORIES:
        summary = category_summary.get(category) if isinstance(category_summary.get(category), dict) else {}
        if summary.get("status") != "proven" or summary.get("expectedEngines") != required_engine_count:
            not_configured.add(category)
        for engine in required_engines if isinstance(required_engines, list) else []:
            engine_coverage = engines.get(engine) if isinstance(engines.get(engine), dict) else {}
            item = engine_coverage.get(category) if isinstance(engine_coverage.get(category), dict) else {}
            if item.get("status") != "proven" or int(item.get("expectedCount") or 0) <= 0:
                not_configured.add(category)
    missing_categories = sorted(missing | not_configured)
    coverage_all_proven = coverage.get("allProven") is True and not missing_categories
    return {
        "ok": verification.get("ok") is True and coverage_all_proven,
        "proofOk": verification.get("ok") is True,
        "allTelemetryProven": coverage_all_proven,
        "configuredTelemetryAllProven": coverage.get("allProven") is True,
        "requiredEngineCount": required_engine_count,
        "strictRequiredCategories": list(STRICT_PRODUCT_TELEMETRY_CATEGORIES),
        "notConfiguredCategories": sorted(not_configured),
        "missingCategories": missing_categories,
    }


def real_runtime_proof_gate(verification: dict[str, Any]) -> dict[str, Any]:
    dashboard = verification.get("dashboard") if isinstance(verification.get("dashboard"), dict) else {}
    checked_fields: dict[str, str] = {}
    for field, value in [
        ("proofBoundary", verification.get("proofBoundary")),
        ("dashboard.proofBoundary", dashboard.get("proofBoundary")),
        ("dashboard.storeProvider", dashboard.get("storeProvider")),
    ]:
        if isinstance(value, str) and value.strip():
            checked_fields[field] = value.strip()

    blocking: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, value in checked_fields.items():
        lower_value = value.lower()
        for marker in REAL_RUNTIME_PROOF_BLOCKING_MARKERS:
            if marker in lower_value and (field, marker) not in seen:
                seen.add((field, marker))
                blocking.append({
                    "field": field,
                    "marker": marker,
                    "value": value,
                })

    if blocking:
        classification = "synthetic-or-fixture"
    elif checked_fields:
        classification = "real-runtime-declared"
    else:
        classification = "real-runtime-unmarked"

    return {
        "ok": verification.get("ok") is True and not blocking,
        "proofOk": verification.get("ok") is True,
        "notSyntheticProof": not blocking,
        "syntheticProof": bool(blocking),
        "classification": classification,
        "checkedFields": sorted(checked_fields),
        "proofBoundary": checked_fields.get("proofBoundary"),
        "dashboardProofBoundary": checked_fields.get("dashboard.proofBoundary"),
        "dashboardStoreProvider": checked_fields.get("dashboard.storeProvider"),
        "blockingMarkers": blocking,
        "requirement": "Proof-boundary fields must not declare fake, synthetic, fixture, or mock runtime proof.",
    }


def _event_key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts if part is not None and str(part) != "")


def _serving_event(
    *,
    topic: str,
    event_type: str,
    key: str,
    payload: dict[str, Any],
    summary: dict[str, Any],
    submission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    campaign_id = submission.get("campaignId") if submission else payload.get("campaignId")
    run_id = submission.get("runId") if submission else payload.get("runId")
    engine = submission.get("engine") if submission else payload.get("engine")
    body = {
        "schemaVersion": SERVING_EVENT_SCHEMA_VERSION,
        "topic": topic,
        "eventType": event_type,
        "partitionKey": key,
        "emittedAtUtc": _utc_now_iso(),
        "model": summary.get("model"),
        "runSuffix": summary.get("runSuffix"),
        "campaignId": campaign_id,
        "runId": run_id,
        "engine": engine,
        "runtimeFramework": submission.get("runtimeFramework") if submission else payload.get("runtimeFramework"),
        "proofSummaryPath": summary.get("proofSummaryPath"),
        "artifactPath": submission.get("artifactPath") if submission else payload.get("artifactPath"),
        "manifestPath": submission.get("manifestPath") if submission else payload.get("manifestPath"),
        "payload": payload,
    }
    body["eventId"] = _sha256_json({
        "schemaVersion": body["schemaVersion"],
        "eventType": body["eventType"],
        "partitionKey": body["partitionKey"],
        "payload": body["payload"],
    })
    return body


def build_serving_event_records(summary: dict[str, Any], *, topic: str = SERVING_EVENT_DEFAULT_TOPIC) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for submission in summary.get("submissions", []):
        if not isinstance(submission, dict):
            continue
        campaign_id = submission.get("campaignId")
        run_id = submission.get("runId")
        engine = submission.get("engine")
        key_base = _event_key(campaign_id, run_id, engine)
        artifact = _read_json_object(str(submission.get("artifactPath") or ""))
        manifest = _read_json_object(str(submission.get("manifestPath") or ""))
        events.append(_serving_event(
            topic=topic,
            event_type="serving.submission",
            key=key_base,
            summary=summary,
            submission=submission,
            payload={
                **{key: value for key, value in submission.items() if key != "errors"},
                "errors": submission.get("errors") or [],
            },
        ))
        if artifact:
            events.append(_serving_event(
                topic=topic,
                event_type="serving.artifact",
                key=_event_key(key_base, "artifact"),
                summary=summary,
                submission=submission,
                payload={
                    "schemaVersion": artifact.get("schemaVersion"),
                    "capturePolicy": artifact.get("capturePolicy"),
                    "requestTrace": artifact.get("requestTrace"),
                    "artifactPath": submission.get("artifactPath"),
                    "artifactSha256": submission.get("artifactSha256"),
                    "manifestPath": submission.get("manifestPath"),
                    "manifestSchemaVersion": manifest.get("schemaVersion"),
                },
            ))
            for measurement_index, row in enumerate(artifact.get("measurements") if isinstance(artifact.get("measurements"), list) else []):
                if not isinstance(row, dict):
                    continue
                surface = str(row.get("surface") or "result")
                request_id = row.get("requestId")
                key = _event_key(key_base, surface, request_id, row.get("chunkIndex"), row.get("tokenIndex"), measurement_index)
                events.append(_serving_event(
                    topic=topic,
                    event_type=f"serving.measurement.{surface}",
                    key=key,
                    summary=summary,
                    submission=submission,
                    payload=row,
                ))
            for name, rows in [
                ("serving.request_trace", artifact.get("requestTrace")),
                ("serving.native_telemetry", artifact.get("nativeTelemetry")),
                ("serving.hardware_telemetry", artifact.get("hardwareTelemetry")),
                ("serving.token_detail_summary", artifact.get("tokenDetails")),
            ]:
                if not isinstance(rows, list):
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    events.append(_serving_event(
                        topic=topic,
                        event_type=name,
                        key=_event_key(key_base, name, row.get("requestId"), index),
                        summary=summary,
                        submission=submission,
                        payload=row,
                    ))

    dashboard = summary.get("dashboard")
    if isinstance(dashboard, dict):
        events.append(_serving_event(
            topic=topic,
            event_type="serving.dashboard_snapshot",
            key=_event_key(summary.get("runSuffix"), "dashboard"),
            summary=summary,
            payload={
                "storeProvider": dashboard.get("storeProvider"),
                "rowCounts": dashboard.get("rowCounts"),
                "surfaceCampaignIds": dashboard.get("surfaceCampaignIds"),
                "submittedCampaignRows": dashboard.get("submittedCampaignRows"),
            },
        ))

    receipt_log = str(summary.get("receiptLogPath") or "")
    if receipt_log and os.path.exists(receipt_log):
        try:
            receipts = load_receipts(receipt_log)
        except ValueError:
            receipts = []
        for receipt in receipts:
            events.append(_serving_event(
                topic=topic,
                event_type="serving.request_receipt",
                key=_event_key(receipt.get("campaignId"), receipt.get("runId"), receipt.get("engine"), receipt.get("requestId")),
                summary=summary,
                payload=receipt,
            ))
    return events


def write_serving_event_log(summary: dict[str, Any], event_log_path: str, *, topic: str = SERVING_EVENT_DEFAULT_TOPIC) -> str:
    parent = os.path.dirname(event_log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    events = build_serving_event_records(summary, topic=topic)
    with open(event_log_path, "w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle, sort_keys=True)
            handle.write("\n")
    return event_log_path


KafkaProducerFactory = Callable[..., Any]


def publish_serving_event_log(
    event_log_path: str,
    *,
    bootstrap_servers: str,
    topic: str | None = None,
    client_id: str = "performance-iq-serving-producer",
    producer_factory: KafkaProducerFactory | None = None,
) -> dict[str, Any]:
    if not bootstrap_servers:
        raise ValueError("bootstrap_servers is required to publish serving events to Kafka.")
    events = load_serving_event_log(event_log_path)
    if not events:
        raise ValueError(f"event log contains no publishable events: {event_log_path}")
    if producer_factory is None:
        try:
            from kafka import KafkaProducer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Kafka publication requires kafka-python. Install with `pip install './python[kafka]'` "
                "or omit --publish-kafka and use the JSONL event log as the durable handoff."
            ) from exc
        producer_factory = KafkaProducer
    producer = producer_factory(
        bootstrap_servers=bootstrap_servers,
        client_id=client_id,
    )
    event_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    try:
        for event in events:
            event_topic = topic or str(event.get("topic") or SERVING_EVENT_DEFAULT_TOPIC)
            event_type = str(event.get("eventType") or "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            topic_counts[event_topic] = topic_counts.get(event_topic, 0) + 1
            future = producer.send(
                event_topic,
                key=str(event.get("partitionKey") or "").encode("utf-8"),
                value=_stable_json(event).encode("utf-8"),
            )
            if hasattr(future, "get"):
                future.get(timeout=30)
        if hasattr(producer, "flush"):
            producer.flush(timeout=30)
    finally:
        if hasattr(producer, "close"):
            producer.close(timeout=30)
    return {
        "schemaVersion": SERVING_KAFKA_PUBLICATION_SCHEMA_VERSION,
        "eventLogPath": event_log_path,
        "bootstrapServers": bootstrap_servers,
        "clientId": client_id,
        "publishedAtUtc": _utc_now_iso(),
        "publishedCount": len(events),
        "eventCounts": event_counts,
        "topicCounts": topic_counts,
    }


def run_serving_smoke(
    *,
    engines: list[dict[str, Any]],
    performance_iq: PerformanceIQ | None,
    model: str,
    prompt: str,
    artifact_dir: str,
    repetitions: int,
    max_tokens: int,
    hardware: str,
    operating_point: str,
    pricing: dict[str, Any],
    run_suffix: str | None = None,
    capture_token_details: bool = False,
    top_logprobs: int | None = None,
    submit: bool = True,
    http_post_json: HttpPostJson | None = None,
    http_stream_json: HttpStreamJson | None = None,
    http_get_text: HttpGetText | None = None,
) -> dict[str, Any]:
    suffix = run_suffix or _utc_slug()
    submissions: list[dict[str, Any]] = []
    capability_gaps: list[dict[str, Any]] = []
    for engine in engines:
        engine_id = engine["engine"]
        engine_capture_token_details = (
            bool(engine["captureTokenDetails"])
            if "captureTokenDetails" in engine
            else capture_token_details
        )
        engine_for_run = dict(engine)
        token_detail_policy = token_detail_capability_policy(engine_for_run, engine_capture_token_details)
        if token_detail_policy:
            engine_for_run["tokenDetailsCapability"] = token_detail_policy
        if token_detail_policy and token_detail_policy.get("safeToRequest") is False:
            engine_capture_token_details = False
            engine_for_run["captureTokenDetails"] = False
            capability_gaps.append({
                "engine": engine_id,
                "category": "outputTokenIdsLogprobs",
                **token_detail_policy,
            })
        engine_top_logprobs = engine.get("topLogprobs", top_logprobs)
        result = run_serving_producer(
            engine=engine_for_run,
            request={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "repetitions": repetitions,
                "maxTokens": max_tokens,
                **({"captureTokenDetails": True} if engine_capture_token_details else {}),
                **({"topLogprobs": engine_top_logprobs} if engine_capture_token_details and engine_top_logprobs is not None else {}),
            },
            performance_iq=performance_iq,
            submit=submit,
            artifact_dir=artifact_dir,
            source_type="other-measured-producer",
            run_class="measured",
            campaign={
                "campaignId": f"serving-{engine_id}-{suffix}",
                "runId": f"serving-{engine_id}-{suffix}-run",
            },
            workload={
                "hardware": hardware,
                "operatingPoint": operating_point,
                "scenario": f"Real endpoint smoke for {serving_engine_label(engine_id)}",
            },
            pricing=pricing,
            http_post_json=http_post_json,
            http_stream_json=http_stream_json,
            http_get_text=http_get_text,
        )
        submissions.append({
            "engine": engine_id,
            "runtimeFramework": result["measurements"][0]["runtimeFramework"],
            "requestCount": len(result["samples"]),
            "successCount": sum(1 for sample in result["samples"] if sample.get("ok")),
            "errorCount": sum(1 for sample in result["samples"] if not sample.get("ok")),
            "status": (result.get("submission") or {}).get("status"),
            "liveProofReady": (result.get("submission") or {}).get("liveProofReady"),
            "campaignId": result["manifest"]["campaign"]["campaignId"],
            "runId": result["manifest"]["campaign"]["runId"],
            "artifactPath": result["artifactPath"],
            "artifactSha256": result["manifest"]["artifacts"][0]["sha256"],
            "manifestPath": result["manifestPath"],
            "errors": [sample.get("error") for sample in result["samples"] if sample.get("error")],
        })
    return {
        "schemaVersion": "performance-iq.serving-smoke-summary.v1",
        "model": model,
        "runSuffix": suffix,
        "artifactDir": artifact_dir,
        "proofBoundary": "configured serving endpoint capture with measured client telemetry and dashboard snapshot when queried",
        "submissions": submissions,
        "telemetryCapabilityGaps": capability_gaps,
    }


def _fake_dashboard(summary: dict[str, Any]) -> dict[str, Any]:
    submissions = [item for item in summary.get("submissions", []) if isinstance(item, dict)]
    campaign_ids = [str(item.get("campaignId")) for item in submissions if item.get("campaignId")]
    runtime_frameworks = sorted({
        str(item.get("runtimeFramework"))
        for item in submissions
        if item.get("runtimeFramework")
    })
    request_sample_count = 0
    token_timeline_count = 0
    telemetry_coverage_rows = []
    run_detail_rows = []
    for item in submissions:
        artifact = _read_json_object(str(item.get("artifactPath") or ""))
        measurements = artifact.get("measurements") if isinstance(artifact.get("measurements"), list) else []
        request_sample_count += sum(1 for row in measurements if isinstance(row, dict) and row.get("surface") == "serving_request_sample")
        token_timeline_count += sum(1 for row in measurements if isinstance(row, dict) and row.get("surface") == "serving_token_timeline")
        telemetry_coverage_rows.extend([
            [
                item.get("campaignId"),
                item.get("runId"),
                item.get("runtimeFramework"),
                row.get("coverageCategory"),
                row.get("coverageStatus"),
                row.get("provenCount"),
                row.get("expectedCount"),
                row.get("allProven"),
            ]
            for row in measurements
            if isinstance(row, dict) and row.get("surface") == "serving_telemetry_coverage"
        ])
        first = measurements[0] if measurements and isinstance(measurements[0], dict) else {}
        run_detail_rows.append([
            item.get("campaignId"),
            item.get("runId"),
            item.get("runtimeFramework"),
            first.get("avgTtftMs"),
            first.get("avgTpotMs"),
            first.get("avgTtfotMs"),
            first.get("metricCompleteness"),
        ])
    engine_count = len(submissions)
    campaign_rows = [[campaign_id] for campaign_id in campaign_ids]
    return {
        "storeProvider": "local-fake-serving-fixture",
        "proofBoundary": "local fake serving engines and synthetic dashboard row snapshot; not real runtime or production dashboard proof",
        "rowCounts": {
            "price_performance": engine_count,
            "capacity_best": engine_count,
            "campaign_provenance": engine_count,
            "run_details": engine_count,
            "serving_request_samples": request_sample_count,
            "serving_token_timeline": token_timeline_count,
            "serving_telemetry_coverage": len(telemetry_coverage_rows),
        },
        "rows": {
            "price_performance": [
                [summary.get("model"), item.get("runtimeFramework"), item.get("engine")]
                for item in submissions
            ],
            "capacity_best": [
                [summary.get("model"), item.get("runtimeFramework")]
                for item in submissions
            ],
            "campaign_provenance": campaign_rows,
            "run_details": run_detail_rows,
            "serving_request_samples": campaign_rows,
            "serving_token_timeline": campaign_rows,
            "serving_telemetry_coverage": telemetry_coverage_rows,
        },
        "campaignIds": campaign_ids,
        "surfaceCampaignIds": {
            "campaign_provenance": campaign_ids,
            "run_details": campaign_ids,
            "serving_request_samples": campaign_ids,
            "serving_token_timeline": campaign_ids,
            "serving_telemetry_coverage": campaign_ids if telemetry_coverage_rows else [],
        },
        "submittedCampaignRows": {
            "campaign_provenance": campaign_rows,
            "run_details": campaign_rows,
            "serving_request_samples": campaign_rows,
            "serving_token_timeline": campaign_rows,
            "serving_telemetry_coverage": telemetry_coverage_rows,
        },
        "runtimeFrameworks": runtime_frameworks,
    }


def _fake_performance_iq_client() -> PerformanceIQ:
    def transport(method: str, url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        payload = json.loads(body.decode("utf-8")) if body else {}
        manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
        campaign = manifest.get("campaign") if isinstance(manifest.get("campaign"), dict) else {}
        return {
            "id": campaign.get("runId") or "fake-serving-run",
            "status": "accepted",
            "liveProofReady": False,
            "validation": {
                "ok": True,
                "liveProofReady": False,
                "proofBoundary": "local fake serving fixture",
            },
        }

    return PerformanceIQ("http://performance-iq.local/fake", token="fake-token", transport=transport)


def _start_fake_engine_servers(model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from performance_iq_sdk.serving_fake_engine import FakeEngineState, FakeOpenAIServer

    engines: list[dict[str, Any]] = []
    servers: list[dict[str, Any]] = []
    for index, engine_id in enumerate(ENGINE_IDS, start=1):
        server = FakeOpenAIServer(("127.0.0.1", 0), FakeEngineState(engine_id, model))
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://{host}:{port}"
        engines.append({
            "engine": engine_id,
            "baseUrl": base_url,
            "metricsUrl": f"{base_url}/prometheus/metrics" if engine_id == "tensorrt-llm" else f"{base_url}/metrics",
            **({"nativeJsonMetricsUrl": f"{base_url}/metrics"} if engine_id == "tensorrt-llm" else {}),
            "hardwareMetricsUrl": f"{base_url}/prometheus/metrics" if engine_id == "tensorrt-llm" else f"{base_url}/metrics",
            "requireNativeTelemetry": True,
            "requireHardwareTelemetry": True,
            "promptTokenIds": [11, 22, 33, 44],
            "frameworkVersion": f"{engine_id}-fake-1.0",
            "modelRevision": "fake-model-revision",
            "imageDigest": f"sha256:{str(index) * 64}",
            "imageTag": f"performance-iq/{engine_id}:fake",
            "serverArgs": [engine_id, "serve", model, "--fake-full-telemetry"],
            "processId": str(9000 + index),
            "containerId": f"fake-{engine_id}-container",
            "podName": f"fake-{engine_id}-pod",
            "nodeName": "fake-node",
            "hostName": "local-fake-host",
        })
        servers.append({"engine": engine_id, "server": server, "thread": thread})
    return engines, servers


def _stop_fake_engine_servers(servers: list[dict[str, Any]]) -> None:
    for item in servers:
        item["server"].shutdown()
    for item in servers:
        item["server"].server_close()
        item["thread"].join(timeout=2)


def run_fake_full_telemetry_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_suffix = args.run_suffix or f"fake-full-telemetry-{_utc_slug()}"
    receipt_log = args.receipt_log or default_receipt_log_path(args.artifact_dir, run_suffix)
    event_log = args.event_log or os.path.join(args.artifact_dir, f"serving-events-{_safe_slug(run_suffix)}.jsonl")
    fake_servers: list[dict[str, Any]] = []
    proxies: list[dict[str, Any]] = []
    try:
        engines, fake_servers = _start_fake_engine_servers(args.model)
        proxied_engines, proxies = start_recording_proxies(
            engines,
            receipt_log,
            listen_host=args.receipt_proxy_host,
        )
        preflight = runtime_preflight(proxied_engines, [], model=args.model)
        if not preflight["ready"]:
            raise ValueError("fake full telemetry preflight failed: " + json.dumps(preflight, indent=2))
        summary = run_serving_smoke(
            engines=attach_endpoint_preflight(proxied_engines, preflight),
            performance_iq=_fake_performance_iq_client(),
            model=args.model,
            prompt=args.prompt,
            artifact_dir=args.artifact_dir,
            repetitions=args.repetitions,
            max_tokens=args.max_tokens,
            hardware="local fake serving engines",
            operating_point="fake-full-telemetry",
            pricing={
                "usdPerGpuHour": args.usd_per_gpu_hour if args.usd_per_gpu_hour is not None else 1.0,
                "gpuCount": args.gpu_count,
                "powerWattsPerGpu": args.power_watts_per_gpu if args.power_watts_per_gpu is not None else 120.0,
            },
            run_suffix=run_suffix,
            capture_token_details=True,
            top_logprobs=args.top_logprobs if args.top_logprobs > 0 else 5,
            submit=True,
        )
        summary["proofBoundary"] = "local fake serving engines and synthetic dashboard row snapshot; not real runtime proof"
        summary["preflight"] = preflight
        summary["receiptProxies"] = proxy_summary(proxies)
        summary["dashboard"] = _fake_dashboard(summary)
        summary["receiptLogPath"] = receipt_log
        summary["eventLogPath"] = event_log
        proof_path = write_proof_summary(summary, args.artifact_dir, summary_out=args.summary_out)
        write_serving_event_log(summary, event_log, topic=args.kafka_topic)
        verification = verify_proof_summary(proof_path)
        return {
            **summary,
            "proofSummaryPath": proof_path,
            "verification": verification,
        }
    finally:
        stop_recording_proxies(proxies)
        _stop_fake_engine_servers(fake_servers)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit vLLM, SGLang, and TensorRT-LLM OpenAI-compatible serving producer smoke data to Performance IQ.",
    )
    parser.add_argument("--piq-base-url", default=_env("PIQ_BASE_URL"), help="Performance IQ base URL; env PIQ_BASE_URL.")
    parser.add_argument("--piq-token", default=_env("PIQ_TOKEN"), help="Performance IQ bearer token; env PIQ_TOKEN.")
    parser.add_argument("--vllm-url", help="vLLM base URL; env PIQ_VLLM_URL.")
    parser.add_argument("--sglang-url", help="SGLang base URL; env PIQ_SGLANG_URL.")
    parser.add_argument("--tensorrt-llm-url", help="TensorRT-LLM base URL; env PIQ_TENSORRT_LLM_URL.")
    parser.add_argument("--model", default=_env("PIQ_SERVING_MODEL", laptop_smoke_model()))
    parser.add_argument("--prompt", default="Return a short acknowledgement.")
    parser.add_argument("--artifact-dir", default=_env("PIQ_ARTIFACT_DIR", ".performance-iq/serving-producers"))
    parser.add_argument("--repetitions", type=int, default=int(_env("PIQ_SERVING_REPETITIONS", "3") or "3"))
    parser.add_argument("--max-tokens", type=int, default=int(_env("PIQ_SERVING_MAX_TOKENS", "64") or "64"))
    parser.add_argument("--hardware", default=_env("PIQ_SERVING_HARDWARE", "local serving endpoint"))
    parser.add_argument("--operating-point", default=_env("PIQ_SERVING_OPERATING_POINT", "serving-smoke"))
    parser.add_argument("--usd-per-gpu-hour", type=float, default=_float_env("PIQ_SERVING_USD_PER_GPU_HOUR"))
    parser.add_argument("--gpu-count", type=float, default=_float_env("PIQ_SERVING_GPU_COUNT", 1.0) or 1.0)
    parser.add_argument("--power-watts-per-gpu", type=float, default=_float_env("PIQ_SERVING_POWER_WATTS_PER_GPU"))
    parser.add_argument("--capture-token-details", action="store_true", default=_env("PIQ_SERVING_CAPTURE_TOKEN_DETAILS", "false").lower() == "true", help="Request response token logprobs/top-logprobs when the serving engine supports them.")
    parser.add_argument("--top-logprobs", type=int, default=int(_env("PIQ_SERVING_TOP_LOGPROBS", "0") or "0"), help="Number of top logprobs to request when token detail capture is enabled.")
    parser.add_argument("--resolve-token-ids-with-tokenizer", action="store_true", default=_env("PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER", "false").lower() == "true", help="Use the configured Hugging Face tokenizer to resolve token IDs when response logprob items omit IDs.")
    parser.add_argument("--tokenizer-model", default=_env("PIQ_SERVING_TOKENIZER_MODEL"), help="Global Hugging Face tokenizer model for token ID resolution; per-engine PIQ_<ENGINE>_TOKENIZER_MODEL overrides it.")
    parser.add_argument("--collect-hardware-metrics", action="store_true", default=_env("PIQ_SERVING_COLLECT_HARDWARE_METRICS", "false").lower() == "true", help="Read DCGM metrics from PIQ_*_HARDWARE_METRICS_URL, or engine /metrics when no dedicated URL is set.")
    parser.add_argument("--require-native-telemetry", action="store_true", default=_env("PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY", "false").lower() == "true", help="Make proof completeness require native serving Prometheus telemetry for configured engines.")
    parser.add_argument("--require-hardware-telemetry", action="store_true", default=_env("PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY", "false").lower() == "true", help="Make proof completeness require DCGM hardware telemetry for configured engines.")
    parser.add_argument("--framework-version")
    parser.add_argument("--model-revision")
    parser.add_argument("--image-digest")
    parser.add_argument("--image-tag")
    parser.add_argument("--server-args", help="Global serving-engine launch args or command line; per-engine PIQ_<ENGINE>_SERVER_ARGS overrides it.")
    parser.add_argument("--process-id", help="Global serving-engine process ID; per-engine PIQ_<ENGINE>_PROCESS_ID overrides it.")
    parser.add_argument("--container-id", help="Global serving-engine container ID; per-engine PIQ_<ENGINE>_CONTAINER_ID overrides it.")
    parser.add_argument("--pod-name", help="Global serving-engine pod name; per-engine PIQ_<ENGINE>_POD_NAME overrides it.")
    parser.add_argument("--node-name", help="Global serving-engine node name; per-engine PIQ_<ENGINE>_NODE_NAME overrides it.")
    parser.add_argument("--host-name", help="Global serving-engine host name; per-engine PIQ_<ENGINE>_HOST_NAME overrides it.")
    parser.add_argument("--run-suffix", default=_env("PIQ_SERVING_RUN_SUFFIX"))
    parser.add_argument("--summary-out", default=_env("PIQ_SERVING_SUMMARY_OUT"), help="Write the overall smoke proof summary to this JSON path.")
    parser.add_argument("--event-log", default=_env("PIQ_SERVING_EVENT_LOG"), help="Write Kafka-ready post-capture serving telemetry events to this JSONL path.")
    parser.add_argument("--kafka-topic", default=_env("PIQ_SERVING_KAFKA_TOPIC", SERVING_EVENT_DEFAULT_TOPIC), help="Topic name embedded in post-capture serving telemetry events.")
    parser.add_argument("--publish-kafka", action="store_true", default=_env("PIQ_SERVING_PUBLISH_KAFKA", "false").lower() == "true", help="Publish post-capture serving telemetry events to Kafka after writing the JSONL event log.")
    parser.add_argument("--kafka-bootstrap-servers", default=_env("PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS"), help="Kafka bootstrap servers for --publish-kafka; env PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS.")
    parser.add_argument("--kafka-client-id", default=_env("PIQ_SERVING_KAFKA_CLIENT_ID", "performance-iq-serving-producer"), help="Kafka client ID for post-capture event publication.")
    parser.add_argument("--receipt-log", default=_env("PIQ_SERVING_RECEIPT_LOG"), help="JSONL receipt log produced by the serving request recorder.")
    parser.add_argument("--record-receipts", action="store_true", help="Start in-process receipt proxies and route engine traffic through them.")
    parser.add_argument("--receipt-proxy-host", default=_env("PIQ_SERVING_RECEIPT_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--verify-after-capture", action="store_true", default=_env("PIQ_SERVING_VERIFY_AFTER_CAPTURE", "false").lower() == "true", help="Run the saved-proof verifier after capture/submission completes.")
    parser.add_argument("--require-telemetry-coverage", action="store_true", default=_env("PIQ_SERVING_REQUIRE_TELEMETRY_COVERAGE", "false").lower() == "true", help="Fail verification unless strictTelemetryGate.ok is true for every full-product telemetry category.")
    parser.add_argument("--require-real-runtime-proof", action="store_true", default=_env("PIQ_SERVING_REQUIRE_REAL_RUNTIME_PROOF", "false").lower() == "true", help="Fail verification when proof-boundary fields declare fake, synthetic, fixture, or mock runtime proof.")
    parser.add_argument("--no-submit", action="store_true", help="Capture artifacts and manifests without submitting to Performance IQ.")
    parser.add_argument("--query-dashboard", action="store_true", help="Query fixed dashboard surfaces after submission.")
    parser.add_argument("--allow-missing-engines", action="store_true", help="Run configured engines only instead of requiring all three URLs.")
    parser.add_argument("--preflight-only", action="store_true", help="Report local runtime and endpoint readiness without sending completion requests.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Report read-only host, cache, port, and endpoint diagnostics for real engine setup.")
    parser.add_argument("--launch-plan-only", action="store_true", help="Print host-aware launch commands for the three serving engines.")
    parser.add_argument("--fake-full-telemetry", action="store_true", help="Run deterministic local fake engines with strict telemetry, receipts, event log, and proof verification.")
    parser.add_argument("--verify-proof", help="Verify a saved serving smoke proof summary JSON. Requires all three engines unless --allow-missing-engines is set.")
    parser.add_argument("--dump-proof-rows", default=_env("PIQ_SERVING_PROOF_ROWS_OUT"), help="With --verify-proof, write all finest-grain proof rows to this JSON path.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the default /v1/models readiness check before sending completion requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.verify_proof:
        report = verify_proof_summary(args.verify_proof, require_all_engines=not args.allow_missing_engines)
        if args.dump_proof_rows:
            write_proof_rows(
                args.verify_proof,
                args.dump_proof_rows,
                require_all_engines=not args.allow_missing_engines,
                verification=report,
            )
            report["proofRowsPath"] = os.path.abspath(args.dump_proof_rows)
        if args.require_telemetry_coverage:
            report["strictTelemetryGate"] = strict_telemetry_gate(report)
        if args.require_real_runtime_proof:
            report["realRuntimeProofGate"] = real_runtime_proof_gate(report)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            return 1
        if args.require_telemetry_coverage and not report["strictTelemetryGate"]["ok"]:
            return 1
        if args.require_real_runtime_proof and not report["realRuntimeProofGate"]["ok"]:
            return 1
        return 0
    engines, missing = engine_configs_from_env(args)
    if args.launch_plan_only:
        print(json.dumps(runtime_launch_plan(args.model), indent=2))
        return 0
    if args.diagnostics_only:
        print(json.dumps(runtime_diagnostics(engines, missing, model=args.model), indent=2))
        return 0
    if args.fake_full_telemetry:
        try:
            report = run_fake_full_telemetry_smoke(args)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        verification = report.get("verification") if isinstance(report.get("verification"), dict) else {}
        report["strictTelemetryGate"] = strict_telemetry_gate(verification)
        report["realRuntimeProofGate"] = real_runtime_proof_gate(verification)
        print(json.dumps(report, indent=2))
        if args.require_real_runtime_proof and not report["realRuntimeProofGate"]["ok"]:
            return 1
        return 0 if report["strictTelemetryGate"]["ok"] else 1
    if args.preflight_only:
        preflight = runtime_preflight(
            engines,
            [] if args.allow_missing_engines else missing,
            model=args.model,
        )
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["ready"] else 1
    if missing and not args.allow_missing_engines:
        print(
            "Missing serving engine URL(s): " + ", ".join(missing) +
            ". Set all three URLs or pass --allow-missing-engines for partial smoke.",
            file=sys.stderr,
        )
        return 2
    if not engines:
        print("No serving engine URLs configured.", file=sys.stderr)
        return 2
    if not args.no_submit and not args.piq_base_url:
        print("PIQ_BASE_URL or --piq-base-url is required unless --no-submit is set.", file=sys.stderr)
        return 2
    if args.publish_kafka:
        if not args.event_log:
            print("--publish-kafka requires --event-log or PIQ_SERVING_EVENT_LOG so publication happens after durable capture.", file=sys.stderr)
            return 2
        if not args.kafka_bootstrap_servers:
            print("--publish-kafka requires --kafka-bootstrap-servers or PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS.", file=sys.stderr)
            return 2

    client = None if args.no_submit else PerformanceIQ(args.piq_base_url, token=args.piq_token)
    pricing = {
        **({"usdPerGpuHour": args.usd_per_gpu_hour} if args.usd_per_gpu_hour is not None else {}),
        "gpuCount": args.gpu_count,
        **({"powerWattsPerGpu": args.power_watts_per_gpu} if args.power_watts_per_gpu is not None else {}),
    }
    run_suffix = args.run_suffix or _utc_slug()
    receipt_log = args.receipt_log
    proxies: list[dict[str, Any]] = []
    try:
        if args.record_receipts:
            receipt_log = receipt_log or default_receipt_log_path(args.artifact_dir, run_suffix)
            engines, proxies = start_recording_proxies(
                engines,
                receipt_log,
                listen_host=args.receipt_proxy_host,
            )

        preflight = None
        if not args.skip_preflight:
            preflight = runtime_preflight(engines, [], model=args.model)
            if not preflight["ready"]:
                print(json.dumps(preflight, indent=2))
                print("Serving engine preflight failed; pass --skip-preflight only for targeted debugging.", file=sys.stderr)
                return 1

        summary = run_serving_smoke(
            engines=attach_endpoint_preflight(engines, preflight),
            performance_iq=client,
            model=args.model,
            prompt=args.prompt,
            artifact_dir=args.artifact_dir,
            repetitions=args.repetitions,
            max_tokens=args.max_tokens,
            hardware=args.hardware,
            operating_point=args.operating_point,
            pricing=pricing,
            run_suffix=run_suffix,
            capture_token_details=args.capture_token_details,
            top_logprobs=args.top_logprobs if args.top_logprobs > 0 else None,
            submit=not args.no_submit,
        )
        if preflight is not None:
            summary["preflight"] = preflight
        if proxies:
            summary["receiptProxies"] = proxy_summary(proxies)
        if args.query_dashboard:
            if not args.piq_base_url:
                raise ValueError("dashboard query requires PIQ_BASE_URL or --piq-base-url")
            summary["dashboard"] = query_dashboard(
                args.piq_base_url,
                token=args.piq_token,
                campaign_ids=[item["campaignId"] for item in summary["submissions"]],
            )
        if receipt_log:
            summary["receiptLogPath"] = receipt_log
        if args.event_log:
            summary["eventLogPath"] = args.event_log
        proof_path = write_proof_summary(summary, args.artifact_dir, summary_out=args.summary_out)
        if args.event_log:
            write_serving_event_log(summary, args.event_log, topic=args.kafka_topic)
        if args.publish_kafka:
            summary["kafkaPublication"] = publish_serving_event_log(
                args.event_log,
                bootstrap_servers=args.kafka_bootstrap_servers,
                topic=args.kafka_topic,
                client_id=args.kafka_client_id,
            )
            proof_path = write_proof_summary(summary, args.artifact_dir, summary_out=args.summary_out)
        if args.verify_after_capture or args.require_telemetry_coverage or args.require_real_runtime_proof:
            verification = verify_proof_summary(proof_path, require_all_engines=not args.allow_missing_engines)
            summary["verification"] = verification
            summary["strictTelemetryGate"] = strict_telemetry_gate(verification)
            summary["realRuntimeProofGate"] = real_runtime_proof_gate(verification)
        print(json.dumps(summary, indent=2))
        if args.verify_after_capture and not summary.get("verification", {}).get("ok"):
            return 1
        if args.require_telemetry_coverage and not summary.get("strictTelemetryGate", {}).get("ok"):
            return 1
        if args.require_real_runtime_proof and not summary.get("realRuntimeProofGate", {}).get("ok"):
            return 1
        failures = [
            item for item in summary["submissions"]
            if item["successCount"] != item["requestCount"] or (not args.no_submit and item["status"] != "accepted")
        ]
        if failures:
            return 1
        if args.query_dashboard:
            row_counts = summary.get("dashboard", {}).get("rowCounts", {})
            surface_campaign_ids = summary.get("dashboard", {}).get("surfaceCampaignIds", {})
            submitted_campaign_ids = {item["campaignId"] for item in summary["submissions"]}
            expected = len(summary["submissions"])
            missing_campaign_surfaces = [
                name for name in CAMPAIGN_ID_QUERY_COLUMN
                if not submitted_campaign_ids.issubset(set(surface_campaign_ids.get(name, [])))
            ]
            if (
                any(row_counts.get(name, 0) < expected for name in QUERY_NAMES) or
                missing_campaign_surfaces
            ):
                return 1
        return 0
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        stop_recording_proxies(proxies)


if __name__ == "__main__":
    raise SystemExit(main())
