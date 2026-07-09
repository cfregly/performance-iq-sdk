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
from typing import Any

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
)
CAMPAIGN_ID_QUERY_COLUMN = {
    "campaign_provenance": 0,
    "run_details": 0,
    "serving_request_samples": 0,
    "serving_token_timeline": 0,
}
ENGINE_DEFAULT_PORT = {
    "vllm": 8000,
    "sglang": 30000,
    "tensorrt-llm": 8001,
}
PROOF_SCHEMA_VERSION = "performance-iq.serving-smoke-proof.v1"
PROOF_VERIFICATION_SCHEMA_VERSION = "performance-iq.serving-smoke-proof-verification.v1"
EVIDENCE_INDEX_SCHEMA_VERSION = "performance-iq.serving-evidence-index.v1"
SERVING_SUMMARY_SCHEMA_VERSION = "performance-iq.serving-producer-summary.v1"
PRODUCER_MANIFEST_SCHEMA_VERSION = "performance-iq.producer-manifest.v1"
LOW_FREE_SPACE_BYTES = 30 * 1024 * 1024 * 1024
SIZE_TIMEOUT_SECONDS = 15
COMMAND_PROBE_TIMEOUT_SECONDS = 30


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


def command_probe_timeout_seconds() -> float:
    value = _env("PIQ_COMMAND_PROBE_TIMEOUT_SECONDS")
    if value is None:
        return COMMAND_PROBE_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return COMMAND_PROBE_TIMEOUT_SECONDS
    return max(1.0, timeout)


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
        metrics_url = _env(ENGINE_METRICS_URL_ENV[engine], f"{url.rstrip('/')}/metrics")
        hardware_metrics_url = _env(ENGINE_HARDWARE_METRICS_URL_ENV[engine])
        if getattr(args, "collect_hardware_metrics", False) and not hardware_metrics_url:
            hardware_metrics_url = metrics_url
        configs.append({
            "engine": engine,
            "baseUrl": url,
            "metricsUrl": metrics_url,
            **({"hardwareMetricsUrl": hardware_metrics_url} if hardware_metrics_url else {}),
            **({"requireHardwareTelemetry": True} if getattr(args, "require_hardware_telemetry", False) else {}),
            **({"apiKey": api_key} if api_key else {}),
            **({"frameworkVersion": args.framework_version} if args.framework_version else {}),
            **({"imageDigest": args.image_digest} if args.image_digest else {}),
            **({"imageTag": args.image_tag} if args.image_tag else {}),
        })
    return configs, missing


def command_probe(command: str, *args: str) -> dict[str, Any]:
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
        "PIQ_TENSORRT_LLM_IMAGE",
        "PIQ_PYTHON_BIN",
        "PIQ_SERVING_BIN_DIR",
        "PIQ_VLLM_SOURCE_PATH",
        "PIQ_SGLANG_SOURCE_PATH",
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
    local = preflight.get("localRuntime", {})
    if not local.get("vllmModule", {}).get("available"):
        blockers.append("Python module 'vllm' is not importable in the smoke-runner Python environment.")
    elif not local.get("vllmExtension", {}).get("available"):
        blockers.append("Python module 'vllm' is importable, but compiled extension 'vllm._C' is missing.")
    elif local.get("vllmExtension", {}).get("imported") is False:
        blockers.append("Python module 'vllm._C' is present, but failed to import in the smoke-runner Python environment.")
    if not local.get("sglangModule", {}).get("available"):
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
    url = f"{str(engine['baseUrl']).rstrip('/')}/v1/models"
    request = urllib.request.Request(
        url,
        headers={
            "accept": "application/json",
            **({"authorization": f"Bearer {engine['apiKey']}"} if engine.get("apiKey") else {}),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_body = response.read(4096).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw_body) if raw_body.strip().startswith("{") else None
            except json.JSONDecodeError:
                body = None
            return _with_model_check({
                "engine": engine["engine"],
                "url": url,
                "reachable": True,
                "status": response.status,
                "ok": 200 <= response.status < 300,
                "bodyPreview": raw_body[:300],
            }, body, model)
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


def runtime_launch_plan(model: str) -> dict[str, Any]:
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
                    f"vllm serve {quoted_model} --host 127.0.0.1 "
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
                    f"SGLANG_USE_MLX=1 python -m sglang.launch_server --model {quoted_model} "
                    f"--disable-cuda-graph --host 127.0.0.1 --port {ENGINE_DEFAULT_PORT['sglang']} "
                    f"--served-model-name {quoted_model}"
                ) if apple_silicon else (
                    f"python -m sglang.launch_server --model-path {quoted_model} --host 127.0.0.1 "
                    f"--port {ENGINE_DEFAULT_PORT['sglang']} --served-model-name {quoted_model}"
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
            "tensorrtLlmServeCommand": command_probe("trtllm-serve", "--help"),
            "nvidiaSmiCommand": command_probe("nvidia-smi"),
        },
        "storage": storage_probe(),
        "missingEngineUrls": missing_urls,
        "endpoints": endpoint_results,
        "configuredEngineCount": len(endpoint_results),
        "launchPlan": runtime_launch_plan(model or laptop_smoke_model()),
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
    for key in ["avgLatencyMs", "p95LatencyMs", "p50LatencyMs", "p99LatencyMs", "avgTtftMs", "avgTpotMs", "avgTtfotMs"]:
        if not _is_number(row.get(key)) or row.get(key) < 0:
            errors.append(f"{engine} measurement {key} must be a non-negative number.")
    if not _is_number(row.get("metricCompleteness")) or not 0 <= row.get("metricCompleteness") <= 1:
        errors.append(f"{engine} measurement metricCompleteness must be between 0 and 1.")

    for key in ["operatingPoint", "latestCapturedAtUtc", "solRigor", "tags"]:
        if not isinstance(row.get(key), str) or not row.get(key):
            errors.append(f"{engine} measurement {key} is required.")
    tags = str(row.get("tags") or "")
    for fragment in ["serving-producer", engine, expected_framework, str(expected_model or "")]:
        if fragment and fragment not in tags:
            errors.append(f"{engine} measurement tags must include {fragment}.")


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
                    "avgTtftMs",
                    "avgTpotMs",
                    "avgTtfotMs",
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
            if not isinstance(samples, list):
                errors.append(f"{engine} summary artifact samples must be a list.")
            else:
                if isinstance(request_count, int) and len(samples) != request_count:
                    errors.append(f"{engine} sample count does not match requestCount.")
                if sum(1 for sample in samples if isinstance(sample, dict) and sample.get("ok")) != success_count:
                    errors.append(f"{engine} successful sample count does not match successCount.")
                sample_trace_ids = []
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
                    if token_row.get("tokenLogprob") is not None and not _is_number(token_row.get("tokenLogprob")):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenLogprob must be numeric when present.")
                    if token_row.get("tokenId") is not None and not isinstance(token_row.get("tokenId"), int):
                        errors.append(f"{engine} tokenTimeline[{token_index}].tokenId must be an integer when present.")
            hardware_telemetry = artifact.get("hardwareTelemetry")
            if not isinstance(hardware_telemetry, list) or len(hardware_telemetry) != request_count:
                errors.append(f"{engine} summary artifact hardwareTelemetry must match requestCount.")
            capture_policy = artifact.get("capturePolicy") if isinstance(artifact.get("capturePolicy"), dict) else {}
            if capture_policy.get("mode") != "operator-full":
                errors.append(f"{engine} summary artifact capturePolicy.mode must be operator-full.")
            raw_artifact_path = _resolve_proof_member_path(capture_policy.get("rawArtifactPath"), proof_dir)
            if not raw_artifact_path or not os.path.exists(raw_artifact_path):
                errors.append(f"{engine} operator-full raw artifact is required.")
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
                    if evidence:
                        evidence_measurement = evidence.get("measurement") if isinstance(evidence.get("measurement"), dict) else {}
                        for key in ["outputTpm", "totalTpm", "avgLatencyMs", "avgTtftMs", "avgTpotMs", "avgTtfotMs", "p95LatencyMs", "metricCompleteness"]:
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

    return {
        "schemaVersion": PROOF_VERIFICATION_SCHEMA_VERSION,
        "proofPath": proof_abs,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "engineCount": len(submissions_by_engine),
        "requiredEngines": expected_engines,
        "campaignIds": sorted(campaign_ids),
        "artifactHashes": artifact_hashes,
        "receiptLogPath": receipt_log_path or None,
        "receiptCounts": receipt_counts,
        "dashboard": proof.get("dashboard") if isinstance(proof.get("dashboard"), dict) else None,
    }


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
    for engine in engines:
        engine_id = engine["engine"]
        result = run_serving_producer(
            engine=engine,
            request={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "repetitions": repetitions,
                "maxTokens": max_tokens,
                **({"captureTokenDetails": True} if capture_token_details else {}),
                **({"topLogprobs": top_logprobs} if top_logprobs is not None else {}),
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
        "submissions": submissions,
    }


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
    parser.add_argument("--usd-per-gpu-hour", type=float)
    parser.add_argument("--gpu-count", type=float, default=1)
    parser.add_argument("--power-watts-per-gpu", type=float)
    parser.add_argument("--capture-token-details", action="store_true", default=_env("PIQ_SERVING_CAPTURE_TOKEN_DETAILS", "false").lower() == "true", help="Request response token logprobs/top-logprobs when the serving engine supports them.")
    parser.add_argument("--top-logprobs", type=int, default=int(_env("PIQ_SERVING_TOP_LOGPROBS", "0") or "0"), help="Number of top logprobs to request when token detail capture is enabled.")
    parser.add_argument("--collect-hardware-metrics", action="store_true", default=_env("PIQ_SERVING_COLLECT_HARDWARE_METRICS", "false").lower() == "true", help="Read DCGM metrics from PIQ_*_HARDWARE_METRICS_URL, or engine /metrics when no dedicated URL is set.")
    parser.add_argument("--require-hardware-telemetry", action="store_true", default=_env("PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY", "false").lower() == "true", help="Make proof completeness require DCGM hardware telemetry for configured engines.")
    parser.add_argument("--framework-version")
    parser.add_argument("--image-digest")
    parser.add_argument("--image-tag")
    parser.add_argument("--run-suffix", default=_env("PIQ_SERVING_RUN_SUFFIX"))
    parser.add_argument("--summary-out", default=_env("PIQ_SERVING_SUMMARY_OUT"), help="Write the overall smoke proof summary to this JSON path.")
    parser.add_argument("--receipt-log", default=_env("PIQ_SERVING_RECEIPT_LOG"), help="JSONL receipt log produced by the serving request recorder.")
    parser.add_argument("--record-receipts", action="store_true", help="Start in-process receipt proxies and route engine traffic through them.")
    parser.add_argument("--receipt-proxy-host", default=_env("PIQ_SERVING_RECEIPT_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--no-submit", action="store_true", help="Capture artifacts and manifests without submitting to Performance IQ.")
    parser.add_argument("--query-dashboard", action="store_true", help="Query fixed dashboard surfaces after submission.")
    parser.add_argument("--allow-missing-engines", action="store_true", help="Run configured engines only instead of requiring all three URLs.")
    parser.add_argument("--preflight-only", action="store_true", help="Report local runtime and endpoint readiness without sending completion requests.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Report read-only host, cache, port, and endpoint diagnostics for real engine setup.")
    parser.add_argument("--launch-plan-only", action="store_true", help="Print host-aware launch commands for the three serving engines.")
    parser.add_argument("--verify-proof", help="Verify a saved serving smoke proof summary JSON. Requires all three engines unless --allow-missing-engines is set.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the default /v1/models readiness check before sending completion requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.verify_proof:
        report = verify_proof_summary(args.verify_proof, require_all_engines=not args.allow_missing_engines)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    engines, missing = engine_configs_from_env(args)
    if args.launch_plan_only:
        print(json.dumps(runtime_launch_plan(args.model), indent=2))
        return 0
    if args.diagnostics_only:
        print(json.dumps(runtime_diagnostics(engines, missing, model=args.model), indent=2))
        return 0
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
        write_proof_summary(summary, args.artifact_dir, summary_out=args.summary_out)
        print(json.dumps(summary, indent=2))
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
