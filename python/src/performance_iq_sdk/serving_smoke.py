from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

from performance_iq_sdk.client import PerformanceIQ
from performance_iq_sdk.producers.serving import (
    HttpPostJson,
    ServingEngineId,
    laptop_smoke_model,
    run_serving_producer,
    serving_engine_label,
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
QUERY_NAMES = ("price_performance", "capacity_best", "campaign_provenance", "run_details")
CAMPAIGN_ID_QUERY_COLUMN = {
    "campaign_provenance": 0,
    "run_details": 0,
}
ENGINE_DEFAULT_PORT = {
    "vllm": 8000,
    "sglang": 30000,
    "tensorrt-llm": 8001,
}
LOW_FREE_SPACE_BYTES = 30 * 1024 * 1024 * 1024
SIZE_TIMEOUT_SECONDS = 15


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
        configs.append({
            "engine": engine,
            "baseUrl": url,
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
            timeout=10,
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
        "PIQ_SGLANG_URL",
        "PIQ_TENSORRT_LLM_URL",
        "PIQ_TENSORRT_LLM_IMAGE",
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
            "vllmCommand": command_probe("vllm", "--version"),
            "vllmModule": module_probe("vllm"),
            "sglangModule": module_probe("sglang"),
            "tensorrtLlmServeCommand": command_probe("trtllm-serve", "--help"),
            "nvidiaSmiCommand": command_probe("nvidia-smi"),
        },
        "storage": storage_probe(),
        "missingEngineUrls": missing_urls,
        "endpoints": endpoint_results,
        "launchPlan": runtime_launch_plan(model or laptop_smoke_model()),
        "ready": not missing_urls and all(item.get("ok") for item in endpoint_results),
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


def query_dashboard(base_url: str, token: str | None = None) -> dict[str, Any]:
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
        surface_campaign_ids = {
            name: sorted({
                row[index]
                for row in body.get(name, {}).get("rows", [])
                if len(row) > index and isinstance(row[index], str)
            })
            for name, index in CAMPAIGN_ID_QUERY_COLUMN.items()
        }
        return {
            "storeProvider": response.headers.get("x-piq-store-provider"),
            "rowCounts": {name: body[name]["rowCount"] for name in QUERY_NAMES if name in body},
            "campaignIds": surface_campaign_ids.get("campaign_provenance", []),
            "surfaceCampaignIds": surface_campaign_ids,
            "runtimeFrameworks": sorted({
                row[2]
                for row in body.get("price_performance", {}).get("rows", [])
                if len(row) > 2
            }),
        }


def default_proof_summary_path(artifact_dir: str, run_suffix: str) -> str:
    return os.path.join(artifact_dir, f"serving-smoke-proof-{_safe_slug(run_suffix)}.json")


def write_proof_summary(summary: dict[str, Any], artifact_dir: str, summary_out: str | None = None) -> str:
    proof_path = summary_out or default_proof_summary_path(artifact_dir, str(summary.get("runSuffix") or _utc_slug()))
    parent = os.path.dirname(proof_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    summary["proofSummaryPath"] = proof_path
    body = {
        **summary,
        "schemaVersion": "performance-iq.serving-smoke-proof.v1",
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
    submit: bool = True,
    http_post_json: HttpPostJson | None = None,
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
    parser.add_argument("--framework-version")
    parser.add_argument("--image-digest")
    parser.add_argument("--image-tag")
    parser.add_argument("--run-suffix", default=_env("PIQ_SERVING_RUN_SUFFIX"))
    parser.add_argument("--summary-out", default=_env("PIQ_SERVING_SUMMARY_OUT"), help="Write the overall smoke proof summary to this JSON path.")
    parser.add_argument("--no-submit", action="store_true", help="Capture artifacts and manifests without submitting to Performance IQ.")
    parser.add_argument("--query-dashboard", action="store_true", help="Query fixed dashboard surfaces after submission.")
    parser.add_argument("--allow-missing-engines", action="store_true", help="Run configured engines only instead of requiring all three URLs.")
    parser.add_argument("--preflight-only", action="store_true", help="Report local runtime and endpoint readiness without sending completion requests.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Report read-only host, cache, port, and endpoint diagnostics for real engine setup.")
    parser.add_argument("--launch-plan-only", action="store_true", help="Print host-aware launch commands for the three serving engines.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the default /v1/models readiness check before sending completion requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    engines, missing = engine_configs_from_env(args)
    if args.launch_plan_only:
        print(json.dumps(runtime_launch_plan(args.model), indent=2))
        return 0
    if args.diagnostics_only:
        print(json.dumps(runtime_diagnostics(engines, missing, model=args.model), indent=2))
        return 0
    if args.preflight_only:
        preflight = runtime_preflight(engines, missing, model=args.model)
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

    preflight = None
    if not args.skip_preflight:
        preflight = runtime_preflight(engines, [], model=args.model)
        if not preflight["ready"]:
            print(json.dumps(preflight, indent=2))
            print("Serving engine preflight failed; pass --skip-preflight only for targeted debugging.", file=sys.stderr)
            return 1

    client = None if args.no_submit else PerformanceIQ(args.piq_base_url, token=args.piq_token)
    pricing = {
        **({"usdPerGpuHour": args.usd_per_gpu_hour} if args.usd_per_gpu_hour is not None else {}),
        "gpuCount": args.gpu_count,
        **({"powerWattsPerGpu": args.power_watts_per_gpu} if args.power_watts_per_gpu is not None else {}),
    }
    try:
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
            run_suffix=args.run_suffix,
            submit=not args.no_submit,
        )
        if preflight is not None:
            summary["preflight"] = preflight
        if args.query_dashboard:
            if not args.piq_base_url:
                raise ValueError("dashboard query requires PIQ_BASE_URL or --piq-base-url")
            summary["dashboard"] = query_dashboard(args.piq_base_url, token=args.piq_token)
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


if __name__ == "__main__":
    raise SystemExit(main())
