import json
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from performance_iq_sdk import PerformanceIQ, build_manifest, laptop_smoke_model, run_serving_producer, validate_run
from performance_iq_sdk.serving_smoke import (
    engine_configs_from_env,
    endpoint_probe,
    external_python_module_probe,
    extract_proof_rows,
    build_serving_event_records,
    huggingface_model_cache_name,
    main as serving_smoke_main,
    parser as serving_smoke_parser,
    publish_serving_event_log,
    query_dashboard,
    real_runtime_proof_gate,
    run_serving_smoke,
    runtime_diagnostics,
    runtime_launch_plan,
    runtime_preflight,
    sha256_file,
    vllm_extension_probe,
    verify_proof_summary,
    write_serving_event_log,
    write_proof_summary,
    write_proof_rows,
)
from performance_iq_sdk.serving_receipts import (
    REQUEST_RECEIPT_SCHEMA_VERSION,
    load_receipts,
    recording_proxy_server,
)


class PerformanceIQSdkTest(unittest.TestCase):
    SERVING_ENV_NAMES = [
        "PIQ_VLLM_URL",
        "PIQ_SGLANG_URL",
        "PIQ_TENSORRT_LLM_URL",
        "PIQ_VLLM_API_KEY",
        "PIQ_SGLANG_API_KEY",
        "PIQ_TENSORRT_LLM_API_KEY",
        "PIQ_TOKEN",
        "PIQ_BASE_URL",
        "PIQ_SERVING_MODEL",
        "PIQ_SERVING_REPETITIONS",
        "PIQ_SERVING_MAX_TOKENS",
        "PIQ_SERVING_USD_PER_GPU_HOUR",
        "PIQ_SERVING_GPU_COUNT",
        "PIQ_SERVING_POWER_WATTS_PER_GPU",
        "PIQ_ARTIFACT_DIR",
        "PIQ_SERVING_SUMMARY_OUT",
        "PIQ_SERVING_EVENT_LOG",
        "PIQ_SERVING_PUBLISH_KAFKA",
        "PIQ_SERVING_KAFKA_BOOTSTRAP_SERVERS",
        "PIQ_SERVING_KAFKA_CLIENT_ID",
        "PIQ_SERVING_KAFKA_TOPIC",
        "PIQ_SERVING_CAPTURE_TOKEN_DETAILS",
        "PIQ_SERVING_TOP_LOGPROBS",
        "PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER",
        "PIQ_SERVING_TOKENIZER_MODEL",
        "PIQ_SERVING_COLLECT_HARDWARE_METRICS",
        "PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY",
        "PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY",
        "PIQ_SERVING_VERIFY_AFTER_CAPTURE",
        "PIQ_SERVING_REQUIRE_TELEMETRY_COVERAGE",
        "PIQ_VLLM_METRICS_URL",
        "PIQ_SGLANG_METRICS_URL",
        "PIQ_TENSORRT_LLM_METRICS_URL",
        "PIQ_VLLM_JSON_METRICS_URL",
        "PIQ_SGLANG_JSON_METRICS_URL",
        "PIQ_TENSORRT_LLM_JSON_METRICS_URL",
        "PIQ_VLLM_HARDWARE_METRICS_URL",
        "PIQ_SGLANG_HARDWARE_METRICS_URL",
        "PIQ_TENSORRT_LLM_HARDWARE_METRICS_URL",
        "PIQ_VLLM_FRAMEWORK_VERSION",
        "PIQ_SGLANG_FRAMEWORK_VERSION",
        "PIQ_TENSORRT_LLM_FRAMEWORK_VERSION",
        "PIQ_VLLM_MODEL_REVISION",
        "PIQ_SGLANG_MODEL_REVISION",
        "PIQ_TENSORRT_LLM_MODEL_REVISION",
        "PIQ_VLLM_IMAGE_TAG",
        "PIQ_SGLANG_IMAGE_TAG",
        "PIQ_TENSORRT_LLM_IMAGE_TAG",
        "PIQ_VLLM_IMAGE_DIGEST",
        "PIQ_SGLANG_IMAGE_DIGEST",
        "PIQ_TENSORRT_LLM_IMAGE_DIGEST",
        "PIQ_VLLM_SERVER_ARGS",
        "PIQ_SGLANG_SERVER_ARGS",
        "PIQ_TENSORRT_LLM_SERVER_ARGS",
        "PIQ_VLLM_TOKENIZER_MODEL",
        "PIQ_SGLANG_TOKENIZER_MODEL",
        "PIQ_TENSORRT_LLM_TOKENIZER_MODEL",
        "PIQ_VLLM_PROCESS_ID",
        "PIQ_SGLANG_PROCESS_ID",
        "PIQ_TENSORRT_LLM_PROCESS_ID",
        "PIQ_VLLM_CONTAINER_ID",
        "PIQ_SGLANG_CONTAINER_ID",
        "PIQ_TENSORRT_LLM_CONTAINER_ID",
        "PIQ_VLLM_POD_NAME",
        "PIQ_SGLANG_POD_NAME",
        "PIQ_TENSORRT_LLM_POD_NAME",
        "PIQ_VLLM_NODE_NAME",
        "PIQ_SGLANG_NODE_NAME",
        "PIQ_TENSORRT_LLM_NODE_NAME",
        "PIQ_VLLM_HOST_NAME",
        "PIQ_SGLANG_HOST_NAME",
        "PIQ_TENSORRT_LLM_HOST_NAME",
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
        "PIQ_SGLANG_BACKEND",
        "PIQ_SGLANG_ALLOW_UNSAFE_TOKEN_DETAILS",
        "PIQ_SERVING_ALLOW_PARTIAL",
        "PIQ_COMMAND_PROBE_TIMEOUT_SECONDS",
    ]

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="piq-python-test-")
        self.artifact_path = os.path.join(self.tmp_dir, "summary.json")
        with open(self.artifact_path, "w", encoding="utf-8") as handle:
            handle.write('{"ok":true}\n')
        self.old_serving_env = {name: os.environ.get(name) for name in self.SERVING_ENV_NAMES}
        for name in self.SERVING_ENV_NAMES:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self.old_serving_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self.tmp_dir)

    def input(self, **overrides):
        payload = {
            "sourceType": "fresh-run",
            "confidentiality": "operator-full",
            "producer": {
                "repo": "producer-runner",
                "tool": "runner",
                "commitSha": "1234567890abcdef",
            },
            "campaign": {
                "campaignId": "campaign-python-test",
                "runId": "run-python-test",
            },
            "workload": {
                "model": "llama-3.1-70b",
                "hardware": "B200 SXM",
                "operatingPoint": "peak",
            },
            "runtime": {
                "imageDigest": "sha256:" + "a" * 64,
            },
            "artifacts": [self.artifact_path],
            "measurements": [{"outputTpm": 1234}],
        }
        payload.update(overrides)
        return payload

    def write_full_serving_proof(self, *, repetitions=1):
        calls = []

        def http_stream_json(url, headers, payload):
            calls.append((url, headers, payload))
            request_index = len(calls)
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 5,
                        "receivedAtUtc": "2026-07-09T12:00:00.005Z",
                        "body": {
                            "id": f"chatcmpl-proof-{request_index}",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
                        },
                    },
                    {
                        "receivedMs": 8,
                        "receivedAtUtc": "2026-07-09T12:00:00.008Z",
                        "body": {
                            "id": f"chatcmpl-proof-{request_index}",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {"content": "o"}, "finish_reason": None}],
                        },
                    },
                    {
                        "receivedMs": 15,
                        "receivedAtUtc": "2026-07-09T12:00:00.015Z",
                        "body": {
                            "id": f"chatcmpl-proof-{request_index}",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {"content": "k"}, "finish_reason": "stop"}],
                        },
                    },
                    {
                        "receivedMs": 20,
                        "receivedAtUtc": "2026-07-09T12:00:00.020Z",
                        "body": {
                            "id": f"chatcmpl-proof-{request_index}",
                            "model": laptop_smoke_model(),
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 7,
                                "total_tokens": 17,
                            },
                        },
                    },
                ],
            }

        def transport(method, url, headers, body):
            payload = json.loads(body.decode("utf-8"))
            return {
                "id": payload["manifest"]["campaign"]["runId"],
                "status": "accepted",
                "liveProofReady": False,
            }

        endpoints = [
            {
                "engine": "vllm",
                "url": "http://127.0.0.1:8000/v1/models",
                "reachable": True,
                "status": 200,
                "ok": True,
                "modelChecked": True,
                "servedModels": [laptop_smoke_model()],
                "modelAvailable": True,
            },
            {
                "engine": "sglang",
                "url": "http://127.0.0.1:30000/v1/models",
                "reachable": True,
                "status": 200,
                "ok": True,
                "modelChecked": True,
                "servedModels": [laptop_smoke_model()],
                "modelAvailable": True,
            },
            {
                "engine": "tensorrt-llm",
                "url": "http://127.0.0.1:8001/v1/models",
                "reachable": True,
                "status": 200,
                "ok": True,
                "modelChecked": True,
                "servedModels": [laptop_smoke_model()],
                "modelAvailable": True,
            },
        ]
        engines = [
            {
                "engine": endpoint["engine"],
                "baseUrl": endpoint["url"].replace("/v1/models", ""),
                "endpointPreflight": endpoint,
            }
            for endpoint in endpoints
        ]
        client = PerformanceIQ("https://performance-iq.example", token="service-token", transport=transport)
        summary = run_serving_smoke(
            engines=engines,
            performance_iq=client,
            model=laptop_smoke_model(),
            prompt="Say ok.",
            artifact_dir=self.tmp_dir,
            repetitions=repetitions,
            max_tokens=16,
            hardware="local endpoints",
            operating_point="laptop-smoke",
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            run_suffix="full-proof",
            http_stream_json=http_stream_json,
        )
        campaign_ids = [item["campaignId"] for item in summary["submissions"]]
        summary["preflight"] = {
            "host": {"system": "unit", "machine": "unit", "platform": "unit", "python": sys.executable},
            "localRuntime": {},
            "storage": {"path": self.tmp_dir, "available": True, "freeGiB": 100, "lowFreeSpace": False},
            "missingEngineUrls": [],
            "endpoints": endpoints,
            "ready": True,
        }
        summary["dashboard"] = {
            "storeProvider": "sdk-ingestion",
            "rowCounts": {
                "price_performance": 3,
                "capacity_best": 3,
                "campaign_provenance": 3,
                "run_details": 3,
                "serving_request_samples": 3 * repetitions,
                "serving_token_timeline": 6 * repetitions,
                "serving_telemetry_coverage": 3,
            },
            "rows": {
                "price_performance": [
                    [laptop_smoke_model(), "local endpoints", "vLLM"],
                    [laptop_smoke_model(), "local endpoints", "SGLang"],
                    [laptop_smoke_model(), "local endpoints", "TensorRT-LLM"],
                ],
                "capacity_best": [
                    [laptop_smoke_model(), "vLLM"],
                    [laptop_smoke_model(), "SGLang"],
                    [laptop_smoke_model(), "TensorRT-LLM"],
                ],
                "campaign_provenance": [[campaign_id] for campaign_id in campaign_ids],
                "run_details": [[campaign_id] for campaign_id in campaign_ids],
                "serving_request_samples": [[campaign_id] for campaign_id in campaign_ids],
                "serving_token_timeline": [[campaign_id] for campaign_id in campaign_ids],
                "serving_telemetry_coverage": [[campaign_id] for campaign_id in campaign_ids],
            },
            "campaignIds": campaign_ids,
            "surfaceCampaignIds": {
                "campaign_provenance": campaign_ids,
                "run_details": campaign_ids,
                "serving_request_samples": campaign_ids,
                "serving_token_timeline": campaign_ids,
                "serving_telemetry_coverage": campaign_ids,
            },
            "submittedCampaignRows": {
                "campaign_provenance": [[campaign_id] for campaign_id in campaign_ids],
                "run_details": [[campaign_id] for campaign_id in campaign_ids],
                "serving_request_samples": [[campaign_id] for campaign_id in campaign_ids],
                "serving_token_timeline": [[campaign_id] for campaign_id in campaign_ids],
                "serving_telemetry_coverage": [[campaign_id] for campaign_id in campaign_ids],
            },
            "runtimeFrameworks": ["SGLang", "TensorRT-LLM", "vLLM"],
        }
        receipt_log_path = os.path.join(self.tmp_dir, "request-receipts.jsonl")
        with open(receipt_log_path, "w", encoding="utf-8") as handle:
            for submission in summary["submissions"]:
                with open(submission["artifactPath"], encoding="utf-8") as artifact_handle:
                    artifact = json.load(artifact_handle)
                for sample in artifact["samples"]:
                    handle.write(json.dumps({
                        "schemaVersion": REQUEST_RECEIPT_SCHEMA_VERSION,
                        "recordedAtUtc": sample["requestCompletedAtUtc"],
                        "engine": submission["engine"],
                        "requestId": sample["requestId"],
                        "campaignId": submission["campaignId"],
                        "runId": submission["runId"],
                        "method": "POST",
                        "path": "/v1/chat/completions",
                        "targetUrl": sample["endpoint"],
                        "status": sample["status"],
                        "latencyMs": sample["latencyMs"],
                        "requestBytes": 100,
                        "responseBytes": 100,
                        "requestHeaders": {
                            "x-performance-iq-engine": submission["engine"],
                            "x-performance-iq-request-id": sample["requestId"],
                        },
                    }) + "\n")
        summary["receiptLogPath"] = receipt_log_path
        proof_path = write_proof_summary(summary, self.tmp_dir)
        return proof_path, summary

    def rewrite_first_receipt(self, receipt_log_path, **overrides):
        receipts = load_receipts(receipt_log_path)
        receipts[0].update(overrides)
        with open(receipt_log_path, "w", encoding="utf-8") as handle:
            for receipt in receipts:
                handle.write(json.dumps(receipt) + "\n")

    def rewrite_first_measurement(self, summary, **overrides):
        artifact_path = summary["submissions"][0]["artifactPath"]
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact["measurements"][0].update(overrides)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

    def rewrite_engine_artifact(self, summary, engine, mutate):
        artifact_path = next(item["artifactPath"] for item in summary["submissions"] if item["engine"] == engine)
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        mutate(artifact)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

    def rewrite_proof_dashboard(self, proof_path, mutate):
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        mutate(proof["dashboard"])
        with open(proof_path, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")

    def refresh_proof_artifact_hashes(self, proof_path, engine):
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        submission = next(item for item in proof["submissions"] if item["engine"] == engine)
        artifact_path = submission["artifactPath"]
        manifest_path = submission["manifestPath"]
        artifact_sha = sha256_file(artifact_path)
        submission["artifactSha256"] = artifact_sha
        proof["evidenceIndex"]["engines"][engine]["artifact"]["sha256"] = artifact_sha
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for manifest_artifact in manifest["artifacts"]:
            if not isinstance(manifest_artifact, dict):
                continue
            path = manifest_artifact.get("path")
            if path == artifact_path:
                manifest_artifact["sha256"] = artifact_sha
            elif isinstance(path, str) and os.path.exists(path):
                manifest_artifact["sha256"] = sha256_file(path)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        with open(proof_path, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")

    def test_build_manifest_hashes_artifacts(self):
        manifest = build_manifest(self.input())

        self.assertEqual(manifest["schemaVersion"], "performance-iq.producer-manifest.v1")
        self.assertEqual(manifest["sourceType"], "fresh-run")
        self.assertEqual(manifest["artifacts"][0]["kind"], "normalized-summary")
        self.assertEqual(manifest["artifacts"][0]["sha256"], "e5f1eb4d806641698a35efe20e098efd20d7d57a9b90ee69079d5bb650920726")
        self.assertEqual(manifest["store"]["sourceTables"], [
            "platform_store.object_store.producer_runner_result_bundles",
            "platform_store.iceberg.intake_store.producer_runner_results",
        ])
        self.assertEqual(manifest["store"]["rowProof"][0]["campaignId"], "campaign-python-test")

    def test_validate_run_live_proof_classification(self):
        result = validate_run(self.input())

        self.assertTrue(result["ok"])
        self.assertTrue(result["liveProofReady"])
        self.assertTrue(result["freshRun"])
        self.assertTrue(result["producerBacked"])

    def test_rejects_non_producer_source_table(self):
        result = validate_run(self.input(store={
            "sourceTables": ["model_store.synthetic_fixture"],
            "modelTables": ["model_store.sdk_pending_ingest"],
            "rowProof": [{"table": "model_store.sdk_pending_ingest", "rowCount": 1}],
        }))

        self.assertFalse(result["ok"])
        self.assertIn("only use latest Producer Runner source tables", " ".join(result["errors"]))

    def test_customer_safe_fails_closed(self):
        result = validate_run(self.input(confidentiality="customer-safe"))

        self.assertFalse(result["ok"])
        self.assertIn("operator-full", " ".join(result["errors"]))

    def test_rejects_sql_keys(self):
        result = validate_run(self.input(measurements=[{"sql": "SELECT * FROM secret"}]))

        self.assertFalse(result["ok"])
        self.assertIn("sql", " ".join(result["errors"]))

    def test_client_submit_run_sends_auth_and_idempotency(self):
        calls = []

        def transport(method, url, headers, body):
            calls.append((method, url, headers, json.loads(body.decode("utf-8"))))
            return {"id": "run-python-test", "status": "accepted"}

        client = PerformanceIQ("https://performance-iq.example", token="service-token", transport=transport)
        result = client.submit_run(self.input(), idempotency_key="idem-python")

        self.assertEqual(result, {"id": "run-python-test", "status": "accepted"})
        method, url, headers, body = calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://performance-iq.example/api/v1/runs")
        self.assertEqual(headers["authorization"], "Bearer service-token")
        self.assertEqual(headers["idempotency-key"], "idem-python")
        self.assertEqual(body["schemaVersion"], "performance-iq.ingestion-request.v1")

    def test_serving_producer_captures_openai_compatible_usage(self):
        calls = []

        def http_post_json(url, headers, payload):
            calls.append((url, headers, payload))
            return {
                "status": 200,
                "body": {
                    "id": "chatcmpl-python-test",
                    "model": laptop_smoke_model(),
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 7,
                        "total_tokens": 17,
                    },
                },
            }

        result = run_serving_producer(
            engine={
                "engine": "sglang",
                "baseUrl": "http://127.0.0.1:30000",
                "endpointPreflight": {
                    "url": "http://127.0.0.1:30000/v1/models",
                    "ok": True,
                    "modelAvailable": True,
                },
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "repetitions": 2,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local mock engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_post_json=http_post_json,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][2]["model"], laptop_smoke_model())
        self.assertEqual(calls[0][1]["x-performance-iq-engine"], "sglang")
        self.assertTrue(calls[0][1]["x-performance-iq-request-id"].startswith("piq-sglang-"))
        self.assertEqual(result["manifest"]["producer"]["tool"], "sglang-serving-producer")
        self.assertEqual(result["manifest"]["runtime"]["framework"], "SGLang")
        self.assertEqual(result["manifest"]["sourceType"], "other-measured-producer")
        self.assertTrue(os.path.exists(result["artifactPath"]))
        self.assertTrue(os.path.exists(result["manifestPath"]))
        with open(result["artifactPath"], encoding="utf-8") as handle:
            artifact = json.load(handle)
        with open(result["manifestPath"], encoding="utf-8") as handle:
            manifest_artifact = json.load(handle)
        self.assertEqual(artifact["endpointPreflight"]["url"], "http://127.0.0.1:30000/v1/models")
        self.assertTrue(artifact["endpointPreflight"]["modelAvailable"])
        self.assertEqual(artifact["samples"][0]["requestId"], calls[0][1]["x-performance-iq-request-id"])
        self.assertEqual(artifact["samples"][0]["endpoint"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(artifact["requestTrace"][0]["requestId"], artifact["samples"][0]["requestId"])
        self.assertEqual(manifest_artifact["campaign"]["campaignId"], result["manifest"]["campaign"]["campaignId"])
        self.assertEqual(manifest_artifact["artifacts"][0]["path"], result["artifactPath"])
        self.assertEqual(manifest_artifact["artifacts"][0]["sha256"], result["manifest"]["artifacts"][0]["sha256"])
        self.assertEqual(manifest_artifact["platform"]["requestTraceIds"], [
            sample["requestId"] for sample in result["samples"]
        ])
        self.assertEqual(result["measurements"][0]["runtimeEngine"], "sglang")
        self.assertEqual(result["measurements"][0]["completionTokens"], 14)
        self.assertTrue(validate_run(result["runInput"])["ok"])

    def test_serving_producer_derives_streaming_ttft_tpot_and_ttfot(self):
        calls = []

        def http_stream_json(url, headers, payload):
            calls.append((url, headers, payload))
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 4,
                        "receivedAtUtc": "2026-07-09T12:00:00.004Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"role": "assistant"}}]},
                    },
                    {
                        "receivedMs": 9,
                        "receivedAtUtc": "2026-07-09T12:00:00.009Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "o"}}]},
                    },
                    {
                        "receivedMs": 19,
                        "receivedAtUtc": "2026-07-09T12:00:00.019Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "k"}, "finish_reason": "stop"}]},
                    },
                    {
                        "receivedMs": 22,
                        "receivedAtUtc": "2026-07-09T12:00:00.022Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "vllm",
                "baseUrl": "http://127.0.0.1:8000",
                "frameworkVersion": "unit-test-runtime",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
        )

        self.assertTrue(calls[0][2]["stream"])
        sample = result["samples"][0]
        self.assertTrue(sample["streaming"])
        self.assertEqual(sample["timeToFirstByteMs"], 4)
        self.assertEqual(sample["ttftMs"], 9)
        self.assertEqual(sample["ttfotMs"], 9)
        self.assertAlmostEqual(sample["tpotMs"], 2.0)
        self.assertEqual(sample["outputTokenCount"], 6)
        self.assertEqual(len(sample["tokenTimeline"]), 2)
        aggregate = result["measurements"][0]
        self.assertEqual(aggregate["avgTtftMs"], 9)
        self.assertEqual(aggregate["avgTtfotMs"], 9)
        self.assertAlmostEqual(aggregate["avgTpotMs"], 2.0)
        self.assertEqual(aggregate["metricCompleteness"], 1)
        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        token_rows = [row for row in result["measurements"] if row.get("surface") == "serving_token_timeline"]
        coverage_rows = [row for row in result["measurements"] if row.get("surface") == "serving_telemetry_coverage"]
        self.assertEqual(len(sample_rows), 1)
        self.assertEqual(len(token_rows), 2)
        self.assertEqual(len(coverage_rows), 8)
        self.assertIn("clientStreamTiming", {row["coverageCategory"] for row in coverage_rows})
        with open(result["artifactPath"], encoding="utf-8") as handle:
            artifact = json.load(handle)
        with open(result["manifestPath"], encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertNotIn("messages", artifact["request"])
        self.assertEqual(artifact["capturePolicy"]["mode"], "operator-full")
        raw_artifacts = [item for item in manifest["artifacts"] if item["kind"] == "operator-full-serving-raw"]
        self.assertEqual(len(raw_artifacts), 1)
        self.assertTrue(os.path.exists(raw_artifacts[0]["path"]))
        self.assertEqual(sample_rows[0]["rawArtifactPath"], raw_artifacts[0]["path"])
        self.assertEqual(
            next(row for row in coverage_rows if row["coverageCategory"] == "operatorFullArtifacts")["proofPath"],
            raw_artifacts[0]["path"],
        )

    def test_serving_producer_fails_closed_on_interrupted_stream(self):
        def http_stream_json(url, headers, payload):
            raise RuntimeError("stream interrupted after role chunk")

        result = run_serving_producer(
            engine={"engine": "vllm", "baseUrl": "http://127.0.0.1:8000"},
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
        )

        sample = result["samples"][0]
        self.assertFalse(sample["ok"])
        self.assertEqual(sample["status"], 0)
        self.assertEqual(sample["streaming"], True)
        self.assertIsInstance(sample["e2eLatencyMs"], float)
        self.assertIsNone(sample["timeToFirstByteMs"])
        self.assertIsNone(sample["ttftMs"])
        self.assertIsNone(sample["ttfotMs"])
        self.assertIsNone(sample["tpotMs"])
        self.assertEqual(sample["streamChunkCount"], 0)
        self.assertEqual(sample["outputTokenCount"], 0)
        self.assertIn("stream interrupted", sample["error"])

        aggregate = result["measurements"][0]
        self.assertEqual(aggregate["successCount"], 0)
        self.assertEqual(aggregate["errorCount"], 1)
        self.assertLess(aggregate["metricCompleteness"], 1)

        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        token_rows = [row for row in result["measurements"] if row.get("surface") == "serving_token_timeline"]
        coverage_rows = [row for row in result["measurements"] if row.get("surface") == "serving_telemetry_coverage"]
        self.assertEqual(sample_rows[0]["ok"], False)
        self.assertIsNone(sample_rows[0]["ttftMs"])
        self.assertTrue(os.path.exists(sample_rows[0]["rawArtifactPath"]))
        self.assertEqual(token_rows, [])
        self.assertEqual(
            next(row for row in coverage_rows if row["coverageCategory"] == "clientStreamTiming")["coverageStatus"],
            "missing",
        )

    def test_serving_producer_captures_tokenizer_prompt_token_rows(self):
        class FakePromptTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return [11, 22, 33]

            def convert_ids_to_tokens(self, token_id):
                return f"tok-{token_id}"

        tokenizer = FakePromptTokenizer()

        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 5,
                        "receivedAtUtc": "2026-07-09T12:00:00.005Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "ok"}}]},
                    },
                    {
                        "receivedMs": 15,
                        "receivedAtUtc": "2026-07-09T12:00:00.015Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "vllm",
                "baseUrl": "http://127.0.0.1:8000",
                "tokenizer": tokenizer,
                "tokenizerModel": "unit-tokenizer",
                "frameworkVersion": "unit-test-runtime",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local tokenizer engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
        )

        sample = result["samples"][0]
        self.assertTrue(sample["promptTokenIdsAvailable"])
        self.assertEqual(sample["promptTokenDetailCount"], 3)
        self.assertEqual(sample["promptTokenIdSource"], "configured-tokenizer")
        self.assertEqual(sample["promptTokenizationSource"], "configured-tokenizer-chat-template")
        self.assertEqual(sample["promptTokenizerModel"], "unit-tokenizer")
        self.assertEqual(len(sample["promptTokenIdsSha256"]), 64)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Say ok."}])
        prompt_rows = [row for row in sample["tokenTimeline"] if row.get("tokenPhase") == "prompt"]
        output_rows = [row for row in sample["tokenTimeline"] if row.get("tokenPhase") == "output"]
        self.assertEqual([row["tokenId"] for row in prompt_rows], [11, 22, 33])
        self.assertEqual([row["tokenIdSource"] for row in prompt_rows], ["configured-tokenizer"] * 3)
        self.assertEqual(len(output_rows), 1)
        aggregate = result["measurements"][0]
        self.assertTrue(aggregate["promptTokenDetailsRequired"])
        self.assertEqual(aggregate["promptTokenIdsAvailableCount"], 1)
        self.assertEqual(aggregate["metricCompleteness"], 1)
        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        token_rows = [row for row in result["measurements"] if row.get("surface") == "serving_token_timeline"]
        self.assertTrue(sample_rows[0]["promptTokenIdsAvailable"])
        self.assertEqual(sample_rows[0]["promptTokenIdSource"], "configured-tokenizer")
        self.assertEqual([row["tokenId"] for row in token_rows if row.get("tokenPhase") == "prompt"], [11, 22, 33])

    def test_serving_producer_derives_native_prometheus_deltas(self):
        metric_snapshots = [
            """
vllm:time_to_first_token_seconds_count{model_name="qwen"} 7
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 1.4
vllm:request_time_per_output_token_seconds_count{model_name="qwen"} 7
vllm:request_time_per_output_token_seconds_sum{model_name="qwen"} 0.21
vllm:e2e_request_latency_seconds_count{model_name="qwen"} 7
vllm:e2e_request_latency_seconds_sum{model_name="qwen"} 3.5
vllm:request_queue_time_seconds_count{model_name="qwen"} 7
vllm:request_queue_time_seconds_sum{model_name="qwen"} 0.07
vllm:request_prefill_time_seconds_count{model_name="qwen"} 7
vllm:request_prefill_time_seconds_sum{model_name="qwen"} 0.35
vllm:request_decode_time_seconds_count{model_name="qwen"} 7
vllm:request_decode_time_seconds_sum{model_name="qwen"} 0.7
vllm:num_requests_running{model_name="qwen"} 1
vllm:num_requests_waiting{model_name="qwen"} 0
vllm:kv_cache_usage_perc{model_name="qwen"} 0.125
vllm:prefix_cache_queries_total{model_name="qwen"} 20
vllm:prefix_cache_hits_total{model_name="qwen"} 5
vllm:prompt_tokens_cached_total{model_name="qwen"} 3
vllm:request_prefill_kv_computed_tokens_sum{model_name="qwen"} 8
""",
            """
vllm:time_to_first_token_seconds_count{model_name="qwen"} 8
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 1.65
vllm:request_time_per_output_token_seconds_count{model_name="qwen"} 8
vllm:request_time_per_output_token_seconds_sum{model_name="qwen"} 0.23
vllm:e2e_request_latency_seconds_count{model_name="qwen"} 8
vllm:e2e_request_latency_seconds_sum{model_name="qwen"} 4.1
vllm:request_queue_time_seconds_count{model_name="qwen"} 8
vllm:request_queue_time_seconds_sum{model_name="qwen"} 0.073
vllm:request_prefill_time_seconds_count{model_name="qwen"} 8
vllm:request_prefill_time_seconds_sum{model_name="qwen"} 0.41
vllm:request_decode_time_seconds_count{model_name="qwen"} 8
vllm:request_decode_time_seconds_sum{model_name="qwen"} 0.82
vllm:num_requests_running{model_name="qwen"} 1
vllm:num_requests_waiting{model_name="qwen"} 0
vllm:kv_cache_usage_perc{model_name="qwen"} 0.25
vllm:prefix_cache_queries_total{model_name="qwen"} 30
vllm:prefix_cache_hits_total{model_name="qwen"} 7
vllm:prompt_tokens_cached_total{model_name="qwen"} 6
vllm:request_prefill_kv_computed_tokens_sum{model_name="qwen"} 16
""",
        ]

        def http_get_text(url, headers):
            self.assertEqual(url, "http://127.0.0.1:8000/metrics")
            self.assertEqual(headers["accept"], "text/plain")
            return metric_snapshots.pop(0)

        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 5,
                        "receivedAtUtc": "2026-07-09T12:00:00.005Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "ok"}}]},
                    },
                    {
                        "receivedMs": 15,
                        "receivedAtUtc": "2026-07-09T12:00:00.015Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "vllm",
                "baseUrl": "http://127.0.0.1:8000",
                "metricsUrl": "http://127.0.0.1:8000/metrics",
                "frameworkVersion": "vllm-test",
                "modelRevision": "revision-a",
                "imageTag": "vllm:test",
                "imageDigest": "sha256:abc",
                "serverArgs": ["vllm", "serve", laptop_smoke_model()],
                "processId": "1234",
                "containerId": "container-a",
                "podName": "pod-a",
                "nodeName": "node-a",
                "hostName": "host-a",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
            http_get_text=http_get_text,
        )

        self.assertEqual(metric_snapshots, [])
        sample = result["samples"][0]
        self.assertTrue(sample["nativeTelemetryAvailable"])
        self.assertAlmostEqual(sample["nativeTelemetry"]["nativeTtftMs"], 250)
        self.assertAlmostEqual(sample["nativeTelemetry"]["nativeTpotMs"], 20)
        self.assertAlmostEqual(sample["nativeTelemetry"]["nativeE2eLatencyMs"], 600)
        self.assertAlmostEqual(sample["queueWaitMs"], 3)
        self.assertAlmostEqual(sample["prefillMs"], 60)
        self.assertAlmostEqual(sample["decodeMs"], 120)
        self.assertEqual(sample["nativeTelemetry"]["prefixCacheQueriesDelta"], 10)
        self.assertEqual(sample["nativeTelemetry"]["prefixCacheHitsDelta"], 2)
        self.assertAlmostEqual(sample["nativeTelemetry"]["cacheHitRate"], 0.2)
        self.assertEqual(sample["runningRequests"], 1)
        self.assertEqual(sample["waitingRequests"], 0)
        self.assertAlmostEqual(sample["kvCacheUsagePct"], 0.25)
        self.assertEqual(sample["promptTokensCachedDelta"], 3)
        self.assertEqual(sample["promptTokensComputedDelta"], 8)
        self.assertEqual(sample["engineVersion"], "vllm-test")
        self.assertEqual(sample["modelRevision"], "revision-a")
        self.assertEqual(sample["imageTag"], "vllm:test")
        self.assertEqual(sample["imageDigest"], "sha256:abc")
        self.assertIsInstance(sample["serverArgsSha256"], str)
        self.assertEqual(sample["processId"], "1234")
        self.assertEqual(sample["containerId"], "container-a")
        self.assertEqual(sample["podName"], "pod-a")
        self.assertEqual(sample["nodeName"], "node-a")
        self.assertEqual(sample["hostName"], "host-a")
        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        self.assertEqual(sample_rows[0]["nativeTtftMs"], 250)
        self.assertEqual(sample_rows[0]["runningRequests"], 1)
        self.assertEqual(sample_rows[0]["promptTokensCachedDelta"], 3)
        self.assertEqual(sample_rows[0]["engineVersion"], "vllm-test")
        self.assertEqual(sample_rows[0]["containerId"], "container-a")
        aggregate = result["measurements"][0]
        self.assertEqual(aggregate["nativeTelemetryAvailableCount"], 1)
        self.assertAlmostEqual(aggregate["avgQueueWaitMs"], 3)
        self.assertAlmostEqual(aggregate["avgPrefillMs"], 60)
        self.assertAlmostEqual(aggregate["avgDecodeMs"], 120)

    def test_serving_producer_derives_sglang_documented_native_metrics(self):
        metric_snapshots = [
            """
sglang:time_to_first_token_seconds_count{model_name="qwen"} 10
sglang:time_to_first_token_seconds_sum{model_name="qwen"} 1.0
sglang:inter_token_latency_seconds_count 10
sglang:inter_token_latency_seconds_sum 0.2
sglang:e2e_request_latency_seconds_count{model_name="qwen"} 10
sglang:e2e_request_latency_seconds_sum{model_name="qwen"} 4.0
sglang:request_queue_time_seconds_count{model_name="qwen"} 10
sglang:request_queue_time_seconds_sum{model_name="qwen"} 0.05
sglang:per_stage_req_latency_seconds_count{stage="prefill_forward"} 10
sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward"} 0.5
sglang:per_stage_req_latency_seconds_count{stage="decode_forward"} 10
sglang:per_stage_req_latency_seconds_sum{stage="decode_forward"} 1.0
sglang:num_running_reqs{model_name="qwen"} 0
sglang:num_queue_reqs{model_name="qwen"} 0
sglang:token_usage{model_name="qwen"} 0.4
sglang:cache_hit_rate{model_name="qwen"} 0.25
sglang:cached_tokens_total 0
sglang:uncached_prompt_tokens_histogram_sum 40
""",
            """
sglang:time_to_first_token_seconds_count{model_name="qwen"} 11
sglang:time_to_first_token_seconds_sum{model_name="qwen"} 1.15
sglang:inter_token_latency_seconds_count 11
sglang:inter_token_latency_seconds_sum 0.23
sglang:e2e_request_latency_seconds_count{model_name="qwen"} 11
sglang:e2e_request_latency_seconds_sum{model_name="qwen"} 4.5
sglang:request_queue_time_seconds_count{model_name="qwen"} 11
sglang:request_queue_time_seconds_sum{model_name="qwen"} 0.052
sglang:per_stage_req_latency_seconds_count{stage="prefill_forward"} 11
sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward"} 0.56
sglang:per_stage_req_latency_seconds_count{stage="decode_forward"} 11
sglang:per_stage_req_latency_seconds_sum{stage="decode_forward"} 1.12
sglang:num_running_reqs{model_name="qwen"} 0
sglang:num_queue_reqs{model_name="qwen"} 0
sglang:token_usage{model_name="qwen"} 0.5
sglang:cache_hit_rate{model_name="qwen"} 0.3
sglang:cached_tokens_total 33
sglang:uncached_prompt_tokens_histogram_sum 41
""",
        ]

        def http_get_text(url, headers):
            self.assertEqual(url, "http://127.0.0.1:30000/metrics")
            return metric_snapshots.pop(0)

        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 7,
                        "receivedAtUtc": "2026-07-09T12:00:00.007Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "ok"}}]},
                    },
                    {
                        "receivedMs": 17,
                        "receivedAtUtc": "2026-07-09T12:00:00.017Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "sglang",
                "baseUrl": "http://127.0.0.1:30000",
                "metricsUrl": "http://127.0.0.1:30000/metrics",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
            http_get_text=http_get_text,
        )

        self.assertEqual(metric_snapshots, [])
        sample = result["samples"][0]
        self.assertTrue(sample["nativeTelemetryAvailable"])
        self.assertAlmostEqual(sample["nativeTtftMs"], 150)
        self.assertAlmostEqual(sample["nativeTpotMs"], 30)
        self.assertAlmostEqual(sample["nativeE2eLatencyMs"], 500)
        self.assertAlmostEqual(sample["queueWaitMs"], 2)
        self.assertAlmostEqual(sample["prefillMs"], 60)
        self.assertAlmostEqual(sample["decodeMs"], 120)
        self.assertEqual(sample["runningRequests"], 0)
        self.assertEqual(sample["waitingRequests"], 0)
        self.assertAlmostEqual(sample["kvCacheUsagePct"], 0.5)
        self.assertAlmostEqual(sample["cacheHitRate"], 0.3)
        self.assertEqual(sample["promptTokensCachedDelta"], 33)
        self.assertEqual(sample["promptTokensComputedDelta"], 1)

    def test_serving_producer_derives_tensorrt_prometheus_and_json_metrics(self):
        prometheus_snapshots = [
            """
trtllm_time_to_first_token_seconds_count{model_name="qwen"} 3
trtllm_time_to_first_token_seconds_sum{model_name="qwen"} 0.3
trtllm_time_per_output_token_seconds_count{model_name="qwen"} 3
trtllm_time_per_output_token_seconds_sum{model_name="qwen"} 0.06
trtllm_e2e_request_latency_seconds_count{model_name="qwen"} 3
trtllm_e2e_request_latency_seconds_sum{model_name="qwen"} 1.2
trtllm_request_queue_time_seconds_count{model_name="qwen"} 3
trtllm_request_queue_time_seconds_sum{model_name="qwen"} 0.03
trtllm_kv_cache_hit_rate{model_name="qwen"} 0.1
trtllm_kv_cache_utilization{model_name="qwen"} 0.2
""",
            """
trtllm_time_to_first_token_seconds_count{model_name="qwen"} 4
trtllm_time_to_first_token_seconds_sum{model_name="qwen"} 0.5
trtllm_time_per_output_token_seconds_count{model_name="qwen"} 4
trtllm_time_per_output_token_seconds_sum{model_name="qwen"} 0.09
trtllm_e2e_request_latency_seconds_count{model_name="qwen"} 4
trtllm_e2e_request_latency_seconds_sum{model_name="qwen"} 1.9
trtllm_request_queue_time_seconds_count{model_name="qwen"} 4
trtllm_request_queue_time_seconds_sum{model_name="qwen"} 0.035
trtllm_kv_cache_hit_rate{model_name="qwen"} 0.2
trtllm_kv_cache_utilization{model_name="qwen"} 0.3
""",
        ]

        json_snapshots = [
            '[{"gpuMemUsage": 1000, "iterLatencyMS": 5, "kvCacheStats": {"usedNumBlocks": 2, "maxNumBlocks": 10, "cacheHitRate": 0.1}, "numActiveRequests": 0}]',
            '[{"gpuMemUsage": 2000, "iterLatencyMS": 7, "kvCacheStats": {"usedNumBlocks": 4, "maxNumBlocks": 10, "cacheHitRate": 0.4}, "numActiveRequests": 1}]',
        ]

        def http_get_combined(url, headers):
            if url == "http://127.0.0.1:8001/prometheus/metrics":
                return prometheus_snapshots.pop(0)
            self.assertEqual(url, "http://127.0.0.1:8001/metrics")
            return json_snapshots.pop(0)

        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 8,
                        "receivedAtUtc": "2026-07-09T12:00:00.008Z",
                        "body": {"id": "chatcmpl-stream", "model": laptop_smoke_model(), "choices": [{"delta": {"content": "ok"}}]},
                    },
                    {
                        "receivedMs": 18,
                        "receivedAtUtc": "2026-07-09T12:00:00.018Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "tensorrt-llm",
                "baseUrl": "http://127.0.0.1:8001",
                "collectNativeMetrics": True,
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
            http_get_text=http_get_combined,
        )

        self.assertEqual(prometheus_snapshots, [])
        self.assertEqual(json_snapshots, [])
        sample = result["samples"][0]
        self.assertTrue(sample["nativeTelemetryAvailable"])
        self.assertAlmostEqual(sample["nativeTtftMs"], 200)
        self.assertAlmostEqual(sample["nativeTpotMs"], 30)
        self.assertAlmostEqual(sample["nativeE2eLatencyMs"], 700)
        self.assertAlmostEqual(sample["queueWaitMs"], 5)
        self.assertAlmostEqual(sample["kvCacheUsagePct"], 0.3)
        self.assertAlmostEqual(sample["cacheHitRate"], 0.2)
        self.assertAlmostEqual(sample["nativeIterationLatencyMs"], 7)
        self.assertAlmostEqual(sample["nativeGpuMemoryBytes"], 2000)
        self.assertAlmostEqual(sample["nativeKvCacheUsedBlocks"], 4)
        self.assertAlmostEqual(sample["nativeKvCacheMaxBlocks"], 10)
        self.assertAlmostEqual(sample["nativeTelemetry"]["trtllmIterationLatencyMs"], 7)
        self.assertAlmostEqual(sample["nativeTelemetry"]["trtllmGpuMemoryBytes"], 2000)
        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        self.assertAlmostEqual(sample_rows[0]["nativeIterationLatencyMs"], 7)
        self.assertAlmostEqual(sample_rows[0]["nativeGpuMemoryBytes"], 2000)
        self.assertAlmostEqual(result["measurements"][0]["avgNativeIterationLatencyMs"], 7)

        json_snapshots = [
            '[{"gpuMemUsage": 1000, "iterLatencyMS": 5, "kvCacheStats": {"usedNumBlocks": 2, "maxNumBlocks": 10, "cacheHitRate": 0.1}, "numActiveRequests": 0}]',
            '[{"gpuMemUsage": 2000, "iterLatencyMS": 7, "kvCacheStats": {"usedNumBlocks": 4, "maxNumBlocks": 10, "cacheHitRate": 0.4}, "numActiveRequests": 1}]',
        ]

        def http_get_json(url, headers):
            self.assertEqual(url, "http://127.0.0.1:8001/metrics")
            return json_snapshots.pop(0)

        json_result = run_serving_producer(
            engine={
                "engine": "tensorrt-llm",
                "baseUrl": "http://127.0.0.1:8001",
                "metricsUrl": "http://127.0.0.1:8001/metrics",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local stream engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
            http_get_text=http_get_json,
        )

        self.assertEqual(json_snapshots, [])
        json_sample = json_result["samples"][0]
        self.assertTrue(json_sample["nativeTelemetryAvailable"])
        self.assertEqual(json_sample["runningRequests"], 1)
        self.assertAlmostEqual(json_sample["kvCacheUsagePct"], 0.4)
        self.assertAlmostEqual(json_sample["cacheHitRate"], 0.4)
        self.assertAlmostEqual(json_sample["nativeIterationLatencyMs"], 7)
        self.assertAlmostEqual(json_sample["nativeGpuMemoryBytes"], 2000)
        self.assertAlmostEqual(json_sample["nativeKvCacheUsedBlocks"], 4)
        self.assertAlmostEqual(json_sample["nativeKvCacheMaxBlocks"], 10)
        self.assertAlmostEqual(json_sample["nativeTelemetry"]["trtllmIterationLatencyMs"], 7)
        self.assertAlmostEqual(json_sample["nativeTelemetry"]["trtllmGpuMemoryBytes"], 2000)
        json_sample_rows = [row for row in json_result["measurements"] if row.get("surface") == "serving_request_sample"]
        self.assertAlmostEqual(json_sample_rows[0]["nativeIterationLatencyMs"], 7)
        self.assertAlmostEqual(json_sample_rows[0]["nativeGpuMemoryBytes"], 2000)
        self.assertAlmostEqual(json_result["measurements"][0]["avgNativeIterationLatencyMs"], 7)

    def test_serving_producer_captures_token_logprobs_and_dcgm_metrics(self):
        metric_snapshots = [
            """
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 100
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 40
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 12
DCGM_FI_DEV_GPU_TEMP{gpu="0"} 60
DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1800
DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5000
DCGM_FI_DEV_FB_USED{gpu="0"} 4096
DCGM_FI_DEV_FB_FREE{gpu="0"} 8192
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 1000
""",
            """
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 120
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 50
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 20
DCGM_FI_DEV_GPU_TEMP{gpu="0"} 61
DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1801
DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5001
DCGM_FI_DEV_FB_USED{gpu="0"} 4097
DCGM_FI_DEV_FB_FREE{gpu="0"} 8191
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 2500
""",
        ]

        def http_get_text(url, headers):
            self.assertEqual(url, "http://127.0.0.1:9400/metrics")
            self.assertEqual(headers["accept"], "text/plain")
            return metric_snapshots.pop(0)

        def http_stream_json(url, headers, payload):
            self.assertTrue(payload["stream"])
            self.assertTrue(payload["logprobs"])
            self.assertEqual(payload["top_logprobs"], 2)
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 10,
                        "receivedAtUtc": "2026-07-09T12:00:00.010Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{
                                "delta": {"content": "o"},
                                "logprobs": {
                                    "content": [{
                                        "token": "o",
                                        "token_id": 101,
                                        "logprob": -0.1,
                                        "top_logprobs": [
                                            {"token": "o", "token_id": 101, "logprob": -0.1},
                                            {"token": "O", "token_id": 102, "logprob": -2.0},
                                        ],
                                    }],
                                },
                            }],
                        },
                    },
                    {
                        "receivedMs": 30,
                        "receivedAtUtc": "2026-07-09T12:00:00.030Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{
                                "delta": {"content": "k"},
                                "finish_reason": "stop",
                                "logprobs": {"content": [{"token": "k", "token_id": 202, "logprob": -0.2}]},
                            }],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "vllm",
                "baseUrl": "http://127.0.0.1:8000",
                "hardwareMetricsUrl": "http://127.0.0.1:9400/metrics",
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "maxTokens": 16,
                "captureTokenDetails": True,
                "topLogprobs": 2,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local dcgm engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1},
            http_stream_json=http_stream_json,
            http_get_text=http_get_text,
        )

        self.assertEqual(metric_snapshots, [])
        sample = result["samples"][0]
        self.assertTrue(sample["hardwareTelemetryAvailable"])
        self.assertTrue(sample["tokenDetailsAvailable"])
        self.assertTrue(sample["tokenIdsAvailable"])
        self.assertTrue(sample["logprobsAvailable"])
        self.assertEqual(sample["tokenDetailCount"], 2)
        self.assertEqual(sample["tokenIdSource"], "response-logprobs")
        self.assertEqual(sample["tokenTimeline"][0]["tokenId"], 101)
        self.assertEqual(sample["tokenTimeline"][0]["tokenIdSource"], "response-logprobs")
        self.assertAlmostEqual(sample["tokenTimeline"][0]["tokenLogprob"], -0.1)
        self.assertIn("topLogprobsJson", sample["tokenTimeline"][0])
        self.assertAlmostEqual(sample["avgPowerWatts"], 120)
        self.assertAlmostEqual(sample["avgPowerWattsPerGpu"], 120)
        self.assertAlmostEqual(sample["gpuUtilizationPct"], 50)
        self.assertAlmostEqual(sample["memoryCopyUtilizationPct"], 20)
        self.assertAlmostEqual(sample["gpuTemperatureC"], 61)
        self.assertAlmostEqual(sample["smClockMHz"], 1801)
        self.assertAlmostEqual(sample["memoryClockMHz"], 5001)
        self.assertAlmostEqual(sample["fbUsedMiB"], 4097)
        self.assertAlmostEqual(sample["fbFreeMiB"], 8191)
        self.assertAlmostEqual(sample["energyJoules"], 1.5)
        aggregate = result["measurements"][0]
        self.assertTrue(aggregate["dcgmGrounded"])
        self.assertEqual(aggregate["hardwareTelemetryAvailableCount"], 1)
        self.assertEqual(aggregate["tokenDetailsAvailableCount"], 1)
        self.assertEqual(aggregate["tokenIdsAvailableCount"], 1)
        self.assertEqual(aggregate["logprobsAvailableCount"], 1)
        self.assertEqual(aggregate["powerSource"], "dcgm")
        token_rows = [row for row in result["measurements"] if row.get("surface") == "serving_token_timeline"]
        self.assertEqual(token_rows[0]["tokenId"], 101)
        self.assertEqual(token_rows[0]["tokenIdSource"], "response-logprobs")
        self.assertAlmostEqual(token_rows[1]["tokenLogprob"], -0.2)
        sample_rows = [row for row in result["measurements"] if row.get("surface") == "serving_request_sample"]
        self.assertEqual(sample_rows[0]["tokenIdSource"], "response-logprobs")
        self.assertAlmostEqual(sample_rows[0]["gpuTemperatureC"], 61)
        self.assertAlmostEqual(sample_rows[0]["smClockMHz"], 1801)
        self.assertAlmostEqual(sample_rows[0]["fbUsedMiB"], 4097)

    def test_serving_producer_resolves_missing_token_ids_with_configured_token_map(self):
        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 10,
                        "receivedAtUtc": "2026-07-09T12:00:00.010Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{
                                "delta": {"content": "o"},
                                "logprobs": {
                                    "content": [{
                                        "token": "o",
                                        "logprob": -0.1,
                                        "top_logprobs": [
                                            {"token": "o", "logprob": -0.1},
                                            {"token": "O", "logprob": -2.0},
                                        ],
                                    }],
                                },
                            }],
                        },
                    },
                    {
                        "receivedMs": 20,
                        "receivedAtUtc": "2026-07-09T12:00:00.020Z",
                        "body": {
                            "id": "chatcmpl-stream",
                            "model": laptop_smoke_model(),
                            "choices": [{
                                "delta": {"content": "k"},
                                "finish_reason": "stop",
                                "logprobs": {"content": [{"token": "k", "logprob": -0.2}]},
                            }],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                        },
                    },
                ],
            }

        result = run_serving_producer(
            engine={
                "engine": "vllm",
                "baseUrl": "http://127.0.0.1:8000",
                "tokenIdMap": {"o": 101, "O": 102, "k": 202},
            },
            request={
                "model": laptop_smoke_model(),
                "messages": [{"role": "user", "content": "Say ok."}],
                "captureTokenDetails": True,
                "topLogprobs": 2,
            },
            artifact_dir=self.tmp_dir,
            workload={"hardware": "local token map engine", "operatingPoint": "laptop-smoke"},
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            http_stream_json=http_stream_json,
        )

        sample = result["samples"][0]
        self.assertTrue(sample["tokenIdsAvailable"])
        self.assertEqual(sample["tokenIdSource"], "configured-token-id-map")
        self.assertEqual(sample["tokenTimeline"][0]["tokenId"], 101)
        self.assertEqual(sample["tokenTimeline"][0]["tokenIdSource"], "configured-token-id-map")
        top_logprobs = json.loads(sample["tokenTimeline"][0]["topLogprobsJson"])
        self.assertEqual(top_logprobs[1]["tokenId"], 102)
        self.assertEqual(top_logprobs[1]["tokenIdSource"], "configured-token-id-map")
        token_rows = [row for row in result["measurements"] if row.get("surface") == "serving_token_timeline"]
        self.assertEqual(token_rows[1]["tokenId"], 202)
        self.assertEqual(token_rows[1]["tokenIdSource"], "configured-token-id-map")

    def test_serving_producer_resolves_token_ids_with_external_tokenizer_python(self):
        tokenizer_python = os.path.join(self.tmp_dir, "fake-tokenizer-python")
        with open(tokenizer_python, "w", encoding="utf-8") as handle:
            handle.write("""#!/usr/bin/env python3
import json
import sys

mode = sys.argv[3]
payload = json.loads(sys.argv[5])
if mode == "token":
    print(json.dumps({"ok": True, "tokenId": {"o": 101, "k": 202, "ok": 303}.get(payload)}))
elif mode == "prompt":
    print(json.dumps({"ok": True, "tokenIds": [501, 502], "tokenTexts": ["Return", " ok"], "mode": "chat-template"}))
else:
    print(json.dumps({"ok": False}))
""")
        os.chmod(tokenizer_python, 0o755)

        def http_stream_json(url, headers, payload):
            return {
                "status": 200,
                "events": [
                    {
                        "receivedMs": 5,
                        "receivedAtUtc": "2026-07-09T12:00:00.005Z",
                        "body": {
                            "id": "chatcmpl-external-tokenizer",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {"content": "o"}, "finish_reason": None, "logprobs": {"content": [{
                                "token": "o",
                                "logprob": -0.1,
                                "top_logprobs": [{"token": "o", "logprob": -0.1}],
                            }]}}],
                        },
                    },
                    {
                        "receivedMs": 8,
                        "receivedAtUtc": "2026-07-09T12:00:00.008Z",
                        "body": {
                            "id": "chatcmpl-external-tokenizer",
                            "model": laptop_smoke_model(),
                            "choices": [{"delta": {"content": "k"}, "finish_reason": "stop", "logprobs": {"content": [{
                                "token": "k",
                                "logprob": -0.2,
                            }]}}],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                        },
                    },
                ],
            }

        with patch("performance_iq_sdk.producers.serving._load_hf_tokenizer", return_value=None):
            result = run_serving_producer(
                engine={
                    "engine": "vllm",
                    "baseUrl": "http://127.0.0.1:8000",
                    "tokenizerModel": laptop_smoke_model(),
                    "tokenizerPythonBin": tokenizer_python,
                    "resolveTokenIdsWithTokenizer": True,
                },
                request={
                    "model": laptop_smoke_model(),
                    "messages": [{"role": "user", "content": "Return ok."}],
                    "captureTokenDetails": True,
                    "topLogprobs": 1,
                    "resolveTokenIdsWithTokenizer": True,
                },
                artifact_dir=self.tmp_dir,
                workload={"hardware": "local tokenizer runtime", "operatingPoint": "laptop-smoke"},
                pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
                http_stream_json=http_stream_json,
            )

        sample = result["samples"][0]
        tokenizer_python_sha256 = hashlib.sha256(tokenizer_python.encode("utf-8")).hexdigest()
        self.assertTrue(sample["promptTokenIdsAvailable"])
        self.assertEqual(sample["promptTokenIdSource"], "external-hf-tokenizer")
        self.assertTrue(sample["tokenIdsAvailable"])
        self.assertEqual(sample["tokenIdSource"], "external-hf-tokenizer")
        self.assertEqual(sample["tokenizerModel"], laptop_smoke_model())
        self.assertEqual(sample["tokenizerPythonBinSha256"], tokenizer_python_sha256)
        output_rows = [
            row for row in sample["tokenTimeline"]
            if row.get("tokenPhase", "output") == "output"
        ]
        self.assertEqual(output_rows[0]["tokenId"], 101)
        self.assertEqual(output_rows[0]["tokenIdSource"], "external-hf-tokenizer")
        self.assertEqual(output_rows[0]["tokenizerModel"], laptop_smoke_model())
        self.assertEqual(output_rows[0]["tokenizerPythonBinSha256"], tokenizer_python_sha256)
        top_logprobs = json.loads(output_rows[0]["topLogprobsJson"])
        self.assertEqual(top_logprobs[0]["tokenId"], 101)
        sample_rows = [
            row for row in result["measurements"]
            if row.get("surface") == "serving_request_sample"
        ]
        self.assertEqual(sample_rows[0]["tokenizerModel"], laptop_smoke_model())
        self.assertEqual(sample_rows[0]["tokenizerPythonBinSha256"], tokenizer_python_sha256)
        prompt_rows = [
            row for row in result["measurements"]
            if row.get("surface") == "serving_token_timeline" and row.get("tokenPhase") == "prompt"
        ]
        self.assertEqual([row["tokenId"] for row in prompt_rows], [501, 502])
        self.assertEqual([row["tokenizerModel"] for row in prompt_rows], [laptop_smoke_model(), laptop_smoke_model()])
        self.assertEqual([row["tokenizerPythonBinSha256"] for row in prompt_rows], [tokenizer_python_sha256, tokenizer_python_sha256])
        output_measurement_rows = [
            row for row in result["measurements"]
            if row.get("surface") == "serving_token_timeline" and row.get("tokenPhase") == "output"
        ]
        self.assertEqual(output_measurement_rows[0]["tokenizerModel"], laptop_smoke_model())
        self.assertEqual(output_measurement_rows[0]["tokenizerPythonBinSha256"], tokenizer_python_sha256)

    def test_serving_smoke_runs_all_configured_engines(self):
        calls = []
        submissions = []

        def http_post_json(url, headers, payload):
            calls.append((url, headers, payload))
            return {
                "status": 200,
                "body": {
                    "id": f"chatcmpl-{len(calls)}",
                    "model": laptop_smoke_model(),
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 7,
                        "total_tokens": 17,
                    },
                },
            }

        def transport(method, url, headers, body):
            payload = json.loads(body.decode("utf-8"))
            submissions.append((method, url, headers, payload))
            return {
                "id": payload["manifest"]["campaign"]["runId"],
                "status": "accepted",
                "liveProofReady": False,
            }

        client = PerformanceIQ("https://performance-iq.example", token="service-token", transport=transport)
        summary = run_serving_smoke(
            engines=[
                {"engine": "vllm", "baseUrl": "http://127.0.0.1:8000"},
                {"engine": "sglang", "baseUrl": "http://127.0.0.1:30000"},
                {"engine": "tensorrt-llm", "baseUrl": "http://127.0.0.1:8001"},
            ],
            performance_iq=client,
            model=laptop_smoke_model(),
            prompt="Say ok.",
            artifact_dir=self.tmp_dir,
            repetitions=2,
            max_tokens=16,
            hardware="local endpoints",
            operating_point="laptop-smoke",
            pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
            run_suffix="unit",
            http_post_json=http_post_json,
        )

        self.assertEqual(len(calls), 6)
        self.assertEqual(len(submissions), 3)
        self.assertEqual({call[2]["model"] for call in calls}, {laptop_smoke_model()})
        self.assertEqual(
            [item["runtimeFramework"] for item in summary["submissions"]],
            ["vLLM", "SGLang", "TensorRT-LLM"],
        )
        self.assertTrue(all(item["status"] == "accepted" for item in summary["submissions"]))
        self.assertTrue(all(item["successCount"] == 2 for item in summary["submissions"]))
        self.assertTrue(all(os.path.exists(item["manifestPath"]) for item in summary["submissions"]))
        self.assertTrue(all(item["artifactSha256"] for item in summary["submissions"]))

    def test_serving_smoke_disables_sglang_mps_token_details_before_request(self):
        calls = []

        def fake_run_serving_producer(**kwargs):
            calls.append(kwargs)
            return {
                "measurements": [{"runtimeFramework": "SGLang"}],
                "samples": [{"ok": True}],
                "manifest": {
                    "campaign": kwargs["campaign"],
                    "artifacts": [{"sha256": "a" * 64}],
                },
                "artifactPath": os.path.join(self.tmp_dir, "sglang-summary.json"),
                "manifestPath": os.path.join(self.tmp_dir, "sglang-manifest.json"),
                "submission": None,
            }

        with patch("performance_iq_sdk.serving_smoke.run_serving_producer", side_effect=fake_run_serving_producer):
            summary = run_serving_smoke(
                engines=[{
                    "engine": "sglang",
                    "baseUrl": "http://127.0.0.1:30000",
                    "endpointPreflight": {"serverInfo": {"device": "mps"}},
                }],
                performance_iq=None,
                model=laptop_smoke_model(),
                prompt="Say ok.",
                artifact_dir=self.tmp_dir,
                repetitions=1,
                max_tokens=16,
                hardware="local endpoints",
                operating_point="laptop-smoke",
                pricing={"usdPerGpuHour": 1, "gpuCount": 1, "powerWattsPerGpu": 100},
                run_suffix="sglang-mps",
                capture_token_details=True,
                top_logprobs=3,
                submit=False,
            )

        self.assertEqual(len(calls), 1)
        self.assertNotIn("captureTokenDetails", calls[0]["request"])
        self.assertNotIn("topLogprobs", calls[0]["request"])
        self.assertFalse(calls[0]["engine"]["captureTokenDetails"])
        capability = calls[0]["engine"]["tokenDetailsCapability"]
        self.assertFalse(capability["supported"])
        self.assertFalse(capability["safeToRequest"])
        self.assertEqual(capability["reason"], "sglang-mps-mlx-logprobs-crash")
        self.assertEqual(summary["telemetryCapabilityGaps"][0]["category"], "outputTokenIdsLogprobs")

    def test_serving_smoke_verifies_full_proof_bundle(self):
        proof_path, summary = self.write_full_serving_proof()

        verification = verify_proof_summary(proof_path)

        self.assertTrue(verification["ok"], json.dumps(verification, indent=2))
        self.assertEqual(verification["engineCount"], 3)
        self.assertEqual(verification["campaignIds"], sorted(item["campaignId"] for item in summary["submissions"]))
        self.assertEqual(set(verification["artifactHashes"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertEqual(verification["receiptCounts"], {"vllm": 1, "sglang": 1, "tensorrt-llm": 1})
        coverage = verification["telemetryCoverage"]
        self.assertFalse(coverage["allProven"])
        self.assertEqual(coverage["categorySummary"]["clientStreamTiming"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["requestReceipts"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["dashboardFineGrainRows"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["operatorFullArtifacts"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["rawMetricSnapshots"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["rawMetricSnapshots"]["expectedEngines"], 0)
        self.assertEqual(coverage["categorySummary"]["nativeRuntimeTelemetry"]["status"], "missing")
        self.assertEqual(coverage["categorySummary"]["dcgmHardwareTelemetry"]["status"], "missing")
        self.assertEqual(coverage["categorySummary"]["promptTokenIds"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["promptTokenIds"]["expectedEngines"], 0)
        self.assertEqual(coverage["categorySummary"]["outputTokenIdsLogprobs"]["status"], "proven")
        self.assertEqual(coverage["categorySummary"]["outputTokenIdsLogprobs"]["expectedEngines"], 0)
        self.assertEqual(coverage["engines"]["vllm"]["clientStreamTiming"]["status"], "proven")
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        self.assertEqual(set(proof["evidenceIndex"]["engines"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertEqual(
            proof["evidenceIndex"]["engines"]["vllm"]["artifact"]["sha256"],
            summary["submissions"][0]["artifactSha256"],
        )
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(serving_smoke_main(["--verify-proof", proof_path]), 0)

        rows = extract_proof_rows(proof_path, verification=verification)
        self.assertEqual(rows["rowCounts"]["submissions"], 3)
        self.assertEqual(rows["rowCounts"]["servingRequestSamples"], 3)
        self.assertEqual(rows["rowCounts"]["servingTokenTimeline"], 6)
        self.assertEqual(rows["rowCounts"]["requestReceipts"], 3)
        self.assertEqual(rows["rowCounts"]["telemetryCoverageRows"], 33)
        self.assertEqual({row["engine"] for row in rows["servingRequestSamples"]}, {"vllm", "sglang", "tensorrt-llm"})
        self.assertEqual({row["coverageSource"] for row in rows["telemetryCoverageRows"]}, {"proof-verifier"})
        self.assertTrue(all(row.get("campaignId") for row in rows["servingTokenTimeline"]))
        rows_path = os.path.join(self.tmp_dir, "proof-rows.json")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(serving_smoke_main(["--verify-proof", proof_path, "--dump-proof-rows", rows_path]), 0)
        with open(rows_path, encoding="utf-8") as handle:
            persisted_rows = json.load(handle)
        self.assertEqual(persisted_rows["schemaVersion"], "performance-iq.serving-proof-rows.v1")
        self.assertEqual(persisted_rows["telemetryCoverage"]["schemaVersion"], "performance-iq.serving-telemetry-coverage.v1")
        self.assertEqual(persisted_rows["rowCounts"]["telemetryCoverageRows"], 33)
        self.assertEqual(persisted_rows["rowCounts"]["servingTokenTimeline"], 6)
        self.assertEqual(write_proof_rows(proof_path, rows_path)["rowCounts"]["servingRequestSamples"], 3)

    def test_serving_smoke_writes_kafka_ready_event_log(self):
        proof_path, summary = self.write_full_serving_proof()
        event_log_path = os.path.join(self.tmp_dir, "serving-events.jsonl")
        summary["eventLogPath"] = event_log_path
        write_proof_summary(summary, self.tmp_dir, summary_out=proof_path)

        events = build_serving_event_records(summary)
        write_serving_event_log(summary, event_log_path)

        event_types = [event["eventType"] for event in events]
        self.assertIn("serving.submission", event_types)
        self.assertIn("serving.measurement.result", event_types)
        self.assertIn("serving.measurement.serving_request_sample", event_types)
        self.assertIn("serving.measurement.serving_token_timeline", event_types)
        self.assertIn("serving.measurement.serving_telemetry_coverage", event_types)
        self.assertIn("serving.native_telemetry", event_types)
        self.assertIn("serving.hardware_telemetry", event_types)
        self.assertIn("serving.request_receipt", event_types)
        self.assertTrue(all(event["topic"] == "performance-iq.serving.telemetry.v1" for event in events))
        self.assertTrue(all(len(event["eventId"]) == 64 for event in events))
        with open(event_log_path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), len(events))
        published: list[dict[str, object]] = []

        class FakeFuture:
            def get(self, timeout=None):
                return {"timeout": timeout}

        class FakeKafkaProducer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def send(self, topic, key=None, value=None):
                published.append({
                    "topic": topic,
                    "key": key.decode("utf-8"),
                    "value": json.loads(value.decode("utf-8")),
                })
                return FakeFuture()

            def flush(self, timeout=None):
                published.append({"flush": timeout})

            def close(self, timeout=None):
                published.append({"close": timeout})

        publication = publish_serving_event_log(
            event_log_path,
            bootstrap_servers="kafka:9092",
            producer_factory=FakeKafkaProducer,
        )
        self.assertEqual(publication["schemaVersion"], "performance-iq.serving-kafka-publication.v1")
        self.assertEqual(publication["publishedCount"], len(events))
        self.assertEqual(
            publication["eventCounts"]["serving.measurement.serving_token_timeline"],
            event_types.count("serving.measurement.serving_token_timeline"),
        )
        sent_events = [item for item in published if "value" in item]
        self.assertEqual(len(sent_events), len(events))
        self.assertTrue(all(item["topic"] == "performance-iq.serving.telemetry.v1" for item in sent_events))
        summary["kafkaPublication"] = publication
        write_proof_summary(summary, self.tmp_dir, summary_out=proof_path)
        verification = verify_proof_summary(proof_path)
        self.assertTrue(verification["ok"], json.dumps(verification, indent=2))
        self.assertGreaterEqual(verification["eventCounts"]["serving.measurement.serving_request_sample"], 3)
        self.assertGreaterEqual(verification["eventCounts"]["serving.measurement.serving_token_timeline"], 3)
        self.assertEqual(verification["telemetryCoverage"]["categorySummary"]["kafkaEventLog"]["status"], "proven")
        lines[0]["payload"]["campaignId"] = "tampered"
        with open(event_log_path, "w", encoding="utf-8") as handle:
            for event in lines:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tampered_verification = verify_proof_summary(proof_path)
        self.assertFalse(tampered_verification["ok"])
        self.assertTrue(any("eventId digest" in error for error in tampered_verification["errors"]))

    def test_serving_smoke_coverage_requires_engine_specific_dashboard_rows(self):
        proof_path, summary = self.write_full_serving_proof()
        sglang_campaign = next(item["campaignId"] for item in summary["submissions"] if item["engine"] == "sglang")

        def remove_sglang_timeline_row(dashboard):
            dashboard["submittedCampaignRows"]["serving_token_timeline"] = [
                row for row in dashboard["submittedCampaignRows"]["serving_token_timeline"]
                if row[0] != sglang_campaign
            ]

        self.rewrite_proof_dashboard(proof_path, remove_sglang_timeline_row)

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"], json.dumps(verification, indent=2))
        coverage = verification["telemetryCoverage"]
        self.assertEqual(coverage["engines"]["vllm"]["dashboardFineGrainRows"]["status"], "proven")
        self.assertEqual(coverage["engines"]["sglang"]["dashboardFineGrainRows"]["status"], "partial")
        self.assertIn(
            "dashboard serving_token_timeline rows are missing for this engine campaign",
            coverage["engines"]["sglang"]["dashboardFineGrainRows"]["missing"],
        )
        self.assertEqual(coverage["categorySummary"]["dashboardFineGrainRows"]["status"], "partial")

    def test_serving_smoke_coverage_requires_engine_specific_event_rows(self):
        proof_path = os.path.join(self.tmp_dir, "fake-full-engine-specific-proof.json")
        event_log_path = os.path.join(self.tmp_dir, "fake-engine-specific-events.jsonl")
        receipt_log_path = os.path.join(self.tmp_dir, "fake-engine-specific-receipts.jsonl")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--fake-full-telemetry",
                "--model", laptop_smoke_model(),
                "--repetitions", "1",
                "--max-tokens", "8",
                "--artifact-dir", self.tmp_dir,
                "--run-suffix", "fake-engine-specific",
                "--summary-out", proof_path,
                "--event-log", event_log_path,
                "--receipt-log", receipt_log_path,
            ])

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        with open(event_log_path, encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        kept_events = [
            event for event in events
            if not (
                event.get("engine") == "tensorrt-llm"
                and event.get("eventType") == "serving.measurement.serving_token_timeline"
            )
        ]
        self.assertLess(len(kept_events), len(events))
        with open(event_log_path, "w", encoding="utf-8") as handle:
            for event in kept_events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"], json.dumps(verification, indent=2))
        self.assertTrue(any(
            "tensorrt-llm event log" in error
            and "serving.measurement.serving_token_timeline" in error
            for error in verification["errors"]
        ))
        self.assertGreaterEqual(verification["eventCounts"].get("serving.measurement.serving_token_timeline", 0), 3)
        self.assertEqual(
            verification["eventCountsByEngine"]["tensorrt-llm"].get("serving.measurement.serving_token_timeline", 0),
            0,
        )
        coverage = verification["telemetryCoverage"]
        self.assertEqual(coverage["engines"]["vllm"]["kafkaEventLog"]["status"], "proven")
        self.assertEqual(coverage["engines"]["sglang"]["kafkaEventLog"]["status"], "proven")
        self.assertEqual(coverage["engines"]["tensorrt-llm"]["kafkaEventLog"]["status"], "partial")
        self.assertIn(
            "Kafka-ready event log is missing token-timeline events for this engine",
            coverage["engines"]["tensorrt-llm"]["kafkaEventLog"]["missing"],
        )
        self.assertEqual(coverage["categorySummary"]["kafkaEventLog"]["status"], "partial")
        self.assertFalse(coverage["allProven"])

    def test_serving_smoke_coverage_requires_request_level_token_timeline_rows(self):
        proof_path, summary = self.write_full_serving_proof(repetitions=2)
        submission = next(item for item in summary["submissions"] if item["engine"] == "vllm")
        artifact_path = submission["artifactPath"]
        manifest_path = submission["manifestPath"]

        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        missing_request_id = artifact["samples"][1]["requestId"]
        artifact["tokenTimeline"] = [
            row for row in artifact["tokenTimeline"]
            if row.get("requestId") != missing_request_id
        ]
        artifact["measurements"] = [
            row for row in artifact["measurements"]
            if not (
                row.get("surface") == "serving_token_timeline"
                and row.get("requestId") == missing_request_id
            )
        ]
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

        artifact_sha = sha256_file(artifact_path)
        submission["artifactSha256"] = artifact_sha
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for manifest_artifact in manifest["artifacts"]:
            if isinstance(manifest_artifact, dict) and manifest_artifact.get("path") == artifact_path:
                manifest_artifact["sha256"] = artifact_sha
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        summary.pop("evidenceIndex", None)
        write_proof_summary(summary, self.tmp_dir, summary_out=proof_path)

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"], json.dumps(verification, indent=2))
        self.assertTrue(any(
            "vllm tokenTimeline is missing output rows for requestIds" in error
            and missing_request_id in error
            for error in verification["errors"]
        ))
        coverage = verification["telemetryCoverage"]
        self.assertEqual(coverage["engines"]["vllm"]["clientStreamTiming"]["status"], "partial")
        self.assertEqual(coverage["engines"]["vllm"]["clientStreamTiming"]["provenCount"], 1)
        self.assertEqual(coverage["engines"]["vllm"]["clientStreamTiming"]["expectedCount"], 2)
        self.assertIn(
            "one or more request samples are missing stream timing fields or request-level output token timeline rows",
            coverage["engines"]["vllm"]["clientStreamTiming"]["missing"],
        )
        self.assertEqual(coverage["categorySummary"]["clientStreamTiming"]["status"], "partial")

    def test_serving_smoke_verifier_requires_request_level_native_rows(self):
        proof_path, summary = self.write_full_serving_proof(repetitions=2)
        submission = next(item for item in summary["submissions"] if item["engine"] == "vllm")
        artifact_path = submission["artifactPath"]

        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        missing_request_id = artifact["samples"][1]["requestId"]
        artifact["nativeTelemetry"][1]["requestId"] = artifact["samples"][0]["requestId"]
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")
        self.refresh_proof_artifact_hashes(proof_path, "vllm")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"], json.dumps(verification, indent=2))
        self.assertTrue(any(
            "vllm nativeTelemetry is missing rows for requestIds" in error
            and missing_request_id in error
            for error in verification["errors"]
        ))

    def test_serving_smoke_coverage_requires_request_level_dcgm_and_raw_snapshots(self):
        proof_path = os.path.join(self.tmp_dir, "fake-full-dcgm-request-proof.json")
        event_log_path = os.path.join(self.tmp_dir, "fake-dcgm-request-events.jsonl")
        receipt_log_path = os.path.join(self.tmp_dir, "fake-dcgm-request-receipts.jsonl")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--fake-full-telemetry",
                "--model", laptop_smoke_model(),
                "--repetitions", "2",
                "--max-tokens", "8",
                "--artifact-dir", self.tmp_dir,
                "--run-suffix", "fake-dcgm-request",
                "--summary-out", proof_path,
                "--event-log", event_log_path,
                "--receipt-log", receipt_log_path,
            ])

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        submission = next(item for item in proof["submissions"] if item["engine"] == "vllm")
        artifact_path = submission["artifactPath"]
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        missing_request_id = artifact["samples"][1]["requestId"]
        artifact["hardwareTelemetry"][1]["requestId"] = artifact["samples"][0]["requestId"]
        raw_artifact_path = artifact["capturePolicy"]["rawArtifactPath"]
        with open(raw_artifact_path, encoding="utf-8") as handle:
            raw_artifact = json.load(handle)
        raw_artifact["captures"][1]["requestId"] = raw_artifact["captures"][0]["requestId"]
        with open(raw_artifact_path, "w", encoding="utf-8") as handle:
            json.dump(raw_artifact, handle, indent=2)
            handle.write("\n")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")
        self.refresh_proof_artifact_hashes(proof_path, "vllm")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"], json.dumps(verification, indent=2))
        self.assertTrue(any(
            "vllm hardwareTelemetry is missing rows for requestIds" in error
            and missing_request_id in error
            for error in verification["errors"]
        ))
        self.assertTrue(any(
            "vllm operator-full raw artifact captures are missing requestIds" in error
            and missing_request_id in error
            for error in verification["errors"]
        ))
        coverage = verification["telemetryCoverage"]
        self.assertEqual(coverage["engines"]["vllm"]["dcgmHardwareTelemetry"]["status"], "partial")
        self.assertEqual(coverage["engines"]["vllm"]["dcgmHardwareTelemetry"]["provenCount"], 1)
        self.assertEqual(coverage["engines"]["vllm"]["dcgmHardwareTelemetry"]["expectedCount"], 2)
        self.assertEqual(coverage["engines"]["vllm"]["rawMetricSnapshots"]["status"], "partial")
        self.assertEqual(coverage["engines"]["vllm"]["rawMetricSnapshots"]["provenCount"], 1)
        self.assertEqual(coverage["engines"]["vllm"]["rawMetricSnapshots"]["expectedCount"], 2)
        self.assertEqual(coverage["categorySummary"]["dcgmHardwareTelemetry"]["status"], "partial")
        self.assertEqual(coverage["categorySummary"]["rawMetricSnapshots"]["status"], "partial")

    def test_serving_smoke_verify_proof_requires_all_telemetry_when_requested(self):
        proof_path, _summary = self.write_full_serving_proof()
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--verify-proof", proof_path,
                "--require-telemetry-coverage",
            ])

        self.assertEqual(code, 1, stderr.getvalue() + stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"], json.dumps(report, indent=2))
        self.assertFalse(report["telemetryCoverage"]["allProven"])
        self.assertFalse(report["strictTelemetryGate"]["ok"])
        self.assertTrue(report["strictTelemetryGate"]["proofOk"])
        self.assertFalse(report["strictTelemetryGate"]["allTelemetryProven"])
        self.assertIn("dcgmHardwareTelemetry", report["strictTelemetryGate"]["missingCategories"])
        self.assertIn("nativeRuntimeTelemetry", report["strictTelemetryGate"]["missingCategories"])

    def test_serving_smoke_require_real_runtime_proof_accepts_configured_endpoint_boundary(self):
        proof_path, _summary = self.write_full_serving_proof()
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--verify-proof", proof_path,
                "--require-real-runtime-proof",
            ])

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"], json.dumps(report, indent=2))
        self.assertEqual(
            report["proofBoundary"],
            "configured serving endpoint capture with measured client telemetry and dashboard snapshot when queried",
        )
        self.assertTrue(report["realRuntimeProofGate"]["ok"], json.dumps(report["realRuntimeProofGate"], indent=2))
        self.assertEqual(report["realRuntimeProofGate"]["classification"], "real-runtime-declared")
        self.assertEqual(report["realRuntimeProofGate"]["blockingMarkers"], [])
        self.assertTrue(real_runtime_proof_gate(report)["ok"])

    def test_serving_smoke_require_telemetry_coverage_rejects_unconfigured_token_details(self):
        proof_path = os.path.join(self.tmp_dir, "fake-full-without-token-detail-proof.json")
        event_log_path = os.path.join(self.tmp_dir, "fake-without-token-detail-events.jsonl")
        receipt_log_path = os.path.join(self.tmp_dir, "fake-without-token-detail-receipts.jsonl")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--fake-full-telemetry",
                "--model", laptop_smoke_model(),
                "--repetitions", "1",
                "--max-tokens", "8",
                "--artifact-dir", self.tmp_dir,
                "--run-suffix", "fake-without-token-detail",
                "--summary-out", proof_path,
                "--event-log", event_log_path,
                "--receipt-log", receipt_log_path,
            ])

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        for submission in proof["submissions"]:
            artifact_path = submission["artifactPath"]
            with open(artifact_path, encoding="utf-8") as handle:
                artifact = json.load(handle)
            artifact["measurements"][0].update({
                "tokenDetailsRequired": False,
                "tokenDetailsAvailableCount": 0,
                "tokenIdsAvailableCount": 0,
                "logprobsAvailableCount": 0,
                "promptTokenDetailsRequired": False,
                "promptTokenIdsAvailableCount": 0,
            })
            for sample in artifact["samples"]:
                sample.update({
                    "tokenDetailsAvailable": False,
                    "tokenIdsAvailable": False,
                    "logprobsAvailable": False,
                    "tokenDetailCount": 0,
                    "tokenDetailSource": "not-requested",
                    "tokenIdSource": None,
                    "promptTokenIdsAvailable": False,
                    "promptTokenDetailCount": 0,
                    "promptTokenIdSource": None,
                    "promptTokenIdsSha256": None,
                    "promptTokenizationSource": "not-requested",
                })
            for row in artifact["tokenTimeline"]:
                row["tokenId"] = None
                row["tokenIdSource"] = None
                row["tokenLogprob"] = None
                row["topLogprobsJson"] = None
                row["tokenDetailSource"] = "not-requested"
            with open(artifact_path, "w", encoding="utf-8") as handle:
                json.dump(artifact, handle, indent=2)
                handle.write("\n")
            self.refresh_proof_artifact_hashes(proof_path, submission["engine"])

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--verify-proof", proof_path,
                "--require-telemetry-coverage",
            ])

        self.assertEqual(code, 1, stderr.getvalue() + stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"], json.dumps(report, indent=2))
        self.assertTrue(report["telemetryCoverage"]["allProven"], json.dumps(report["telemetryCoverage"], indent=2))
        self.assertFalse(report["strictTelemetryGate"]["ok"])
        self.assertTrue(report["strictTelemetryGate"]["configuredTelemetryAllProven"])
        self.assertIn("promptTokenIds", report["strictTelemetryGate"]["notConfiguredCategories"])
        self.assertIn("outputTokenIdsLogprobs", report["strictTelemetryGate"]["notConfiguredCategories"])
        self.assertIn("promptTokenIds", report["strictTelemetryGate"]["missingCategories"])
        self.assertIn("outputTokenIdsLogprobs", report["strictTelemetryGate"]["missingCategories"])

    def test_serving_smoke_require_real_runtime_proof_rejects_fake_full_telemetry(self):
        proof_path = os.path.join(self.tmp_dir, "fake-full-real-runtime-rejected-proof.json")
        event_log_path = os.path.join(self.tmp_dir, "fake-real-runtime-rejected-events.jsonl")
        receipt_log_path = os.path.join(self.tmp_dir, "fake-real-runtime-rejected-receipts.jsonl")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--fake-full-telemetry",
                "--model", laptop_smoke_model(),
                "--repetitions", "1",
                "--max-tokens", "8",
                "--artifact-dir", self.tmp_dir,
                "--run-suffix", "fake-real-runtime-rejected",
                "--summary-out", proof_path,
                "--event-log", event_log_path,
                "--receipt-log", receipt_log_path,
                "--require-real-runtime-proof",
            ])

        self.assertEqual(code, 1, stderr.getvalue() + stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["verification"]["ok"], json.dumps(report["verification"], indent=2))
        self.assertTrue(report["strictTelemetryGate"]["ok"], json.dumps(report["strictTelemetryGate"], indent=2))
        gate = report["realRuntimeProofGate"]
        self.assertFalse(gate["ok"], json.dumps(gate, indent=2))
        self.assertTrue(gate["syntheticProof"])
        markers = {item["marker"] for item in gate["blockingMarkers"]}
        self.assertIn("fake", markers)
        self.assertIn("synthetic", markers)
        self.assertTrue(os.path.exists(proof_path))

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            verify_code = serving_smoke_main([
                "--verify-proof", proof_path,
                "--require-telemetry-coverage",
                "--require-real-runtime-proof",
            ])

        self.assertEqual(verify_code, 1, stderr.getvalue() + stdout.getvalue())
        verify_report = json.loads(stdout.getvalue())
        self.assertTrue(verify_report["strictTelemetryGate"]["ok"], json.dumps(verify_report["strictTelemetryGate"], indent=2))
        self.assertFalse(verify_report["realRuntimeProofGate"]["ok"], json.dumps(verify_report["realRuntimeProofGate"], indent=2))
        self.assertEqual(verify_report["realRuntimeProofGate"]["classification"], "synthetic-or-fixture")

    def test_serving_smoke_fake_full_telemetry_proves_all_coverage(self):
        proof_path = os.path.join(self.tmp_dir, "fake-full-proof.json")
        event_log_path = os.path.join(self.tmp_dir, "fake-events.jsonl")
        receipt_log_path = os.path.join(self.tmp_dir, "fake-receipts.jsonl")
        rows_path = os.path.join(self.tmp_dir, "fake-proof-rows.json")
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = serving_smoke_main([
                "--fake-full-telemetry",
                "--model", laptop_smoke_model(),
                "--repetitions", "1",
                "--max-tokens", "8",
                "--artifact-dir", self.tmp_dir,
                "--run-suffix", "fake-full",
                "--summary-out", proof_path,
                "--event-log", event_log_path,
                "--receipt-log", receipt_log_path,
            ])

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["proofBoundary"], "local fake serving engines and synthetic dashboard row snapshot; not real runtime proof")
        self.assertTrue(os.path.exists(proof_path))
        self.assertTrue(os.path.exists(event_log_path))
        self.assertTrue(os.path.exists(receipt_log_path))
        verification = report["verification"]
        self.assertTrue(verification["ok"], json.dumps(verification, indent=2))
        self.assertTrue(report["strictTelemetryGate"]["ok"], json.dumps(report["strictTelemetryGate"], indent=2))
        self.assertEqual(report["strictTelemetryGate"]["notConfiguredCategories"], [])
        coverage = verification["telemetryCoverage"]
        self.assertTrue(coverage["allProven"], json.dumps(coverage, indent=2))
        for category, summary in coverage["categorySummary"].items():
            self.assertEqual(summary["status"], "proven", category)
        self.assertEqual(coverage["engines"]["tensorrt-llm"]["dcgmHardwareTelemetry"]["status"], "proven")
        self.assertEqual(coverage["engines"]["tensorrt-llm"]["rawMetricSnapshots"]["status"], "proven")
        self.assertEqual(verification["eventCounts"]["serving.measurement.serving_request_sample"], 3)
        self.assertGreaterEqual(verification["eventCounts"]["serving.request_receipt"], 3)

        rows = write_proof_rows(proof_path, rows_path, verification=verification)
        self.assertTrue(rows["telemetryCoverage"]["allProven"])
        self.assertEqual(rows["rowCounts"]["servingRequestSamples"], 3)
        self.assertEqual(rows["rowCounts"]["telemetryCoverageRows"], 33)
        self.assertGreaterEqual(rows["rowCounts"]["servingTokenTimeline"], 18)
        first_sample_row = rows["servingRequestSamples"][0]
        self.assertIn(first_sample_row["runtimeEngine"], {"vllm", "sglang", "tensorrt-llm"})
        self.assertTrue(first_sample_row["rawArtifactPath"].endswith("-operator-full.json"))
        self.assertTrue(os.path.exists(first_sample_row["rawArtifactPath"]))
        self.assertEqual(
            next(row for row in rows["measurements"] if row.get("surface") == "serving_request_sample")["rawArtifactPath"],
            first_sample_row["rawArtifactPath"],
        )
        raw_artifact_path = rows["submissions"][0]["artifactPath"].replace(".json", "-operator-full.json")
        with open(raw_artifact_path, encoding="utf-8") as handle:
            raw_artifact = json.load(handle)
        self.assertIn("nativeMetricsRaw", raw_artifact["captures"][0])
        self.assertIn("hardwareMetricsRaw", raw_artifact["captures"][0])
        self.assertTrue(raw_artifact["captures"][0]["nativeMetricsRaw"]["before"]["available"])
        self.assertTrue(raw_artifact["captures"][0]["hardwareMetricsRaw"]["after"]["available"])

    def test_serving_smoke_verifies_partial_proof_only_when_allowed(self):
        proof_path, summary = self.write_full_serving_proof()
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        proof["submissions"] = [summary["submissions"][0]]
        proof["preflight"]["endpoints"] = [
            item for item in proof["preflight"]["endpoints"]
            if item.get("engine") == "vllm"
        ]
        proof["evidenceIndex"]["engines"] = {
            "vllm": proof["evidenceIndex"]["engines"]["vllm"],
        }
        with open(proof_path, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")

        strict = verify_proof_summary(proof_path)
        partial = verify_proof_summary(proof_path, require_all_engines=False)

        self.assertFalse(strict["ok"])
        self.assertIn("missing required engine submissions", " ".join(strict["errors"]))
        self.assertTrue(partial["ok"], json.dumps(partial, indent=2))
        self.assertEqual(partial["engineCount"], 1)
        self.assertEqual(partial["requiredEngines"], ["vllm"])
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(serving_smoke_main(["--verify-proof", proof_path, "--allow-missing-engines"]), 0)

    def test_serving_smoke_verify_proof_rejects_missing_evidence_index(self):
        proof_path, _summary = self.write_full_serving_proof()
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        del proof["evidenceIndex"]
        with open(proof_path, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("proof evidenceIndex is required", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_inconsistent_evidence_index(self):
        proof_path, _summary = self.write_full_serving_proof()
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        proof["evidenceIndex"]["engines"]["vllm"]["runtimeFramework"] = "other"
        with open(proof_path, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("evidenceIndex runtimeFramework must match", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_tampered_artifact(self):
        proof_path, summary = self.write_full_serving_proof()
        with open(summary["submissions"][0]["artifactPath"], "a", encoding="utf-8") as handle:
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("artifactSha256 does not match", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_missing_receipt(self):
        proof_path, summary = self.write_full_serving_proof()
        os.remove(summary["receiptLogPath"])

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("receipt log does not exist", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_failed_receipt_status(self):
        proof_path, summary = self.write_full_serving_proof()
        self.rewrite_first_receipt(summary["receiptLogPath"], status=502)

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("successful 2xx status", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_wrong_receipt_path(self):
        proof_path, summary = self.write_full_serving_proof()
        self.rewrite_first_receipt(summary["receiptLogPath"], path="/v1/models")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("path must be /v1/chat/completions", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_missing_measurement_metric(self):
        proof_path, summary = self.write_full_serving_proof()
        self.rewrite_first_measurement(summary, outputTpm=None)

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("measurement outputTpm must be a positive number", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_wrong_measurement_framework(self):
        proof_path, summary = self.write_full_serving_proof()
        self.rewrite_first_measurement(summary, runtimeFramework="other")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("measurement runtimeFramework must be vLLM", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_missing_required_hardware_telemetry(self):
        proof_path, summary = self.write_full_serving_proof()
        artifact_path = summary["submissions"][0]["artifactPath"]
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact["measurements"][0].update({
            "hardwareTelemetryRequired": True,
            "hardwareTelemetryAvailableCount": 0,
            "dcgmGrounded": False,
            "metricCompleteness": 0.9,
        })
        artifact["samples"][0]["hardwareTelemetryAvailable"] = False
        artifact["hardwareTelemetry"] = [{"requestId": artifact["samples"][0]["requestId"], "available": False}]
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        joined = " ".join(verification["errors"])
        self.assertIn("hardwareTelemetryAvailableCount must equal successCount", joined)
        self.assertIn("dcgmGrounded must be true", joined)

    def test_serving_smoke_verify_proof_rejects_missing_required_native_telemetry(self):
        proof_path, summary = self.write_full_serving_proof()
        artifact_path = summary["submissions"][0]["artifactPath"]
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact["measurements"][0].update({
            "nativeTelemetryRequired": True,
            "nativeTelemetryAvailableCount": 0,
            "metricCompleteness": 0.9,
        })
        artifact["samples"][0].update({
            "nativeTelemetryAvailable": False,
            "nativeTtftMs": None,
            "queueWaitMs": None,
            "kvCacheUsagePct": None,
        })
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        joined = " ".join(verification["errors"])
        self.assertIn("nativeTelemetryAvailableCount must equal successCount", joined)
        self.assertIn("nativeTelemetryAvailable must be true", joined)
        self.assertIn("nativeTtftMs must be numeric", joined)

    def test_serving_smoke_verify_proof_rejects_missing_required_token_details(self):
        proof_path, summary = self.write_full_serving_proof()
        artifact_path = summary["submissions"][0]["artifactPath"]
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact["measurements"][0].update({
            "tokenDetailsRequired": True,
            "tokenDetailsAvailableCount": 0,
            "tokenIdsAvailableCount": 0,
            "logprobsAvailableCount": 0,
            "metricCompleteness": 0.9,
        })
        artifact["samples"][0].update({
            "tokenDetailsAvailable": False,
            "tokenIdsAvailable": False,
            "logprobsAvailable": False,
            "tokenDetailSource": "requested-not-exposed",
        })
        for row in artifact["tokenTimeline"]:
            row["tokenId"] = None
            row["tokenLogprob"] = None
            row["tokenDetailSource"] = "requested-not-exposed"
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        joined = " ".join(verification["errors"])
        self.assertIn("tokenIdsAvailableCount must equal successCount", joined)
        self.assertIn("tokenTimeline[0].tokenId must be an integer", joined)

    def test_serving_smoke_coverage_respects_engine_token_detail_not_required(self):
        proof_path, summary = self.write_full_serving_proof()

        def disable_sglang_output_token_details(artifact):
            artifact["measurements"][0]["tokenDetailsRequired"] = False
            artifact["measurements"][0]["tokenDetailsAvailableCount"] = 0
            artifact["measurements"][0]["tokenIdsAvailableCount"] = 0
            artifact["measurements"][0]["logprobsAvailableCount"] = 0
            for sample in artifact["samples"]:
                sample["tokenDetailsAvailable"] = False
                sample["tokenIdsAvailable"] = False
                sample["logprobsAvailable"] = False
                sample["tokenDetailSource"] = "not-requested"
                sample["tokenIdSource"] = None
            for row in artifact["tokenTimeline"]:
                if row.get("tokenPhase") != "prompt":
                    row["tokenId"] = None
                    row["tokenIdSource"] = None
                    row["tokenLogprob"] = None

        self.rewrite_engine_artifact(summary, "sglang", disable_sglang_output_token_details)

        verification = verify_proof_summary(proof_path)
        sglang_coverage = verification["telemetryCoverage"]["engines"]["sglang"]["outputTokenIdsLogprobs"]
        output_summary = verification["telemetryCoverage"]["categorySummary"]["outputTokenIdsLogprobs"]

        self.assertEqual(sglang_coverage["expectedCount"], 0)
        self.assertEqual(output_summary["expectedEngines"], 0)
        self.assertEqual(output_summary["status"], "proven")
        self.assertNotIn("sglang measurement tokenIdsAvailableCount must equal successCount", " ".join(verification["errors"]))

    def test_serving_smoke_coverage_flags_unsupported_requested_token_details(self):
        proof_path, summary = self.write_full_serving_proof()

        def mark_sglang_output_token_details_unsupported(artifact):
            artifact["tokenDetailsCapability"] = {
                "requested": True,
                "supported": False,
                "safeToRequest": False,
                "status": "unsupported-runtime",
                "reason": "sglang-mps-mlx-logprobs-crash",
            }
            artifact["measurements"][0]["tokenDetailsRequired"] = False
            artifact["measurements"][0]["tokenDetailsAvailableCount"] = 0
            artifact["measurements"][0]["tokenIdsAvailableCount"] = 0
            artifact["measurements"][0]["logprobsAvailableCount"] = 0
            for sample in artifact["samples"]:
                sample["tokenDetailsAvailable"] = False
                sample["tokenIdsAvailable"] = False
                sample["logprobsAvailable"] = False
                sample["tokenDetailSource"] = "not-requested"
                sample["tokenDetailsCapabilityStatus"] = "unsupported-runtime"
                sample["tokenDetailsUnsupportedReason"] = "sglang-mps-mlx-logprobs-crash"
            for row in artifact["tokenTimeline"]:
                if row.get("tokenPhase") != "prompt":
                    row["tokenId"] = None
                    row["tokenIdSource"] = None
                    row["tokenLogprob"] = None

        self.rewrite_engine_artifact(summary, "sglang", mark_sglang_output_token_details_unsupported)

        verification = verify_proof_summary(proof_path)
        sglang_coverage = verification["telemetryCoverage"]["engines"]["sglang"]["outputTokenIdsLogprobs"]
        output_summary = verification["telemetryCoverage"]["categorySummary"]["outputTokenIdsLogprobs"]

        self.assertEqual(sglang_coverage["expectedCount"], 1)
        self.assertEqual(sglang_coverage["status"], "missing")
        self.assertIn("unsupported by runtime", " ".join(sglang_coverage["missing"]))
        self.assertEqual(output_summary["expectedEngines"], 1)
        self.assertEqual(output_summary["status"], "missing")

    def test_serving_smoke_verify_proof_rejects_missing_dashboard_rows(self):
        proof_path, _summary = self.write_full_serving_proof()
        self.rewrite_proof_dashboard(proof_path, lambda dashboard: dashboard["rows"].pop("price_performance"))

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("dashboard price_performance rows must include inspectable row data", " ".join(verification["errors"]))

    def test_serving_smoke_verify_proof_rejects_missing_submitted_campaign_rows(self):
        proof_path, _summary = self.write_full_serving_proof()
        self.rewrite_proof_dashboard(
            proof_path,
            lambda dashboard: dashboard["submittedCampaignRows"].update({"run_details": []}),
        )

        verification = verify_proof_summary(proof_path)

        self.assertFalse(verification["ok"])
        self.assertIn("dashboard run_details submittedCampaignRows is missing", " ".join(verification["errors"]))

    def test_serving_smoke_reports_missing_engine_urls(self):
        class Args:
            vllm_url = "http://127.0.0.1:8000"
            sglang_url = None
            tensorrt_llm_url = None
            framework_version = None
            image_digest = None
            image_tag = None

        configs, missing = engine_configs_from_env(Args())

        self.assertEqual([config["engine"] for config in configs], ["vllm"])
        self.assertEqual(missing, ["sglang (PIQ_SGLANG_URL)", "tensorrt-llm (PIQ_TENSORRT_LLM_URL)"])

    def test_serving_smoke_reads_per_engine_token_detail_overrides(self):
        class Args:
            vllm_url = "http://127.0.0.1:8000"
            sglang_url = "http://127.0.0.1:30000"
            tensorrt_llm_url = None
            framework_version = None
            model_revision = None
            image_digest = None
            image_tag = None
            server_args = None
            tokenizer_model = None
            resolve_token_ids_with_tokenizer = False
            collect_hardware_metrics = False
            require_native_telemetry = False
            require_hardware_telemetry = False
            process_id = None
            container_id = None
            pod_name = None
            node_name = None
            host_name = None

        os.environ["PIQ_VLLM_CAPTURE_TOKEN_DETAILS"] = "true"
        os.environ["PIQ_VLLM_TOP_LOGPROBS"] = "3"
        os.environ["PIQ_SGLANG_CAPTURE_TOKEN_DETAILS"] = "false"

        configs, _missing = engine_configs_from_env(Args())
        by_engine = {config["engine"]: config for config in configs}

        self.assertTrue(by_engine["vllm"]["captureTokenDetails"])
        self.assertEqual(by_engine["vllm"]["topLogprobs"], 3)
        self.assertFalse(by_engine["sglang"]["captureTokenDetails"])

    def test_serving_smoke_passes_runtime_python_for_tokenizer_resolution(self):
        class Args:
            vllm_url = "http://127.0.0.1:8000"
            sglang_url = None
            tensorrt_llm_url = None
            framework_version = None
            model_revision = None
            image_digest = None
            image_tag = None
            server_args = None
            tokenizer_model = None
            resolve_token_ids_with_tokenizer = True
            collect_hardware_metrics = False
            require_native_telemetry = False
            require_hardware_telemetry = False
            process_id = None
            container_id = None
            pod_name = None
            node_name = None
            host_name = None

        with patch("performance_iq_sdk.serving_smoke.preferred_runtime_python", return_value="/tmp/runtime-python"):
            configs, _missing = engine_configs_from_env(Args())

        self.assertEqual(configs[0]["tokenizerPythonBin"], "/tmp/runtime-python")
        self.assertTrue(configs[0]["resolveTokenIdsWithTokenizer"])

    def test_serving_smoke_preflight_reports_missing_without_requests(self):
        os.environ["PIQ_COMMAND_PROBE_TIMEOUT_SECONDS"] = "2.5"
        preflight = runtime_preflight([], ["vllm (PIQ_VLLM_URL)"], model=laptop_smoke_model())

        self.assertFalse(preflight["ready"])
        self.assertEqual(preflight["missingEngineUrls"], ["vllm (PIQ_VLLM_URL)"])
        self.assertEqual(preflight["endpoints"], [])
        self.assertEqual(preflight["configuredEngineCount"], 0)
        self.assertIn("vllmCommand", preflight["localRuntime"])
        self.assertIn("vllmExtension", preflight["localRuntime"])
        self.assertIn("runtimeCandidates", preflight["localRuntime"])
        self.assertIn("vllm", preflight["localRuntime"]["runtimeCandidates"])
        self.assertIn("sglang", preflight["localRuntime"]["runtimeCandidates"])
        self.assertEqual(preflight["localRuntime"]["commandProbeTimeoutSeconds"], 2.5)
        self.assertIn("python", preflight["host"])
        self.assertIn("freeGiB", preflight["storage"])
        self.assertEqual(preflight["launchPlan"]["model"], laptop_smoke_model())
        self.assertEqual(preflight["launchPlan"]["endpointEnv"]["PIQ_VLLM_URL"], "http://127.0.0.1:8000")

    def test_external_python_module_probe_imports_from_selected_interpreter(self):
        probe = external_python_module_probe(sys.executable, "json", import_module=True)

        self.assertTrue(probe["available"])
        self.assertTrue(probe["imported"])
        self.assertEqual(probe["python"], sys.executable)

    def test_serving_smoke_diagnostics_uses_local_runtime_candidates(self):
        runtime_candidates = {
            "vllm": {
                "pythonEnv": "PIQ_VLLM_PYTHON_BIN",
                "usable": True,
                "preferred": {"python": "/tmp/vllm/bin/python", "usable": True},
                "candidates": [{"python": "/tmp/vllm/bin/python", "usable": True}],
            },
            "sglang": {
                "pythonEnv": "PIQ_SGLANG_PYTHON_BIN",
                "usable": True,
                "preferred": {"python": "/tmp/sglang/bin/python", "usable": True},
                "candidates": [{"python": "/tmp/sglang/bin/python", "usable": True}],
            },
        }
        with patch("performance_iq_sdk.serving_smoke.local_runtime_discovery", return_value=runtime_candidates):
            diagnostics = runtime_diagnostics([], ["vllm (PIQ_VLLM_URL)"], model=laptop_smoke_model())

        blockers = " ".join(diagnostics["blockers"])
        self.assertTrue(diagnostics["preflight"]["localRuntime"]["runtimeCandidates"]["vllm"]["usable"])
        self.assertTrue(diagnostics["preflight"]["localRuntime"]["runtimeCandidates"]["sglang"]["usable"])
        self.assertIn("Missing configured endpoint URL for vllm (PIQ_VLLM_URL).", blockers)
        self.assertNotIn("Python module 'vllm' is not importable", blockers)
        self.assertNotIn("Python module 'sglang' is not importable", blockers)

    def test_serving_smoke_vllm_extension_probe_handles_missing_parent_module(self):
        with patch(
            "performance_iq_sdk.serving_smoke.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'vllm'"),
        ):
            probe = vllm_extension_probe()

        self.assertFalse(probe["available"])
        self.assertIn("findError", probe)
        self.assertNotIn("torchCInitCpuMemoryEnv", probe)

    def test_serving_smoke_vllm_extension_probe_reports_import_failure(self):
        with (
            patch(
                "performance_iq_sdk.serving_smoke.importlib.util.find_spec",
                return_value=SimpleNamespace(origin="/tmp/vllm/_C.so"),
            ),
            patch(
                "performance_iq_sdk.serving_smoke.importlib.import_module",
                side_effect=RuntimeError("extension load failed"),
            ),
        ):
            probe = vllm_extension_probe()

        self.assertTrue(probe["available"])
        self.assertFalse(probe["imported"])
        self.assertEqual(probe["importError"], "extension load failed")
        self.assertNotIn("torchCInitCpuMemoryEnv", probe)

    def test_serving_smoke_main_preflight_allows_configured_partial_engine(self):
        calls = {"get": 0, "post": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                calls["get"] += 1
                payload = json.dumps({
                    "object": "list",
                    "data": [{"id": laptop_smoke_model(), "object": "model"}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                calls["post"] += 1
                self.send_response(200)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stdout = StringIO()
        stderr = StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = serving_smoke_main([
                    "--preflight-only",
                    "--vllm-url", f"http://127.0.0.1:{server.server_address[1]}",
                    "--allow-missing-engines",
                    "--model", laptop_smoke_model(),
                ])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        self.assertEqual(calls["get"], 1)
        self.assertEqual(calls["post"], 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ready"])
        self.assertEqual(report["missingEngineUrls"], [])
        self.assertEqual(report["configuredEngineCount"], 1)

    def test_serving_smoke_launch_plan_includes_all_engine_commands(self):
        plan = runtime_launch_plan(laptop_smoke_model())

        self.assertEqual(set(plan["engines"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertIn("vllm serve", plan["engines"]["vllm"]["serve"])
        self.assertIn("sglang.launch_server", plan["engines"]["sglang"]["serve"])
        self.assertIn("trtllm-serve", plan["engines"]["tensorrt-llm"]["serve"])
        self.assertEqual(plan["runtimeDiscovery"]["vllm"]["pythonEnv"], "PIQ_VLLM_PYTHON_BIN")
        self.assertEqual(plan["runtimeDiscovery"]["sglang"]["pythonEnv"], "PIQ_SGLANG_PYTHON_BIN")
        self.assertIn("usable", plan["runtimeDiscovery"]["vllm"])
        self.assertIn("freeGiB", plan["storage"])
        self.assertEqual(plan["endpointEnv"]["PIQ_SGLANG_URL"], "http://127.0.0.1:30000")
        self.assertEqual(plan["strictProof"]["mode"], "strict-recorded-smoke")
        self.assertIn("strict-recorded-smoke", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_USD_PER_GPU_HOUR", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_REQUIRE_NATIVE_TELEMETRY=true", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_REQUIRE_HARDWARE_TELEMETRY=true", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_VERIFY_AFTER_CAPTURE=true", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_REQUIRE_TELEMETRY_COVERAGE=true", plan["strictProof"]["command"])
        self.assertIn("PIQ_SERVING_REQUIRE_REAL_RUNTIME_PROOF=true", plan["strictProof"]["command"])
        self.assertIn("--require-telemetry-coverage", plan["strictProof"]["verify"])
        self.assertIn("--require-real-runtime-proof", plan["strictProof"]["verify"])
        self.assertIn("serving_request_samples", plan["strictProof"]["dashboardSurfaces"])
        self.assertIn("serving_token_timeline", plan["strictProof"]["dashboardSurfaces"])
        self.assertIn("serving_telemetry_coverage", plan["strictProof"]["dashboardSurfaces"])
        self.assertIn("stream=true", plan["telemetryModel"]["streamingCollection"])
        self.assertIn("post-capture", plan["telemetryModel"]["kafkaBoundary"])
        self.assertIn("serving_token_timeline", plan["telemetryModel"]["requestSurfaces"])
        self.assertIn("serving_telemetry_coverage", plan["telemetryModel"]["requestSurfaces"])
        self.assertIn("DCGM hardware counters", plan["telemetryModel"]["strictTelemetry"])
        self.assertIn("tokenizer-exact prompt token IDs", plan["telemetryModel"]["strictTelemetry"])

    def test_serving_smoke_parser_reads_pricing_env(self):
        os.environ["PIQ_SERVING_USD_PER_GPU_HOUR"] = "2.5"
        os.environ["PIQ_SERVING_GPU_COUNT"] = "4"
        os.environ["PIQ_SERVING_POWER_WATTS_PER_GPU"] = "700"
        os.environ["PIQ_SERVING_RESOLVE_TOKEN_IDS_WITH_TOKENIZER"] = "true"
        os.environ["PIQ_SERVING_TOKENIZER_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"
        os.environ["PIQ_SERVING_VERIFY_AFTER_CAPTURE"] = "true"
        os.environ["PIQ_SERVING_REQUIRE_TELEMETRY_COVERAGE"] = "true"

        args = serving_smoke_parser().parse_args([])

        self.assertEqual(args.usd_per_gpu_hour, 2.5)
        self.assertEqual(args.gpu_count, 4)
        self.assertEqual(args.power_watts_per_gpu, 700)
        self.assertTrue(args.resolve_token_ids_with_tokenizer)
        self.assertEqual(args.tokenizer_model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(args.verify_after_capture)
        self.assertTrue(args.require_telemetry_coverage)

    def test_serving_smoke_diagnostics_reports_cache_ports_and_blockers(self):
        old_home = os.environ.get("HOME")
        old_hf_home = os.environ.get("HF_HOME")
        hf_home = os.path.join(self.tmp_dir, "hf")
        os.makedirs(os.path.join(hf_home, "hub", huggingface_model_cache_name(laptop_smoke_model())))
        os.makedirs(os.path.join(self.tmp_dir, ".cache", "huggingface", "hub"))
        try:
            os.environ["HOME"] = self.tmp_dir
            os.environ["HF_HOME"] = hf_home
            os.environ["PIQ_VLLM_SOURCE_PATH"] = "/tmp/vllm"

            report = runtime_diagnostics([], ["vllm (PIQ_VLLM_URL)"], laptop_smoke_model())
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old_hf_home

        self.assertFalse(report["preflight"]["ready"])
        self.assertEqual(
            report["diagnostics"]["caches"]["modelCache"]["huggingFaceCacheName"],
            "models--Qwen--Qwen2.5-0.5B-Instruct",
        )
        self.assertTrue(report["diagnostics"]["caches"]["modelCache"]["candidates"][0]["exists"])
        self.assertEqual(set(report["diagnostics"]["ports"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertFalse(report["diagnostics"]["environment"]["PIQ_TOKEN"]["set"])
        self.assertTrue(report["diagnostics"]["environment"]["PIQ_VLLM_SOURCE_PATH"]["set"])
        self.assertTrue(report["blockers"])

    def test_serving_smoke_endpoint_preflight_rejects_not_found_models_route(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(404)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"not found"}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = endpoint_probe({
                "engine": "vllm",
                "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
            })
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["reachable"])
        self.assertEqual(result["status"], 404)
        self.assertFalse(result["ok"])

    def test_serving_smoke_endpoint_preflight_rejects_wrong_served_model(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"object":"list","data":[{"id":"other-model","object":"model"}]}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = endpoint_probe({
                "engine": "sglang",
                "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
            }, model=laptop_smoke_model())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["reachable"])
        self.assertTrue(result["modelChecked"])
        self.assertFalse(result["modelAvailable"])
        self.assertFalse(result["ok"])

    def test_serving_smoke_endpoint_preflight_captures_sglang_server_info(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path == "/v1/models":
                    payload = {
                        "object": "list",
                        "data": [{"id": laptop_smoke_model(), "object": "model"}],
                    }
                elif self.path == "/model_info":
                    payload = {
                        "model_path": laptop_smoke_model(),
                        "tokenizer_path": laptop_smoke_model(),
                        "is_generation": True,
                        "weight_version": "default",
                        "model_type": "qwen2",
                    }
                elif self.path == "/server_info":
                    payload = {
                        "device": "mps",
                        "model_path": laptop_smoke_model(),
                        "served_model_name": laptop_smoke_model(),
                        "context_length": 512,
                        "max_total_tokens": 512,
                        "disable_overlap_schedule": True,
                        "api_key": "must-not-leak",
                    }
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = endpoint_probe({
                "engine": "sglang",
                "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
            }, model=laptop_smoke_model())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["serverInfo"]["device"], "mps")
        self.assertEqual(result["serverInfo"]["context_length"], 512)
        self.assertNotIn("api_key", result["serverInfo"])
        self.assertEqual(result["modelInfo"]["model_path"], laptop_smoke_model())
        self.assertIsInstance(result["serverInfoSha256"], str)

    def test_serving_smoke_endpoint_preflight_rejects_nonstandard_model_list(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"object":"list","models":["not-standard"]}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = endpoint_probe({
                "engine": "tensorrt-llm",
                "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
            }, model=laptop_smoke_model())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["reachable"])
        self.assertFalse(result["ok"])
        self.assertFalse(result["modelChecked"])
        self.assertIsNone(result["modelAvailable"])
        self.assertIn("data[].id", result["error"])

    def test_serving_smoke_endpoint_preflight_rejects_auth_error(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(401)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = endpoint_probe({
                "engine": "vllm",
                "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
            }, model=laptop_smoke_model())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result["reachable"])
        self.assertEqual(result["status"], 401)
        self.assertTrue(result["authFailed"])
        self.assertFalse(result["ok"])

    def test_serving_receipt_proxy_forwards_and_records_trace_headers(self):
        backend_headers = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                backend_headers["request_id"] = self.headers.get("x-performance-iq-request-id")
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        receipt_log = os.path.join(self.tmp_dir, "proxy-receipts.jsonl")
        proxy = recording_proxy_server(
            engine="vllm",
            target_base_url=f"http://127.0.0.1:{backend.server_address[1]}",
            receipt_log=receipt_log,
        )
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
                data=b'{"model":"test"}',
                headers={
                    "content-type": "application/json",
                    "x-performance-iq-request-id": "piq-vllm-unit-request-1",
                    "x-performance-iq-campaign-id": "campaign-unit",
                    "x-performance-iq-run-id": "run-unit",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read().decode("utf-8")), {"ok": True})
        finally:
            proxy.shutdown()
            proxy.server_close()
            proxy_thread.join(timeout=2)
            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=2)

        self.assertEqual(backend_headers["request_id"], "piq-vllm-unit-request-1")
        receipts = load_receipts(receipt_log)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["schemaVersion"], REQUEST_RECEIPT_SCHEMA_VERSION)
        self.assertEqual(receipts[0]["engine"], "vllm")
        self.assertEqual(receipts[0]["requestId"], "piq-vllm-unit-request-1")

    def test_serving_smoke_main_preflight_blocks_wrong_model_before_request(self):
        calls = {"post": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"object":"list","data":[{"id":"other-model","object":"model"}]}')

            def do_POST(self):
                calls["post"] += 1
                self.send_response(200)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = serving_smoke_main([
                    "--vllm-url", f"http://127.0.0.1:{server.server_address[1]}",
                    "--allow-missing-engines",
                    "--no-submit",
                    "--model", laptop_smoke_model(),
                    "--artifact-dir", self.tmp_dir,
                ])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(code, 1)
        self.assertEqual(calls["post"], 0)

    def test_serving_smoke_main_preflight_allows_matching_model_request(self):
        calls = {"post": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "object": "list",
                    "data": [{"id": laptop_smoke_model(), "object": "model"}],
                }).encode("utf-8"))

            def do_POST(self):
                calls["post"] += 1
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                payload = (
                    "data: " + json.dumps({
                        "id": "chatcmpl-main-test",
                        "model": laptop_smoke_model(),
                        "choices": [{"delta": {"role": "assistant"}}],
                    }) + "\n\n" +
                    "data: " + json.dumps({
                        "id": "chatcmpl-main-test",
                        "model": laptop_smoke_model(),
                        "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                    }) + "\n\n" +
                    "data: " + json.dumps({
                        "id": "chatcmpl-main-test",
                        "model": laptop_smoke_model(),
                        "choices": [],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    }) + "\n\n" +
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stdout = StringIO()
        stderr = StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = serving_smoke_main([
                    "--vllm-url", f"http://127.0.0.1:{server.server_address[1]}",
                    "--allow-missing-engines",
                    "--no-submit",
                    "--model", laptop_smoke_model(),
                    "--repetitions", "1",
                    "--artifact-dir", self.tmp_dir,
                    "--run-suffix", "unit-summary",
                ])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(code, 0, stderr.getvalue() + stdout.getvalue())
        self.assertEqual(calls["post"], 1)
        proof_path = os.path.join(self.tmp_dir, "serving-smoke-proof-unit-summary.json")
        self.assertTrue(os.path.exists(proof_path))
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        self.assertEqual(proof["schemaVersion"], "performance-iq.serving-smoke-proof.v1")
        self.assertEqual(proof["runSuffix"], "unit-summary")
        self.assertEqual(proof["proofSummaryPath"], proof_path)
        self.assertEqual(proof["submissions"][0]["campaignId"], "serving-vllm-unit-summary")
        self.assertTrue(os.path.exists(proof["submissions"][0]["manifestPath"]))
        self.assertTrue(proof["submissions"][0]["artifactSha256"])
        self.assertTrue(proof["preflight"]["ready"])

    def test_serving_smoke_main_recorded_smoke_proves_receipts_and_dashboard_rows(self):
        submitted_campaigns = []

        class EngineHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                payload = json.dumps({
                    "object": "list",
                    "data": [{"id": laptop_smoke_model(), "object": "model"}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                payload = (
                    "data: " + json.dumps({
                        "id": "chatcmpl-recorded-test",
                        "model": laptop_smoke_model(),
                        "choices": [{"delta": {"role": "assistant"}}],
                    }) + "\n\n" +
                    "data: " + json.dumps({
                        "id": "chatcmpl-recorded-test",
                        "model": laptop_smoke_model(),
                        "choices": [{"delta": {"content": "o"}}],
                    }) + "\n\n" +
                    "data: " + json.dumps({
                        "id": "chatcmpl-recorded-test",
                        "model": laptop_smoke_model(),
                        "choices": [{"delta": {"content": "k"}, "finish_reason": "stop"}],
                    }) + "\n\n" +
                    "data: " + json.dumps({
                        "id": "chatcmpl-recorded-test",
                        "model": laptop_smoke_model(),
                        "choices": [],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
                    }) + "\n\n" +
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class PiqHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                if self.path == "/api/v1/runs":
                    payload = json.loads(body.decode("utf-8"))
                    campaign = payload["manifest"]["campaign"]["campaignId"]
                    submitted_campaigns.append(campaign)
                    response = {
                        "id": payload["manifest"]["campaign"]["runId"],
                        "status": "accepted",
                        "liveProofReady": False,
                    }
                elif self.path == "/api/store/queries":
                    response = {
                        "price_performance": {
                            "rowCount": 3,
                            "rows": [
                                [laptop_smoke_model(), "local endpoints", "vLLM"],
                                [laptop_smoke_model(), "local endpoints", "SGLang"],
                                [laptop_smoke_model(), "local endpoints", "TensorRT-LLM"],
                            ],
                        },
                        "capacity_best": {
                            "rowCount": 3,
                            "rows": [[laptop_smoke_model()], [laptop_smoke_model()], [laptop_smoke_model()]],
                        },
                        "campaign_provenance": {
                            "rowCount": 3,
                            "rows": [[campaign] for campaign in submitted_campaigns],
                        },
                        "run_details": {
                            "rowCount": 3,
                            "rows": [[campaign] for campaign in submitted_campaigns],
                        },
                        "serving_request_samples": {
                            "rowCount": 3,
                            "rows": [[campaign] for campaign in submitted_campaigns],
                        },
                        "serving_token_timeline": {
                            "rowCount": 6,
                            "rows": [[campaign] for campaign in submitted_campaigns],
                        },
                        "serving_telemetry_coverage": {
                            "rowCount": 3,
                            "rows": [[campaign] for campaign in submitted_campaigns],
                        },
                    }
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("x-piq-store-provider", "sdk-ingestion")
                self.end_headers()
                self.wfile.write(payload)

        engine_servers = [ThreadingHTTPServer(("127.0.0.1", 0), EngineHandler) for _ in range(3)]
        piq_server = ThreadingHTTPServer(("127.0.0.1", 0), PiqHandler)
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in [*engine_servers, piq_server]
        ]
        for thread in threads:
            thread.start()
        proof_path = os.path.join(self.tmp_dir, "recorded-proof.json")
        receipt_log = os.path.join(self.tmp_dir, "recorded-receipts.jsonl")
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = serving_smoke_main([
                    "--vllm-url", f"http://127.0.0.1:{engine_servers[0].server_address[1]}",
                    "--sglang-url", f"http://127.0.0.1:{engine_servers[1].server_address[1]}",
                    "--tensorrt-llm-url", f"http://127.0.0.1:{engine_servers[2].server_address[1]}",
                    "--piq-base-url", f"http://127.0.0.1:{piq_server.server_address[1]}",
                    "--model", laptop_smoke_model(),
                    "--repetitions", "1",
                    "--artifact-dir", self.tmp_dir,
                    "--run-suffix", "recorded-unit",
                    "--summary-out", proof_path,
                    "--receipt-log", receipt_log,
                    "--record-receipts",
                    "--query-dashboard",
                ])
        finally:
            for server in [*engine_servers, piq_server]:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

        self.assertEqual(code, 0)
        verification = verify_proof_summary(proof_path)
        self.assertTrue(verification["ok"], json.dumps(verification, indent=2))
        self.assertEqual(verification["receiptCounts"], {"vllm": 1, "sglang": 1, "tensorrt-llm": 1})
        self.assertTrue(os.path.exists(receipt_log))

    def test_serving_smoke_dashboard_query_reports_campaign_surfaces(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps({
                    "price_performance": {
                        "rowCount": 1,
                        "rows": [["model", "hardware", "vLLM"]],
                    },
                    "capacity_best": {"rowCount": 1, "rows": [["model"]]},
                    "campaign_provenance": {"rowCount": 1, "rows": [["campaign-a"]]},
                    "run_details": {"rowCount": 1, "rows": [["campaign-a"]]},
                    "serving_request_samples": {"rowCount": 1, "rows": [["campaign-a"]]},
                    "serving_token_timeline": {"rowCount": 1, "rows": [["campaign-a"]]},
                    "serving_telemetry_coverage": {"rowCount": 1, "rows": [["campaign-a"]]},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("x-piq-store-provider", "sdk-ingestion")
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = query_dashboard(f"http://127.0.0.1:{server.server_address[1]}")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(result["storeProvider"], "sdk-ingestion")
        self.assertEqual(result["rows"]["price_performance"], [["model", "hardware", "vLLM"]])
        self.assertEqual(result["campaignIds"], ["campaign-a"])
        self.assertEqual(result["surfaceCampaignIds"], {
            "campaign_provenance": ["campaign-a"],
            "run_details": ["campaign-a"],
            "serving_request_samples": ["campaign-a"],
            "serving_token_timeline": ["campaign-a"],
            "serving_telemetry_coverage": ["campaign-a"],
        })
        self.assertEqual(result["submittedCampaignRows"], {})
        self.assertEqual(result["runtimeFrameworks"], ["vLLM"])

    def test_serving_smoke_dashboard_query_filters_submitted_campaign_rows(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps({
                    "price_performance": {
                        "rowCount": 1,
                        "rows": [["model", "hardware", "vLLM"]],
                    },
                    "capacity_best": {"rowCount": 1, "rows": [["model"]]},
                    "campaign_provenance": {"rowCount": 2, "rows": [["campaign-a"], ["other"]]},
                    "run_details": {"rowCount": 2, "rows": [["campaign-a"], ["other"]]},
                    "serving_request_samples": {"rowCount": 2, "rows": [["campaign-a"], ["other"]]},
                    "serving_token_timeline": {"rowCount": 2, "rows": [["campaign-a"], ["other"]]},
                    "serving_telemetry_coverage": {"rowCount": 2, "rows": [["campaign-a"], ["other"]]},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("x-piq-store-provider", "sdk-ingestion")
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = query_dashboard(
                f"http://127.0.0.1:{server.server_address[1]}",
                campaign_ids=["campaign-a"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(result["submittedCampaignRows"], {
            "campaign_provenance": [["campaign-a"]],
            "run_details": [["campaign-a"]],
            "serving_request_samples": [["campaign-a"]],
            "serving_token_timeline": [["campaign-a"]],
            "serving_telemetry_coverage": [["campaign-a"]],
        })


if __name__ == "__main__":
    unittest.main()
