import json
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
    huggingface_model_cache_name,
    main as serving_smoke_main,
    query_dashboard,
    run_serving_smoke,
    runtime_diagnostics,
    runtime_launch_plan,
    runtime_preflight,
    vllm_extension_probe,
    verify_proof_summary,
    write_proof_summary,
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
        "PIQ_ARTIFACT_DIR",
        "PIQ_SERVING_SUMMARY_OUT",
        "PIQ_TENSORRT_LLM_IMAGE",
        "PIQ_PYTHON_BIN",
        "PIQ_SERVING_BIN_DIR",
        "PIQ_VLLM_SOURCE_PATH",
        "PIQ_SGLANG_SOURCE_PATH",
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

    def write_full_serving_proof(self):
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
            repetitions=1,
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
                "serving_request_samples": 3,
                "serving_token_timeline": 6,
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
            },
            "campaignIds": campaign_ids,
            "surfaceCampaignIds": {
                "campaign_provenance": campaign_ids,
                "run_details": campaign_ids,
                "serving_request_samples": campaign_ids,
                "serving_token_timeline": campaign_ids,
            },
            "submittedCampaignRows": {
                "campaign_provenance": [[campaign_id] for campaign_id in campaign_ids],
                "run_details": [[campaign_id] for campaign_id in campaign_ids],
                "serving_request_samples": [[campaign_id] for campaign_id in campaign_ids],
                "serving_token_timeline": [[campaign_id] for campaign_id in campaign_ids],
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

    def rewrite_proof_dashboard(self, proof_path, mutate):
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        mutate(proof["dashboard"])
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
        self.assertEqual(len(sample_rows), 1)
        self.assertEqual(len(token_rows), 2)
        with open(result["artifactPath"], encoding="utf-8") as handle:
            artifact = json.load(handle)
        with open(result["manifestPath"], encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertNotIn("messages", artifact["request"])
        self.assertEqual(artifact["capturePolicy"]["mode"], "operator-full")
        raw_artifacts = [item for item in manifest["artifacts"] if item["kind"] == "operator-full-serving-raw"]
        self.assertEqual(len(raw_artifacts), 1)
        self.assertTrue(os.path.exists(raw_artifacts[0]["path"]))

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
vllm:kv_cache_usage_perc{model_name="qwen"} 0.125
vllm:prefix_cache_queries_total{model_name="qwen"} 20
vllm:prefix_cache_hits_total{model_name="qwen"} 5
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
vllm:kv_cache_usage_perc{model_name="qwen"} 0.25
vllm:prefix_cache_queries_total{model_name="qwen"} 30
vllm:prefix_cache_hits_total{model_name="qwen"} 7
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
        aggregate = result["measurements"][0]
        self.assertEqual(aggregate["nativeTelemetryAvailableCount"], 1)
        self.assertAlmostEqual(aggregate["avgQueueWaitMs"], 3)
        self.assertAlmostEqual(aggregate["avgPrefillMs"], 60)
        self.assertAlmostEqual(aggregate["avgDecodeMs"], 120)

    def test_serving_producer_captures_token_logprobs_and_dcgm_metrics(self):
        metric_snapshots = [
            """
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 100
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 40
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 12
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 1000
""",
            """
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 120
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 50
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 20
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
        self.assertEqual(sample["tokenTimeline"][0]["tokenId"], 101)
        self.assertAlmostEqual(sample["tokenTimeline"][0]["tokenLogprob"], -0.1)
        self.assertIn("topLogprobsJson", sample["tokenTimeline"][0])
        self.assertAlmostEqual(sample["avgPowerWatts"], 120)
        self.assertAlmostEqual(sample["avgPowerWattsPerGpu"], 120)
        self.assertAlmostEqual(sample["gpuUtilizationPct"], 50)
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
        self.assertAlmostEqual(token_rows[1]["tokenLogprob"], -0.2)

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

    def test_serving_smoke_verifies_full_proof_bundle(self):
        proof_path, summary = self.write_full_serving_proof()

        verification = verify_proof_summary(proof_path)

        self.assertTrue(verification["ok"], json.dumps(verification, indent=2))
        self.assertEqual(verification["engineCount"], 3)
        self.assertEqual(verification["campaignIds"], sorted(item["campaignId"] for item in summary["submissions"]))
        self.assertEqual(set(verification["artifactHashes"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertEqual(verification["receiptCounts"], {"vllm": 1, "sglang": 1, "tensorrt-llm": 1})
        with open(proof_path, encoding="utf-8") as handle:
            proof = json.load(handle)
        self.assertEqual(set(proof["evidenceIndex"]["engines"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertEqual(
            proof["evidenceIndex"]["engines"]["vllm"]["artifact"]["sha256"],
            summary["submissions"][0]["artifactSha256"],
        )
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(serving_smoke_main(["--verify-proof", proof_path]), 0)

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

    def test_serving_smoke_preflight_reports_missing_without_requests(self):
        os.environ["PIQ_COMMAND_PROBE_TIMEOUT_SECONDS"] = "2.5"
        preflight = runtime_preflight([], ["vllm (PIQ_VLLM_URL)"], model=laptop_smoke_model())

        self.assertFalse(preflight["ready"])
        self.assertEqual(preflight["missingEngineUrls"], ["vllm (PIQ_VLLM_URL)"])
        self.assertEqual(preflight["endpoints"], [])
        self.assertEqual(preflight["configuredEngineCount"], 0)
        self.assertIn("vllmCommand", preflight["localRuntime"])
        self.assertIn("vllmExtension", preflight["localRuntime"])
        self.assertEqual(preflight["localRuntime"]["commandProbeTimeoutSeconds"], 2.5)
        self.assertIn("python", preflight["host"])
        self.assertIn("freeGiB", preflight["storage"])
        self.assertEqual(preflight["launchPlan"]["model"], laptop_smoke_model())
        self.assertEqual(preflight["launchPlan"]["endpointEnv"]["PIQ_VLLM_URL"], "http://127.0.0.1:8000")

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
        self.assertIn("freeGiB", plan["storage"])
        self.assertEqual(plan["endpointEnv"]["PIQ_SGLANG_URL"], "http://127.0.0.1:30000")

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
        })


if __name__ == "__main__":
    unittest.main()
