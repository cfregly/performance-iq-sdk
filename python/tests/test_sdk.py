import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from performance_iq_sdk import PerformanceIQ, build_manifest, laptop_smoke_model, run_serving_producer, validate_run
from performance_iq_sdk.serving_smoke import (
    engine_configs_from_env,
    endpoint_probe,
    run_serving_smoke,
    runtime_launch_plan,
    runtime_preflight,
)


class PerformanceIQSdkTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="piq-python-test-")
        self.artifact_path = os.path.join(self.tmp_dir, "summary.json")
        with open(self.artifact_path, "w", encoding="utf-8") as handle:
            handle.write('{"ok":true}\n')

    def tearDown(self):
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
        self.assertFalse(result["snapshotBacked"])

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
            engine={"engine": "sglang", "baseUrl": "http://127.0.0.1:30000"},
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
        self.assertEqual(result["manifest"]["producer"]["tool"], "sglang-serving-producer")
        self.assertEqual(result["manifest"]["runtime"]["framework"], "SGLang")
        self.assertEqual(result["manifest"]["sourceType"], "other-measured-producer")
        self.assertTrue(os.path.exists(result["artifactPath"]))
        self.assertEqual(result["measurements"][0]["runtimeEngine"], "sglang")
        self.assertEqual(result["measurements"][0]["completionTokens"], 14)
        self.assertTrue(validate_run(result["runInput"])["ok"])

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
        preflight = runtime_preflight([], ["vllm (PIQ_VLLM_URL)"], model=laptop_smoke_model())

        self.assertFalse(preflight["ready"])
        self.assertEqual(preflight["missingEngineUrls"], ["vllm (PIQ_VLLM_URL)"])
        self.assertEqual(preflight["endpoints"], [])
        self.assertIn("vllmCommand", preflight["localRuntime"])
        self.assertIn("python", preflight["host"])
        self.assertIn("freeGiB", preflight["storage"])
        self.assertEqual(preflight["launchPlan"]["model"], laptop_smoke_model())
        self.assertEqual(preflight["launchPlan"]["endpointEnv"]["PIQ_VLLM_URL"], "http://127.0.0.1:8000")

    def test_serving_smoke_launch_plan_includes_all_engine_commands(self):
        plan = runtime_launch_plan(laptop_smoke_model())

        self.assertEqual(set(plan["engines"].keys()), {"vllm", "sglang", "tensorrt-llm"})
        self.assertIn("vllm serve", plan["engines"]["vllm"]["serve"])
        self.assertIn("sglang.launch_server", plan["engines"]["sglang"]["serve"])
        self.assertIn("trtllm-serve", plan["engines"]["tensorrt-llm"]["serve"])
        self.assertIn("freeGiB", plan["storage"])
        self.assertEqual(plan["endpointEnv"]["PIQ_SGLANG_URL"], "http://127.0.0.1:30000")

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


if __name__ == "__main__":
    unittest.main()
