import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from performance_iq_sdk import PerformanceIQ, build_manifest, validate_run


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
            "confidentiality": "internal-full",
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
        self.assertEqual(manifest["store"]["rowProof"][0]["campaignId"], "campaign-python-test")

    def test_validate_run_live_proof_classification(self):
        result = validate_run(self.input())

        self.assertTrue(result["ok"])
        self.assertTrue(result["liveProofReady"])
        self.assertTrue(result["freshRun"])
        self.assertFalse(result["snapshotBacked"])

    def test_customer_safe_fails_closed(self):
        result = validate_run(self.input(confidentiality="customer-safe"))

        self.assertFalse(result["ok"])
        self.assertIn("internal-full", " ".join(result["errors"]))

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


if __name__ == "__main__":
    unittest.main()
