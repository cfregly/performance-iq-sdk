from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from performance_iq_kafka.publisher import (  # noqa: E402
    SERVING_EVENT_SCHEMA_VERSION,
    _sha256_json,
    publish_serving_event_log,
)


class PublisherTests(unittest.TestCase):
    def test_publishes_validated_event_log(self) -> None:
        payload = {"campaignId": "campaign-1", "runId": "run-1"}
        event = {
            "schemaVersion": SERVING_EVENT_SCHEMA_VERSION,
            "topic": "performance-iq.serving.telemetry.v1",
            "eventType": "serving.submission",
            "partitionKey": "campaign-1:run-1",
            "payload": payload,
        }
        event["eventId"] = _sha256_json(
            {
                "schemaVersion": event["schemaVersion"],
                "eventType": event["eventType"],
                "partitionKey": event["partitionKey"],
                "payload": event["payload"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            sent: list[dict[str, object]] = []

            class Future:
                def get(self, timeout=None):
                    return {"timeout": timeout}

            class Producer:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def send(self, topic, key=None, value=None):
                    sent.append({"topic": topic, "key": key, "value": value})
                    return Future()

                def flush(self, timeout=None):
                    return None

                def close(self, timeout=None):
                    return None

            receipt = publish_serving_event_log(
                path,
                bootstrap_servers="kafka:9092",
                producer_factory=Producer,
            )

        self.assertEqual(receipt["publishedCount"], 1)
        self.assertEqual(receipt["acks"], "all")
        self.assertEqual(receipt["deliverySemantics"], "at-least-once-with-event-id-deduplication")
        self.assertEqual(sent[0]["topic"], "performance-iq.serving.telemetry.v1")


if __name__ == "__main__":
    unittest.main()
