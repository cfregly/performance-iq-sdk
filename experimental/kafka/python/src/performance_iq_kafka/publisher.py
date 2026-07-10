from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

SERVING_EVENT_SCHEMA_VERSION = "performance-iq.serving-telemetry-event.v1"
SERVING_EVENT_DEFAULT_TOPIC = "performance-iq.serving.telemetry.v1"
KAFKA_PUBLICATION_SCHEMA_VERSION = "performance-iq.serving-kafka-publication.v1"

KafkaProducerFactory = Callable[..., Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _validation_errors(event: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    if event.get("schemaVersion") != SERVING_EVENT_SCHEMA_VERSION:
        errors.append(
            f"event log line {line_number} schemaVersion must be {SERVING_EVENT_SCHEMA_VERSION}."
        )
    if not isinstance(event.get("topic"), str) or not event["topic"]:
        errors.append(f"event log line {line_number} topic is required.")
    event_type = event.get("eventType")
    event_id = event.get("eventId")
    partition_key = event.get("partitionKey")
    payload = event.get("payload")
    if not isinstance(event_type, str) or not event_type:
        errors.append(f"event log line {line_number} eventType is required.")
    if not isinstance(event_id, str) or len(event_id) != 64:
        errors.append(f"event log line {line_number} eventId must be a 64-character digest.")
    if not isinstance(partition_key, str) or not partition_key:
        errors.append(f"event log line {line_number} partitionKey is required.")
    if not isinstance(payload, dict):
        errors.append(f"event log line {line_number} payload must be an object.")
    if (
        isinstance(event_type, str)
        and isinstance(event_id, str)
        and len(event_id) == 64
        and isinstance(partition_key, str)
        and isinstance(payload, dict)
    ):
        expected_event_id = _sha256_json(
            {
                "schemaVersion": event.get("schemaVersion"),
                "eventType": event_type,
                "partitionKey": partition_key,
                "payload": payload,
            }
        )
        if event_id != expected_event_id:
            errors.append(f"event log line {line_number} eventId digest does not match event payload.")
    return errors


def load_serving_event_log(event_log_path: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    with open(event_log_path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value:
                continue
            try:
                event = json.loads(value)
            except json.JSONDecodeError as exc:
                errors.append(f"event log line {line_number} is not valid JSON: {exc}")
                continue
            if not isinstance(event, dict):
                errors.append(f"event log line {line_number} must be a JSON object.")
                continue
            errors.extend(_validation_errors(event, line_number))
            events.append(event)
    if errors:
        raise ValueError("; ".join(errors))
    return events


def publish_serving_event_log(
    event_log_path: str,
    *,
    bootstrap_servers: str,
    topic: str | None = None,
    client_id: str = "performance-iq-serving-exporter",
    producer_factory: KafkaProducerFactory | None = None,
) -> dict[str, Any]:
    """Publish a validated, already-captured event log with at-least-once delivery."""
    if not bootstrap_servers:
        raise ValueError("bootstrap_servers is required to publish serving events to Kafka.")
    events = load_serving_event_log(event_log_path)
    if not events:
        raise ValueError(f"event log contains no publishable events: {event_log_path}")
    if producer_factory is None:
        try:
            from kafka import KafkaProducer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Kafka publication requires the experimental package. Install with "
                "`pip install ./experimental/kafka/python`."
            ) from exc
        producer_factory = KafkaProducer
    producer = producer_factory(
        bootstrap_servers=bootstrap_servers,
        client_id=client_id,
        acks="all",
        retries=5,
        max_in_flight_requests_per_connection=1,
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
        "schemaVersion": KAFKA_PUBLICATION_SCHEMA_VERSION,
        "eventLogPath": event_log_path,
        "bootstrapServers": bootstrap_servers,
        "clientId": client_id,
        "deliverySemantics": "at-least-once-with-event-id-deduplication",
        "acks": "all",
        "retries": 5,
        "publishedAtUtc": _utc_now_iso(),
        "publishedCount": len(events),
        "eventCounts": event_counts,
        "topicCounts": topic_counts,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a captured Performance IQ serving JSONL event log to Kafka."
    )
    parser.add_argument("--event-log", required=True)
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("PIQ_KAFKA_BOOTSTRAP_SERVERS"),
        required=os.getenv("PIQ_KAFKA_BOOTSTRAP_SERVERS") is None,
    )
    parser.add_argument("--topic", default=os.getenv("PIQ_KAFKA_TOPIC"))
    parser.add_argument(
        "--client-id",
        default=os.getenv("PIQ_KAFKA_CLIENT_ID", "performance-iq-serving-exporter"),
    )
    return parser


def main() -> int:
    args = parser().parse_args()
    receipt = publish_serving_event_log(
        args.event_log,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        client_id=args.client_id,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
