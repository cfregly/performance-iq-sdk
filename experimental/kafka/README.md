# Experimental Kafka Exporter

This package is an optional, post-capture exporter for the stable Performance
IQ serving-event JSONL contract. It is not part of the product ingestion path,
the serving smoke image, or the Performance IQ Helm chart.

The producer always measures vLLM, SGLang, or TensorRT-LLM directly and writes
the immutable JSONL event log only after timing, token, artifact, and telemetry
capture has completed. This exporter may replay that log to Kafka when a future
deployment has a demonstrated need for multiple independent consumers or
replayable fan-out. It must never be placed between the measuring client and a
serving engine.

## Install and run

```bash
python -m pip install ./experimental/kafka/python
piq-kafka-publish-serving-events \
  --event-log ./performance-iq-output/serving-producers/serving-events.jsonl \
  --bootstrap-servers kafka-1:9092,kafka-2:9092
```

The publisher validates every JSONL event ID before transmission, uses
`acks=all`, and records an at-least-once publication receipt. Consumers must
deduplicate on the stable `eventId`.

## Promotion gate

Do not add Kafka to the product Helm chart or make it a required SDK dependency
until a buyer requirement or two independently-operated consumers justify the
operational cost. The durable product source of truth remains the ingestion
service's Iceberg/Nessie commit, not a Kafka offset.
