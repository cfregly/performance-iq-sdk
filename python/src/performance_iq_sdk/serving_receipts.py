from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

REQUEST_RECEIPT_SCHEMA_VERSION = "performance-iq.serving-request-receipt.v1"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SAFE_REQUEST_HEADERS = {
    "content-type",
    "x-performance-iq-engine",
    "x-performance-iq-campaign-id",
    "x-performance-iq-run-id",
    "x-performance-iq-request-id",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def filtered_forward_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }


def safe_request_headers(headers: Any) -> dict[str, str]:
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in SAFE_REQUEST_HEADERS
    }


def write_receipt(receipt_log: str, receipt: dict[str, Any]) -> None:
    parent = os.path.dirname(receipt_log)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(receipt_log, "a", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True)
        handle.write("\n")


def load_receipts(receipt_log: str) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    with open(receipt_log, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                receipt = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"receipt log line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(receipt, dict):
                raise ValueError(f"receipt log line {line_number} must be a JSON object")
            receipts.append(receipt)
    return receipts


def receipt_ids_by_engine(receipts: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_engine: dict[str, set[str]] = {}
    for receipt in receipts:
        if receipt.get("schemaVersion") != REQUEST_RECEIPT_SCHEMA_VERSION:
            continue
        engine = receipt.get("engine")
        request_id = receipt.get("requestId")
        if isinstance(engine, str) and isinstance(request_id, str) and request_id:
            by_engine.setdefault(engine, set()).add(request_id)
    return by_engine


def _copy_response_headers(handler: BaseHTTPRequestHandler, headers: Any) -> None:
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
            continue
        handler.send_header(key, value)
    handler.send_header("connection", "close")


def _relay_body(handler: BaseHTTPRequestHandler, source: Any) -> int:
    total = 0
    while True:
        chunk = source.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if handler.command != "HEAD":
            handler.wfile.write(chunk)
            handler.wfile.flush()
    return total


def make_recording_proxy_handler(engine: str, target_base_url: str, receipt_log: str):
    target_base = normalize_base_url(target_base_url)

    class RecordingProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: N802 - BaseHTTPRequestHandler API
            return

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self._proxy()

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self._proxy()

        def do_HEAD(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self._proxy()

        def _proxy(self) -> None:
            started = time.perf_counter()
            length = int(self.headers.get("content-length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            target_url = f"{target_base}{self.path}"
            status = 502
            response_bytes = 0
            response_headers: Any = {}
            error: str | None = None
            try:
                request = urllib.request.Request(
                    target_url,
                    data=body if body or self.command not in {"GET", "HEAD"} else None,
                    headers=filtered_forward_headers(self.headers),
                    method=self.command,
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = response.status
                    response_headers = response.headers
                    self.send_response(status)
                    _copy_response_headers(self, response_headers)
                    self.end_headers()
                    response_bytes = _relay_body(self, response)
            except urllib.error.HTTPError as exc:
                status = exc.code
                response_headers = exc.headers
                self.send_response(status)
                _copy_response_headers(self, response_headers)
                self.end_headers()
                response_bytes = _relay_body(self, exc)
            except Exception as exc:
                error = str(exc)
                response_body = json.dumps({"error": error}).encode("utf-8")
                response_headers = {"content-type": "application/json"}
                response_bytes = len(response_body)
                self.send_response(status)
                _copy_response_headers(self, response_headers)
                self.send_header("content-length", str(len(response_body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(response_body)
                    self.wfile.flush()
            finally:
                self.close_connection = True

            safe_headers = safe_request_headers(self.headers)
            write_receipt(receipt_log, {
                "schemaVersion": REQUEST_RECEIPT_SCHEMA_VERSION,
                "recordedAtUtc": utc_now_iso(),
                "engine": engine,
                "requestId": safe_headers.get("x-performance-iq-request-id"),
                "campaignId": safe_headers.get("x-performance-iq-campaign-id"),
                "runId": safe_headers.get("x-performance-iq-run-id"),
                "method": self.command,
                "path": self.path,
                "targetUrl": target_url,
                "status": status,
                "latencyMs": (time.perf_counter() - started) * 1000,
                "requestBytes": len(body),
                "responseBytes": response_bytes,
                "requestHeaders": safe_headers,
                **({"error": error} if error else {}),
            })

    return RecordingProxyHandler


def recording_proxy_server(
    *,
    engine: str,
    target_base_url: str,
    receipt_log: str,
    listen_host: str = "127.0.0.1",
    listen_port: int = 0,
) -> ThreadingHTTPServer:
    handler = make_recording_proxy_handler(engine, target_base_url, receipt_log)
    return ThreadingHTTPServer((listen_host, listen_port), handler)


def serve_recording_proxy(
    *,
    engine: str,
    target_base_url: str,
    receipt_log: str,
    listen_host: str = "127.0.0.1",
    listen_port: int,
) -> None:
    server = recording_proxy_server(
        engine=engine,
        target_base_url=target_base_url,
        receipt_log=receipt_log,
        listen_host=listen_host,
        listen_port=listen_port,
    )
    host, port = server.server_address
    print(json.dumps({
        "schemaVersion": "performance-iq.serving-request-recorder.v1",
        "engine": engine,
        "listenUrl": f"http://{host}:{port}",
        "targetBaseUrl": normalize_base_url(target_base_url),
        "receiptLog": receipt_log,
    }, indent=2), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record request receipts while proxying Performance IQ serving producer traffic.")
    parser.add_argument("--engine", required=True, choices=["vllm", "sglang", "tensorrt-llm"])
    parser.add_argument("--target-url", required=True, help="Real engine base URL to forward to, for example http://127.0.0.1:8000.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--receipt-log", required=True, help="JSONL path for request receipts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    serve_recording_proxy(
        engine=args.engine,
        target_base_url=args.target_url,
        receipt_log=args.receipt_log,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
