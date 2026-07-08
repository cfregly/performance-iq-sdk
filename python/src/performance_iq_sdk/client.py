from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .models import build_envelope, validate_manifest, validate_run

Transport = Callable[[str, str, dict[str, str], bytes | None], Any]


class PerformanceIQError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class PerformanceIQ:
    def __init__(self, base_url: str, token: str | None = None, transport: Transport | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.transport = transport

    def validate_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return validate_run(payload)

    def submit_run(self, payload: dict[str, Any], *, idempotency_key: str | None = None, dry_run: bool = False) -> Any:
        local = validate_run(payload)
        if not local["ok"]:
            raise PerformanceIQError("run failed local validation: " + "; ".join(local["errors"]))
        if dry_run:
            return local
        manifest = local["manifest"]
        envelope = build_envelope(manifest, payload.get("measurements"))
        return self._post_json(
            "/api/v1/evidence/runs",
            envelope,
            {"idempotency-key": idempotency_key or manifest["campaign"]["runId"]},
        )

    def submit_manifest(self, manifest_path: str, *, idempotency_key: str | None = None, dry_run: bool = False) -> Any:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        local = validate_manifest(manifest)
        if not local["ok"]:
            raise PerformanceIQError("manifest failed local validation: " + "; ".join(local["errors"]))
        if dry_run:
            return local
        return self._post_json(
            "/api/v1/evidence/runs",
            build_envelope(manifest),
            {"idempotency-key": idempotency_key or manifest["campaign"]["runId"]},
        )

    def get_run_status(self, run_id: str) -> Any:
        return self._get_json(f"/api/v1/evidence/runs/{urllib.parse.quote(run_id, safe='')}")

    def get_evidence_status(self, request: dict[str, Any]) -> Any:
        if _find_disallowed_key(request):
            raise PerformanceIQError("evidence status request must not include SQL or query keys")
        return self._post_json("/api/downstream/evidence-status", request)

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _post_json(self, path: str, body: Any, extra_headers: dict[str, str] | None = None) -> Any:
        return self._request("POST", path, self._headers(extra_headers), json.dumps(body).encode("utf-8"))

    def _get_json(self, path: str) -> Any:
        return self._request("GET", path, self._headers(), None)

    def _request(self, method: str, path: str, headers: dict[str, str], body: bytes | None) -> Any:
        url = f"{self.base_url}{path}"
        if self.transport:
            return self.transport(method, url, headers, body)
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8")
            raise PerformanceIQError(_error_message(payload) or exc.reason, exc.code) from exc


def _error_message(payload: str) -> str | None:
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return payload.strip() or None
    detail = body.get("detail") or body.get("error") or body.get("message")
    return detail if isinstance(detail, str) else json.dumps(detail) if detail else None


def _find_disallowed_key(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            match = _find_disallowed_key(item)
            if match:
                return match
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"sql", "queryName", "queries"}:
                return key
            match = _find_disallowed_key(child)
            if match:
                return match
    return None
