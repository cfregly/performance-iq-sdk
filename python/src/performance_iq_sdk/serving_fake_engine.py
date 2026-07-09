from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from performance_iq_sdk.producers.serving import laptop_smoke_model


TOKEN_EVENTS = [
    {"text": "o", "token_id": 101, "logprob": -0.10},
    {"text": "k", "token_id": 202, "logprob": -0.20},
    {"text": ".", "token_id": 303, "logprob": -0.30},
]


class FakeEngineState:
    def __init__(self, engine: str, model: str) -> None:
        self.engine = engine
        self.model = model
        self.request_count = 0
        self.lock = threading.Lock()

    def increment_requests(self) -> int:
        with self.lock:
            self.request_count += 1
            return self.request_count

    def current_requests(self) -> int:
        with self.lock:
            return self.request_count


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "PerformanceIQFakeEngine/1.0"

    @property
    def state(self) -> FakeEngineState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._write_json(200, {"object": "list", "data": [{"id": self.state.model, "object": "model"}]})
            return
        if self.path == "/metrics" and self.state.engine == "tensorrt-llm":
            self._write_tensorrt_json_metrics()
            return
        if self.path in {"/metrics", "/prometheus/metrics"}:
            self._write_metrics()
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("content-length") or "0")
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            request = {}
        if request.get("model") != self.state.model:
            self._write_json(400, {"error": f"model {request.get('model')} is not served"})
            return
        request_number = self.state.increment_requests()
        if request.get("stream") is False:
            self._write_json(200, self._completion_body(request_number))
            return
        self._write_stream(request_number)

    def _completion_body(self, request_number: int) -> dict[str, Any]:
        return {
            "id": f"{self.state.engine}-fake-{request_number}",
            "object": "chat.completion",
            "model": self.state.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok."},
                "finish_reason": "stop",
                "logprobs": {"content": [self._token_logprob(item) for item in TOKEN_EVENTS]},
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": len(TOKEN_EVENTS), "total_tokens": 8 + len(TOKEN_EVENTS)},
        }

    def _write_stream(self, request_number: int) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for item in TOKEN_EVENTS:
            body = {
                "id": f"{self.state.engine}-fake-{request_number}",
                "object": "chat.completion.chunk",
                "model": self.state.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": item["text"]},
                    "logprobs": {"content": [self._token_logprob(item)]},
                }],
            }
            self._write_sse(body)
        self._write_sse({
            "id": f"{self.state.engine}-fake-{request_number}",
            "object": "chat.completion.chunk",
            "model": self.state.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": len(TOKEN_EVENTS), "total_tokens": 8 + len(TOKEN_EVENTS)},
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse(self, body: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(body)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _token_logprob(self, item: dict[str, Any]) -> dict[str, Any]:
        token = str(item["text"])
        token_id = int(item["token_id"])
        return {
            "token": token,
            "token_id": token_id,
            "logprob": float(item["logprob"]),
            "bytes": list(token.encode("utf-8")),
            "top_logprobs": [
                {
                    "token": token,
                    "token_id": token_id,
                    "logprob": float(item["logprob"]),
                    "bytes": list(token.encode("utf-8")),
                },
                {
                    "token": token.upper(),
                    "token_id": token_id + 1,
                    "logprob": float(item["logprob"]) - 2.0,
                    "bytes": list(token.upper().encode("utf-8")),
                },
            ],
        }

    def _write_metrics(self) -> None:
        count = self.state.current_requests()
        if self.state.engine == "sglang":
            engine_metrics = [
                f'sglang:time_to_first_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'sglang:time_to_first_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.125}',
                f'sglang:time_per_output_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'sglang:time_per_output_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.015}',
                f'sglang:e2e_request_latency_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'sglang:e2e_request_latency_seconds_sum{{model_name="{self.state.model}"}} {count * 0.240}',
                f'sglang:request_queue_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'sglang:request_queue_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.002}',
                f'sglang:per_stage_req_latency_seconds_count{{stage="prefill_forward",model_name="{self.state.model}"}} {count}',
                f'sglang:per_stage_req_latency_seconds_sum{{stage="prefill_forward",model_name="{self.state.model}"}} {count * 0.070}',
                f'sglang:per_stage_req_latency_seconds_count{{stage="decode_forward",model_name="{self.state.model}"}} {count}',
                f'sglang:per_stage_req_latency_seconds_sum{{stage="decode_forward",model_name="{self.state.model}"}} {count * 0.110}',
                f'sglang:num_running_reqs{{model_name="{self.state.model}"}} 1',
                f'sglang:num_queue_reqs{{model_name="{self.state.model}"}} 0',
                f'sglang:token_usage{{model_name="{self.state.model}"}} {0.05 + count * 0.01}',
                f'sglang:cache_hit_rate{{model_name="{self.state.model}"}} 0.2',
                f'sglang:prompt_tokens_cached_total{{model_name="{self.state.model}"}} {count * 3}',
                f'sglang:request_prefill_kv_computed_tokens_sum{{model_name="{self.state.model}"}} {count * 8}',
            ]
        elif self.state.engine == "tensorrt-llm":
            engine_metrics = [
                f'trtllm_time_to_first_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_time_to_first_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.125}',
                f'trtllm_time_per_output_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_time_per_output_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.015}',
                f'trtllm_e2e_request_latency_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_e2e_request_latency_seconds_sum{{model_name="{self.state.model}"}} {count * 0.240}',
                f'trtllm_request_queue_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_request_queue_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.002}',
                f'trtllm_request_prefill_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_request_prefill_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.070}',
                f'trtllm_request_decode_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'trtllm_request_decode_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.110}',
                f'trtllm_num_active_requests{{model_name="{self.state.model}"}} 1',
                f'trtllm_num_queued_requests{{model_name="{self.state.model}"}} 0',
                f'trtllm_kv_cache_utilization{{model_name="{self.state.model}"}} {0.05 + count * 0.01}',
                f'trtllm_kv_cache_hit_rate{{model_name="{self.state.model}"}} 0.2',
                f'trtllm_prompt_tokens_cached_total{{model_name="{self.state.model}"}} {count * 3}',
                f'trtllm_request_prefill_kv_computed_tokens_sum{{model_name="{self.state.model}"}} {count * 8}',
            ]
        else:
            engine_metrics = [
                f'vllm:time_to_first_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:time_to_first_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.125}',
                f'vllm:request_time_per_output_token_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:request_time_per_output_token_seconds_sum{{model_name="{self.state.model}"}} {count * 0.015}',
                f'vllm:e2e_request_latency_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:e2e_request_latency_seconds_sum{{model_name="{self.state.model}"}} {count * 0.240}',
                f'vllm:request_queue_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:request_queue_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.002}',
                f'vllm:request_prefill_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:request_prefill_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.070}',
                f'vllm:request_decode_time_seconds_count{{model_name="{self.state.model}"}} {count}',
                f'vllm:request_decode_time_seconds_sum{{model_name="{self.state.model}"}} {count * 0.110}',
                f'vllm:num_requests_running{{model_name="{self.state.model}"}} 1',
                f'vllm:num_requests_waiting{{model_name="{self.state.model}"}} 0',
                f'vllm:kv_cache_usage_perc{{model_name="{self.state.model}"}} {0.05 + count * 0.01}',
                f'vllm:prefix_cache_queries_total{{model_name="{self.state.model}"}} {count * 10}',
                f'vllm:prefix_cache_hits_total{{model_name="{self.state.model}"}} {count * 2}',
                f'vllm:prompt_tokens_cached_total{{model_name="{self.state.model}"}} {count * 3}',
                f'vllm:request_prefill_kv_computed_tokens_sum{{model_name="{self.state.model}"}} {count * 8}',
            ]
        metrics = "\n".join([
            *engine_metrics,
            f'DCGM_FI_DEV_POWER_USAGE{{gpu="0",modelName="{self.state.model}"}} {120 + count}',
            f'DCGM_FI_DEV_GPU_UTIL{{gpu="0",modelName="{self.state.model}"}} {50 + count}',
            f'DCGM_FI_DEV_MEM_COPY_UTIL{{gpu="0",modelName="{self.state.model}"}} {20 + count}',
            f'DCGM_FI_DEV_GPU_TEMP{{gpu="0",modelName="{self.state.model}"}} {60 + count}',
            f'DCGM_FI_DEV_SM_CLOCK{{gpu="0",modelName="{self.state.model}"}} {1800 + count}',
            f'DCGM_FI_DEV_MEM_CLOCK{{gpu="0",modelName="{self.state.model}"}} {5000 + count}',
            f'DCGM_FI_DEV_FB_USED{{gpu="0",modelName="{self.state.model}"}} {4096 + count}',
            f'DCGM_FI_DEV_FB_FREE{{gpu="0",modelName="{self.state.model}"}} {8192 - count}',
            f'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{gpu="0",modelName="{self.state.model}"}} {count * 1500}',
            f'DCGM_FI_PROF_SM_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.50 + count * 0.01}',
            f'DCGM_FI_PROF_DRAM_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.25 + count * 0.01}',
            f'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.70 + count * 0.01}',
            f'DCGM_FI_PROF_PIPE_FP64_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.02 + count * 0.001}',
            f'DCGM_FI_PROF_PIPE_FP32_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.35 + count * 0.01}',
            f'DCGM_FI_PROF_PIPE_FP16_ACTIVE{{gpu="0",modelName="{self.state.model}"}} {0.55 + count * 0.01}',
            f'DCGM_FI_DEV_PCIE_TX_THROUGHPUT{{gpu="0",modelName="{self.state.model}"}} {2100 + count}',
            f'DCGM_FI_DEV_PCIE_RX_THROUGHPUT{{gpu="0",modelName="{self.state.model}"}} {3200 + count}',
            f'DCGM_FI_PROF_PCIE_TX_BYTES{{gpu="0",modelName="{self.state.model}"}} {count * 6000}',
            f'DCGM_FI_PROF_PCIE_RX_BYTES{{gpu="0",modelName="{self.state.model}"}} {count * 8000}',
            f'DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{gpu="0",modelName="{self.state.model}"}} {count * 2}',
            f'DCGM_FI_PROF_NVLINK_TX_BYTES{{gpu="0",modelName="{self.state.model}"}} {count * 12000}',
            f'DCGM_FI_PROF_NVLINK_RX_BYTES{{gpu="0",modelName="{self.state.model}"}} {count * 13000}',
            f'DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL{{gpu="0",modelName="{self.state.model}"}} {950 + count}',
            f'DCGM_FI_DEV_ENC_UTIL{{gpu="0",modelName="{self.state.model}"}} {8 + count}',
            f'DCGM_FI_DEV_DEC_UTIL{{gpu="0",modelName="{self.state.model}"}} {11 + count}',
            f'DCGM_FI_DEV_XID_ERRORS{{gpu="0",modelName="{self.state.model}"}} {count}',
            f'DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{{gpu="0",modelName="{self.state.model}"}} {count * 2}',
            f'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{gpu="0",modelName="{self.state.model}"}} {count}',
            f'DCGM_FI_DEV_POWER_VIOLATION{{gpu="0",modelName="{self.state.model}"}} {count * 30}',
            f'DCGM_FI_DEV_THERMAL_VIOLATION{{gpu="0",modelName="{self.state.model}"}} {count * 40}',
        ]) + "\n"
        payload = metrics.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/plain; version=0.0.4")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_tensorrt_json_metrics(self) -> None:
        count = self.state.current_requests()
        payload = json.dumps([{
            "gpuMemUsage": 2_000_000_000 + count,
            "iterLatencyMS": 7 + count,
            "numActiveRequests": 1,
            "kvCacheStats": {
                "usedNumBlocks": 4 + count,
                "maxNumBlocks": 10,
                "cacheHitRate": 0.2,
            },
        }]).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FakeOpenAIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: FakeEngineState) -> None:
        super().__init__(server_address, FakeOpenAIHandler)
        self.state = state


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic OpenAI-compatible fake serving engine for Performance IQ smoke tests.")
    parser.add_argument("--engine", default="vllm", choices=["vllm", "sglang", "tensorrt-llm"])
    parser.add_argument("--model", default=laptop_smoke_model())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    server = FakeOpenAIServer((args.host, args.port), FakeEngineState(args.engine, args.model))
    host, port = server.server_address
    print(json.dumps({"engine": args.engine, "model": args.model, "baseUrl": f"http://{host}:{port}"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
