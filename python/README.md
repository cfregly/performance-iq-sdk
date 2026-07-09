# performance-iq-sdk for Python

Python SDK for building, validating, and submitting Performance IQ producer
results packets. See the repository root README for full examples.

## Serving producer smoke

After installing the package, run all configured OpenAI-compatible serving
engines with:

```bash
PIQ_BASE_URL=http://127.0.0.1:3002 \
PIQ_VLLM_URL=http://127.0.0.1:8000 \
PIQ_SGLANG_URL=http://127.0.0.1:30000 \
PIQ_TENSORRT_LLM_URL=http://127.0.0.1:8001 \
piq-serving-smoke --query-dashboard
```

From this checkout, use:

```bash
PYTHONPATH=src python -m performance_iq_sdk.serving_smoke --query-dashboard
```

The command fails unless all three engine URLs are configured and pass the
model-aware `/v1/models` preflight, sends the same model and prompt to each
runtime, writes normalized summary artifacts, submits producer runs, and
verifies the fixed Performance IQ dashboard query surfaces. The normalized
summary artifact for each engine includes the endpoint preflight evidence used
for that run. Use `--skip-preflight` only for targeted endpoint debugging.

Use `--preflight-only` to check local runtime availability and configured
`/v1/models` endpoints without sending inference requests or submitting runs.
The preflight validates the requested model when the endpoint returns a
standard OpenAI-compatible model list.

Use `--launch-plan-only` to print host-aware launch commands and endpoint env
vars for vLLM, SGLang, and TensorRT-LLM:

```bash
PYTHONPATH=src python -m performance_iq_sdk.serving_smoke \
  --launch-plan-only \
  --model Qwen/Qwen2.5-0.5B-Instruct
```
