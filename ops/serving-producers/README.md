# Serving Producer Operations

This folder is the operator path for proving real vLLM, SGLang, and
TensorRT-LLM serving engines into Performance IQ.

The SDK adapters are endpoint-driven: each engine must expose an
OpenAI-compatible API with:

- `GET /v1/models`
- `POST /v1/chat/completions`

The smoke runner uses the same model and prompt across all three engines,
preflights `/v1/models`, sends completion requests, writes per-engine
normalized summary artifacts, submits producer runs, and verifies the fixed
Performance IQ dashboard query surfaces.

## Files

- `performance-iq-serving.env.example` - environment contract for engines and
  Performance IQ.
- `docker-compose.nvidia.yaml` - NVIDIA/Linux local stack template for the
  three serving engines.
- `kubernetes-smoke-job.yaml` - cluster job template for running the smoke
  from inside the same network as deployed engines. Replace
  `performance-iq-sdk:latest` with your built SDK image before applying it.
- `run-smoke.sh` - local wrapper around `performance_iq_sdk.serving_smoke`.

## Local NVIDIA Host

Copy the env template and set the TensorRT-LLM image:

```bash
cp ops/serving-producers/performance-iq-serving.env.example .env.serving-producers
$EDITOR .env.serving-producers
```

Start the three engines:

```bash
docker compose \
  --env-file .env.serving-producers \
  -f ops/serving-producers/docker-compose.nvidia.yaml \
  up -d vllm sglang tensorrt-llm
```

Run the model-aware endpoint preflight:

```bash
set -a
. ./.env.serving-producers
set +a

bash ops/serving-producers/run-smoke.sh preflight
```

Submit producer runs and verify dashboard materialization:

```bash
bash ops/serving-producers/run-smoke.sh smoke
```

Success requires:

- all three `/v1/models` endpoints serve `PIQ_SERVING_MODEL`;
- each engine returns successful chat completions;
- Performance IQ accepts all three producer runs;
- `price_performance`, `capacity_best`, `campaign_provenance`, and
  `run_details` row counts increase;
- submitted campaign IDs appear in both `campaign_provenance` and
  `run_details`.

## Mac / Apple Silicon

Use the launch plan for local vLLM and SGLang source-build commands:

```bash
bash ops/serving-producers/run-smoke.sh launch-plan
```

TensorRT-LLM requires a Linux x86_64/aarch64 target with supported NVIDIA GPUs.
For a Mac workflow, run TensorRT-LLM remotely and set
`PIQ_TENSORRT_LLM_URL` to the reachable OpenAI-compatible endpoint.

## Remote Engines

When engines are already deployed, skip compose and set only the endpoint env:

```bash
export PIQ_BASE_URL=https://performance-iq.example.com
export PIQ_VLLM_URL=https://vllm.example.com
export PIQ_SGLANG_URL=https://sglang.example.com
export PIQ_TENSORRT_LLM_URL=https://trtllm.example.com
export PIQ_SERVING_MODEL=Qwen/Qwen2.5-0.5B-Instruct

bash ops/serving-producers/run-smoke.sh preflight
bash ops/serving-producers/run-smoke.sh smoke
```

Do not use `--skip-preflight` for proof runs. It exists only for targeted
debugging of nonstandard endpoints.
