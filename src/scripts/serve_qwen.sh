#!/usr/bin/env bash
# Serve the fixed generator: Qwen3-Coder-30B-A3B-Instruct, BF16, 16k context.
# The served name must match config.yaml's "vllm/qwen3-coder-30b-a3b" (litellm
# strips the "vllm/" prefix and sends "qwen3-coder-30b-a3b").
# BF16 weights are ~57 GB -> one 80 GB A100, or TP=2 on smaller cards.
set -euo pipefail
export HF_HUB_OFFLINE=1
# FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel on first use, and
# these nodes have no nvcc (no /usr/local/cuda) and no internet for
# `vllm download-kernels`, so the engine died in warmup with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
# vLLM's own detection passes the arch check and misses the missing toolkit, so
# say it explicitly. forward_native is the reference top-k/top-p implementation,
# so temperature/top_p semantics -- and the fixed protocol -- are unchanged.
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=${MODEL:-$HOME/flash/col8385/models/Qwen3-Coder-30B-A3B-Instruct}
PORT=${PORT:-8000}
TP=${TP:-1}
# --kernel_config.enable_jit_warmup=False: vLLM ends warm-up by installing JIT
# monitor hooks, which imports tilelang; tilelang's bundled TVM blocks forever
# in tvm/tirx/__init__.py on this cluster, and vLLM's `except Exception` around
# that import catches a failure but not a hang. Startup sat silent for 28 min
# before this flag; with it the server is up ~100 s in. The warm-up it also
# skips logged "finished in 0.00s" anyway.
EXTRA=${SERVE_EXTRA:-}   # extra serve flags; not VLLM_* or vllm warns about it
vllm serve "$MODEL" \
  --served-model-name qwen3-coder-30b-a3b \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --kernel_config.enable_jit_warmup=False \
  --port "$PORT" $EXTRA 2>&1
