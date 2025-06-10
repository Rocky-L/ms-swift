#!/bin/bash
# Multi-node inference for MedGemma-4B-IT using ms-swift.
# Adjust ``NODE_RANK`` and ``MASTER_ADDR`` for each node.

# This version uses ``vllm`` as the backend to enable efficient inference.
# The ``tensor_parallel_size`` is set to the number of GPUs per node.
# ``gpu_memory_utilization`` and ``max_model_len`` can be tuned based on the
# available hardware.

nnodes=2
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
swift infer \
    --model google/medgemma-4b-it \
    --infer_backend vllm \
    --val_dataset /path/to/medgemma_dataset.jsonl \
    --torch_dtype bfloat16 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --tensor_parallel_size $((nnodes * nproc_per_node)) \
    --max_new_tokens 512 \
    --system "$(cat examples/infer/multimodal/medgemma_prompt.txt)"
