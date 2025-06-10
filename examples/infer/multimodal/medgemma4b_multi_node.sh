#!/bin/bash
# Multi-node inference for MedGemma-4B-IT using ms-swift.
# Adjust NODE_RANK and MASTER_ADDR for each node.

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
    --infer_backend pt \
    --val_dataset /path/to/medgemma_dataset.jsonl \
    --torch_dtype bfloat16 \
    --max_batch_size 1 \
    --max_new_tokens 512 \
    --system "$(cat examples/infer/multimodal/medgemma_prompt.txt)"
