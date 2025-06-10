#!/bin/bash
# Multi-node GRPO training for MedGemma 27B
# Run this script on every node; it detects its NODE_RANK from HOSTFILE.

HOSTFILE=${HOSTFILE:-$(dirname "$0")/hosts.txt}
if [ ! -f "$HOSTFILE" ]; then
  echo "Host file $HOSTFILE not found" >&2
  exit 1
fi

NNODES=$(wc -l < "$HOSTFILE")
MASTER_ADDR=$(head -n1 "$HOSTFILE")
LOCAL_IP=$(hostname -I | awk '{print $1}')
NODE_RANK=0
rank=0
while read -r ip; do
  if [ "$ip" = "$LOCAL_IP" ]; then
    NODE_RANK=$rank
    break
  fi
  rank=$((rank+1))
done < "$HOSTFILE"

if [ -z "$NODE_RANK" ]; then
  echo "Local IP $LOCAL_IP not found in $HOSTFILE" >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPU_COUNT-1)))}
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${GPU_COUNT}
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=INFO

swift rlhf \
  --rlhf_type grpo \
  --model google/medgemma-27b-text-it \
  --reward_funcs accuracy \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_max_model_len 4096 \
  --train_type full \
  --torch_dtype bfloat16 \
  --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT#5000' \
  --max_completion_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 2 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 4096 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --num_generations 8 \
  --system 'examples/train/grpo/prompt.txt' \
  --deepspeed zero2 \
  --log_completions true
