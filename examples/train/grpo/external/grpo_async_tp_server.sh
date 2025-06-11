#!/bin/bash
set -x

# Environment setup
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
# GRPO training might spend long time on inference
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2097152

# Parse command line arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case $1 in
            --PER_DEV_TRAIN_BS=*)
                PER_DEV_TRAIN_BS="${1#*=}"
                shift
                ;;
            --GRAD_ACC=*)
                GRAD_ACC="${1#*=}"
                shift
                ;;
            --PER_DEV_EVAL_BS=*)
                PER_DEV_EVAL_BS="${1#*=}"
                shift
                ;;
            --NUM_GENERATIONS=*)
                NUM_GENERATIONS="${1#*=}"
                shift
                ;;
            --NUM_INFER_WORKERS=*)
                NUM_INFER_WORKERS="${1#*=}"
                shift
                ;;
            --GRPO_MODE=*)
                GRPO_MODE="${1#*=}"
                shift
                ;;
            --DEEPSPEED=*)
                DEEPSPEED="${1#*=}"
                shift
                ;;
            --DDP_TIMEOUT=*)
                DDP_TIMEOUT="${1#*=}"
                shift
                ;;
            --VLLM_SERVER_TIMEOUT=*)
                VLLM_SERVER_TIMEOUT="${1#*=}"
                shift
                ;;
            --VLLM_SERVER_PORT=*)
                VLLM_SERVER_PORT="${1#*=}"
                shift
                ;;
            --VLLM_SERVER_HOST=*)
                VLLM_SERVER_HOST="${1#*=}"
                shift
                ;;
            *)
                echo "Unknown argument: $1"
                shift
                ;;
        esac
    done
}

parse_args "$@"

# 添加调试信息
echo "解析后的命令行参数:"
echo "  PER_DEV_TRAIN_BS: $PER_DEV_TRAIN_BS"
echo "  GRAD_ACC: $GRAD_ACC"
echo "  PER_DEV_EVAL_BS: $PER_DEV_EVAL_BS"
echo "  NUM_GENERATIONS: $NUM_GENERATIONS"
echo "  TP_SIZE: $TP_SIZE"
echo "  NUM_INFER_WORKERS: $NUM_INFER_WORKERS"
echo "  GRPO_MODE: $GRPO_MODE"
echo "  DEEPSPEED: $DEEPSPEED"
echo "  DDP_TIMEOUT: $DDP_TIMEOUT"

# Default values
PER_DEV_TRAIN_BS=${PER_DEV_TRAIN_BS:-1}
GRAD_ACC=${GRAD_ACC:-1}
PER_DEV_EVAL_BS=${PER_DEV_EVAL_BS:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-6}
GRPO_MODE=${GRPO_MODE:-auto}
DEEPSPEED=${DEEPSPEED:-zero3_offload}
DDP_TIMEOUT=${DDP_TIMEOUT:-3600}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-3600}

# Setup working paths
DIR=$(pwd)
MODEL_PATH="$PRIMUS_SOURCE_CHECKPOINT_DIR/$PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME"
DATA_PATH="$PRIMUS_DATA_PATH.jsonl"
export PRIMUS_TENSORBOARD_LOG_DIR="$PRIMUS_SAVE_CHECKPOINT_DIR/runs"

# Query GPUs with free memory >17G
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=memory.free \
  --format=csv,noheader,nounits | \
  awk '{if ($1 > 17000) print NR-1}' | paste -sd "," -)
if [ -z "$AVAILABLE_GPUS" ]; then
  echo "No GPU with sufficient free memory found!" && exit 1
fi
TOTAL_GPUS_COUNT=$(echo $AVAILABLE_GPUS | awk -F',' '{print NF}')
TOTAL_GPUS_LIST=$(echo $AVAILABLE_GPUS | tr ',' ' ')

# NUM_INFER_WORKERS auto config
if [ -z "$NUM_INFER_WORKERS" ]; then
    if [ "$GRPO_MODE" = "colocate" ]; then
        NUM_INFER_WORKERS=$TOTAL_GPUS_COUNT
    elif [ "$GRPO_MODE" = "async" ]; then
        NUM_INFER_WORKERS=2
    else
        if [ "$TOTAL_GPUS_COUNT" -le 4 ]; then
            GRPO_MODE="colocate"
            NUM_INFER_WORKERS=$TOTAL_GPUS_COUNT
        else
            GRPO_MODE="async"
            NUM_INFER_WORKERS=2
        fi
    fi
fi

# GPU allocation
if [ "$NUM_INFER_WORKERS" -eq "$TOTAL_GPUS_COUNT" ]; then
    GRPO_MODE="colocate"
    TRAIN_CUDA_LIST="$AVAILABLE_GPUS"
    VLLM_DEVICE="auto"
    ASYNC_GENERATE="false"
    echo "=== Using COLOCATE mode ==="
    echo "  - Training and inference share all GPUs"
    echo "  - TRAIN_CUDA_LIST: $TRAIN_CUDA_LIST"
    echo "  - VLLM_DEVICE: auto"
    echo "  - async_generate: false"
else
    GRPO_MODE="async"
    IDX=0
    TRAIN_CUDA_LIST=""
    VLLM_DEVICE_STRING=""
    for g in $TOTAL_GPUS_LIST; do
      if [ "$IDX" -ge $((TOTAL_GPUS_COUNT - NUM_INFER_WORKERS)) ]; then
        VLLM_DEVICE_STRING="${VLLM_DEVICE_STRING}cuda:${g} "
      else
        if [ -z "$TRAIN_CUDA_LIST" ]; then
          TRAIN_CUDA_LIST="${g}"
        else
          TRAIN_CUDA_LIST="${TRAIN_CUDA_LIST},${g}"
        fi
      fi
      IDX=$((IDX+1))
    done
    VLLM_DEVICE=$(echo "$VLLM_DEVICE_STRING" | xargs)
    ASYNC_GENERATE="true"
    echo "=== Using ASYNC mode ==="
    echo "  - Training and inference use separate GPUs"
    echo "  - TRAIN_CUDA_LIST: $TRAIN_CUDA_LIST"
    echo "  - VLLM_DEVICE: $VLLM_DEVICE"
    echo "  - async_generate: true"
fi

NPROC_PER_NODE=$(echo $TRAIN_CUDA_LIST | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES="$AVAILABLE_GPUS"
export NPROC_PER_NODE

export NNODES=$WORLD_SIZE
export NODE_RANK=$RANK

# Start external vLLM server when using async mode
if [ "$GRPO_MODE" = "async" ]; then
    VLLM_CUDA_IDS=$(echo "$VLLM_DEVICE" | sed 's/cuda://g' | tr ' ' ',')
    CUDA_VISIBLE_DEVICES=$VLLM_CUDA_IDS \
    swift rollout \
      --model $MODEL_PATH \
      --tensor_parallel_size $NUM_INFER_WORKERS \
      --data_parallel_size 1 \
      --gpu_memory_utilization 0.9 \
      --max_model_len 15000 \
      --max_num_seqs 128 \
      --host $VLLM_SERVER_HOST \
      --port $VLLM_SERVER_PORT \
      > "$PRIMUS_OUTPUT_DIR/vllm_server.log" 2>&1 &

    VLLM_SERVER_PID=$!
    timeout=$VLLM_SERVER_TIMEOUT

    while ! nc -z "$VLLM_SERVER_HOST" "$VLLM_SERVER_PORT"; do
        if [ "$timeout" -le 0 ]; then
            echo "❌ vLLM 未就绪"
            kill "$VLLM_SERVER_PID"
            exit 1
        fi

        timeout=$((timeout - 1))
        sleep 1
    done

    echo "✅ vLLM ($VLLM_SERVER_PID) ready"
fi

# 使用 swift rlhf 而不是直接调用 torchrun
# ms-swift 会根据环境变量自动判断是否使用 torchrun
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --train_type full \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0.005 \
    --torch_dtype bfloat16 \
    --external_plugins scripts/rl/v2/orm.py \
    --reward_funcs any_valid_drug_mention drug_only_f1 structure soft_overlong json_integrity \
    --reward_weights 1.0 3.0 1.0 1.0 1.0 \
    --num_train_epochs 4 \
    --max_length 9000 \
    --max_completion_length 4096 \
    --soft_cache_length 256 \
    --num_generations $NUM_GENERATIONS \
    --max_resample_times 3 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $VLLM_SERVER_HOST \
    --vllm_server_port $VLLM_SERVER_PORT \
    --vllm_server_timeout $VLLM_SERVER_TIMEOUT \
    --per_device_train_batch_size $PER_DEV_TRAIN_BS \
    --per_device_eval_batch_size $PER_DEV_EVAL_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --temperature 0.6 \
    --top_p 0.85 \
    --top_k 50 \
    --beta 0.01 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type bnpo \
    --dynamic_sample true \
    --overlong_filter true \
    --log_completions true \
    --gc_collect_after_offload true \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --eval_steps 50 \
    --output_dir $PRIMUS_SAVE_CHECKPOINT_DIR \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_iterations 1 \
    --async_generate $ASYNC_GENERATE \
    --deepspeed $DEEPSPEED \
    --ddp_timeout $DDP_TIMEOUT \
    --save_on_each_node false \
    --add_version false

echo "--------- FINISHED Training -----------"
