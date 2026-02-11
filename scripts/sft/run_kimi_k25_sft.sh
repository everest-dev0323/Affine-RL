#!/bin/bash
# =============================================================================
# Kimi K2.5 SFT Training Script
#
# This script supports TWO training approaches:
#
# 1. KTransformers + LLaMA-Factory (RECOMMENDED for limited VRAM)
#    - Requires: 4× RTX 4090 (96GB total), 2TB RAM, Intel CPU with AMX
#    - Use: ./run_kimi_k25_sft.sh --mode ktransformers --yaml_config your_config.yaml
#
# 2. DeepSpeed ZeRO-3 (requires substantial GPU resources)
#    - Requires: 8× H100 80GB or equivalent
#    - Use: ./run_kimi_k25_sft.sh --mode deepspeed --dataset_name your-dataset
#
# =============================================================================

set -e

# Default mode
MODE="ktransformers"  # Options: ktransformers, deepspeed

# Configuration
export WANDB_PROJECT="kimi-k25-sft"
export HF_HOME="${HF_HOME:-/scratch/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Model and data paths
MODEL_NAME="moonshotai/Kimi-K2.5"
DATASET_NAME=""  # Set your dataset name or path
DATASET_PATH=""  # Or set local path to JSONL file
OUTPUT_DIR="./outputs/kimi_k25_sft"

# Training configuration (for DeepSpeed mode)
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}

# LoRA configuration (for DeepSpeed mode)
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# DeepSpeed config
DS_CONFIG="config/ds_kimi_k25_zero3.json"

# KTransformers config
KT_YAML_CONFIG="scripts/sft/kimik25_kt_lora.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --yaml_config)
            KT_YAML_CONFIG="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE_PER_GPU="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --help|-h)
            echo "Kimi K2.5 SFT Training Script"
            echo ""
            echo "Usage:"
            echo "  ./run_kimi_k25_sft.sh --mode ktransformers [--yaml_config config.yaml]"
            echo "  ./run_kimi_k25_sft.sh --mode deepspeed --dataset_name your-dataset"
            echo ""
            echo "Options:"
            echo "  --mode          Training mode: 'ktransformers' (recommended) or 'deepspeed'"
            echo "  --yaml_config   YAML config file for KTransformers mode"
            echo "  --dataset_name  HuggingFace dataset name (for deepspeed mode)"
            echo "  --dataset_path  Local dataset path (for deepspeed mode)"
            echo "  --output_dir    Output directory"
            echo "  --num_gpus      Number of GPUs (for deepspeed mode)"
            echo "  --max_seq_length Maximum sequence length"
            echo "  --batch_size    Batch size per GPU"
            echo "  --lora_r        LoRA rank"
            echo ""
            echo "KTransformers mode (RECOMMENDED for limited VRAM):"
            echo "  Requires: 4× RTX 4090, 2TB RAM, Intel CPU with AMX"
            echo "  Edit scripts/sft/kimik25_kt_lora.yaml to configure your dataset"
            echo ""
            echo "DeepSpeed mode (requires substantial GPU resources):"
            echo "  Requires: 8× H100 80GB or equivalent"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===========================================================================
# MODE: KTransformers (RECOMMENDED for limited VRAM)
# ===========================================================================
if [[ "$MODE" == "ktransformers" ]]; then
    echo "=========================================="
    echo "Kimi K2.5 SFT Training (KTransformers)"
    echo "=========================================="
    echo "Mode: KTransformers + LLaMA-Factory"
    echo "Config: $KT_YAML_CONFIG"
    echo ""
    echo "Hardware Requirements:"
    echo "  - GPU: 4× RTX 4090 24GB (~48GB+ available VRAM)"
    echo "  - CPU: Intel with AMX support (Sapphire Rapids+)"
    echo "  - RAM: At least 2TB system memory"
    echo "=========================================="

    # Check if config file exists
    if [[ ! -f "$KT_YAML_CONFIG" ]]; then
        echo "Error: YAML config not found: $KT_YAML_CONFIG"
        echo "Please create the config file or specify --yaml_config path"
        exit 1
    fi

    # Check if llamafactory-cli is available
    if ! command -v llamafactory-cli &> /dev/null; then
        echo "Error: llamafactory-cli not found!"
        echo ""
        echo "Please install LLaMA-Factory and KTransformers:"
        echo "  git clone https://github.com/hiyouga/LlamaFactory.git"
        echo "  cd LlamaFactory && pip install -e ."
        echo "  pip install ktransformers-<version>.whl  # from KTransformers releases"
        exit 1
    fi

    echo "Starting KTransformers training..."
    USE_KT=1 llamafactory-cli train "$KT_YAML_CONFIG"

    echo "=========================================="
    echo "Training complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Verify the model with chat:"
    echo "   llamafactory-cli chat scripts/sft/kimik25_kt_inference.yaml"
    echo ""
    echo "2. For production serving with SGLang, see:"
    echo "   https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/SFT_Installation_Guide_KimiK2.5.md"

    exit 0
fi

# ===========================================================================
# MODE: DeepSpeed ZeRO-3 (requires substantial GPU resources)
# ===========================================================================
if [[ "$MODE" == "deepspeed" ]]; then
    # Validate dataset for DeepSpeed mode
    if [[ -z "$DATASET_NAME" && -z "$DATASET_PATH" ]]; then
        echo "Error: Either --dataset_name or --dataset_path must be provided for DeepSpeed mode"
        echo "Example: ./run_kimi_k2_sft.sh --mode deepspeed --dataset_name your-org/your-dataset"
        exit 1
    fi

    # Build dataset argument
    DATASET_ARG=""
    if [[ -n "$DATASET_NAME" ]]; then
        DATASET_ARG="--dataset_name $DATASET_NAME"
    else
        DATASET_ARG="--dataset_path $DATASET_PATH"
    fi

    echo "=========================================="
    echo "Kimi-K2-Instruct SFT Training (DeepSpeed ZeRO-3)"
    echo "=========================================="
    echo "Mode: DeepSpeed ZeRO-3 with CPU Offloading"
    echo "Model: $MODEL_NAME (Text-Only)"
    echo "Dataset: ${DATASET_NAME:-$DATASET_PATH}"
    echo "Output: $OUTPUT_DIR"
    echo "GPUs: $NUM_GPUS"
    echo "Max Seq Length: $MAX_SEQ_LENGTH"
    echo "Batch Size/GPU: $BATCH_SIZE_PER_GPU"
    echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
    echo "Effective Batch Size: $((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION * NUM_GPUS))"
    echo "LoRA Rank: $LORA_R"
    echo ""
    echo "Hardware Requirements:"
    echo "  - GPU: 8× H100 80GB (or equivalent ~640GB VRAM)"
    echo "  - RAM: >500GB system memory"
    echo "=========================================="

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Launch training with DeepSpeed
    deepspeed --num_gpus=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        scripts/sft/sft_kimi_k25.py \
        --model_name_or_path "$MODEL_NAME" \
        $DATASET_ARG \
        --output_dir "$OUTPUT_DIR" \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_EPOCHS \
        --warmup_ratio 0.1 \
        --bf16 true \
        --gradient_checkpointing true \
        --use_lora true \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 3 \
        --deepspeed "$DS_CONFIG" \
        --report_to wandb \
        --do_train true \
        --trust_remote_code true \
        --use_flash_attention_2 true \
        --torch_dtype bfloat16

    echo "Training complete! Model saved to: $OUTPUT_DIR"
    exit 0
fi

echo "Error: Invalid mode '$MODE'. Use 'ktransformers' or 'deepspeed'"
exit 1

