#!/bin/bash
# Run SFT experiments with different dataset sizes
# Usage: bash scripts/run_sft_experiments.sh

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Math-1.5B"
TRAIN_DATA="data/gsm8k/train.jsonl"
EVAL_DATA="data/gsm8k/test.jsonl"
OUTPUT_DIR="outputs/sft"
WANDB_PROJECT="sft-gsm8k"

# Training hyperparameters (tune these for best results)
LEARNING_RATE=1e-5
BATCH_SIZE=4
MICROBATCH_SIZE=2
NUM_EPOCHS=3
EVAL_EVERY=50
MAX_GEN_TOKENS=1024
NUM_EVAL_EXAMPLES=500

# Dataset sizes to experiment with
SIZES=(128 256 512 1024)

echo "====================================="
echo "Running SFT Experiments"
echo "====================================="
echo "Model: $MODEL_NAME"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "====================================="

# Run experiments for each dataset size
for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "====================================="
    echo "Training with $SIZE examples"
    echo "====================================="

    python scripts/sft_train.py \
        --model_name "$MODEL_NAME" \
        --train_data "$TRAIN_DATA" \
        --eval_data "$EVAL_DATA" \
        --num_examples $SIZE \
        --num_eval_examples $NUM_EVAL_EXAMPLES \
        --batch_size $BATCH_SIZE \
        --microbatch_size $MICROBATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --eval_every $EVAL_EVERY \
        --max_gen_tokens $MAX_GEN_TOKENS \
        --output_dir "$OUTPUT_DIR" \
        --use_wandb \
        --wandb_project "$WANDB_PROJECT" \
        --run_name "sft_n${SIZE}_lr${LEARNING_RATE}" \
        --policy_device cuda:0 \
        --vllm_device cuda:1
done

# Run with full dataset
echo ""
echo "====================================="
echo "Training with FULL dataset"
echo "====================================="

python scripts/sft_train.py \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --num_eval_examples $NUM_EVAL_EXAMPLES \
    --batch_size $BATCH_SIZE \
    --microbatch_size $MICROBATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --eval_every $EVAL_EVERY \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "sft_full_lr${LEARNING_RATE}" \
    --policy_device cuda:0 \
    --vllm_device cuda:1

# Run with filtered (correct only) dataset
echo ""
echo "====================================="
echo "Training with FILTERED (correct only) dataset"
echo "====================================="

python scripts/sft_train.py \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --num_eval_examples $NUM_EVAL_EXAMPLES \
    --filter_correct \
    --batch_size $BATCH_SIZE \
    --microbatch_size $MICROBATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --eval_every $EVAL_EVERY \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "sft_filtered_lr${LEARNING_RATE}" \
    --policy_device cuda:0 \
    --vllm_device cuda:1

echo ""
echo "====================================="
echo "All experiments completed!"
echo "====================================="
