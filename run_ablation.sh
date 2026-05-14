#!/bin/bash
# Sequential 2x2 ablation: (conv3d | laguerre) x (linear | poly)
# Usage: bash run_ablation.sh [gpus] [dataset] [avg_splits]
#   gpus:       comma-separated GPU ids (default: 4,5,6,7)
#   dataset:    ucf101 | hmdb51 (default: ucf101)
#   avg_splits: 1 = average over all 3 splits (default: 0)

GPUS=${1:-4,5,6,7}
DATASET=${2:-ucf101}
AVG_SPLITS=${3:-0}
N_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
LR=$(python3 -c "print(f'{int($N_GPUS) * 4e-4:.0e}')")

echo "==> GPUs: $GPUS  (${N_GPUS}x)  dataset: $DATASET  lr: $LR  avg_splits: $AVG_SPLITS"
echo ""

AVG_SPLITS_FLAG=""
[ "$AVG_SPLITS" = "1" ] && AVG_SPLITS_FLAG="--avg_splits"

COMMON="
  --dataset $DATASET --model r3d_adapted
  --adapter_bottleneck_rank 64
  --batch_size 32 --lr $LR --epochs 50 --warmup_epochs 5
  --no_wandb $AVG_SPLITS_FLAG
"

TORCHRUN="NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS \
  torchrun --nproc_per_node=$N_GPUS train_par.py"

run() {
    local name=$1; shift
    echo "==> Starting: $name"
    eval $TORCHRUN --run_name ${DATASET}_${name} $COMMON "$@"
    if [ $? -ne 0 ]; then
        echo "ERROR: $name failed, stopping."
        exit 1
    fi
    echo "==> Done: $name"
    echo ""
}

run lora            --adapter_conv conv3d   --adapter_mode linear
run lora_laguerre   --adapter_conv laguerre --adapter_mode linear
run poly_conv3d     --adapter_conv conv3d   --adapter_mode poly
run poly_laguerre   --adapter_conv laguerre --adapter_mode poly

echo "==> All runs complete."
