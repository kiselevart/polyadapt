#!/bin/bash
# Mode ablation: linear (LoRA) / relu / cross_poly / poly on each backbone.
#
# Usage: bash run_ablation.sh [gpus] [dataset] [avg_splits] [backbone]
#   gpus     : comma-separated GPU ids (default: 4,5,6,7)
#   dataset  : ucf101 | hmdb51 (default: ucf101)
#   avg_splits: 1 = average over splits 1-3 (default: 0)
#   backbone : all | r3d | swin3d_t | swin3d_s | swin3d_b (default: all)
#
# Bottleneck rank is chosen per backbone so that linear / relu / cross_poly
# all have approximately the same adapter param count within that backbone
# (within ~15%). Poly is structurally ~3x larger (expand_ch = 2*Q*dim) and
# cannot be matched without collapsing Q to 1.
#
# linear/relu/cross_poly run at 3× rank to roughly match poly's naturally larger
# expand_ch (= 2*Q*dim vs 2*Q). For linear the rank is btl; for relu/cross_poly it's Q.
#
# Approximate adapter params (poly: Q=4 btl=base; others: Q=12 btl=3×base):
#   R3D-18    btl=64→192, Q=12 : linear≈1.21M  relu≈906K  cross_poly≈908K  poly≈1.27M
#   Swin3D-T  btl=8→24,   Q=12 : linear≈424K   relu≈376K  cross_poly≈378K  poly≈389K
#   Swin3D-S  btl=8→24,   Q=12 : linear≈867K   relu≈769K  cross_poly≈771K  poly≈794K
#   Swin3D-B  btl=8→24,   Q=12 : linear≈1.16M  relu≈1.01M cross_poly≈1.01M poly≈1.06M

GPUS=${1:-4,5,6,7}
DATASET=${2:-ucf101}
AVG_SPLITS=${3:-0}
BACKBONE=${4:-all}

N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
LR=$(python3 -c "print(f'{int($N_GPUS) * 4e-4:.0e}')")

echo "==> GPUs: $GPUS (${N_GPUS}x)  dataset: $DATASET  lr: $LR  avg_splits: $AVG_SPLITS  backbone: $BACKBONE"
echo ""

AVG_FLAG=""
[ "$AVG_SPLITS" = "1" ] && AVG_FLAG="--avg_splits"

TORCHRUN="NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS \
  torchrun --nproc_per_node=$N_GPUS train_par.py"

BASE="--dataset $DATASET --batch_size 32 --lr $LR --epochs 50 --warmup_epochs 5
  --no_wandb $AVG_FLAG"

run() {
    local name=$1; shift
    echo "==> Starting: $name"
    eval $TORCHRUN --run_name "${DATASET}_${name}" $BASE "$@"
    if [ $? -ne 0 ]; then
        echo "ERROR: $name failed, stopping."
        exit 1
    fi
    echo "==> Done: $name"
    echo ""
}

# Runs all four modes for a given backbone. poly keeps the base rank; the other
# three use 3× rank so their param counts roughly match poly.
run_modes() {
    local tag=$1      # run name prefix, e.g. "r3d"
    local model=$2    # model name, e.g. "r3d_adapted"
    local btl=$3      # base bottleneck rank (poly uses this; others use 3×)
    local Q=$4        # base adapter rank    (poly uses this; others use 3×)
    local btl3=$((btl * 3))
    local Q3=$((Q * 3))
    run ${tag}_lora       --model $model --adapter_rank $Q  --adapter_bottleneck_rank $btl3 --adapter_mode linear
    run ${tag}_relu       --model $model --adapter_rank $Q3 --adapter_bottleneck_rank $btl3 --adapter_mode relu
    run ${tag}_cross_poly --model $model --adapter_rank $Q3 --adapter_bottleneck_rank $btl3 --adapter_mode cross_poly
    run ${tag}_poly       --model $model --adapter_rank $Q  --adapter_bottleneck_rank $btl  --adapter_mode poly
}

[[ "$BACKBONE" == "all" || "$BACKBONE" == "r3d"      ]] && run_modes r3d      r3d_adapted      64 4
[[ "$BACKBONE" == "all" || "$BACKBONE" == "swin3d_t" ]] && run_modes swin3d_t swin3d_t_adapted  8 4
[[ "$BACKBONE" == "all" || "$BACKBONE" == "swin3d_s" ]] && run_modes swin3d_s swin3d_s_adapted  8 4
[[ "$BACKBONE" == "all" || "$BACKBONE" == "swin3d_b" ]] && run_modes swin3d_b swin3d_b_adapted  8 4

echo "==> All ablation runs complete."
