#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_deep_mask
#SBATCH --output=logs/gap2_deep_mask_%j.out
#SBATCH --error=logs/gap2_deep_mask_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 — Deep-Layer Targeted Masking (P0/P1)
# 裁决："深层 attention 回看视觉 token 是功能性必需还是 attention sink？"
# 实验：保留前 24 层完整视觉，仅在 24-47 层 mask 视觉 token

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/deep_layer_masking_chartqa"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "GAP 2.0 — Deep-Layer Targeted Masking (P0/P1)"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

conda run -n gap2 python scripts/run_deep_layer_masking.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --dataset chartqa \
  --split test \
  --max_samples 300 \
  --boundary_layer 24 \
  --extra_boundaries 12 18 30 36 \
  --mask_strategies zero norm_matched_noise \
  --max_new_tokens 512 \
  --seed 42 \
  --output_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "Deep-Layer Masking complete | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
