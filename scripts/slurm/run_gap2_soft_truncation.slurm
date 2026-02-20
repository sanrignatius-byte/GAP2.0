#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_soft_trunc
#SBATCH --output=logs/gap2_soft_trunc_%j.out
#SBATCH --error=logs/gap2_soft_trunc_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 — Norm-matched Soft Truncation (P0)
# 裁决："hard truncation 的 accuracy drop 是真实的视觉依赖，还是分布突变伪影？"
# 对比: zero vs norm_matched_noise vs pca_matched_noise vs shuffle

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/soft_truncation_chartqa"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "GAP 2.0 — Soft Truncation Experiment (P0)"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

conda run -n gap2 python scripts/run_soft_truncation.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --dataset chartqa \
  --split test \
  --max_samples 300 \
  --strategies zero norm_matched_noise pca_matched_noise shuffle \
  --truncation_layers 0 6 12 18 24 30 36 42 47 \
  --max_new_tokens 512 \
  --seed 42 \
  --output_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "Soft Truncation complete | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
