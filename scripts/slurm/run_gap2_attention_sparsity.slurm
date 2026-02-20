#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_attn_sparse
#SBATCH --output=logs/gap2_attn_sparse_%j.out
#SBATCH --error=logs/gap2_attn_sparse_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 — Attention Sparsity & Entropy Analysis (P1)
# 裁决 U-shape attention：re-grounding（稀疏/语义导向）vs attention sink（均匀/结构性）

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/attention_sparsity"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "GAP 2.0 — Attention Sparsity Analysis (P1)"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

conda run -n gap2 python scripts/run_attention_sparsity.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --num_samples 30 \
  --attn_implementation eager \
  --max_pixels 401408 \
  --fallback_max_pixels "200704,100352,62720" \
  --top_k_values 5 10 50 \
  --seed 42 \
  --output_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "Attention Sparsity complete | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
