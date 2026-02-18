#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_acc_orand_lg
#SBATCH --output=logs/gap2_acc_orand_lg_%j.out
#SBATCH --error=logs/gap2_acc_orand_lg_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/model_accuracy_chartqa_orig_vs_random_large"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "GAP 2.0 Phase 2 — Model Accuracy (Large: Original vs Random)"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

conda run -n gap2 python scripts/run_phase2_model_accuracy.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --dataset chartqa \
  --split test \
  --max_samples 1200 \
  --image_modes original blind_randomimage \
  --seed 42 \
  --max_new_tokens 128 \
  --output_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "Model Accuracy Large evaluation complete | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"

