#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_full
#SBATCH --output=logs/gap2_full_%j.out
#SBATCH --error=logs/gap2_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 Phase 1 多数据集完整测试
# ChartQA(hard) + DocVQA(hard) + TextVQA(medium)
# 每数据集 10 hard + 10 easy 样本

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/full_test"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR/plots"

echo "========================================"
echo "GAP 2.0 Phase 1 — Multi-Dataset Full Test"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

conda run -n gap2 python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  {i}: {torch.cuda.get_device_name(i)} {p.total_memory/1024**3:.1f}GB')
"

CFG="--config configs/default.yaml --model_config configs/qwen_vl_full.yaml"

echo ""
echo "===== Stage 1/4: Causal Tracing (10 samples × 4 datasets) ====="
conda run -n gap2 python scripts/run_phase1_causal.py \
  $CFG \
  --num_samples 10 \
  --output_dir "$OUT_DIR" \
  --adaptive_tau \
  --tau_ratio 0.1

echo ""
echo "===== Stage 2/4: Truncation (10 samples × 4 datasets) ====="
conda run -n gap2 python scripts/run_phase1_truncation.py \
  $CFG \
  --num_samples 10 \
  --output_dir "$OUT_DIR" \
  --step 4

echo ""
echo "===== Stage 3/4: Geometry (10 samples) ====="
conda run -n gap2 python scripts/run_phase1_geometry.py \
  $CFG \
  --num_samples 10 \
  --output_dir "$OUT_DIR"

echo ""
echo "===== Stage 4/4: Checkpoint Evaluation ====="
conda run -n gap2 python scripts/run_checkpoint_eval.py \
  --results_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "All stages complete! | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
