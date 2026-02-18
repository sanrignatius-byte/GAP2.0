#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_smoke
#SBATCH --output=logs/gap2_smoke_%j.out
#SBATCH --error=logs/gap2_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 Phase 1 冒烟测试
# 环境: gap2 | 模型: Qwen3-VL-30B-A3B-Thinking
# 数据: ChartQA 5 hard + 5 easy

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/smoke_test"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR/plots"

echo "========================================"
echo "Node: $(hostname) | Start: $(date)"
echo "========================================"

# 打印 GPU 信息
conda run -n gap2 python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  {i}: {torch.cuda.get_device_name(i)} {p.total_memory/1024**3:.1f}GB')
"

# 合并配置: default.yaml + qwen_vl_smoke.yaml (含模型参数 + 冒烟测试覆盖)
CFG="--config configs/default.yaml --model_config configs/qwen_vl_smoke.yaml"

echo ""
echo "===== Stage 1/4: Causal Tracing ====="
conda run -n gap2 python scripts/run_phase1_causal.py \
  $CFG \
  --num_samples 5 \
  --output_dir "$OUT_DIR" \
  --adaptive_tau \
  --tau_ratio 0.1

echo ""
echo "===== Stage 2/4: Truncation ====="
conda run -n gap2 python scripts/run_phase1_truncation.py \
  $CFG \
  --num_samples 5 \
  --output_dir "$OUT_DIR" \
  --step 4

echo ""
echo "===== Stage 3/4: Geometry ====="
conda run -n gap2 python scripts/run_phase1_geometry.py \
  $CFG \
  --num_samples 5 \
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
