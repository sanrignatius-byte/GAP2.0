#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_attn_flow
#SBATCH --output=logs/gap2_attn_flow_%j.out
#SBATCH --error=logs/gap2_attn_flow_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 Phase 2 — Attention Flow Experiment
# Distinguishes Late-Stage Readout vs Early Assimilation hypotheses
# 30 samples (ChartQA + DocVQA + TextVQA), eager attention, 48 layers

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/attention_flow"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR/plots"

echo "========================================"
echo "GAP 2.0 Phase 2 — Attention Flow"
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

echo ""
echo "===== Phase 2: Attention Flow (30 samples, eager attn) ====="
conda run -n gap2 python scripts/run_phase2_attention_flow.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --num_samples 30 \
  --output_dir "$OUT_DIR" \
  --attn_implementation eager

echo ""
echo "========================================"
echo "Attention Flow complete! | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
