#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_text_probe_brand
#SBATCH --output=logs/gap2_text_probe_brand_%j.out
#SBATCH --error=logs/gap2_text_probe_brand_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# GAP 2.0 Phase 2 — Text Token Probing (Blind Random-Image Baseline)

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
OUT_DIR="${REPO_ROOT}/results/text_probing_chartqa_blind_random"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

echo "========================================"
echo "GAP 2.0 Phase 2 — Text Token Probing (Blind Random Image)"
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
echo "===== Phase 2: Text Probing Blind Random (ChartQA yes/no, 300 samples) ====="
conda run -n gap2 python scripts/run_phase2_text_probing.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl_full.yaml \
  --dataset chartqa \
  --split test \
  --max_samples 300 \
  --probe_task yes_no \
  --image_mode blind_randomimage \
  --seed 42 \
  --cv_folds 5 \
  --output_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "Text Probing Blind Random complete! | End: $(date)"
echo "Results: $OUT_DIR"
echo "========================================"
