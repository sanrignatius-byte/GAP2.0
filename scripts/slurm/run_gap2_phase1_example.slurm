#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=gap2_phase1
#SBATCH --output=logs/gap2_phase1_%j.out
#SBATCH --error=logs/gap2_phase1_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/GAP2.0
cd "$REPO_ROOT"
mkdir -p logs

# 你可替换为本地部署的 Qwen3-VL-30B 路径
MODEL_PATH="${REPO_ROOT}/Qwen3-VL-30B-A3B-Thinking"

echo "Node: $(hostname) | Start: $(date)"
conda run -n minerU python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(f'  {i}: {torch.cuda.get_device_name(i)} {p.total_memory/1024**3:.1f}GB')
"

# 注意：当前仓库默认 runner 仍偏向 LLaVA 输入流。
# 若直接切到 Qwen3-VL，请先补齐 Qwen 输入预处理与 visual token 定位逻辑。

conda run -n minerU python scripts/run_phase1_causal.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl.yaml \
  --num_samples 10

conda run -n minerU python scripts/run_phase1_truncation.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl.yaml \
  --step 2

conda run -n minerU python scripts/run_phase1_geometry.py \
  --config configs/default.yaml \
  --model_config configs/qwen_vl.yaml \
  --num_samples 10

conda run -n minerU python scripts/run_checkpoint_eval.py \
  --results_dir ./results

echo "Done: $(date)"
