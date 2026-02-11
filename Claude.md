# GAP 2.0 — Claude Development Notes

> 本文件记录项目的开发思路、架构设计和当前进度。

---

## 项目概述

GAP 2.0 (Geometric Alignment for Preventing Information Collapse) 旨在通过几何视角诊断、解释和缓解 decoder-only MLLMs 中的 Modality Cliff 现象——即视觉 token 的因果贡献在中间层后急剧衰减的问题。

## 架构设计思路

### 整体设计原则

1. **模块化**：诊断（causal）、分析（geometry）、数据（data）、可视化（visualization）严格分离
2. **配置驱动**：通过 YAML 配置文件控制实验参数，避免硬编码
3. **渐进式验证**：Phase 1 先验证假设，通过 Checkpoint 机制做 go/no-go 决策
4. **可复用 Hook 系统**：ActivationExtractor、PatchingHook、TruncationHook 三个 hook 类覆盖所有实验需求

### 代码结构

```
GAP2.0/
├── configs/              # 实验配置
│   ├── default.yaml      # 默认参数
│   ├── llava_7b.yaml     # LLaVA-1.5-7B 特定参数
│   └── qwen_vl.yaml      # Qwen2.5-VL 特定参数
├── src/
│   ├── models/
│   │   ├── hooks.py       # ActivationExtractor, PatchingHook, TruncationHook
│   │   └── model_loader.py # 模型加载 (LLaVA, Qwen-VL)
│   ├── causal/
│   │   ├── patching.py    # CausalPatcher — 因果干预实验
│   │   ├── evd.py         # EVD 计算和分析
│   │   └── truncation.py  # TruncationExperiment — 截断实验
│   ├── geometry/
│   │   ├── effective_rank.py      # SVD 有效秩
│   │   ├── cosine_concentration.py # 余弦集中度 (ICC)
│   │   └── cka.py                 # 跨模态 CKA
│   ├── data/
│   │   ├── dataset_loader.py  # 数据集加载 (ChartQA, DocVQA, TextVQA, VQAv2, ScienceQA)
│   │   └── subset_sampler.py  # Hard/Easy 子集采样
│   └── visualization/
│       └── plots.py       # GAPVisualizer — 所有可视化
├── scripts/
│   ├── run_phase1_causal.py      # Phase 1 Week 1: 因果追踪
│   ├── run_phase1_truncation.py  # Phase 1 Week 1: 截断实验
│   ├── run_phase1_geometry.py    # Phase 1 Week 2: 几何分析
│   └── run_checkpoint_eval.py    # 综合 Checkpoint 评估
├── requirements.txt
├── pyproject.toml
├── discussion.md          # 讨论记录
└── Claude.md              # 本文件
```

### 关键设计决策

#### Hook 系统 (`src/models/hooks.py`)

核心挑战：需要在同一个模型上支持三种不同的干预操作：
- **提取** hidden states（只读）
- **替换** hidden states（因果干预）
- **置零** hidden states（截断）

解决方案：三个独立的 Hook 类，都实现 context manager 协议（`__enter__`/`__exit__`），支持 `with` 语句安全使用。共享 transformer layer 发现逻辑。

#### 模型架构适配 (`src/models/model_loader.py`)

支持两个模型家族：
- LLaVA-1.5 (7B/13B)：通过 HuggingFace `LlavaForConditionalGeneration` 加载
- Qwen2.5-VL：通过 `AutoModelForCausalLM` 加载

关键差异：
- LLaVA 的视觉 token 数量固定（576 = 24×24 patches）
- Qwen2.5-VL 使用动态分辨率，视觉 token 数量不固定

#### CKA 实现 (`src/geometry/cka.py`)

由于视觉 token (N_v=576) 和文本 token (N_t≈20-200) 数量不同，标准 CKA 无法直接计算。实现了两个策略：
1. **Debiased CKA**：子采样较大集合使 Gram 矩阵大小匹配，使用去偏 HSIC 估计器
2. **Minibatch CKA**：随机配对子采样后平均多次计算结果

## 当前进度

### ✅ 已完成

- [x] 项目结构设计和初始化
- [x] 配置文件系统 (default.yaml, llava_7b.yaml, qwen_vl.yaml)
- [x] Activation extraction pipeline (ActivationExtractor + PatchingHook + TruncationHook)
- [x] 模型加载工具 (LLaVA + Qwen-VL)
- [x] 因果干预系统 (CausalPatcher — 三种腐败方法)
- [x] EVD 计算和分析工具
- [x] 截断实验框架 (TruncationExperiment)
- [x] 几何指标: Effective Rank (SVD-based)
- [x] 几何指标: Inter-token Cosine Concentration
- [x] 几何指标: Cross-modal CKA (debiased + minibatch)
- [x] 数据加载 (ChartQA, DocVQA, TextVQA, VQAv2, ScienceQA)
- [x] Hard/Easy 子集采样器
- [x] 可视化 pipeline (8种图表类型)
- [x] Phase 1 实验 runner 脚本 (causal, truncation, geometry, checkpoint eval)

### 🔲 待完成

- [ ] 在集群上测试 pipeline 可运行性（模型加载、数据下载）
- [ ] 根据初步实验结果校准 EVD 阈值 τ
- [ ] 运行完整 Phase 1 实验
- [ ] Phase 1 go/no-go 决策
- [ ] Phase 2: Geometric Adapter 设计与实现 (OT barycenter)
- [ ] Phase 2: Layer-wise Barycenter Propagation 实现
- [ ] Checkpoint 4: 训练目标混淆分析（对比 visual grounding loss 模型）

### 运行指南

```bash
# 安装依赖
pip install -r requirements.txt

# Phase 1 Week 1: 因果追踪
python scripts/run_phase1_causal.py --config configs/default.yaml --model_config configs/llava_7b.yaml --num_samples 50

# Phase 1 Week 1: 截断实验
python scripts/run_phase1_truncation.py --config configs/default.yaml --model_config configs/llava_7b.yaml

# Phase 1 Week 2: 几何分析
python scripts/run_phase1_geometry.py --config configs/default.yaml --model_config configs/llava_7b.yaml --num_samples 20

# 综合评估
python scripts/run_checkpoint_eval.py --results_dir ./results
```

### 注意事项

1. **GPU 需求**：LLaVA-1.5-7B 需要约 14GB VRAM (float16)，13B 需要约 26GB
2. **存储需求**：每个模型每个数据集的 hidden states 约 10-50GB（可选存储）
3. **首次运行**：会自动下载模型和数据集，需要网络连接
4. **调试建议**：先用 `--num_samples 5` 运行验证 pipeline，再做大规模实验
