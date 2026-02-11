# GAP 2.0 Discussion Log

> 本文件记录 GAP 2.0 项目的讨论过程、设计决策和关键争论点。

---

## 2026-02-11: 实验设计启动讨论

### 背景

Ignatius 提出了 GAP 2.0 的完整研究方案，旨在通过几何视角诊断和缓解 decoder-only MLLMs 中的 Modality Cliff 现象。该方案经过三位独立审稿人的反馈整合，形成了一份可执行的实验计划。

### 核心讨论点

#### 1. 实验框架的整体架构

**讨论**：如何组织实验代码，使其既能服务 Phase 1（诊断验证），又能为 Phase 2（Geometric Adapter）做好扩展准备？

**决策**：
- 采用模块化架构，将功能明确分为 `models/`、`causal/`、`geometry/`、`data/`、`visualization/` 五个子模块
- 每个实验阶段有独立的 runner 脚本（`scripts/run_phase1_*.py`），通过 YAML 配置文件驱动
- Hook 系统设计为可复用的上下文管理器，同时支持 activation extraction、causal patching 和 truncation 三种用途

**理由**：
- Phase 1 的 hook 基础设施（activation extraction）是 Phase 2 OT adapter 所必需的——在 intermediate layer 插入 adapter 需要精确的 hidden states 访问
- 配置驱动的设计允许快速切换模型（LLaVA → Qwen-VL）和数据集

#### 2. Causal Patching 的三种腐败方法

**讨论**：方案中提出了 zero-out、Gaussian noise、mean-substitution 三种腐败方法。它们的理论含义和实际差异是什么？

**分析**：
- **Zero-out**：最激进的干预，完全消除视觉信息。可能高估 causal effect，因为零向量在 hidden space 中是不自然的分布
- **Gaussian noise**：保留视觉 token 的统计分布但破坏具体信息。sigma = clean activations 的标准差，确保噪声量级合理
- **Mean substitution**：用所有视觉 token 的均值替换每个 token，保留"存在视觉输入"的信号但消除 token 间差异。这是对 over-smoothing 假说最直接的模拟

**决策**：默认使用 Gaussian noise（与 LLaVA-CAM 等先前工作一致），但实现全部三种以便做消融实验。

#### 3. EVD 阈值 τ 的选择

**讨论**：EVD = max{l : Δ^(l) ≥ τ} 中的阈值 τ 如何确定？

**初步决策**：默认 τ = 0.01（log-prob 变化）。但这个值需要根据实际运行结果进行校准——如果所有层的 Δ 都很小，τ 需要相应降低。

**后续行动**：在首批实验运行后，绘制 Δ 的分布直方图，基于数据选择合理的 τ。

#### 4. Cliff Boundary 的估计方法

**讨论**：如何从多个样本的 causal trace 中估计统一的 cliff boundary？

**决策**：实现了三种方法：
- `median_evd`：取所有样本 EVD 的中位数（鲁棒、默认）
- `gradient`：找 mean Delta 曲线下降最陡的层
- `threshold_crossing`：mean Delta 首次低于 τ 的层

**理由**：不同方法适用于不同的 Delta 曲线形状。如果 cliff 是突变型的，gradient 方法最准确；如果是渐变型的，threshold_crossing 更合理。

#### 5. 几何指标的设计选择

**Effective Rank**：
- 使用 Shannon entropy 定义（而非阈值截断），因为它对 singular value 的分布形状敏感
- 对 representation matrix 做中心化（减均值），避免 mean shift 主导 first singular value

**CKA**：
- 使用 debiased HSIC 估计器（Nguyen et al. 2020），因为视觉和文本 token 数量不同（576 vs ~几十到几百），小样本偏差不可忽略
- 对不同大小的 token 集采用 minibatch CKA（随机配对子采样后平均）

**ICC**：
- 直接计算所有 pair 的 cosine similarity 均值
- 对 visual tokens（N_v=576）计算量为 O(576²)≈165K pairs，完全可行

#### 6. 数据准备：Hard vs Easy 的划分标准

**讨论**：如何客观地定义"需要深层视觉推理"的 hard 样本？

**决策**：
- 使用问题文本的关键词启发式（比较、空间关系、跨区域引用等）作为初始划分
- ChartQA 和 DocVQA 整体视为 "hard" 数据集（inherently require spatial/structural reasoning）
- VQAv2 中的简单属性问题视为 "easy" 对照组

**已知局限**：关键词启发式是粗粒度的。更精确的方法是根据 truncation 实验的结果反向标注——那些在 truncation 下准确率显著下降的样本才是真正的 "hard" 样本。

#### 7. Checkpoint 4（训练目标混淆）的讨论

**讨论**：如何区分"几何坍缩是由表示空间的内在特性导致"和"只是因为 next-token prediction loss 只作用于文本 token"？

**当前立场**：
- 这是一个重要的混淆变量，但完全证明因果关系超出了单篇论文的范围
- 初步方案：对比有 visual grounding loss 的模型（如 Shikra、Kosmos-2）和标准 LLaVA 的 rank decay 曲线
- 如果带 grounding loss 的模型 rank decay 更慢 → 支持训练信号假说
- 这作为"支持性证据"而非硬性 go/no-go checkpoint

**注意**：如果没有现成的 grounding-loss 模型 checkpoint 可用，这个实验可以降低优先级。

### 技术风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 模型加载需要大量 GPU 显存 | 使用 float16 + device_map auto |
| 576 个视觉 token 的 activation 存储量大 | store_on_cpu=True，按需加载 |
| 不同样本的视觉 token 位置可能不同 | 每次 forward 前动态检测 image token 位置 |
| ChartQA/DocVQA 的 answer 格式复杂 | 使用简单的 containment check 作为初始评估方法 |

### 下一步

1. 在集群上部署实验代码
2. 先跑少量样本（10-20个）验证 pipeline 正确性
3. 根据初步结果校准 EVD 阈值 τ
4. 全量运行 Phase 1 实验
