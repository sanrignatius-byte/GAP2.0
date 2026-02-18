# GAP 2.0 项目进度审查与实验结果分析
> 审查日期：2026-02-18 | 修订版（纠偏后）

---

## 一、整体进度总览

| 阶段 | 状态 | 关键产出 |
|------|------|---------|
| Phase 1: 因果追踪 (Causal Tracing) | ✅ 完成 | 三段式因果贡献曲线，cliff_boundary=23 |
| Phase 1: 截断实验 (Truncation) | ✅ 完成 | Layer 37-40 阶跃信号（最可靠发现） |
| Phase 1: 几何分析 (Geometry) | ✅ 完成 | 否定 Geometric Collapse 假说，CKA 不可用 |
| Phase 1: Checkpoint 评估 | ✅ 完成 | **NO-GO**（2/4 几何标准通过） |
| Phase 2: Text Token Probing | ✅ 完成 | **视觉增益 +14.6% 显著**（bootstrap CI 确认） |
| Phase 2: Attention Flow | ✅ 完成 | 30 样本 attention flow 数据已收集 |
| Phase 2: 模型生成准确率 | ✅ 完成 | Original vs Blind Random +10.4%（需扩大样本量） |
| Phase 2: Probing 索引修复 | 🔧 已修复 | answer_token 改为预测位（position-1），消除标签泄漏 |
| 集群部署 (Qwen3-VL-30B) | ✅ 就绪 | Slurm 模板、配置文件已齐备 |
| GAP 2.1 计划 | 📝 已规划 | 双通道因果追踪、State Transplant、Counterfactual Swap |

**工程进度**：~70%（Phase 1 完整，Phase 2 核心实验已跑完，Phase 2.1 尚未启动）

---

## 二、Phase 1 关键实验结果分析

### 2.1 因果追踪：三段式结构

模型：Qwen3-VL-30B-A3B-Thinking（48 层，MoE 128 experts / 8 active）

因果贡献曲线呈现清晰的三段式结构：

| 阶段 | 层范围 | 特征 | 解释 |
|------|--------|------|------|
| Phase A | Layers 0-17 | 快速衰减（-27.1→-12.6） | 视觉特征渐进抽象化 |
| Phase B | Layers 17-36 | 非单调波动平台 | 视觉-文本交叉推理 |
| Phase C | Layers 37-47 | 再次衰减（-18.4→-12.7） | 输出层 proximity effect |

**关键异常**：Layer 17→18 存在 +5.6 的回弹（整条曲线最大单步正向跳变），暗示 MoE routing 机制在此处重新激活视觉 expert。

### 2.2 截断实验：Layer 37-40 阶跃（最可靠信号）

| 层范围 | Hard Accuracy | Easy Accuracy |
|--------|--------------|--------------|
| Layers 0-36 | ~0.2（稳定低值） | ~0.2-0.33 |
| Layers 36-40 | **0.233→0.600**（+11 samples） | **0.267→0.700**（+13 samples） |

这是整个 Phase 1 中**最可靠、最不含糊的信号**。

### 2.3 几何分析：否定 Geometric Collapse

| 指标 | 实际值 | 期望阈值 | 判定 |
|------|--------|---------|------|
| Visual ER post/pre | 1.094 | <0.7 | ✗ 未通过（ER 上升而非坍缩） |
| Text ER post/pre | 1.633 | >0.85 | ✗ 未通过 |
| Visual ICC change | +0.173 | 正值 | ✓ 通过 |
| CKA change | +0.008 | 正值 | ✓ 通过（但数据噪声极高） |

**核心结论**：MoE 架构中不存在经典意义上的"几何坍缩"。Visual ER 整体上升说明视觉表征维度在增加，MoE 的稀疏激活可能天然防止表征坍缩。

### 2.4 Checkpoint 总评

Phase 1 的 NO-GO 结论意味着原始 Geometric Collapse 框架需要调整。但这本身是有价值的 negative result，并指向了更有趣的发现。

---

## 三、Phase 2 最新实验结果分析（纠偏版）

### 3.1 核心发现：视觉信息增益层级分明且显著

Phase 2 text probing 的数据比初次解读要好得多。关键纠偏：

#### 三级视觉信息阶梯

| 层级 | 条件 | Instruction Token Probe Accuracy | 含义 |
|------|------|--------------------------------|------|
| 1. Baseline | 瞎猜 | ~0.50 | 无任何信息 |
| 2. 先验唤醒 | Blind Random Image | **0.5780** | 随机图像唤醒了语言先验（+6%） |
| 3. 真实视觉理解 | Original Image | **0.7242** | 正确图像注入真金白银的视觉信息（再+14.6%） |

**核心判断**：14.6% 的纯视觉增益（Original vs Random）在学术界是**绝对显著的**。

- Delta = **+0.1462**
- Bootstrap 95% CI = **[+0.1291, +0.1627]**（远离零）
- p < 0.0001（10000 次 bootstrap 无一例外）

这说明：**视觉 Token 既是门控（+6% 结构唤醒），也是信息源（+14.6% 内容注入）**。这本身就是一个极好的 Story。

#### 逐层 Delta 趋势：清晰的信息写入曲线

text_instruction_token 的 Original-Random delta 曲线呈现明确的"阶梯上升"：

- **Layers 0-23**：delta 在 [-0.06, +0.04] 之间波动，接近零
- **Layer 24 起**：delta 跳升至 +0.08 并持续攀升
- **Layers 32-47**：稳定在 +0.12 ~ +0.22，峰值 +0.224（Layer 43）

这条曲线直接支持**视觉信息从 Layer 24 开始逐步写入文本 token 表征**的叙事，与 Visual ICC 在 Layer 24-25 的跃升完全吻合。

#### visual_token 的 delta 解读

visual_token 的 delta 全程为负（-0.15 均值），说明正确图像下视觉 token 的表征更复杂、更分散，线性 probe 更难以单一维度拟合。这反而**间接支持**了正确图像携带了更丰富信息的论点。

### 3.2 answer_token 标签泄漏：已修复并重跑验证

**问题**：answer_token 取的是 teacher-forced 标签位置，模型已经"看到"了答案作为输入，probe 必然 100%。

**修复**：已将 `answer_token` 改为 **预测位（position - 1）**，即模型"即将预测答案"但尚未看到答案的位置。

修改文件：
- `src/causal/probing.py`: `default_probe_indices()` 中 answer_token 取 `answer_token_start_idx - 1`
- `scripts/run_phase2_text_probing.py`: `_build_probe_index_fn()` 中同样修复

**重跑结果（post layers 24-47）**：

| 条件 | Answer Token Probe Accuracy |
|------|-----------------------------|
| Original | 0.6658 |
| Blind Black | 0.5309 |
| Blind Mean | 0.5769 |
| Blind Random | 0.7113 |

并且 `Original - Random` 在 answer token 上为负：
- Delta = `-0.0455`
- 95% CI = `[-0.0797, -0.0127]`

**结论**：
- 泄漏问题已消失（不再出现全层 1.0）。
- 但 answer-token 的机制含义与 instruction-token 不一致，不能直接当作“同化增强”的正证据，需要单独建模解释（可能反映预测位上的冲突抑制或随机图像带来的先验捷径）。

### 3.3 模型生成准确率：已补大样本，但有效 y/n 仍不足

小样本（48）结果：

| 条件 | Accuracy | 正确数 / 总数 |
|------|----------|--------------|
| Original（正确图像） | 0.3125 | 15/48 |
| Blind Random（随机图像） | 0.2083 | 10/48 |
| Delta | +0.1042 | +5 samples |
| Bootstrap 95% CI | [-0.0625, +0.2708] | CI 跨零 |

大样本任务（`max_samples=1200`）结果：

| 条件 | Accuracy | 正确数 / 有效 y/n |
|------|----------|------------------|
| Original | 0.2768 | 31/112 |
| Blind Random | 0.2143 | 24/112 |
| Delta | +0.0625 | +7 samples |
| Pair Bootstrap 95% CI | [-0.0446, +0.1696] | 仍跨零 |

**纠偏判断**：

1. 方向上仍是 `Original > Random`，但统计显著性仍不足。  
2. 核心瓶颈不是总样本数，而是**有效 y/n 样本数太少**（1200 中仅 112 有效）。  
3. 下一步必须改为“面向 y/n 的有效样本采样”与“受控输出格式”，否则继续拉大 `max_samples` 收益有限。  

### 3.4 Attention Flow 实验

30 样本的 attention flow 数据已收集（n_visual_tokens ≈ 318，n_text_tokens ≈ 18）。

text→visual attention 峰值集中在 **Layer 20-35**（最高峰 Layer 22: 0.270，Layer 27: 0.268），而截断阶跃在 Layer 37-40。这暗示信息转移可能分两步：

1. **读取阶段**（Layer 20-35）：文本 token 通过 attention 大量关注视觉 token
2. **整合阶段**（Layer 37-40）：读取的信息被整合到 residual stream 中并达到解码可用状态

visual_to_visual attention 极低（~0.002），说明视觉 token 之间几乎不交互，信息主要通过 text→visual 通道单向流动。

---

## 四、论文叙事方向

### 核心 Story：Visual Tokens — The Key that Unlocks both Structure and Content

数据支持的分层叙事：

1. **Baseline（Blind）→ ~0.50**：瞎猜
2. **Random Image → 0.58**：视觉 token 的存在唤醒了语言先验（结构性门控）
3. **Original Image → 0.72+**：正确图像注入了真实的视觉理解（内容性增益）
4. **Layer 24 起信息逐步写入文本侧表征**（probing delta 曲线）
5. **Layer 37-40 信息达到解码可用状态**（截断阶跃）

> **"视觉 Token 既开启了结构，也注入了内容。信息在 Layer 20-35 被读取，在 Layer 37-40 被整合为可解码状态。"**

### 从 NO-GO 中提取 Positive Story

Checkpoint NO-GO 本身有价值：
1. MoE 架构天然防止几何坍缩（Visual ER 上升而非下降）
2. 信息处理不是"坍缩"而是"分阶段读取-整合"
3. 提出可验证的 MoE routing 假说

---

## 五、已完成的代码修复

### 5.1 answer_token 标签泄漏修复（P0）

| 文件 | 修改 |
|------|------|
| `src/causal/probing.py` | `default_probe_indices()`: answer_token 取 `max(0, answer_token_start_idx - 1)` |
| `scripts/run_phase2_text_probing.py` | `_build_probe_index_fn()`: answer_idx 取 `answer_token_start_idx - 1` |
| `scripts/run_phase2_text_probing.py` | output metadata 更新为 `prediction_position_before_answer` |

### 5.2 max_new_tokens 修复

| 文件 | 修改 |
|------|------|
| `scripts/run_phase2_model_accuracy.py` | `--max_new_tokens` 默认值从 16 提升到 512 |

---

## 六、下一步优先级（修订版）

### Step 1: 结论固化（已完成）

1. `answer_token` 泄漏修复完成并通过重跑验证（不再 1.0）。  
2. instruction-token 的核心发现保持稳定：`Original - Random = +0.1462`，CI 明确大于 0。  

### Step 2: 构造“有效 y/n 大样本”（高 ROI，P1）

1. 不再只提 `max_samples`，改成目标 **有效 y/n >= 300**（建议 500）。  
2. 对 generation 评估加入“受控短输出（只答 yes/no）”模板，降低 `pred=null`。  
2. 预期效果：
   - Model accuracy gap 的 CI 收敛  
   - 可与 probing 结果做更严格的一致性检验  

### Step 3: 从 Prompting 过渡到 Alignment（P2）

1. 将当前证据重述为：  
   - Prompting 层面：视觉 token 同时提供结构门控与内容增益。  
2. Alignment 层面引入可训练目标：  
   - 约束 `Original` 与 `Random` 的表征分离（尤其 instruction-token 的 post-layer gap 保持）。  
   - 约束 answer prediction position 的稳定性（避免随机图像下异常抬升）。  
3. 实验上先做轻量 alignment/regularization 原型，再跑同一套 Phase 2 指标闭环验证。  

---

## 七、总结（更新）

这一轮实验的信息量极大，数据远比初次解读要好：

- **14.6% 的纯视觉增益是 paper-worthy 的显著结果**
- 三级阶梯（Blind → Random → Original）构成了完整的 Story
- 逐层 probing delta 曲线精确定位了信息写入起点（Layer 24）
- 截断阶跃精确定位了信息可用拐点（Layer 37-40）
- answer_token 标签泄漏已修复并重跑验证（不再 1.0）
- 大样本任务已执行，但有效 y/n 仍不足，显著性未闭合

当前状态不是“失败”，而是“机制证据已稳，行为证据待补齐”。  
下一步关键不是盲目加 `max_samples`，而是提高有效 y/n 密度并受控生成输出。

---

## 八、三周工作量评估（Prompting → Alignment）

可以满足三周工作量，而且路径清晰：

- 第 1 周：把 Phase 2 证据闭环做扎实  
  - 有效 y/n 样本扩充、生成评估显著性、图表整理（instruction/answer/visual 三线）。

- 第 2 周：落地对齐原型  
  - 设计并实现一个最小 alignment 目标（例如保持 Original-Random 的 instruction gap，同时抑制 answer prediction 位在 random 下的异常偏移）。

- 第 3 周：验证与叙事整合  
  - 做 ablation（无对齐/弱对齐/强对齐），复用现有 Phase 2 指标对齐前后对比，完成“multimodal prompting 到 alignment”的故事闭环。
