# GAP 2.0 项目进度审查与实验结果分析
> 审查日期：2026-02-18

---

## 一、整体进度总览

| 阶段 | 状态 | 关键产出 |
|------|------|---------|
| Phase 1: 因果追踪 (Causal Tracing) | ✅ 完成 | 三段式因果贡献曲线，cliff_boundary=23 |
| Phase 1: 截断实验 (Truncation) | ✅ 完成 | Layer 37-40 阶跃信号（最可靠发现） |
| Phase 1: 几何分析 (Geometry) | ✅ 完成 | 否定 Geometric Collapse 假说，CKA 不可用 |
| Phase 1: Checkpoint 评估 | ✅ 完成 | **NO-GO**（2/4 几何标准通过） |
| Phase 2: Text Token Probing | ✅ 完成 | 文本侧表征差异显著（bootstrap CI 确认） |
| Phase 2: Attention Flow | ✅ 完成 | 30 样本 attention flow 数据已收集 |
| Phase 2: 模型生成准确率 | ✅ 完成 | Original vs Blind Random 差异未显著 |
| Phase 2: Probing 索引修复 | ✅ 完成 | answer_token 改为 teacher-forced 标签位置 |
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

⚠️ Easy layers 20-32 的"梯度恢复"（每步 +0.033）仅对应 1 个 sample 变化，是统计噪声。

### 2.3 几何分析：否定 Geometric Collapse

| 指标 | 实际值 | 期望阈值 | 判定 |
|------|--------|---------|------|
| Visual ER post/pre | 1.094 | <0.7 | ✗ 未通过（ER 上升而非坍缩） |
| Text ER post/pre | 1.633 | >0.85 | ✗ 未通过 |
| Visual ICC change | +0.173 | 正值 | ✓ 通过 |
| CKA change | +0.008 | 正值 | ✓ 通过（但数据噪声极高） |

**核心结论**：MoE 架构中不存在经典意义上的"几何坍缩"。Visual ER 整体上升说明视觉表征维度在增加，MoE 的稀疏激活可能天然防止表征坍缩。CKA 在 0-0.078 之间剧烈跳动，基本不可用。

### 2.4 Checkpoint 总评：NO-GO

Phase 1 的 NO-GO 结论意味着原始 "Geometric Alignment for Preventing Information Collapse" 假说在数据层面不成立。但这本身是有价值的——它指向了一个更有趣的发现：**Late-Stage Visual Readout**。

---

## 三、Phase 2 最新实验结果分析

### 3.1 Text Token Probing（48 样本 ChartQA yes/no）

这是当前最关键的 Phase 2 结果。线性 probe 在每层文本 token hidden states 上训练，预测正确答案。

#### 核心数据

| Target | Original | Blind Random | Delta | 95% CI | 显著性 |
|--------|----------|-------------|-------|--------|--------|
| text_instruction_token (post L24-47) | 0.7242 | 0.5780 | **+0.1462** | [+0.1291, +0.1627] | ✅ **显著** |
| visual_token (post L24-47) | 0.3611 | 0.5131 | **-0.1520** | [-0.1751, -0.1294] | ✅ **显著** |
| answer_token | 1.0 (全层) | 1.0 (全层) | 0.0 | — | ⚠️ 标签泄漏 |

#### 逐层 Delta 趋势分析

**text_instruction_token 的 delta 曲线** 呈现明显的"阶梯上升"模式：
- Layers 0-23：delta 在 [-0.06, +0.04] 之间波动，接近零（视觉信息尚未写入文本侧）
- **Layer 24 起**：delta 跳升至 +0.08 并持续攀升
- Layers 35-47：稳定在 +0.14 ~ +0.22 区间

这条曲线直接支持**视觉信息从 Layer 24 开始逐步写入文本 token 表征**的叙事。

**visual_token 的 delta 曲线** 全程为负（Original 低于 Random），均值 -0.15，无明显层级趋势。这说明视觉 token 本身的 probe accuracy 在有正确图像时反而更低——可能因为正确图像引入了更复杂的表征分布，线性 probe 更难拟合。

#### answer_token 标签泄漏问题

answer_token 全层 delta=0 且 accuracy=1.0，说明当前 answer_token 定义（teacher-forced 标签位置）使 probe 直接读取了标签信号。**这不是实验 bug，而是定义问题**——需要将 answer_token 改为"预测位（答案前一位）"才能获得有意义的信号。

### 3.2 模型生成准确率（Original vs Blind Random）

| 条件 | Accuracy | 正确数 / 总数 |
|------|----------|--------------|
| Original（正确图像） | 0.3125 | 15/48 |
| Blind Random（随机图像） | 0.2083 | 10/48 |
| Delta | +0.1042 | +5 samples |
| Bootstrap 95% CI | [-0.0625, +0.2708] | **未显著** |

**关键判断**：

1. **表征层面差异显著，行为层面差异不显著**——这是当前最核心的 gap。文本 token 在中后层确实获得了更多视觉信息（probe delta 显著），但模型最终生成的答案正确率提升仅 +5 个样本，置信区间跨零。

2. **Original accuracy 仅 31.25%** 说明 Qwen3-VL-30B 在 ChartQA yes/no 上的 baseline 性能本身偏低。检查生成内容可以发现，模型经常进行了正确推理但因 max_new_tokens=128 截断而未输出最终 yes/no 答案（大量 `pred: null`）。**这很可能是 max_new_tokens 设置不足导致的系统性 artifact。**

3. 模型启用了 thinking mode（`<think>` 标签），推理链很长，128 tokens 远不够。建议提升到 256-512 tokens 重跑。

### 3.3 Attention Flow 实验

30 样本的 attention flow 数据已收集（n_visual_tokens ≈ 318，n_text_tokens ≈ 18）。

#### text_to_visual attention 趋势

| 层段 | t2v_normalized 均值 | 特征 |
|------|-------------------|------|
| Layer 0 | 0.293 | 初始层高关注（视觉特征初始读取） |
| Layers 1-6 | 0.04-0.08 | 快速下降（早期抽象化） |
| Layers 7-21 | 0.08-0.20 | 逐步回升（交叉推理需求增加） |
| **Layer 22** | **0.270** | **局部峰值（与 EVD cliff=23 吻合）** |
| **Layer 27** | **0.268** | **第二峰值** |
| Layers 28-35 | 0.17-0.25 | 持续中高水平 |
| **Layer 34** | **0.248** | **第三峰值** |
| Layers 40-47 | 0.11-0.15 | 逐步下降（解码准备阶段） |

#### 分析

1. **text→visual attention 并非在 layer 37-40 出现峰值**，这与"Late-Stage Visual Readout 通过注意力机制实现"的最直接假说不一致。

2. text→visual attention 的峰值集中在 **layer 20-35**，而截断实验的阶跃在 **layer 37-40**。这暗示信息转移可能分两步：
   - Step 1（Layer 20-35）：文本 token 通过 attention 读取视觉信息
   - Step 2（Layer 37-40）：读取的信息被整合到文本 token residual stream 中并达到解码可用状态

3. visual_to_visual attention 极低（~0.002），说明视觉 token 之间几乎不交互，信息主要通过 text→visual 通道流动。

---

## 四、假说竞争现状

### 4.1 原始假说（Modality Assimilation at L20-25）

| 证据 | 方向 |
|------|------|
| Visual ICC L24-25 跃升 | 支持 |
| Visual ER 单调增长 | 弱支持 |
| **截断 L24 后 accuracy 不变** | **强反对** |
| Layer 18 因果贡献回弹 | 反对 |
| CKA 无系统性上升 | 反对 |
| Attention peak at L22 | 部分支持（但无截断阶跃配合） |

**判定：原始假说在时间定位上不成立。**

### 4.2 替代假说 A：Late-Stage Visual Readout

- 截断实验阶跃（L37-40）直接支持
- Text probing delta 从 L24 起上升，在 L35+ 稳定，与两阶段模型一致
- Attention flow 在 L20-35 活跃，提供了"读取窗口"
- 但 attention 峰值不在 L37-40，需要解释 gap

**当前最优假说，但需要 Phase 2.1 实验验证。**

### 4.3 替代假说 B：MoE Routing-Driven Phase Transition

- Layer 18 的因果贡献回弹可由 router 重新激活视觉 expert 解释
- 截断阶跃可由 L37-40 特定 expert 集合负责 visual-to-text transfer 解释
- 需要 expert routing 数据验证（已在 GAP 2.1 计划中，成本极低）

---

## 五、当前瓶颈与风险

### 5.1 技术瓶颈

1. **max_new_tokens 不足**：Thinking mode 下 128 tokens 远不够，导致大量 pred=null，系统性低估 accuracy。这是**最需要立即修复的问题**。

2. **answer_token 标签泄漏**：当前定义导致 probe accuracy 全层 1.0，需改为预测位（答案前一位）。

3. **样本量限制**：Phase 1 仅 10 samples/category，Phase 2 probing 48 samples。统计功效有限，精细层级定位（如 cliff=23）不可信。

4. **EVD 定义语义不清**：mean_deltas 全为负值下 EVD 的计算逻辑不透明，cliff_boundary=23 的可靠性存疑。

### 5.2 论文方向风险

- 原始 GAP（Geometric Alignment Prevention）框架已被 Phase 1 数据否定
- 需要明确 pivot 到 "Late-Stage Visual Readout in MoE VLMs"
- 如果 Phase 2.1 的双通道因果追踪不能清晰定位 transfer window，论文叙事将缺乏 clean story

---

## 六、下一步优先级建议

### 立即执行（本周）

1. **提升 max_new_tokens 到 512 并重跑模型准确率实验**——解决 pred=null 问题，获得真实 accuracy baseline
2. **修复 answer_token 为预测位（答案前一位）**——消除标签泄漏
3. **扩展样本量到 100+**——提升统计功效

### 短期执行（1-2 周）

4. **实现双通道因果追踪**（GAP 2.1 最高优先级）——同时 patch visual/text tokens，寻找交叉层
5. **MoE Expert Routing 分析**（成本极低）——验证 routing-driven phase transition 假说
6. **精细截断 L36-44 步长 1**——精确定位 readout 窗口边界

### 中期执行（2-4 周）

7. **State Transplant 截断**——区分"信息已转移"vs"仍需实时读取"
8. **Counterfactual Swap 实验**——增强因果结论的语义可解释性
9. **撰写论文 Methods + Results 初稿**

---

## 七、总结

GAP 2.0 项目在 Phase 1 中产生了一个有价值的 negative result（否定 Geometric Collapse）和一个强 positive signal（Layer 37-40 阶跃）。Phase 2 的 text probing 结果进一步确认了视觉信息确实在中后层写入文本侧表征，但模型行为层面的验证尚未达到统计显著。当前最大的技术障碍是 max_new_tokens 导致的系统性评估 artifact，以及 answer_token 标签泄漏。

**最有把握的论文叙事**：放弃 Geometric Collapse Prevention，转向 "Late-Stage Visual Readout in MoE Vision-Language Models"，以截断实验阶跃 + text probing 层级曲线作为核心证据，辅以 attention flow 和 expert routing 分析。
