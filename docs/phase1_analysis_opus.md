# GAP 2.0 Phase 1 实验深度分析报告
> 由 Claude Opus 4.6 分析 | 2026-02-18

---

## 实验背景

- **模型**: Qwen3-VL-30B-A3B-Thinking (MoE: 128 experts, 8 active), 48 layers
- **数据集**: ChartQA (hard×10), DocVQA (hard×10), TextVQA (medium×10)
- **三项实验**: 因果追踪 (Causal Tracing) + 几何分析 (Geometry) + 截断实验 (Truncation)

---

## 一、数据层面的精细解读

### 1.1 因果贡献曲线的三段式结构

Hard tasks 的 mean_delta 曲线呈现三段式结构：

**Phase A (Layers 0-17): 快速衰减段**
- 从 -27.1 下降到 -12.6（layer 17），幅度约 14.5
- 解释：早期层视觉 token 直接承载原始视觉特征，随层加深信息通过 cross-attention 逐渐扩散，单独破坏视觉 token 的边际影响下降

**Phase B (Layers 17-36): 波动平台期**
- 在 -12.6 到 -22.1 之间非单调波动
- **关键异常**：layer 17 局部最小值 (-12.6) → layer 18 回弹到 -18.2（单步变化 +5.6）
- 这是整条曲线最大的单步正向跳变，暗示 layer 18 存在特殊机制（可能与 MoE routing 有关）

**Phase C (Layers 37-47): 再次衰减**
- 从 -18.4 下降到 -12.7（output head proximity effect）

**Hard vs Easy 差异模式**：
- Phase A (layers 0-7)：Hard 比 Easy 大约 5-6 点（视觉依赖更强）
- Phase B 后半段 (layers 29-36)：差异收窄到 1-2 点（通用推理层）
- Phase C (layers 40-47)：差异再次扩大（Hard 需重新参考视觉细节）

### 1.2 Visual ICC 在 Layer 24-25 的跃升

- Layer 22→26: ICC 增长 0.087 (0.446 → 0.533)
- 此前 20 层（layer 2-22）总增长仅 0.059
- 与 EVD cliff=23 吻合，但需注意：
  - **反例**：layer 24 CKA=0.073（局部高点），layer 25 暴跌到 0.002
  - 这种 ICC↑ 但 CKA 不稳定的组合严重削弱了 assimilation 假说

### 1.3 Cross-Modal CKA 的高噪声问题

CKA 值在 0 到 0.078 之间剧烈跳动，**基本不可用**。

噪声来源：
1. 样本量不足（N=10），debiased estimator 需要对大集合子采样，引入随机性
2. 视觉 token (~775) 远多于文本 token，Gram 矩阵维度不匹配
3. Linear CKA 在此场景下理论上就无法可靠度量跨模态相似性

### 1.4 Text ER 末层急剧上升

- Layers 0-40: 19.6 → 1.5 → 4.2（U 型底部）
- Layers 40-47: 4.2 → 17.5（急剧回升）

这是 transformer 通用行为：output head 需要区分不同 token，hidden state 必须在末层重新分化为词汇空间表征。**与 assimilation 假说无关。**

### 1.5 截断实验的关键解读

**注意统计精度**：30 samples 基数下，accuracy 最小分辨率 = 1/30 ≈ 0.033（每步变化仅代表 1 个 sample）。

- Layers 0-36：Hard ~0.2, Easy ~0.2-0.33（稳定低值）
- Layer 36-40：阶跃跳变
  - Hard: 0.233 → 0.600（+11 samples）
  - Easy: 0.267 → 0.700（+13 samples）

**核心推论**：Easy layers 20-32 的"梯度恢复"（每步+0.033）只是 1 个 sample 的随机波动，不应过度解读。**Layer 36-40 的阶跃才是真正的信号。**

---

## 二、假说验证

### 2.1 Modality Assimilation 假说（视觉信息在 layer 20-25 转移）

| 证据类型 | 内容 |
|---------|------|
| 强支持 | Visual ICC 在 layer 24-25 加速上升 |
| 强支持 | Visual ER 单调增长（可解释为同化到更高维空间） |
| 弱支持 | EVD gap Hard-Easy = +3.6 层（方向正确但偏小） |
| **强反对** | **截断实验：截断 layer 24 后 accuracy 仍为 0.2，与截断 layer 0 相同** |
| 强反对 | layer 18 因果贡献回弹（若转移在 layer 17 完成则不应回升） |
| 反对 | CKA 在 layer 20-25 无系统性上升 |

**综合判断：Modality Assimilation 假说在时间定位上是错误的。** 信息转移不发生在 layer 20-25，而是发生在 layer 37-40。

### 2.2 替代假说

**假说 A：Late-Stage Visual Readout（晚期视觉读出）**

视觉信息处理四阶段：
1. **Layers 0-17**: 视觉特征渐进抽象化（在视觉 token 内部完成）
2. **Layers 18-36**: 视觉-文本交叉推理（双方 residual stream 都在更新，但信息尚未转移）
3. **Layers 37-40**: 关键读出窗口（视觉信息被大规模读出到文本 token）
4. **Layers 41-47**: 解码准备（文本 token 主导）

**截断实验的阶跃点直接支持此假说。**

**假说 B：MoE Routing-Driven Phase Transition**

观察到的所有分段现象（因果贡献波动、ICC 跃升、截断阶跃）本质上由 MoE expert 路由模式决定。若 layer 37-40 有专门负责 "visual-to-text transfer" 的 expert 集合，阶跃自然产生。Layer 18 的回弹也可由 router 重新激活视觉 expert 解释。

---

## 三、方法论问题

### 3.1 截断实验的 artifact 风险

- **零值替换问题**：hidden state 设为 0 经过 LayerNorm 后变为非零值，向后传播虚假信号
- **测量对象混淆**：截断实验同时测量了"信息读取时间点"和"视觉 token 存在对 attention 的结构性影响"

建议改用 mean substitution 做对照实验。

### 3.2 EVD 定义的语义问题

- `EVD = max{l : delta(l) >= tau}`，tau=1e-6
- 但 mean_deltas 数组全为负值，理论上 EVD 应为 0
- cliff_boundary=23 的计算方式不透明，结论存疑
- **建议**：从根本上重新定义 EVD，或放弃此指标

### 3.3 统计功效限制

- 10 samples per category，30 samples per difficulty
- 截断实验分辨率 = 1/30 ≈ 0.033
- CKA 在 N=10 下不可靠
- **10 samples 足以检测 gross patterns（截断阶跃），但不足以精确定位层级（cliff=23）或检测细微趋势（CKA 变化）**

---

## 四、下一步实验优先级

### ① Attention Flow Quantification（最高优先级，成本低）

**目的**：直接验证 Late-Stage Readout vs Assimilation 竞争假说

**方法**：
- 每层提取 attention 权重
- 计算 text-to-visual 和 visual-to-text attention flow 总量
- 对于 MoE 模型额外追踪 router logit 分布

**预期**：
- Late-Stage Readout 成立 → text-to-visual attention 峰值在 layer 37-40
- Assimilation 成立 → 峰值在 layer 20-25

**成本**：一次 forward pass，提取 attention weights，无需修改模型

---

### ② 精细截断（Layer 36-44，步长 1）（高优先级，成本中等）

**目的**：精确定位 readout 窗口边界

**方法**：
- Layer 36-44 步长 1 截断（vs 当前步长 4）
- 样本扩展到 50+（分辨率提升到 0.02）
- 同时实验 zero-out 和 mean substitution 两种方式

**预期**：精确定位 2-3 层转折窗口；两种截断方式结果一致则 robust

**成本**：~900 次推理，4-8 GPU 小时

---

### ③ Text Token Visual Probing（中优先级，成本中高）

**目的**：检验文本 token 是否及何时获取视觉信息

**方法**：
- 每层提取文本 token hidden states
- 训练线性 probe 预测视觉属性（图表类型、文档布局等）

**预期**：
- Late-Stage Readout → probe accuracy 在 layer 37-40 急剧上升
- Assimilation → probe accuracy 在 layer 20-25 开始上升
- 信息不转移 → probe accuracy 始终低

**成本**：需存储所有层 hidden states（VRAM 密集），需 100+ samples，约 1 天

---

### ④ MoE Expert 路由分析（中优先级，成本低）

**目的**：验证 MoE Routing-Driven Phase Transition 假说

**方法**：
- 提取每层 router logits 和 expert assignment
- 计算视觉 token vs 文本 token 的 expert overlap（Jaccard index）
- 观察视觉 token 高频激活的 expert 集合如何随层变化

**预期**：若假说成立，layer 37-40 的 expert overlap 应急剧增加

**成本**：一次 forward pass + hook，极低

---

### ⑤ 纯文本截断对照实验（低优先级，成本低）

**目的**：验证截断阶跃是否是 output head proximity effect 的架构 artifact

**方法**：对纯文本 QA 任务做相同截断实验

**预期**：若是 proximity effect → 纯文本任务也在 ~layer 40 出现阶跃；若视觉特异 → 纯文本截断无此模式

---

## 五、论文 Narrative 建议

### 5.1 最 Defensible 的 Contribution

**放弃**：Modality Cliff / Geometry Collapse（数据不支持）

**采纳**：
> **"Late-Stage Visual Readout in MoE Vision-Language Models"**
>
> 在 Qwen3-VL-30B MoE 模型中，视觉 token 信息并非在中间层被逐步吸收（如 assimilation 假说），而是保持在视觉 token residual stream 中直到 layer 37-40 才被集中读出。这一 late-stage readout 模式通过截断实验明确支持，且与 MoE 架构的 expert routing 机制存在潜在关联。

**价值**：
1. 修正直觉性错误假设（"信息在中间层转移"）
2. 对 MoE VLM 效率优化有直接 implications：前 36 层可考虑视觉 token 稀疏化
3. 提出可验证的机制性假说（expert routing 驱动 readout 窗口）

### 5.2 从 NO-GO 中提取 Positive Story

Checkpoint NO-GO 本身有价值：
1. **几何坍缩框架不适用于 MoE 架构**：Visual ER 上升（非坍缩），MoE 稀疏激活可能天然防止表征坍缩
2. **提出 refined 版本**：MoE 模型中 modality 信息处理通过 expert routing convergence（"功能性同化"），而非 "几何坍缩"

### 5.3 建议论文结构

1. **Introduction**: MoE VLM 如何处理多模态信息？先前假说 vs 本文发现
2. **Method**: 因果追踪 + 截断实验 + 几何分析 pipeline
3. **Results**:
   - 3.1: 因果追踪揭示三段式处理模式
   - 3.2: 截断实验发现 late-stage readout 窗口（layer 37-40）
   - 3.3: 几何分析否定经典坍缩假说
4. **Analysis**: Late-Stage Readout + MoE Routing 假说
5. **Discussion**: 对 MoE VLM 效率优化的 implications

---

## 总结

| 实验 | 最可靠的信号 | 核心结论 |
|------|------------|---------|
| 截断实验 | ✅ 强信号 | 视觉信息在 layer 37-40 被集中读出 |
| 因果追踪 | ⚠️ 需谨慎 | 三段式处理模式；EVD cliff=23 解释存疑 |
| 几何分析 | ⚠️ 部分有效 | 否定坍缩假说；CKA 数据不可用 |

**最紧迫后续工作**：实验①（Attention Flow）+ 实验④（MoE Routing），成本低且能直接区分竞争假说。
