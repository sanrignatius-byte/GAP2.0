# GAP 2.1 下一步实验计划（基于当前代码与 Phase 1 结果）

## 1) 先统一术语：从“Collapse”改为“Transfer/Assimilation”

当前数据不支持“rank collapse”（visual ER 整体上升），更稳健的主线是：

- `visual token` 的**直接可干预性**随层变化；
- 信息可能发生了**跨模态转移/同化**，而不是简单“消失”。

论文叙事建议改为：

> We study *where* and *how* visual evidence is transferred into language-side states, and whether this transfer is controllable.

---

## 2) 先修两个定义问题（否则后续结论会继续打架）

### 2.1 Truncation 的真实实现语义

当前实现不是 drop token，而是：

- 对 `layer > L` 的每层 forward 输出，将视觉 token hidden states 置零；
- 视觉 token 仍然留在序列里并参与后续注意力结构（仅内容为 0 向量）。

这意味着该实验测的是“**视觉内容被置零后的可恢复性**”，不是“彻底移除视觉 token 对图结构的影响”。

### 2.2 Causal delta 的符号与指标口径

当前定义：

- `delta = s_clean - s_patched`；
- `s` 是正确答案 token 的 log-prob 累积和。

理论上，若 patch 造成伤害，delta 应偏正；但你结果里大量为负，优先考虑是评估位点/answer token 对齐或 patch 噪声标定导致的量纲问题，而不应直接解释为“patch 让模型更好”。

---

## 3) GAP 2.1 三个必须做的实验（按优先级）

## 3.1 双通道因果追踪：Visual vs Text patch transfer curve（最高优先级）

在同一层 `l`，并行测两条曲线：

1. patch visual tokens（已存在）；
2. patch text tokens（新增，与 visual 使用同口径 corruption）。

目标信号：

- 若存在转移，随着层数增加应出现：
  - `|Δ_visual(l)|` 下降；
  - `|Δ_text(l)|` 上升；
  - 两者在某层附近交叉。

该交叉层将比 `cliff_boundary` 更有解释力。

## 3.2 State Transplant 截断（解决“分布漂移”争议）

流程：

1. 正常前向到层 `l`，缓存 text hidden states；
2. 从层 `l+1` 起视觉置零/删除，但把 text states 固定为缓存值继续前向；
3. 评估性能。

解释：

- 若性能几乎不变，说明到 `l` 为止视觉信息已写入 text states；
- 若性能显著下降，说明后续仍需视觉流继续参与计算。

## 3.3 Counterfactual swap 替代 gaussian（增强语义可解释性）

在 patch 时把样本 A 的视觉 states 替换成样本 B（同 batch 或检索匹配）而不是高斯噪声。

判断：

- 输出跟着 B 变化：仍强依赖视觉通道；
- 输出基本不变：视觉信号在该层后不再被读取（或已被转移到文本侧）。

---

## 4) 指标层面的最小重构（避免“23 vs 40”再冲突）

- 弱化单点 `cliff_boundary`；改为 **transfer window**（例如 [L_start, L_end]）；
- EVD 从 hard threshold 改为积分型面积指标（AUC above baseline）；
- 报告必须分三层：
  - token-level causal（visual/text 双通道）；
  - behavior-level truncation（含 state transplant 版本）；
  - geometry-level（ER/ICC 仅作辅助，不作为因果证据）。

---

## 5) 两周执行清单（可直接排期）

### Week 1

- [ ] 新增 `TextPatchingHook`，复用现有 `PatchingHook` 框架；
- [ ] 新增 dual-trace runner：同批样本同时输出 `delta_visual` 与 `delta_text`；
- [ ] 修复并校验 answer log-prob 对齐逻辑（至少做 3 个手工 case）；
- [ ] 重新跑 30-50 样本的小规模 sanity。

### Week 2

- [ ] 实现 `state_transplant` 模式 truncation；
- [ ] 实现 `counterfactual_swap` corruption；
- [ ] 绘制 transfer curve + crossing layer + transfer window；
- [ ] 产出 GAP 2.1 问题定义与方法段草稿。

---

## 6) 论文 framing（可直接替换摘要主句）

> The core issue is not geometric rank collapse. Instead, visual evidence undergoes an often-uncontrolled cross-modal transfer into language-side states. GAP 2.1 targets this transfer trajectory—its timing, structure, and robustness—via causal dual-tracing and transfer-aware regularization.
