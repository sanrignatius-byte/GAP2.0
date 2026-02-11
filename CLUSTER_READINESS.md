# GAP 2.0 集群就绪性评估（面向 Qwen3-VL-30B + Slurm）

> 目的：基于 `discussion.md` 与 `Claude.md` 的研究意图，结合当前代码状态，给出“可以怎么在集群上落地”的可执行建议。

## 一句话结论

- **实验设计是靠谱的**：Phase 1 的因果-几何双证据链（causal/truncation/geometry/checkpoint）是完整的。
- **工程上已经具备“可迁移骨架”**：现在脚本已支持按模型家族分流输入构造（LLaVA / Qwen-VL）。
- **上 Qwen3-VL-30B 前仍建议先小样本冒烟**：重点确认视觉 token 定位在你本地部署权重下是否稳定。

## 我对“具体实现”的建议（按优先级）

### P0：先把 Phase 1 跑成“稳定流水线”

1. **最小集冒烟（每类 5~10 样本）**
   - 顺序固定：`causal -> truncation -> geometry -> checkpoint_eval`。
   - 目标不是指标高，而是确保每个阶段都产出 JSON + 图，避免大作业后期才发现中间环节断裂。

2. **把 cliff boundary 固化为输入参数**
   - 第一次 causal 算出来后，在 truncation 和 geometry 阶段显式传 `--cliff_boundary`（或在配置里固化），减少重复不确定性。

3. **为 Qwen3-VL 做一次“视觉 token 区间 sanity check”**
   - 随机抽 3 个样本，打印 `visual_range` 和 token 长度，确认范围不是空、也不是越界。

### P1：提高实验可信度（你论文最需要）

4. **把答案评估从 containment 升级为“任务敏感打分”**
   - ChartQA/DocVQA 先做数值归一化 + 单位容错；
   - VQAv2 保留宽松匹配。

5. **τ（EVD 阈值）做数据驱动校准**
   - 先跑小样本画 Delta 分布，再决定是否从 0.01 调到 0.005 或 0.02。

6. **加 bootstrap 置信区间**
   - 关键图（EVD gap、truncation drop、ER ratio）给 95% CI，论文说服力会明显提升。

### P2：面向 Qwen3-VL-30B 的工程细化

7. **显存/吞吐策略参数化**
   - 将 `max_model_len`、batch size、tp-size（若你后续接 vLLM/并行后端）写入配置文件，避免脚本里硬编码。

8. **结果目录标准化**
   - 用 `results/{model}/{dataset}/{timestamp}/...`，便于多轮实验追踪和回溯。

## 推荐执行路径（你现在就能做）

1. 用 `scripts/slurm/run_gap2_phase1_example.slurm` 先跑一个 10 样本版本。
2. 若四阶段都成功产出，再把样本拉到 50/100。
3. 产出第一轮结果后再校准 τ 和评估器，进入“论文级”结果生产。

## 风险提醒

- Qwen3-VL 的 tokenizer / special token 定义在不同版本可能有差异；如果视觉 token 区间异常，优先检查 chat template 与 special token id。
- 首次下载数据集可能受集群网络影响，建议预热缓存。

## 总评

- **现在是“可以上集群试跑”的状态**，但建议先小样本验证关键路径，再做全量。
- 你的方案本身方向对，下一步价值最大的是“稳住流程 + 提升评估严谨性”。
