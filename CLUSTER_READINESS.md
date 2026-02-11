# GAP 2.0 集群就绪性评估（面向 Qwen3-VL-30B + Slurm）

> 目的：基于 `discussion.md` 与 `Claude.md` 的研究意图，以及当前仓库代码实现状态，判断“是否可以直接 clone 到集群跑实验”，并给出最短落地路径。

## 结论（先看）

- **研究方案完整度：高（8/10）**。Phase 1 的假设链路、checkpoint 逻辑、指标体系（EVD/ER/ICC/CKA）都比较清晰。
- **工程实现完整度：中高（7/10）**。LLaVA 路径基本能跑通，模块边界清晰。
- **你当前集群目标（Qwen3-VL-30B）可用性：中低（4/10）**。
  - 主要不是 Slurm 问题，而是**代码里仍有 LLaVA 假设**，对 Qwen-VL（尤其 Qwen3-VL）尚未完成端到端适配。

## 你想做的事 vs 代码现状

### 你想做的事

你希望把仓库 clone 到集群后，用已部署的 `qwen3-vl-30B` 通过 Slurm 直接跑批处理/实验。

### 现状匹配度

当前仓库虽然有 `qwen_vl.yaml`，但关键执行路径还存在两类阻塞：

1. **视觉 token 位置定位未实现（Qwen 分支直接 `NotImplementedError`）**
2. **Phase 1 runner 里的输入构造是 `prepare_llava_input(...)`（LLaVA prompt 模板）**

所以：
- 你可以 clone、建环境、提交作业。
- 但若直接切到 Qwen3-VL 运行 Phase 1 runner，**高概率在前处理中断**。

## 风险清单（按优先级）

### P0（必须先处理）

1. **Qwen-VL 视觉 token 区间识别缺失**
   - `src/models/model_loader.py` 的 Qwen 分支未实现视觉 token 定位逻辑。
2. **Runner 强绑定 LLaVA 输入模板**
   - `run_phase1_causal.py`/`run_phase1_truncation.py` 当前调用的是 `prepare_llava_input`。

### P1（建议尽快处理）

3. **Qwen3-VL-30B 的显存/并行策略未在主实验脚本中参数化**
   - 你给出的脚本是 `tp-size 4` 的批处理范式，但当前 GAP2.0 runner 并未统一支持这套并行入口。
4. **数据集在线下载依赖**
   - 首次运行会拉 HF 数据，集群网络策略可能影响稳定性。

### P2（可后置）

5. **评估指标较“研究原型”**
   - 例如 answer 的 containment check 在 DocVQA/ChartQA 上可能偏粗，建议后续换更严格 evaluator。

## 建议落地路线（最短可执行）

1. **先做 LLaVA 小样本冒烟**（确认 pipeline、结果文件、plot 都正常）
2. **再做 Qwen3-VL 适配**（至少补齐：输入模板 + visual token mask/range）
3. **最后在 Slurm 上跑 Phase 1 全流程**（causal → truncation → geometry → checkpoint）

如果你追求“最快出第一批结果”，建议先在 LLaVA 上拿到全流程，再迁移到 Qwen3-VL 做主实验。

## 与你给的 Slurm 脚本如何对接

你的 Slurm 模板思路是对的：
- 资源配置（4GPU / 64G / 2h）
- 先做 CUDA 设备探测
- 再执行主任务 + 校验脚本

对 GAP2.0 建议替换为：
- 分阶段脚本（`run_phase1_causal.py`、`run_phase1_truncation.py`、`run_phase1_geometry.py`、`run_checkpoint_eval.py`）
- 每阶段落盘到 `results/`，方便断点恢复和复查。

## 你现在可以直接做的三件事

1. `pip install -r requirements.txt` / 或按你现有 conda 环境补依赖。
2. 用 `--num_samples 5` 先跑一轮（任意可用模型）验证工程链路。
3. 再决定是否立刻补 Qwen3-VL 适配代码，或者先用 LLaVA 拿 baseline。

## 总评（给你一个可决策版本）

- **科研设计：值得继续，结构扎实。**
- **工程可迁移性：需要一次“Qwen3-VL 接口补齐”才能真正上集群稳定跑。**
- **是否建议现在 clone 到集群：建议 clone，但不要直接假设 Qwen3-VL 可无缝跑通。**

