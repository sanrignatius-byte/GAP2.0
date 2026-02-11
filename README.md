# GAP2.0

## 集群运行与评估

- 项目当前状态与集群迁移建议见 `CLUSTER_READINESS.md`。
- Slurm 任务模板见 `scripts/slurm/run_gap2_phase1_example.slurm`。
- Phase 1 runner 现已支持通过 `--model_name` 覆盖模型路径，便于对接集群本地权重。

- Causal 阶段支持 `--adaptive_tau --tau_ratio 0.1` 进行 EVD 阈值自适应校准。
