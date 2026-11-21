# AutoCkt 评测协议（EVAL PROTOCOL）

> 本协议为所有后续实验（baseline、变体、各 checkpoint）的**唯一评测标准**。任何实验结果只有在完全遵守本协议时才可比。

## 1. 评测集
- 文件：`autockt/gen_specs/ngspice_specs_gen_two_stage_opamp_eval`
- 规模：1000 条 unseen design specs，由 `gen_specs.py` 采样逻辑生成（生成后固定为文件，所有实验共用同一份，绝不重新生成、绝不用于训练）
- 训练集：`autockt/gen_specs/ngspice_specs_gen_two_stage_opamp`（备份见 `backup_20261112/`），仅用于训练

## 2. 环境与算法
- 环境：`two_stage_opamp`（`autockt/envs/ngspice_vanilla_opamp.py`），reward/obs 定义与训练时完全一致
- 算法：PPO（Ray RLlib，checkpoint 对应训练配置，见 `params.json`）
- 动作选择：`compute_action`（greedy，取概率最大动作）

## 3. 成功判据
- 单条 spec 轨迹长度上限 `traj_len = 60`
- episode 内**任一时刻**环境返回 `done=True`（即 reward >= 10，全部指标同时达标）判定为**达标（SUCCESS）**，与论文统计 reached specs 的口径一致
- ⚠️ 不使用 `reward_total >= 9` 判据：先前步骤的负分会把"先扣分后达标"的 near-miss 误杀（smoke run 实测 10 条中 3 条被误杀）
- 成功率 = 达标条数 / 评测条数

## 4. 记录指标（每条 spec 写入 CSV）
`case_id, goal_specs, final_specs, steps, reward_total, success, elapsed_sec`
- 汇总指标：成功率、平均达标步数（仅成功 case）、失败 case 的 spec 维度分布（Phase 4 归因用）

## 5. 对照实验规范
- baseline 与变体使用**同一份评测集**、相同 `traj_len`、相同成功判据
- 变体仅改动声明的变量（如 reward shaping），达标判定逻辑不得修改
- 结论表述仅限"在本评测集上 X% → Y%"，不与论文 80–96% 数字直接对比（spec 分布不同）

---
生成日期：2026-11-12 | 环境：<conda>\envs\autockt（Python 3.7）| 硬件：16 逻辑核 / 32GB RAM / RTX 5060 Ti（不使用）
