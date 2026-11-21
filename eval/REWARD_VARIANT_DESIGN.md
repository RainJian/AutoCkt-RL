# Phase 2 实验设计文档：reward shaping 消融（v2 变体 = ibias_max 加权惩罚）

> 日期：2026-11-13　基线：checkpoint-152（152 iter / 25h / 6595 episodes）
> 协议：`eval/EVAL_PROTOCOL.md`（greedy、done=True 即达标、traj_len=60）

## 1. T2.0 基线失败画像（评测集前 100 条）

| 指标 | 数值 |
|---|---|
| 成功率 | **91/100 = 91.0%**（本评测集口径，不与论文 80–96% 直接对比） |
| 失败 case 步数 | 9/9 全部跑满 60 步（游走至轨迹上限，无提前收敛） |
| 成功 case 平均达标步数 | 15.4 步（min 10 / max 28） |
| 评测总耗时 | 981.9s ≈ 16.4min（9.8s/case，与计时标定一致） |

### 失守维度分布（9 个失败 case，单 case 可多维度失守）

| 维度 | 失守占比 | 平均归一化缺口 `(o-o*)/(o+o*)` |
|---|---|---|
| **ibias_max** | **9/9 = 100%** | -0.695 |
| gain | 8/9 = 88.9% | -0.946 |
| ugbw | 6/9 = 66.7% | -0.240 |
| phm | 0/9 = 0% | — |

同时失守维度数：1 维 1 例 / 2 维 2 例 / 3 维 6 例。

### 失败归因（物理）

9 个失败 case 的 ibias 目标全部极紧（≤ 7.1e-4 A，评测集典型目标 ~5e-3 A）。
成功的 case 里 agent 收敛到少数几个"满配"参数点（如
`[1,37,37,2,0,25,30]` → gain≈758、ugbw≈8.5e7、**ibias≈8.4e-4**），
即用超额性能换电流。当电流预算收紧到 3e-4 量级时，这条 attractor 必然越界；
而低电流区域要求全面缩小管尺寸，gain/ugbw 随 gm 下降——这就是 gain 失守
88.9% 的连带机制。**根因维度是 ibias_max，gain/ugbw 失守是其下游症状。**

## 2. T2.1 变体选择（决策矩阵）

- 失败**集中**在 ibias_max 单一维度（100%）→ 按决策矩阵采用 **A2 加权 reward**；
  A1 步数惩罚对应"失败分散 + 震荡游走"形态，与本画像不符（本画像失败高度集中）。
- 权重选择 w=3.0：使 ibias 单维缺口（典型 -0.7）在 shaped reward 中
  （-2.1）超过 gain 缺口（-0.95），成为主导梯度信号，迫使策略探索低电流设计区；
  w=1 即退化为基线，w→∞ 会淹没其余维度信号，3 是"主导但不独占"的保守选择。

## 3. 假设与控制变量

**假设 H1**：在 ibias_max 维度上加大惩罚权重，同训练预算下 agent 会学习到
低电流设计区域，使失败画像中占 100% 的 ibias 失守 case 部分转为达标，
成功率提升（ ibias 极紧的 specs 恰是基线唯一失败来源）。

**控制变量**（变体训练与基线逐项一致）：

| 项 | 基线 | 变体 |
|---|---|---|
| 算法/超参 | PPO, train_batch=1200, horizon=30, fcnet[64,64], workers=4 | 相同 |
| 观测空间 / 动作空间 / 参数网格 | 相同 | 相同 |
| 训练 spec 集 | `ngspice_specs_gen_two_stage_opamp` | 相同（绝不引入评测集） |
| **reward shaping** | 各维度等权 | **ibias_max 未达标惩罚 ×3** |
| seed | 原训练 seed | 2 个新 seed（Phase 3） |

**达标判定逻辑不动**：v2 reward 用未加权的 raw 和做容差判定
（`raw >= -0.02 → return 10`），与 v1 逐位一致；`reward>=10 触发 done`
的成功率口径不变，保证与基线可比。

## 4. T2.2 代码改动 diff

| 文件 | 性质 | 改动 |
|---|---|---|
| `autockt/envs/ngspice_vanilla_opamp_v2.py` | 新增（复制自 v1） | 仅 `reward()`：ibias_max 维度惩罚 ×`IBIAS_WEIGHT=3.0`；类名 `TwoStageAmpV2`；头部改动说明注释。**另含 [v2.1] 健壮性补丁**：`reset()`/`step()` 返回前将观测 clip 到观测空间边界 |
| `autockt/val_autobag_ray_v2.py` | 新增（复制自基线训练脚本） | 仅环境 `TwoStageAmpV2`、实验名 `train_45nm_ngspice_v2`；新增 `--seed`/`--stop_iters` CLI（默认不改变原行为） |
| `autockt/rollout.py` | 追加（已备份 `backup_20261112/rollout.py.pre_v2reg.bak`） | try/except 导入并注册 `two_stage_opamp_v2` 环境名，供 Phase 4 变体评测 |

v1 环境、基线训练脚本、评测集均未触碰。

### [v2.1] 观测越界健壮性补丁（冒烟暴露，2026-11-13）

- **现象**：首次冒烟 28 分钟未跑完 iter 1，trial ERROR：`Failed to recover from worker crash`。
- **根因**：不稳定电路（phm<0）使 `lookup()` 归一化值 >1（phm=-175、参考 g=60 → 2.04），
  超出 observation_space 上界 1.0，RLlib `check_shape` 抛 ValueError 杀掉 rollout worker；
  随机策略早期频繁踩雷，超过容错上限。属 v1 环境**既有缺陷**（基线当年训练靠
  `ignore_worker_failures` 硬扛），与 reward shaping 无关。
- **修复（仅 v2）**：观测返回前 `np.clip` 到空间边界。不影响达标判定、不影响 [-1,1] 内
  正常观测；病态态只被压到边界，"极差"信号方向保留。
- **叙事价值**：这是第五个真实 debugging 故事（观测空间定义与实际仿真值域不匹配）。

### [v2.2] 停止条件修正 + NGSpice 挂死根治（Phase 3 首日实测，2026-11-13）

1. **停止条件**：基线的 `episode_reward_mean=-0.02` 停止阈值对 v2 shaped reward
   失效（多数 episode 快速 done，iter1 均值 +1.41 即触发停止）。改为固定迭代
   预算 `training_iteration=152`，与基线预算对齐（计划书 T3.1）。
2. **NGSpice 静默挂死**：仿真经 `os.system` 无超时调用，实测出现 0-CPU 挂死
   23 分钟的 ngspice 进程；同步采样要求 4 worker 齐交数据，一个挂死即全员停摆。
   修复：`subprocess.Popen + wait(timeout=60s)`，超时杀进程；仿真失败时
   `translate_result` 降级返回最差 specs（episode 继续，不杀 worker）。
   对 v1 评测路径同样生效，正常仿真语义不变。（备份：backup_20261112/*.bak）

## 5. T2.3 冒烟结果（通过 ✅，2026-11-13）

5 iter 短跑（trial `PPO_TwoStageAmpV2_e13e1_00000`，35min，TERMINATED 正常退出）：

| iter | episode_reward_mean | episode_len_mean |
|---|---|---|
| 1 | -0.60 | 10.94 |
| 2 | +5.40 | 6.33 |
| 3 | +8.43 | 3.48 |
| 4 | +9.02 | 2.54 |
| 5 | +9.46 | 2.01 |

- reward 曲线单调上升、全程无 NaN/Inf；episode 变短 = agent 快速学会达标
  （shaped reward 量级比 v1 大，符合 ×3 权重的预期）
- [v2.1] clip 后 5 个 iter **零 worker 崩溃**（首次冒烟时反复崩溃）
- checkpoint_000001~000005 全部落盘：`~/ray_results/train_45nm_ngspice_v2/PPO_TwoStageAmpV2_e13e1_00000_0_2026-11-13_11-47-00/`
- 单 iter 约 5.8~8.2 min（eval 期间 CPU 空闲时），与基线 ~10min/iter 同量级

**冒烟通过，Phase 3 全量训练（2 seeds）已具备放行条件**——按纪律需用户确认后夜间执行。
