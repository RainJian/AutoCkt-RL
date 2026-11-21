# T0.5 计时标定结论（2026-11-12）

## 标定方法
checkpoint-152，评测集前 10 条，`--traj_len 60`，greedy（explore=False），两次复跑验证确定性。

## 实测数据
| 项目 | 数值 |
|---|---|
| agent 加载 + 环境初始化 | ~50 s（一次性开销） |
| 成功 case 平均耗时 | ~3 s（2–4 步达标） |
| 失败 case 平均耗时 | ~30–34 s（60 步跑满） |
| 平均每条 spec | ~11 s（9/10 成功率下） |

## 规模外推（含启动开销）
| 评测规模 | 预计耗时 |
|---|---|
| 100 条 | ~20 分钟 |
| 500 条 | ~1.6 小时 |
| 1000 条 | ~3 小时 |

结论：**Phase 1 直接跑 1000 条完全可行**，无需切片并行；多个 checkpoint 依次串行，一个下午/晚上可全部完成。

## 过程修正记录（重要）
1. **成功判据修正**：原 `reward_total >= 9` 会把"先扣分后达标"的 near-miss 误杀（10 条中 3 条）。改为论文口径：episode 内任一时刻 `done=True`（reward>=10）即达标。实测同 checkpoint 5/10 → 8/10。
2. **动作选择修正**：原 `compute_action(state)` 默认随机采样，同一 checkpoint 两次结果不一致（5/10 vs 8/10）。改 `explore=False`（greedy/mode 动作）后两次完全复现（9/10），且成功率更高。
3. **环境问题**：在 WorkBuddy bash 中运行老 Python 3.7 需 `env -u PYTHONPATH`，否则被注入的 sitecustomize 与 pathlib 冲突导致 ray.init 失败。此规则适用于本机器上所有 autockt 环境命令。
