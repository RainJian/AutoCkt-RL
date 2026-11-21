import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= 配置区域 =================
# 请替换为你最新的日志文件夹路径
log_dir = r"C:\Users\<user>\ray_results\train_45nm_ngspice\PPO_TwoStageAmp_42d37_00000_0_2025-12-19_00-26-10"

# 你的 Batch Size (在 val_autobag_ray.py 里写的那个数字)
BATCH_SIZE = 1200
# ===========================================

csv_path = os.path.join(log_dir, "progress.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # --- 关键步骤 1: 构造 X 轴数据 (仿真步数) ---
    # 如果 csv 里有 'timesteps_total' 列，直接用；如果没有，通过迭代轮数换算
    if 'timesteps_total' in df.columns:
        x_data = df['timesteps_total']
    else:
        # 公式: 仿真步数 = 迭代轮数 * 每轮的 Batch Size
        x_data = df['training_iteration'] * BATCH_SIZE

    # --- 关键步骤 2: 提取 Y 轴数据 (奖励) ---
    mean_reward = df['episode_reward_mean']
    min_reward = df['episode_reward_min']
    max_reward = df['episode_reward_max']

    # --- 关键步骤 3: 开始画图 (完全复刻论文风格) ---
    plt.figure(figsize=(10, 6))

    # [修改点] 兼容旧版 Matplotlib 的样式名称
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')  # 如果seaborn样式也没有，就用ggplot兜底

    plt.grid(True, linestyle='-', alpha=0.3, color='gray')

    # 1. 画阴影区域 (Min-Max Range) - 浅绿色
    plt.fill_between(x_data,
                     min_reward,
                     max_reward,
                     color='#8FBC8F',  # DarkSeaGreen
                     alpha=0.3,  # 透明度
                     label='Min-Max Range')

    # 2. 画均值曲线 (Mean Reward) - 深绿色实线
    plt.plot(x_data,
             mean_reward,
             color='#006400',  # DarkGreen
             linewidth=2,
             label='Mean reward per trajectory')

    # 3. 添加 0 分基准线 (论文里那条黑线)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # --- 图表装饰 ---
    plt.title("Mean reward per trajectory for the two-stage transimpedance amplifier", fontsize=12)
    plt.xlabel("Number of simulations", fontsize=11)  # X轴标题改成仿真步数
    plt.ylabel("Mean reward per trajectory", fontsize=11)

    # 设置图例
    plt.legend(loc='lower right', frameon=True)

    # 调整坐标轴范围 (让图看起来更像)
    # 论文大概是从 -60 到 +10
    # 注意：如果数据很少导致min_reward没有那么低，这个ylim可能会让图看起来空空的
    # 建议先注释掉 ylim 看看自动缩放的效果，如果需要再打开
    # plt.ylim(bottom=min(min_reward.min(), -60), top=15)

    plt.xlim(left=0, right=x_data.max())

    plt.tight_layout()
    plt.show()

    print(f"✅ 图表绘制完成！X轴已转换为总仿真步数 (最大值: {x_data.max()})")

else:
    print(f"❌ 错误：找不到文件 {csv_path}")