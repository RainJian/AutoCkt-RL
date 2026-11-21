import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 请确保这个路径是你存放 progress.csv 的文件夹路径
log_dir = r"C:\Users\<user>\ray_results\train_45nm_ngspice\PPO_TwoStageAmp_42d37_00000_0_2025-12-19_00-26-10"
# ===========================================

csv_path = os.path.join(log_dir, "progress.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # 尝试使用美观的样式
    try:
        plt.style.use('ggplot')
    except:
        pass

    # ==========================================
    # 图表 1: 训练收敛趋势 (含最高点标注)
    # ==========================================
    plt.figure(1, figsize=(12, 7))  # 稍微大一点

    # 1. 画主曲线
    plt.plot(df['training_iteration'], df['episode_reward_mean'], color='#1f77b4', linewidth=2, label='Mean Reward')

    # 2. 画阴影 (Min-Max)
    if 'episode_reward_min' in df.columns and 'episode_reward_max' in df.columns:
        plt.fill_between(df['training_iteration'],
                         df['episode_reward_min'],
                         df['episode_reward_max'],
                         color='#1f77b4', alpha=0.15, label='Min-Max Range')

    # 3. 寻找并标注最高点 (最好的模型) !!! 核心修改部分 !!!
    # 找到平均奖励最大的一行
    max_row = df.loc[df['episode_reward_mean'].idxmax()]
    max_reward = max_row['episode_reward_mean']
    max_iter = int(max_row['training_iteration'])

    # 画一个红色的点
    plt.scatter(max_iter, max_reward, color='red', s=100, zorder=10, edgecolors='white', linewidth=2,
                label='Best Model')

    plt.title('1. Learning Convergence & Best Model Selection', fontsize=16, fontweight='bold')
    plt.xlabel('Training Iterations (Checkpoint ID)', fontsize=12)
    plt.ylabel('Episode Reward Mean', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    print("=" * 30)
    print(f"最好的模型出现在第 【{max_iter}】 轮")
    print(f"对应的奖励分数为：{max_reward:.4f}")
    print(f"请去 ray_results 文件夹里找：checkpoint_{max_iter:06d}")
    print("=" * 30)

    # ==========================================
    # 图表 2: 设计效率
    # ==========================================
    plt.figure(2, figsize=(10, 6))
    plt.plot(df['training_iteration'], df['episode_len_mean'], color='#ff7f0e', linewidth=2)
    plt.title('2. Design Efficiency (Episode Length)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

else:
    print(f"错误：找不到CSV文件，请检查路径：\n{csv_path}")