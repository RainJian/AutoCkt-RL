import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init(num_gpus=0)

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter
config_train = {
            #"sample_batch_size": 200,
            "train_batch_size": 1200,
            #"sgd_minibatch_size": 1200,
            #"num_sgd_iter": 3,
            #"lr":1e-3,
            #"vf_loss_coeff": 0.5,
            "horizon":  30,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 4,
            "env_config":{"generalize":True, "run_valid":False},
            "ignore_worker_failures": True,   # 忽略单个 Worker 崩溃，尝试继续训练
            "recreate_failed_workers": True, # 如果 Worker 崩溃，自动重新创建一个
            }

#Runs training and saves the result in ~/ray_results/train_ngspice_45nm
#If checkpoint fails for any reason, training can be restored
#在循环里不断调用底层的 train() 函数（这个函数里包含了随机探索/噪声逻辑）。
#只要是用 tune 跑起来的，默认就是开启 Exploration（探索/噪声）的。
#PPO 算法在训练迭代过程中，内部机制会自动在动作输出上叠加随机概率分布，以实现探索，这也是为什么训练初期分数低的原因。”
if not args.checkpoint_dir:
    trials = tune.run_experiments({
        "train_45nm_ngspice": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": -0.02},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    tune.run_experiments({
        "restore_ppo": {
        "run": "PPO",
        "config": config_train,
        "env": TwoStageAmp,
        #"restore": trials[0]._checkpoint.value},
        "restore": args.checkpoint_dir,
        "checkpoint_freq":1},
    })
