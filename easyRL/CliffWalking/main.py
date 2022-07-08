import datetime
import os
import sys

import gym
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from easyRL.CliffWalking.agent import QLearning
from easyRL.CliffWalking.train import TrainTask
from easyRL.envs.gridworld_env import CliffWalkingWapper

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
algo_name = "Q-learning"
env_name = "CliffWalking-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QlearningConfig:
    def __init__(self):
        self.algo_name = algo_name
        self.env_name = env_name
        self.device = device
        self.train_eps = 400
        self.self_eps = 30
        self.gama = 0.9
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 300
        self.lr = 0.1


class PlotConfig:
    def __init__(self):
        self.algo_name = algo_name
        self.env_name = env_name
        self.device = device
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    # 包装环境，后续可以研究下有什么用
    env = CliffWalkingWapper(env)
    env.seed(seed)
    # 状态维度
    state_dim = env.observation_space.n
    # 动作维度
    action_dim = env.action_space.n
    agent = QLearning(cfg, state_dim, action_dim)
    return env, agent


def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()


if __name__ == '__main__':
    learning_cfg = QlearningConfig()
    plot_cfg = PlotConfig()
    # 开始训练
    env, agent = env_agent_config(learning_cfg)
    task = TrainTask(learning_cfg, env, agent)
    rewards, ma_rewards = task.train()
    # todo 保持训练结果到本地文件
    plot_rewards(rewards, ma_rewards, plot_cfg)
    # 测试
    env, agent = env_agent_config(learning_cfg, seed=10)
    rewards, ma_rewards = task.test(learning_cfg, env, agent)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

