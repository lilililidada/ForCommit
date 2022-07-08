import datetime
import os
import sys

import torch

# 当前文件所在绝对路径
curr_path = os.path.dirname(os.path.abspath(__file__))
# 父路径
parent_path = os.path.dirname(curr_path)
# 添加路径到系统路径
sys.path.append(parent_path)
# 获取当前时间
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Config:

    def __init__(self):
        # 算法名称
        self.algo_name = "DQN"
        self.env_name = "CartPole-v0"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 随机种子
        self.seed = 123
        # 训练回合数
        self.train_eps = 200
        # 测试回合数
        self.test_eps = 20

        """
        算法超参数
        """
        # 强化学习中的折扣因子
        self.gamma = 0.95
        # e-greedy策略中的初始epsilon
        self.epsilon_start = 0.9
        # e-greedy策略中的终止epsilon
        self.epsilon_end = 0.01
        # e-greedy策略中epsilon的衰减率
        self.epsilon_decay = 500
        # 学习率
        self.lr = 0.0001
        # 经验回放的容量
        self.memory_capacity = 100000
        # mini-batch SGD中的批量大小
        self.batch_size = 64
        # 目标网络的更新频率
        self.target_update = 4
        # 网络隐层
        self.hidden_dim = 256
        # 保存结果的路径
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        # 保存模型的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        # 是否保存图片
        self.save = True
