import math
from collections import defaultdict

import numpy as np


class QLearning:
    def __init__(self, cfg, state_dim, action_dim):
        self.action_dim = action_dim
        # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表
        self.Q_table = defaultdict(lambda: np.zeros(action_dim))
        self.epsilon_decay = cfg.epsilon_decay
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon = 0
        self.gamma = cfg.gama
        self.lr = cfg.lr

    def choose_action(self, state):
        """
        选择策略
        """
        self.sample_count += 1
        # 计算ε，每次迭代后，ε应该递减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1 * self.sample_count / self.epsilon_decay)
        # Q_table中存着在某一状态下，所有动作的reward值
        if np.random.uniform(0, 1) > self.epsilon:
            # 选择Q(s,a)最大值对应的动作
            action = np.argmax(self.Q_table[str(state)])
        else:
            # 随机选择动作
            action = np.random.choice(self.action_dim)
        return action

    def update(self, state, reward, next_state, done, action):
        """
        更新策略
        """
        q_predict = self.Q_table[str(state)][action]
        if done:
            q_target = reward
        else:
            q_target = self.gamma * np.max(self.Q_table[str(next_state)]) + reward
        self.Q_table[str(state)][action] += self.lr * (q_target - q_predict)
