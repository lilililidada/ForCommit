import math
import random

import numpy as np
import torch
from torch import nn, optim

from easyRL.CartPole.DQN.Config import Config
from easyRL.CartPole.DQN.MLP import MLP
from easyRL.CartPole.memory import ExpBuffer


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 策略网络，每次用该网络与环境交互
        self.policy_network = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device=cfg.device)
        # 目标网络，待策略函数更新到一定程度后，用策略网络更新该网络
        self.target_network = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device=cfg.device)
        # 优化器
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=cfg.lr)
        # 经验回放
        self.exp_memory = ExpBuffer(cfg.memory_capacity)
        # e-greedy 策略
        self.epsilon_gen = self.__epsilon()
        self.epsilon = lambda: next(self.epsilon_gen)
        # 动作维度
        self.action_dim = action_dim

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        self.exp_memory.push(state, action, reward, next_state, done)

    def update(self):
        # 当memory中不满足一个批量时，不更新策略
        if len(self.exp_memory) < self.cfg.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.exp_memory.sample(
            self.cfg.batch_size)
        # 构造tensor
        states = torch.tensor(state_batch, device=self.cfg.device, dtype=torch.float)
        actions = torch.tensor(action_batch, device=self.cfg.device).unsqueeze(1)
        rewards = torch.tensor(reward_batch, device=self.cfg.device, dtype=torch.float)
        next_states = torch.tensor(next_state_batch, device=self.cfg.device, dtype=torch.float)
        done = torch.tensor(np.float32(done_batch), device=self.cfg.device)
        # 计算当前状态(s_t,a)对应的Q(s_t, a)
        p_values = self.policy_network(states).gather(dim=1, index=actions)
        # 计算下一时刻的状态(s_t_,a)对应的Q值
        next_values = self.target_network(next_states).max(1)[0].detach()
        # 计算期望数值,对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expect_rewards = self.cfg.gamma * next_values * (1 - done) + rewards
        # 损失函数，方根损失
        loss = nn.MSELoss()(p_values, expect_rewards.unsqueeze(1))
        # 优化更新模型
        self.optimizer.zero_grad()
        # 误差逆传播
        loss.backward()
        # clip防止梯度爆炸  没看懂，后续研究
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        # 进行一步优化
        self.optimizer.step()

    def choose_action(self, state):
        """
        e-greedy 策略

        @param state:  状态
        @return: action
        """
        if random.random() > self.epsilon():
            # 用网络计算出动作值
            with torch.no_grad():
                t_state = torch.tensor(state, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_network(t_state)
                # 选择Q值最大的动作
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def __epsilon(self):
        time = 0
        while True:
            time += 1
            yield self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * math.exp(
                -1. * time / self.cfg.epsilon_decay)

    def save(self, path):
        torch.save(self.target_network.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_network.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            param.data.copy_(target_param.data)


if __name__ == '__main__':
    instance = DQN(1, 1, Config())
    for i in range(500):
        instance.test()
