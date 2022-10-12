import math
import random

import torch
import numpy as np

from easyRL.AtariGame.AirRaid.model.network import NeuralNetwork, NeuralNetwork2, AdvantageActorCritic
from easyRL.AtariGame.AirRaid.model.memory import ExperiencePool, PPOExperiencePool
from easyRL.bean.myrl import Reinforcement


class DQNetwork(Reinforcement):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        # 配置信息
        self.hidden_dim = 32
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 200
        self.gamma = 0.5
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # e贪婪
        self.epsilon = lambda study_round: 0.01 + (0.95 - 0.01) * math.exp(-1. * study_round / 1000)
        self.update_time = 0
        # 初始化模型
        self.policy = NeuralNetwork(self.state_dim, self.action_dim).to(device=self.device)
        self.target = NeuralNetwork(self.state_dim, self.action_dim).to(device=self.device)
        # 经验池
        self.buffer = ExperiencePool(100000)
        # 优化参数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        sample_buffer = self.buffer.sample(self.batch_size)
        states, rewards, actions, next_states, dones = zip(*sample_buffer)
        # 将数据包装成tensor，方便运算
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        next_sate_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(np.float32(dones), dtype=torch.float32, device=self.device)
        # 获取当前状态的预测值
        q_values = self.policy(states_tensor)
        # 待入action获取当前所选择的动作的价值
        q_values = torch.gather(q_values, index=action_tensor, dim=1)
        # 下一状态的预测值
        # next_values = self.target(next_sate_tensor)
        # 获取下一状态所预测的动作
        # next_action_tensor = torch.unsqueeze(torch.max(next_values, dim=1)[1], dim=1)
        # 获取下一状态价值
        # next_values = torch.gather(next_values, dim=1, index=next_action_tensor).squeeze(1)
        next_q_values = self.target(next_sate_tensor).max(1)[0].detach()
        # 时序差分
        q_target = self.gamma * (1 - done_tensor) * next_q_values + reward_tensor
        # 计算损失函数
        loss: torch.Tensor = torch.nn.MSELoss()(q_values, torch.unsqueeze(q_target, dim=1))
        # 优化模型
        # zero_grad清除上一步所有旧的gradients from the last step
        self.optimizer.zero_grad()
        # 误差逆传播
        loss.backward()
        # 限制梯度，防止梯度爆炸
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        # 进行一步优化
        self.optimizer.step()
        return loss.item()

    def choose_action(self, state) -> int:
        self.update_time += 1
        if random.random() < self.epsilon(self.update_time):
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
                # 用神经网络算出动作
                action = torch.max(self.policy(state_tensor), dim=1)[1].item()
        return action

    def push(self, state, reward, action, next_state, done):
        self.buffer.put(state, reward, action, next_state, done)

    def load(self, path):
        self.target.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            param.data.copy_(target_param.data)

    def save(self, path):
        torch.save(self.target.state_dict(), path + 'dqn_checkpoint.pth')


class Config:

    def __init__(self) -> None:
        super().__init__()
        # 配置信息
        self.hidden_dim = 32
        self.batch_size = 500
        self.gamma = 0.99
        self.lr = 0.00001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleDQN(Reinforcement):

    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.cfg = Config()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 初始化模型
        self.policy = NeuralNetwork2(self.state_dim, self.action_dim).to(device=self.cfg.device)
        self.target = NeuralNetwork2(self.state_dim, self.action_dim).to(device=self.cfg.device)
        # e贪婪
        self.epsilon = lambda study_round: 0.01 + (0.95 - 0.01) * math.exp(-1. * study_round / 1000)
        self.choose_time = 0
        # 经验池
        self.buffer = ExperiencePool(2000)
        # 优化参数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.cfg.lr)

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return
        states, rewards, actions, next_states, dones = zip(*self.buffer.sample(self.cfg.batch_size))
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.cfg.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.cfg.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.cfg.device).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.cfg.device)
        dones_tensor = torch.tensor(np.int64(dones), dtype=torch.int64, device=self.cfg.device)
        q_values = torch.gather(self.policy(states_tensor), dim=1, index=actions_tensor)
        next_q_values = self.policy(next_states_tensor)
        next_target_values = self.target(next_states_tensor)
        next_target_q_values = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = rewards_tensor + self.cfg.gamma * next_target_q_values * (1 - dones_tensor)
        loss = torch.nn.MSELoss()(q_values, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def choose_action(self, state):
        self.choose_time += 1
        if random.random() < self.epsilon(self.choose_time):
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(dim=0)
                # 用神经网络算出动作
                action = torch.max(self.policy(state_tensor), dim=1)[1].item()
        return action

    def push(self, state, reward, action, next_state, done):
        self.buffer.put(state, reward, action, next_state, done)

    def load(self, path):
        self.target.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            param.data.copy_(target_param.data)

    def save(self, path):
        torch.save(self.target.state_dict(), path + 'dqn_checkpoint.pth')


class A2CAlgorithm(Reinforcement):
    def __init__(self, state_dim, action_dim):
        self.cfg = Config()
        # hyperparameter
        self.batch_size = self.cfg.batch_size
        self.learn_rate = self.cfg.lr
        self.gamma = self.cfg.gamma
        self.expected_repeat_time = 1
        self.pool_size = (self.batch_size ** 2) // self.expected_repeat_time
        self.epsilon = lambda study_round: 0.05 + (0.9 - 0.05) * math.exp(-1. * study_round / 1000)
        # env
        self.state_dim = state_dim
        self.action_dim = action_dim
        # other
        self.experience_pool = ExperiencePool(self.pool_size)
        self.choose_time = 1
        self.device = self.cfg.device
        # model
        self.actor_critic = AdvantageActorCritic(self.state_dim, self.action_dim, (32, 32)).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), self.learn_rate)
        self.loss_function = torch.nn.MSELoss()

    def update(self):
        states, rewards, actions, next_states, dones = zip(*self.experience_pool.sample(self.batch_size))
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(np.int64(dones), dtype=torch.int64, device=self.device)
        dist, predict_values = self.actor_critic(states_tensor)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        _, predict_next_values = self.actor_critic(next_states_tensor)
        advantage = rewards_tensor + predict_next_values * (1 - dones_tensor) - predict_values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = self.loss_function(predict_values, rewards_tensor + predict_next_values * (1 - dones_tensor))
        loss = actor_loss + critic_loss - 0.001 * entropy
        self.optimize(loss)
        return loss.item()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.actor_critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def choose_action(self, state):
        self.choose_time += 1
        if random.random() < self.epsilon(self.choose_time):
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
            dist, _ = self.actor_critic(state_tensor)
        return dist.sample().cpu().numpy()[0]

    def interaction(self, env):
        reward_sum = 0
        step = 0
        state = env.reset(seed=int(1000 * random.random()), return_info=False)
        env.step()
        done = False
        loss_sum = []
        while not done:
            if random.random() < self.epsilon(self.choose_time):
                action = random.randrange(self.action_dim)
            else:
                action = self.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # 存入经验回放池
            self.push(state, reward, action, next_state, done)
            # 保证学习之前，经验池里面有足够多的经验
            if len(self.experience_pool) > self.batch_size * 5:
                # 更新网络
                loss_sum.append(self.update())
            state = next_state
            reward_sum += reward
            step += 1
        return reward_sum, loss_sum, step

    def push(self, state, reward, action, next_state, done):
        self.experience_pool.put(state, reward, action, next_state, done)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path + 'dqn_checkpoint.pth')


class PPO2Algorithm(A2CAlgorithm):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.experience_pool = PPOExperiencePool(self.pool_size)
        self.ppo_epsilon = 0.5

    def update(self):
        states, rewards, actions, next_states, dones, old_probs = zip(
            *self.experience_pool.sample(self.batch_size))
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(np.int64(dones), dtype=torch.int64, device=self.device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        dist, predict_values = self.actor_critic(states_tensor)
        log_probs = dist.log_prob(actions_tensor)
        ratio = torch.exp(log_probs - old_probs_tensor)
        entropy = dist.entropy().mean()
        next_dist, predict_next_values = self.actor_critic(next_states_tensor)
        advantages = rewards_tensor + self.gamma * predict_next_values * (1 - dones_tensor) - predict_values
        critic_loss = torch.mean(((rewards_tensor + self.gamma * predict_next_values - predict_values) ** 2) / 2)
        actor_loss = - torch.mean(
            torch.min(ratio, torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)) * advantages)
        total_loss = actor_loss + critic_loss - 0.001 * entropy
        self.optimize(total_loss)
        return total_loss.item()

    def interaction(self, env):
        loss_sum = []
        steps = []
        all_rewards = []
        while sum(steps) < 500:
            reward_sum = 0
            step = 0
            state = env.reset(seed=int(1000 * random.random()))
            done = False
            transactions = []
            rewards = []
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
                dist, predict_value = self.actor_critic(state_tensor)
                action = dist.sample().cpu().numpy()[0]
                action_tensor = torch.tensor(action, dtype=torch.int8, device=self.device).unsqueeze(dim=0)
                log_prob = dist.log_prob(action_tensor)
                next_state, reward, done, _ = env.step(action)
                transactions.append([state, None, action, next_state, done, log_prob.item()])
                rewards.append(reward)
                state = next_state
                reward_sum += reward
                step += 1
                # 打破摆烂
                # if self.is_nothing_to_do(rewards):
                #     done = True
                #     for i in range(-1, -100, -1):
                #         rewards[i] = -10
            rewards = self._compute_reward(rewards)
            for i in range(len(rewards)):
                transactions[i][1] = rewards[i]
            for transaction in transactions:
                self.experience_pool.put(transaction)
            all_rewards.append(reward_sum)
            steps.append(step)
            for _ in range(step // 5):
                # 保证学习之前，经验池里面有足够多的经验
                if len(self.experience_pool) > self.batch_size * 15:
                    # 更新网络
                    loss_sum.append(self.update())
        return np.mean(all_rewards), loss_sum, np.mean(steps)

    def _compute_reward(self, rewards):
        result = [rewards[-1]]
        for i in range(len(rewards) - 2, -1, -1):
            if rewards[i - 1] < 0:
                reward = rewards[i] + 0.3 * result[-1]
            else:
                reward = rewards[i] + self.gamma * result[-1]
            result.append(reward)
        return result[::-1]

    def is_nothing_to_do(self, rewards):
        if len(rewards) < 1000:
            return False
        for i in range(-1, -1000, -1):
            if rewards[i] != 0:
                return False
        return True
