import gym

from easyRL.CartPole.DQN.Config import Config
from easyRL.CartPole.DQN.DQNetwork import DQN


class TrainTask:
    def __init__(self, env: gym.Env, cfg: Config, agent: DQN):
        self.agent = agent
        self.env = env
        self.cfg = cfg

    def train(self):
        print('开始训练!')
        print(f'环境：{self.cfg.env_name}, 算法：{self.cfg.algo_name}, 设备：{self.cfg.device}')
        # 记录奖励
        rewards = []
        # 记录加权后的奖励
        ma_reward = []
        steps = []
        # 训练步数
        for r in range(self.cfg.train_eps):
            reward_sum = 0
            rou_step = 0
            # 重置环境
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # 将经验放入缓存
                self.agent.push(state, action, reward, next_state, done)
                # 更新网络
                self.agent.update()
                reward_sum += reward
                rou_step += 1
                state = next_state
            if not ma_reward:
                ma_reward.append(reward_sum)
            else:
                ma_reward.append(ma_reward[-1] * 0.9 + 0.1 * reward_sum)
            rewards.append(reward_sum)
            steps.append(rou_step)
            # 更新智能体目标网络
            if r % self.cfg.target_update is 0:
                self.agent.target_network.load_state_dict(self.agent.policy_network.state_dict())
            print(
                f'Episode：{r + 1}/{self.cfg.test_eps}, Reward:{reward_sum:.2f}, Step:{rou_step:.2f} Epislon:{self.agent.epsilon():.3f}')
        return {'rewards': rewards, 'ma_rewards': ma_reward, 'steps': steps}
