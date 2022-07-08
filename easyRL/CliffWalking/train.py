import gym
from easyRL.envs.gridworld_env import CliffWalkingWapper


class TrainTask:
    def __init__(self, cfg, env, agent):
        self.cfg = cfg
        self.env = env
        self.agent = agent

    def train(self):
        # 记录奖励
        rewards = []
        # 记录滑动平均数
        ma_rewards = []
        for i_ep in range(self.cfg.train_eps):
            # 记录每个回合的奖励
            ep_reward = 0
            # 重置环境，开始新回合
            state = self.env.reset()
            done = False
            while not done:
                # 根据算法选择一个动作
                action = self.agent.choose_action(state)
                # 根据选择的动作与环境进行一次交互
                next_state, reward, done, _ = self.env.step(action)
                # 根据交互的结果优化更新算法
                self.agent.update(state, reward, next_state, done, action)
                state = next_state
                ep_reward += reward
            rewards.append(ep_reward)
            if ma_rewards:
                # 这是在干嘛？
                ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
            else:
                ma_rewards.append(ep_reward)
            print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, self.cfg.train_eps, ep_reward))
        print('完成训练！')
        return rewards, ma_rewards

    def test(self, cfg, env, agent):
        print('开始测试！')
        print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
        for item in agent.Q_table.items():
            print(item)
        rewards = []  # 记录所有回合的奖励
        ma_rewards = []  # 滑动平均的奖励
        for i_ep in range(cfg.test_eps):
            ep_reward = 0  # 记录每个episode的reward
            state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
            while True:
                action = agent.predict(state)  # 根据算法选择一个动作
                next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
                state = next_state  # 更新状态
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
            else:
                ma_rewards.append(ep_reward)
            print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
        print('完成测试！')
        return rewards, ma_rewards
