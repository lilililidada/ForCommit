import datetime
import os
import sys
from pathlib import Path

import gym
import numpy as np

from easyRL.AtariGame.AirRaid.agent import DoubleDQN, DQNetwork
from easyRL.AtariGame.AirRaid.wrapper import EnvWrapper
from easyRL.util.utils import save_result_figure, plot_losses


class TrainTask:

    def __init__(self, game_env: gym.Env, train_round) -> None:
        super().__init__()
        self.env = game_env
        self.channel_size = 1
        self.action_dim = self.env.action_space.n
        self.agent = DQNetwork(self.channel_size, self.action_dim)
        self.train_round = train_round
        self.update_time = 1

    def train(self):
        print(f"开始训练")
        rewards = []
        # 加权值，用于对比观察震荡
        ma_reward = []
        steps = []
        loss_avg = []
        for i in range(self.train_round):
            print(f"round {i} start")
            reward_sum = 0
            step = 0
            state = self.env.reset()
            done = False
            loss_sum = []
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # 存入经验回放池
                self.agent.push(state, reward, action, next_state, done)
                if len(self.agent.buffer) > self.agent.batch_size:
                    # 更新网络
                    loss_sum.append(self.agent.update())
                state = next_state
                reward_sum += reward
                step += 1
            print(f"the total of reward is {reward_sum}, run step is {step}")
            # 更新目标网络
            if i % self.update_time == 0:
                self.agent.target.load_state_dict(self.agent.policy.state_dict())
            rewards.append(reward_sum)
            steps.append(step)
            if not ma_reward:
                ma_reward.append(reward_sum)
            else:
                ma_reward.append(ma_reward[-1] * 0.9 + 0.1 * reward_sum)
            if loss_sum:
                loss_avg.append(np.mean(loss_sum))
        return {'rewards': rewards, 'steps': steps, 'ma_reward': ma_reward, 'loss_avg': loss_avg}

    def test(self, test_round):
        for i in range(test_round):
            state = self.env.reset()
            done = False
            reward_sum = 0
            step = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                reward_sum += reward
                step += 1

    def load(self, path):
        self.agent.load(path)


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    env: gym.Env = gym.make("ALE/AirRaid-v5")
    env = EnvWrapper(env)
    task = TrainTask(env, 500)
    # 当前文件所在绝对路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 父路径
    parent_path = os.path.dirname(curr_path)
    # 添加路径到系统路径
    sys.path.append(parent_path)
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = curr_path + "/outputs/" + "model" + '/' + curr_time + '/models/'
    # 保存结果的路径
    result_path = curr_path + "/outputs/" + '/' + curr_time + '/results/'
    result = task.train()
    env.close()
    make_dir(model_path, result_path)
    print("train is end, start to save model")
    # 存储模型
    task.agent.save(path=model_path)
    # 存储训练结果
    save_result_figure(result['rewards'], result['ma_reward'], result_path)
    # 存储损失曲线
    plot_losses(result['loss_avg'], path=result_path)
