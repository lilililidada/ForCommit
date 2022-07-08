import gym
import torch
import numpy as np

from easyRL.CartPole.DQN.Config import Config
from easyRL.CartPole.DQN.DQNetwork import DQN
from easyRL.CartPole.DQN.MLP import MLP
from easyRL.CartPole.TrainTask import TrainTask
from easyRL.util.utils import save_results_1, make_dir
from easyRL.util.utils import plot_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step += 1
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode：{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}')
    print('完成测试！')
    env.close()
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps}


def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    print(f"n states: {n_states}, n actions: {n_actions}")
    agent = DQN(n_states, n_actions, cfg)  # 创建智能体
    if cfg.seed != 0:  # 设置随机种子
        torch.manual_seed(cfg.seed)
        env.reset(seed=cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


if __name__ == '__main__':
    cfg = Config()
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"n states: {state_dim}, n actions: {action_dim}")
    model = MLP(state_dim, action_dim)
    agent = DQN(state_dim, action_dim, cfg)
    if cfg.seed != 0:
        torch.manual_seed(cfg.seed)
        env.reset(seed=cfg.seed)
        np.random.seed(cfg.seed)
    res_dic = TrainTask(env, cfg, agent).train()
    env.close()
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results_1(res_dic, tag="train", path=cfg.result_path)
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")
    env, agent = env_agent_config(cfg)
    # 导入模型
    agent.load(path=cfg.model_path)
    res_dic = test(cfg, env, agent)
    # 保存结果
    save_results_1(res_dic, tag='test',
                   path=cfg.result_path)
    # 画出结果
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="test")
