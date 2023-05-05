import os.path

import gym
from stable_baselines3 import DQN

from easyRL.AtariGame.breakout.wrapper import BreakOutWrapper

model_dir = r'trained_models_test'


def test_model(num_episodes):
    done = False
    env = BreakOutWrapper(gym.make("ALE/Breakout-v5", render_mode="human"))
    # model = DQN.load(os.path.join(model_dir, "breakout__187500_steps.zip"), env=env)
    model = DQN(
        policy="CnnPolicy",
        env=env,
        device="cuda",
        verbose=1,
        buffer_size=1024,
        batch_size=128,
        gamma=0.9,
        tensorboard_log='logs',
        learning_starts=4096
    )
    for i in range(num_episodes):
        state = env.reset()
        while not done:
            action, predict_state = model.predict(state)
            state, reward, done, info = env.step(action)


if __name__ == '__main__':
    test_model(500)
