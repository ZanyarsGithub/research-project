import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from rewardcallback import RewardThresholdCallback

from stable_baselines3.common.evaluation import evaluate_policy

seed = 42
reward_threshold = -2
# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("MountainCar-v0")
env = Monitor(env, log_dir, allow_early_resets=False)

cb = RewardThresholdCallback(reward_threshold=-2, frame=4000, verbose=1, validations=5)
# eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=cb, verbose=1)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save('dqn-mountaincar-v0-optimal')

del model 

env = gym.make("MountainCar-v0")
model = DQN.load('dqn-mountaincar-v0-optimal')
m, st = evaluate_policy(model=model, env=env, n_eval_episodes=100)

print(m, "     ", st)
results_plotter.plot_results([log_dir], int(1e10), results_plotter.X_TIMESTEPS, "Cart pole reward improvement")
plt.show()

