import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

class RewardThresholdCallback(BaseCallback):

    def __init__(self, reward_threshold: int, frame:int, validations: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.frame = frame
        self.validations = validations
        self.reward_threshold = reward_threshold
        self.last_n_mean_rewards = np.zeros(frame)
        self.counter = 0
        self.andagain = 0


    def _on_step(self) -> bool:
        continue_training = True
        if self.counter == self.frame-1:
            rew_mean = np.mean(self.last_n_mean_rewards) * 500
            print(rew_mean)
            if rew_mean < self.reward_threshold:
                self.last_n_mean_rewards = np.zeros(self.frame)
                self.andagain = 0
            else:
                print(self.andagain, '       ', self.validations)
                self.andagain += 1
                if self.andagain >= self.validations:
                    continue_training = False
            self.counter = 0
        else:
            self.last_n_mean_rewards[self.counter] = self.locals['rewards'][-1]
            self.counter += 1

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because reward threshold exceeded"
            )
        return continue_training
