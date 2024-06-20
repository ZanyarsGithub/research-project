from stable_baselines3.common.logger import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


for (num_demos, num_comparisons) in [(20, 200)]:
    arr = []
    times = []
    for seed in [27, 35, 42, 51]:
        df = read_csv(f"./ppo_cartpole_tensorboard/irlhf/noisy demos/{num_demos}/{seed}/{num_comparisons}/progress.csv")
        arr.append(df['rollout/ep_rew_mean'].to_numpy())
        times = df['time/time_elapsed'].to_numpy()
    rew_avg = np.mean(arr, axis=0)
    plt.plot(times, rew_avg, label=f"{num_demos} demonstrations {num_comparisons} comparisons seed {seed}")
# plt.rcParams.update({'font.size': 7})
plt.title("AIRL + RLHF with sub-optimal demonstrations reward learning evaluation")
plt.xlabel("Time elapsed")
plt.ylabel("Reward mean")
plt.legend()
plt.show()