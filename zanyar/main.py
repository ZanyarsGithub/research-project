import numpy as np
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms import preference_comparisons
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.policies.exploration_wrapper import ExplorationWrapper

import wandb

# Change the seed, number of expert demos
# and number of comparisons here. Rest of the
# code does not change. 
SEED = 27
NUM_DEMOS = 60
NUM_COMPARISONS = 500

rng = np.random.default_rng(0)

FAST = False

if FAST:
    N_RL_TRAIN_STEPS = 100_000
else:
    N_RL_TRAIN_STEPS = 500_000

venv = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)
expert = DQN.load('dqn-cartpole-v0-subop-474+-64-rew.zip')

rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=NUM_DEMOS),
    rng=np.random.default_rng(SEED),
)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)
venv.seed(SEED)
airl_trainer.train(N_RL_TRAIN_STEPS)
venv.seed(SEED)

learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)
irl_learner = PPO(
    env=learned_reward_venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
)
tmp_path = f"./ppo_cartpole_tensorboard/irl/noisy demos/{NUM_DEMOS}/{SEED}/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
irl_learner.set_logger(new_logger)
irl_learner.learn(700_000)

n_eval_episodes = 100
reward_mean, reward_std = evaluate_policy(irl_learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)
learner = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    n_steps=2048 // venv.num_envs,
    clip_range=0.1,
    ent_coef=0.01,
    gae_lambda=0.95,
    n_epochs=10,
    gamma=0.97,
    learning_rate=2e-3,
)

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng,
)
gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
preference_model = preference_comparisons.PreferenceModel(reward_net)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=learner,
    reward_fn=reward_net,
    venv=venv,
    exploration_frac=0.3,
    rng=rng,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=60,  # Set to 60 for better performance
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=100,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
)

pref_comparisons.train(
    total_timesteps=500_000,
    total_comparisons=NUM_COMPARISONS,
)

venv = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)
learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)

rlhf_learner = PPO(
    env=learned_reward_venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5
)

tmp_path = f"./ppo_cartpole_tensorboard/irlhf/noisy demos/{NUM_DEMOS}/{SEED}/{NUM_COMPARISONS}/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
rlhf_learner.set_logger(new_logger)
rlhf_learner.learn(1_000_000)

n_eval_episodes = 100
reward_mean, reward_std = evaluate_policy(rlhf_learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

