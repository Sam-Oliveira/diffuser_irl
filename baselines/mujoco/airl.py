import numpy as np
import gymnasium as gym
#import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet,BasicShapedRewardNet
from imitation.util.networks import RunningNorm
import json
import torch
import os

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.types import Trajectory
import d4rl
from gymnasium.spaces import Box
from imitation.data.rollout import rollout as roll_traject
from gymnasium import spaces
from collections import OrderedDict
import diffuser.utils as utils
import diffuser.sampling as sampling
from datasets import load_dataset
# NOTE! NEED TO INSTALL GYMNASIUM-ROBOTICS FOR THE UMAZE ENV!


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

rng = np.random.default_rng(13)
seed=13
#print(gym.envs.registry.keys())


env = make_vec_env(
    'HalfCheetah-v4',
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    env_make_kwargs={'exclude_current_positions_from_observation':False}
)

# This is the env with 18 obs. But dataset only has 100k obs, not 1M like for diffuser I believe
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals-HalfCheetah-v1",
    venv=env,
)


rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=50000), 
    rng=rng,
)


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=seed,
)
reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
env.seed(seed)

learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
airl_trainer.train(800000)  # Train for 800_000 steps to match expert.

learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training),u"\u00B1",np.std(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training),u"\u00B1",np.std(learner_rewards_before_training))

