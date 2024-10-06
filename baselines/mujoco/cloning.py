import numpy as np
import gymnasium as gym
#import gym
from stable_baselines3.common.evaluation import evaluate_policy
import json
import torch
import os
from os.path import join

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

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')


rng = np.random.default_rng(13)
seed=13

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

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

learner_rewards_before_training, _ = evaluate_policy(
    bc_trainer.policy, env, 100, return_episode_rewards=True,)

bc_trainer.train(n_epochs=30)

learner_rewards_after_training, _ = evaluate_policy(
    bc_trainer.policy, env, 100, return_episode_rewards=True,
)

print("mean reward before training:", np.mean(learner_rewards_before_training),u"\u00B1",np.std(learner_rewards_before_training))
print("mean reward after training:", np.mean(learner_rewards_after_training),u"\u00B1",np.std(learner_rewards_after_training))

policy=rollout.policy_to_callable(bc_trainer.policy,env)
torch.save(bc_trainer.policy,'logs/'+args.dataset+'/learnt_behaviour/BC/weights.pt')