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
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

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
    rollout.make_sample_until(min_timesteps=50000),  #usually 50k
    rng=rng,
)

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.99,
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
    demo_batch_size=2048, #usually 2048
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


# Now will see reward of trajectories under the AIRL learnt reward model (for ERC calculation)
value_experiment = utils.load_diffusion_learnt_reward(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.env_seed,
)
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

diffusion=diffusion_experiment.ema
dataset = value_experiment.dataset
value_function = value_experiment.ema
env=dataset.env
renderer = value_experiment.renderer

guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()


# this was previously in unguided planning, but I dont think this works like that anymore
#policy = Policy(diffusion, dataset.normalizer)


## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs (idk what these mean)
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

# calls the guided policy class, instead of the normal policy class that was used for unguided planning
policy_diff = policy_config()

# CODE TO GET REWARDS OF BASE DIFFUSER TRAJ ACCORDING TO AIRL REWARD NET (USED TO CALCULATE ERC)
subset_indices=[i for i in range(100)]
train_dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0,sampler=SubsetRandomSampler(subset_indices))
#print(reward_net.__dict__)
values=torch.empty((0))
for i,data in enumerate(train_dataloader):
    curr_reward=0
    #print(data.trajectories.shape)
    traj=torch.cat((data.trajectories[:,:,:6],torch.zeros((data.trajectories.shape[0],data.trajectories.shape[1],1),dtype=torch.float32),data.trajectories[:,:,6:]),dim=-1)
    for step in range(3):
        done=torch.zeros(1, dtype=torch.bool)
        #if step==382:
         #   done=torch.ones(1, dtype=torch.bool)
        curr_reward+=reward_net.predict(traj[0,step,6:].unsqueeze(0).detach().numpy(),traj[0,step,:6].unsqueeze(0).detach().numpy(),traj[0,step+1,6:].unsqueeze(0).detach().numpy(),done)
    values=torch.cat((values,torch.tensor(curr_reward)))
torch.save(values.unsqueeze(dim=0),'logs/'+args.dataset+'/values_base_traj/values_AIRL.pt')
