import json
import numpy as np
from os.path import join
import pdb
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import gym

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling
from torch.utils.data import DataLoader
from diffuser.models.helpers import MMD
from torch.utils.data import SubsetRandomSampler

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'


"""
This script outputs multiple trajectories based on unguided diffusion.
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('unguided_plan')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

logger = logger_config()
#---------------------------------- main loop ----------------------------------#
env=dataset.env
num_envs=100

# create multiple envs
envs=gym.vector.SyncVectorEnv([

    lambda: gym.make(args.dataset) for i in range(num_envs)

])

observation=envs.reset()

## observations for rendering
rollout = [observation.copy()] 

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
max_steps=1000
learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))

for t in range(max_steps):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state=envs.observations.copy()

    ## format current observation for conditioning (NO IMPAINTING)
    conditions = {0: envs.observations}

    action, samples = policy(conditions, batch_size=args.batch_size,diff_conditions=True)

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()
    trajectories.append(np.concatenate((actions,envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((torch.from_numpy(actions),torch.from_numpy(envs.observations)),axis=-1))

    next_observation, reward, terminal, _ = envs.step(samples.actions[:,0].detach().cpu().numpy())

    total_reward += reward*(0.99**t)

    ## update rollout observations. Note this does not include actions! Rollout is a list of nparrays, each of them is the current state at a step
    rollout.append(next_observation.copy())

    if terminal.any():
        break

## write results to json file at `args.savepath`
print(total_reward)
print(total_reward.tolist())
torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/Unguided/trajectories.pt')
