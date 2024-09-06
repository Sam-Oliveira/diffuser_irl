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


#---------------------------------- main loop ----------------------------------#
env=dataset.env
num_envs=20

# create multiple envs
envs=gym.vector.SyncVectorEnv([

    lambda: gym.make(args.dataset) for i in range(num_envs)

])

observation=envs.reset()


## observations for rendering
rollout = [observation.copy()] #1st observation I think

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
#max_steps=128
#max_steps=200
max_steps=5
learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))
for t in range(max_steps):

    if t % 10 == 0: print(args.savepath, flush=True)


    ## save state for rendering only
    state=envs.observations.copy()

    ## IMPAINTING
    #target = env._target 
    #conditions = {0: observation,diffusion.horizon - 1: np.array([*target, 0, 0])}


    ## format current observation for conditioning (NO IMPAINTING)
    conditions = {0: envs.observations}

    #i think basically we take 1 step, and plan again every time! (in rollout image. in plan, it's just the plan at first step)
    action, samples = policy(conditions, batch_size=args.batch_size,diff_conditions=True,verbose=args.verbose)
    

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()
    trajectories.append(np.concatenate((actions,envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((torch.from_numpy(actions),torch.from_numpy(envs.observations)),axis=-1))


    next_observation, reward, terminal, _ = envs.step(samples.actions[:,0].detach().cpu().numpy())

    ## print reward and score
    total_reward += reward

    ## update rollout observations. Note this does not include actions! Rollout is a list of nparrays, each of them is the current state at a step
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps. Just basically renders one of the items in batch.
    # can change render method if i want to render the entire batch. not worth it now
    #samples=samples._replace(observations=samples.observations[[0]])
    #samples=samples._replace(actions=samples.actions[[0]])
    #samples=samples._replace(values=samples.values[[0]])
    #logger.log(t, samples, state[[0]], np.stack(rollout,axis=1)[[0],:,:])


    if terminal.any():
        break

## write results to json file at `args.savepath`
logger.finish(t, 0, total_reward.tolist(), bool(terminal.any()), diffusion_experiment, value_experiment)

