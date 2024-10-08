import json
import numpy as np
from os.path import join
import pdb
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling
from torch.utils.data import DataLoader
from diffuser.models.helpers import MMD
from torch.utils.data import SubsetRandomSampler
from diffuser.models.helpers import MMD,MMD_loss

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-replay-v2'
    config: str = 'config.locomotion'

"""
This script calculates the mean (and std) action norm of trajectories generated by different algorithms
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion( # changed this function, instead of just being load_diffusion()
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.env_seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = value_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema

#ValueGuide (guiddes.py) takes ValueFunction (temporal.py) as its model
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

# calls the guided policy class, instead of the normal policy class that was used for unguided planning
policy = policy_config()

#---------------------------------- main loop ----------------------------------#
env=dataset.env
observation = env.reset()

learnt_trajectories_1=torch.load('logs/'+args.dataset+'/learnt_behaviour/MSE/trajectories.pt')
learnt_trajectories_2=torch.load('logs/'+args.dataset+'/learnt_behaviour/MMD_Gauss/trajectories.pt')
learnt_trajectories_3=torch.load('logs/'+args.dataset+'/learnt_behaviour/MMD_Matern/trajectories.pt')
learnt_trajectories_4=torch.load('logs/'+args.dataset+'/learnt_behaviour/Unguided/trajectories.pt')
#learnt_trajectories=torch.cat((learnt_trajectories_1,learnt_trajectories_2,learnt_trajectories_3,learnt_trajectories_4),dim=0)

print('MSE')
norm=torch.linalg.vector_norm(learnt_trajectories_1[:,:,:6], ord=2, dim=(2))
print(torch.mean(norm))
print(torch.std(norm))

print('MMD_Gauss')
norm=torch.linalg.vector_norm(learnt_trajectories_2[:,:,:6], ord=2, dim=(2))
print(torch.mean(norm))
print(torch.std(norm))

print('MMD_Matern')
norm=torch.linalg.vector_norm(learnt_trajectories_3[:,:,:6], ord=2, dim=(2))
print(torch.mean(norm))
print(torch.std(norm))

print('Unguided')
norm=torch.linalg.vector_norm(learnt_trajectories_4[:,:,:6], ord=2, dim=(2))
print(torch.mean(norm))
print(torch.std(norm))
