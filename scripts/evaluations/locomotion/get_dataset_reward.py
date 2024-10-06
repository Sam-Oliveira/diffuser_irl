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
from torch.utils.data import SubsetRandomSampler,RandomSampler,SequentialSampler
from diffuser.models.helpers import MMD,MMD_loss

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-replay-v2'
    config: str = 'config.locomotion'


"""
This script outputs the mean (and std) reward and action norm of trajectories in a given dataset (argument).
"""


#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion( 
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

# Load expert trajectories
# dataset has 996000 4-step parts of trajectories. here we just select 10k first ones
subset_indices=[i for i in range(10000)]
#train_dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0,sampler=SequentialSampler(dataset))
train_dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0,sampler=RandomSampler(dataset,True,250*1000))
values=torch.empty((0))
actions=torch.empty((0,4))
stds=[]
values_episode=torch.empty((0))

stop= 250*1000
for i,targets in enumerate(train_dataloader):

    #get reward
    val=torch.sum(targets.values[0,:]*(0.99**(i%250)),dim=0,keepdim=True)

    values_episode=torch.cat((values_episode,val))

    # get action norm
    norm=torch.linalg.vector_norm(targets.trajectories[:,:,:6], ord=2, dim=2)
    actions=torch.cat((actions,norm),dim=0)

    if (i+1)%250==0:
        print((i+1)/250)
        values_episode=torch.sum(values_episode,dim=0,keepdim=True)
        values=torch.cat((values,values_episode))
        values_episode=torch.empty((0))
    if i>=stop:
        break

print(torch.mean(actions))
print(torch.std(actions))
print(torch.mean(values))
print(torch.std(values))






