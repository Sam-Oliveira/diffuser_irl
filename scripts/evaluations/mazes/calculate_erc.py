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
import scipy

"""
This script loads trajectories from the base diffuser's dataset (for umaze or large maze) and calculates 
the value of those trajectories under a certain (possibly learnt) reward model, saving these values (line 114).
After values have been created for all algorithms, it calculates the ERC between the true reward model's values
and a learnt reward function's values. 
"""


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion_learnt_reward( # changed this function, instead of just being load_diffusion()
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
subset_indices=[i for i in range(100)]
train_dataloader=DataLoader(dataset, batch_size=100, shuffle=False,num_workers=0,sampler=SubsetRandomSampler(subset_indices))

# TO GENERATE FILES

for data in train_dataloader:
    trajectories=data.trajectories[:,:384,:]
    conditions=data.conditions[0].detach().cpu()
    time=torch.zeros((trajectories.shape[0]),dtype=torch.float)
    trajectories=trajectories.to(torch.float)

    # NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING

    # NEED TO MAKE SURE VALUE FUNCTION IN FOLDER IS THE TRUE REWARD MODEL AND NOT SMTHG WE LEARNT
    values=value_function(trajectories,conditions,time)
    print(values)
    print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))
    torch.save(values,'logs/'+args.dataset+'/values_base_traj/values_MMD_Matern.pt')

# TO CALCULATE METRICS AFTER VALUE FILES HAVE BEEN CREATED
path = 'logs/'+args.dataset+'/values_base_traj/'
files = [pos_json for pos_json in sorted(os.listdir(path))]
values=torch.empty((0,100))
for file in files:
    print(file)
    if file=='values_AIRL.pt':
        values=torch.cat((values,torch.load(path+file)))
    else:
        values=torch.cat((values,torch.load(path+file).transpose(0,1)))

list=['AIRL','MMD_Gauss','MMD_Matern','MSE']
for value_index in range(len(values)-1):
    pearson_coeff=scipy.stats.pearsonr(np.ravel(values[-1].detach().cpu().numpy()),np.ravel(values[value_index].detach().cpu().numpy()))
    pearson_dist=np.sqrt(1-pearson_coeff.statistic)/np.sqrt(2)
    print(list[value_index]+": ")
    print('Pearson coeff: '+str(pearson_coeff.statistic))
    print('Pearson distance: '+str(pearson_dist)+"\n")