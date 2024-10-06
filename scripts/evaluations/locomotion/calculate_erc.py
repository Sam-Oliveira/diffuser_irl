import json
import numpy as np
from os.path import join
import pdb
import torch
import os
import pandas as pd
import gym
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
import gym


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'


"""
This script loads trajectories from the base diffuser's dataset (for umaze or large maze) and calculates 
the value of those trajectories under a certain (possibly learnt) reward model, saving these values (line 114).
After values have been created for all algorithms, it calculates the ERC between the true reward model's values
and a learnt reward function's values. 
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion_learnt_reward( 
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

subset_indices=[i for i in range(100)]
train_dataloader=DataLoader(dataset, batch_size=100, shuffle=False,num_workers=0,sampler=SubsetRandomSampler(subset_indices))


#Getting value according to value function
values_true=torch.empty((0))
"""
for data in train_dataloader:
    
    #Getting value according to value function
    trajectories=data.trajectories[:,:,:]
    conditions=data.conditions[0].detach().cpu()
    time=torch.zeros((trajectories.shape[0]),dtype=torch.float)
    trajectories=trajectories.to(torch.float)
    values=value_function(trajectories,conditions,time)
    print(values)
    print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))
    torch.save(values,'logs/'+args.dataset+'/values_base_traj/values_MMD_Matern.pt')
    
    # Generate returns of each trajectory
    for i in range(data.trajectories.shape[0]):
        #print(i)
        #print(trajectory.shape)
        total_reward=0
        #for trajectory in shape
        for step in range(data.trajectories.shape[1]):
            observation=env.set_state(np.concatenate((np.asarray([0.0]),np.asarray(data.trajectories[i,step,6:14]))),np.asarray(data.trajectories[i,step,14:]))
            next_observation, reward, terminal, _ = env.step(data.trajectories[i,step,:6].detach().cpu().numpy())
            ## print reward and score
            total_reward += reward
        #print(values_true.shape)
        #print(torch.from_numpy(np.asarray(total_reward)).shape)
        #print(total_reward)
        values_true=torch.cat((values_true,torch.unsqueeze(torch.from_numpy(np.asarray(total_reward)),dim=0)))
torch.save(values_true,'logs/'+args.dataset+'/values_base_traj/values_True.pt')
"""


# TO CALCULATE METRICS AFTER VALUE FILES HAVE BEEN CREATED
path = 'logs/'+args.dataset+'/values_base_traj/'
files = [pos_json for pos_json in sorted(os.listdir(path))]
values=torch.empty((0,100))
for file in files:
    print(file)
    if file=='values_AIRL.pt':
        values=torch.cat((values,torch.load(path+file)))
    elif file=='values_True.pt':
       values=torch.cat((values,torch.unsqueeze(torch.load(path+file),dim=0)))
    else:
        values=torch.cat((values,torch.load(path+file).transpose(0,1)))

list=['AIRL','MMD_Gauss','MMD_Matern','MSE']
for value_index in range(len(values)-1):
    pearson_coeff=scipy.stats.pearsonr(np.ravel(values[-1].detach().cpu().numpy()),np.ravel(values[value_index].detach().cpu().numpy()))
    pearson_dist=np.sqrt(1-pearson_coeff.statistic)/np.sqrt(2)
    print(list[value_index]+": ")
    print('Pearson coeff: '+str(pearson_coeff.statistic))
    print('Pearson distance: '+str(pearson_dist)+"\n")




