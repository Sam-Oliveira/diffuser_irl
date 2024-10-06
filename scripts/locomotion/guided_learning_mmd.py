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
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'


"""
This script learns a reward model using the MMD Loss.
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
train_dataloader=DataLoader(dataset, batch_size=256, shuffle=False,num_workers=0,sampler=SubsetRandomSampler(subset_indices))

epochs=500
loss = MMD_loss()
optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-3)

loss_array=[]
for e in range(epochs):
    print("EPOCH "+str(e))
    curr_loss=0
    terms=0

    for targets in train_dataloader:

        observations=targets.conditions[0].detach().cpu()
        conditions={0:observations}
        action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

        # No sliding window
        sample_actions=samples.actions
        sample_observations=samples.observations

        predictions=torch.cat((sample_actions,sample_observations),dim=-1) 
        targets.trajectories.to(args.device)

        loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets.trajectories,start_dim=1))

        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data
        print(loss_value.detach().cpu().numpy())
        curr_loss+=loss_value.detach().cpu().numpy()
        terms+=1

        optimizer.step()

        optimizer.zero_grad()

    loss_array.append(curr_loss/terms)

    if e%10==0 or e==epochs-1:
        torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD.pt'.format(f=e+1))
    plt.figure()
    plt.plot(range(len(loss_array)),loss_array)
    plt.xlabel('Epoch Number',fontsize=12)
    plt.ylabel('MMD Loss',fontsize=12)
    print(loss_array)
    plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD.pdf',format="pdf", bbox_inches="tight")

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD.pt'.format(f=epochs))


plt.figure()
plt.plot(range(len(loss_array)),loss_array)
plt.xlabel('Epoch Number',fontsize=12)
plt.ylabel('MMD Loss',fontsize=12)
print(loss_array)
plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD.pdf',format="pdf", bbox_inches="tight")
plt.show()