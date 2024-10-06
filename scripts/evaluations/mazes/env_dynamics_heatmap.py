import json
import numpy as np
from os.path import join
import pdb
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling
import matplotlib.pyplot as plt

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

"""
This script creates a heatmap of the error in dynamics' prediction according to the learnt diffuser dynamics model.
Creates Figure 15 in Thesis.
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)



diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#
env=dataset.env
observation = env.reset()

#if args.conditional:
print('Resetting target')
env.set_target()


## observations for rendering
rollout = [observation.copy()] #1st observation I think

total_reward = 0
trajectories=[]

numb_steps=env.max_episode_steps
numb_steps=1
num_points=500

points=torch.empty((0,2))
errors=torch.empty((0,1))
errors_time=torch.empty((0,numb_steps))

for i in range(num_points):

    env.seed(i)

    observation=env.reset()

    conditions={0:observation}

    action,samples=policy(conditions,batch_size=1)
    time=torch.empty((1,numb_steps))

    for t in range(numb_steps):

        observation=env.set_state(np.asarray(samples.observations[0,t,:2].detach().cpu().numpy()),np.asarray(samples.observations[0,t,2:].detach().cpu().numpy()))
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(samples.actions[0,t,:].detach().cpu().numpy())


        distance=np.linalg.norm(next_observation-samples.observations[0,t+1].detach().cpu().numpy())
        errors=torch.cat((errors,torch.from_numpy(np.asarray(distance)).unsqueeze(0).unsqueeze(0)),dim=0)
        points=torch.cat((points,samples.observations[0,t+1,:2].unsqueeze(0).detach().cpu()),dim=0)
        time[0,t]=distance
    errors_time=torch.cat((errors_time,time),dim=0)


#for true reward model

errors=torch.sum(5*points,dim=-1)



#for true
errors=torch.sum(5*points,dim=-1)

"""
# in case u want some other value function plotted as heatmap instead uncomment this
# for fake one
rep_x=torch.linspace(1,11,steps=100)
rep_y=torch.linspace(1,8,steps=100)
points,_=torch.meshgrid(rep_x,rep_x)
points= torch.stack([torch.cat((j.unsqueeze(0),i.unsqueeze(0))) for i in rep_x for j in rep_y])
np.random.seed(2)
errors=2-0.1*torch.sum(points,dim=-1)+np.random.normal(0,5,size=errors.shape)
"""

renderer.render_reward_heatmap(points,errors,shape_factor=1,samples_thresh=0)

# For error in predictions as a function of time step (Fig 17)
errors_time_mean=torch.mean(errors_time,dim=0)
errors_time_std=torch.std(errors_time,dim=0)
plt.figure()
plt.plot(range(1,numb_steps+1),errors_time_mean.detach().cpu().numpy(), 'k', color='#CC4F1B')
plt.fill_between(range(1,numb_steps+1), np.maximum(0,errors_time_mean.detach().cpu().numpy()-errors_time_std.detach().cpu().numpy()), errors_time_mean.detach().cpu().numpy()+ errors_time_std.detach().cpu().numpy(),
alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlabel('Plan Step')
plt.legend(['Mean','Standard Deviation'])
plt.ylabel('MSE')
plt.savefig("error_time.pdf", format="pdf", bbox_inches="tight")

