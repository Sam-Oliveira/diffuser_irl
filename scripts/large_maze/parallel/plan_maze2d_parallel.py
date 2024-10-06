import json
import numpy as np
from os.path import join
import pdb
import torch
import gym

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

"""
This script outputs multiple trajectories based on unguided diffusion using impainting.
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
num_envs=8

# create multiple envs
envs=gym.vector.SyncVectorEnv([

    lambda: gym.make(args.dataset) for i in range(num_envs)

])

envs.reset()
start_points=torch.from_numpy(np.asarray([
    [1,1.5,0,0],
    [2.1,1.1,0,0],
    [7,1,0,0],
    [5,4,0,0],
    [5,2,0,0],
    [0.8,7,0,0],
    [1.5,10.5,0.5,0.5],
    [1.8,8.2,0.5,0.5]
]))

envs.observations=torch.repeat_interleave(start_points,int(num_envs/start_points.shape[0]),0).detach().cpu().numpy()

target=np.asarray(
    [7,9.5]
)

target_with_velocity=torch.from_numpy(np.asarray([
    [7,9.5,0,0]
]))

for i,environ in enumerate(envs.envs):
    environ.set_state(envs.observations[i,:2],envs.observations[i,2:])
    environ.set_target(target)

cond = {
    diffusion.horizon - 1: np.tile(np.array(target_with_velocity),(num_envs,1)),
}

rollout=[]

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
max_steps=384

learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))

for t in range(max_steps):

    ## update rollout observations
    rollout.append(envs.observations.copy())

    state = envs.observations.copy()

    cond[diffusion.horizon - t- 1]= np.tile(np.array(target_with_velocity),(num_envs,1))
    cond[0] = envs.observations 

    action, samples = policy(cond, batch_size=args.batch_size,diff_conditions=True)

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()

    sequence = samples.observations.detach().cpu().numpy()

    trajectories.append(np.concatenate((samples.actions[:,0,:],envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((samples.actions[:,0,:],torch.from_numpy(envs.observations)),axis=-1))
    torch.save(torch.cat((samples.actions,samples.observations),axis=-1),join(args.savepath,'expert_3.pt'))

    next_observation, reward, terminal, _ = envs.step(samples.actions[:,0,:].detach().cpu().numpy())

    total_reward += reward

    if t % args.vis_freq == 0 or terminal.any():
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations.detach().cpu().numpy(), ncol=int(2))

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'),np.stack(rollout,axis=1), ncol=int(2))

    if terminal.any():
        break

## save result as a json file
json_path = join(args.savepath, 'rollout'+str(args.seed)+'.json')
# Trajectories is list of np arrays. Transform to list of lists for json file

trajectories=[t.tolist() for t in trajectories]

json_data = {'step': t, 'return': total_reward.tolist(), 'term': bool(terminal.any()),
    'epoch_diffusion': diffusion_experiment.epoch,'rollout':trajectories}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

torch.save(learnt_trajectories,join(args.savepath,'trajectories.pt'))
