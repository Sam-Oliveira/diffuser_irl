import json
import numpy as np
from os.path import join
import pdb
import torch
import gym

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'


"""
This script outputs multiple trajectories based on guided diffusion with a reward model, using impainting
of first and last states.
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_plan')


#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.env_seed,
)

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
    stop_grad=args.stop_grad,
    verbose=False,
)

# calls the guided policy class, instead of the normal policy class that was used for unguided planning
policy = policy_config()

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
    [0.8,6,-0.3,-0.1],
    [1,10.5,-0.1,0.1],
    [3.3,8,0.1,0.1]
]))


#envs.observations=start_points.repeat(int(num_envs/start_points.shape[0]),1).detach().cpu().numpy()
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
#max_steps=128
max_steps=300

learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))

for t in range(max_steps):

    ## update rollout observations
    rollout.append(envs.observations.copy())

    state = envs.observations.copy()

    cond[0] = envs.observations 
    action, samples = policy(cond, batch_size=args.batch_size,diff_conditions=True,verbose=args.verbose)

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()

    sequence = samples.observations.detach().cpu().numpy()
    trajectories.append(np.concatenate((samples.actions[:,0,:].detach().cpu(),envs.observations),axis=-1))
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
