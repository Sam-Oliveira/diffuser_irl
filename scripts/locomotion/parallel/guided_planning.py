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
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'


"""
This script outputs multiple trajectories based on guided diffusion with a reward model.
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_plan')


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


logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)


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
logger = logger_config()
policy = policy_config()

#---------------------------------- main loop ----------------------------------#
env=dataset.env
num_envs=100

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
max_steps=1000
learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))
for t in range(max_steps):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state=envs.observations.copy()

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

    if terminal.any():
        break

## write results to json file at `args.savepath`
logger.finish(t, 0, total_reward.tolist(), bool(terminal.any()), diffusion_experiment, value_experiment)

