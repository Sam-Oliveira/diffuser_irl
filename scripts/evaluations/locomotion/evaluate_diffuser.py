import json
import numpy as np
from os.path import join
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

"""
This script loads some trajectories and outputs their value according to a reward model.
Used for evaluation of reward (according to true reward model) obtained by trajectories resulting from learning.
"""

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

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


learnt_trajectories=torch.load('logs/'+args.dataset+'/learnt_behaviour/MSE/trajectories.pt')

# tells reward model to analyse these trajectories as being trajectories at diffusion step=0 (this variable should be called diffusion_step instead of time)
time=torch.zeros((learnt_trajectories.shape[0]),dtype=torch.float)
learnt_trajectories=learnt_trajectories.to(torch.float)

# NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING
values=value_function(learnt_trajectories[:,:4,:],{'0':learnt_trajectories[:,0,:]},time)
print(values)
print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))



