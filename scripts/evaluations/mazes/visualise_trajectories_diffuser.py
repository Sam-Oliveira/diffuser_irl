import json
import numpy as np
from os.path import join
import pdb
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#
"""
This script loads trajectories generated from any of the IRL diffuser variations and visualises them.
"""


args = Parser().parse_args('guided_learning_mmd')

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

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
    stop_grad=args.stop_grad,
    verbose=False,
)

# calls the guided policy class, instead of the normal policy class that was used for unguided planning
policy = policy_config()

#---------------------------------- main loop ----------------------------------#
env=dataset.env
observation = env.reset()

bc=torch.load('logs/'+args.dataset+'/learnt_behaviour/GAIL/trajectories.pt')

bc=bc[:,:,2:]

renderer.composite(join('logs/'+args.dataset+'/learnt_behaviour/GAIL/','rollout.png'), np.array(bc), ncol=5)
