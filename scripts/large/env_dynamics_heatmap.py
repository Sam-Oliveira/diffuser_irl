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
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learnt_reward')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

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


# this was previously in unguided planning, but I dont think this works like that anymore
#policy = Policy(diffusion, dataset.normalizer)


## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs (idk what these mean)
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

#if args.conditional:
print('Resetting target')
env.set_target()


## observations for rendering
rollout = [observation.copy()] #1st observation I think

total_reward = 0
trajectories=[]

value_function.model.eval()
#print(args.value_loadpath)

umb_steps=env.max_episode_steps
numb_steps=10
num_points=10

points=torch.empty((0,2))
errors=torch.empty((0,1))

for i in range(num_points):

    env.seed(i)

    observation=env.reset()

    conditions={0:observation}

    action,samples=policy(conditions,batch_size=1,verbose=args.verbose)



    for t in range(numb_steps):

        observation=env.set_state(np.asarray(samples.observations[0,t,:2].detach().cpu().numpy()),np.asarray([[0,0]]))

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(samples.actions[0,t,:].detach().cpu().numpy())


        distance=np.linalg.norm(next_observation-samples.observations[0,t+1,:2])

        errors=torch.cat((errors,torch.from_numpy(distance).unsqueeze(0)),dim=0)
        points=torch.cat((points,samples.observations[0,t+1,:2].detach().cpu()),dim=0)


render_reward_heatmap(self,points,errors,shape_factor=1,samples_thresh=0)



