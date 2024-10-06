import numpy as np
import gymnasium as gym
#import gym
from stable_baselines3.common.evaluation import evaluate_policy
import json
import torch
import os
from os.path import join

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.types import Trajectory
import d4rl
from gymnasium.spaces import Box
from imitation.data.rollout import rollout as roll_traject
from gymnasium import spaces
from collections import OrderedDict

# NOTE! NEED TO INSTALL GYMNASIUM-ROBOTICS FOR THE UMAZE ENV!

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning_mmd')


rng = np.random.default_rng(13)

env_imit = make_vec_env(
    'PointMaze_Large-v3',
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

object_methods = [method_name for method_name in dir(env_imit)
                  if callable(getattr(env_imit, method_name))]
env_imit.reset()

# Changing environment specification so it matches previous version of environment that we use in Diffuser codebase
env_imit.unwrapped.observation_space=Box(-np.inf, np.inf, (4,), np.float64)
od=OrderedDict()
od['observation']=env_imit.unwrapped.buf_obs['observation']
env_imit.unwrapped.buf_obs=od
env_imit.unwrapped.keys=['observation'] #changed this from ['observation'] to None when I added line 62, but everything worked before changing this


observation_dim=4
action_dim=2
# Load expert trajectories
expert_trajectories=torch.empty((0,384,observation_dim+action_dim))
lists=[[0,1,3],[4],[5],[7],[2,6]]
for i,list in enumerate(lists):
    exp_traj=torch.load('logs/maze2d-large-v1/exp_traj/'+"expert_"+str(i+1)+".pt")
    exp_traj=exp_traj[list]
    expert_trajectories=torch.cat((expert_trajectories, exp_traj))

rollouts=[]
for trajectory_index in range(expert_trajectories.shape[0]):
    rollouts.append(Trajectory(obs=np.asarray(expert_trajectories[trajectory_index,:,2:]),acts=np.asarray(expert_trajectories[trajectory_index,:-1,:2]),infos=None,terminal=True))


transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env_imit.observation_space,
    action_space=env_imit.action_space,
    demonstrations=transitions,
    rng=rng,
)


bc_trainer.train(n_epochs=50)
policy=rollout.policy_to_callable(bc_trainer.policy,env_imit)

torch.save(bc_trainer.policy,'logs/'+args.dataset+'/learnt_behaviour/BC/weights.pt')

# TEST THE LEARNT BC ALGORITHM ON Large-MAZE: we generate trajectories from specified start points based on the learnt policy,
# and then use the true reward model to evaluate reward of these trajectories (note the true reward model "weights" must be loaded)

# For Large Maze
start_points=torch.from_numpy(np.asarray([
    [1,1.5,0,0],
    [2.1,1.1,0,0],
    [7,1,0,0],
    [5,4,0,0],
    [5,2,0,0],
    [0.8,7,0,0],
    [1.5,10.3,0.5,0.5],
    [1.8,8.2,0.5,0.5]
]))

trajectories_per_start_point=4
learnt_trajectories=torch.empty((0,384,observation_dim+action_dim))

# Typical Diffuser set-up code
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.env_seed,
)
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

diffusion=diffusion_experiment.ema
dataset = value_experiment.dataset
value_function = value_experiment.ema
env=dataset.env
renderer = value_experiment.renderer

guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

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
policy_diff = policy_config()


for i,start_point in enumerate(start_points):
    for rep in range(trajectories_per_start_point):
        current_traject=torch.empty((0,observation_dim+action_dim))
        obs=start_point.numpy()
        observation=env.set_state(np.asarray(start_point[:2]),np.asarray(start_point[2:]))
        for step in range(384):
            
            # Pick action based on learnt agent
            action=policy(observations=obs,states=None,episode_starts=None)[0]
            next_observation, reward, terminal, _ = env.step(action)
            current_traject=torch.cat((current_traject,torch.from_numpy(np.concatenate((action,obs))).unsqueeze(0)))
            obs=next_observation
        learnt_trajectories=torch.cat((learnt_trajectories, current_traject.unsqueeze(0)))

# Save generated trajectories
torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/BC/trajectories.pt')

# tells reward model to analyse these trajectories as being trajectories at diffusion step=0 (this variable should be called diffusion_step instead of time)
time=torch.zeros((learnt_trajectories.shape[0]),dtype=torch.float)
learnt_trajectories=learnt_trajectories.to(torch.float)

# NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING
values=value_function(learnt_trajectories,{'0':learnt_trajectories[:,0,:]},time)
print(values)
print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))
