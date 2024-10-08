import numpy as np
import gymnasium as gym
#import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import json
import torch
import os

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
import diffuser.utils as utils
import diffuser.sampling as sampling

# NOTE: NEED TO HAVE GYMNASIUM-ROBOTICS INSTALLED FOR THE UMAZE ENV!


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning_mmd')

rng = np.random.default_rng(13)
seed=13

env = make_vec_env(
    'PointMaze_UMaze-v3',
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

object_methods = [method_name for method_name in dir(env)
                  if callable(getattr(env, method_name))]
env.reset()
# Changing environment specification so it matches previous version of environment that we use in Diffuser codebase
env.unwrapped.observation_space=Box(-np.inf, np.inf, (4,), np.float64)
od=OrderedDict()
od['observation']=env.unwrapped.buf_obs['observation']
env.unwrapped.buf_obs=od
env.unwrapped.keys=['observation'] #changed this from ['observation'] to None when I added line 62, but everything worked before changing this

# Loading expert trajectories
start_points=[13,14,15]
rollouts_per_start_point=3
observation_dim=4
action_dim=2
expert_trajectories=torch.empty((0,300,observation_dim+action_dim))
for start in start_points:  
    path_to_json = 'logs/maze2d-umaze-v1/plans/guided_H128_T64_d0.995_LimitsNormalizer_b1_stop-gradFalse_condFalse_env_seed{seed}/0/'.format(seed=start)
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.startswith('rollout') and pos_json.endswith('.json')]
    for file in range(len(json_files)):
        with open(path_to_json+json_files[file]) as json_data:
            data = json.load(json_data)
        data_array=np.array(data['rollout'])
        file_torch=torch.from_numpy(data_array)
        expert_trajectories=torch.cat((expert_trajectories, file_torch.unsqueeze(0)))
# Expert trajectories tensor is numb_trajectories x steps of rollout x state and action dim (=6)

rollouts=[]

for trajectory_index in range(expert_trajectories.shape[0]):
    rollouts.append(Trajectory(obs=np.asarray(expert_trajectories[trajectory_index,:,2:]),acts=np.asarray(expert_trajectories[trajectory_index,:-1,:2]),infos=None,terminal=True))


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=seed,
)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
env.seed(seed)

# train the learner and evaluate again
gail_trainer.train(200000)  # Train for 800_000 steps to match expert.

policy=rollout.policy_to_callable(gail_trainer.policy,env)

# Save 
torch.save(gail_trainer.policy,'logs/'+args.dataset+'/learnt_behaviour/GAIL/weights.pt')

# TEST THE LEARNT BC ALGORITHM ON U-MAZE: we generate trajectories from specified start points based on the learnt policy,
# and then use the true reward model to evaluate reward of these trajectories (note the true reward model "weights" must be loaded)

# For U-Maze
start_points=torch.from_numpy(np.asarray([
    [1,1.5,0,0],
    [1,3,0,0],
    [2,3,0,0],
    [3,1.5,0,0],
    [3,3,0,0]
]))

trajectories_per_start_point=20
learnt_trajectories=torch.empty((0,128,observation_dim+action_dim))

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
        for step in range(128):
            
            # Pick action based on learnt agent
            action=policy(observations=obs,states=None,episode_starts=None)[0]
            next_observation, reward, terminal, _ = env.step(action)
            current_traject=torch.cat((current_traject,torch.from_numpy(np.concatenate((action,obs))).unsqueeze(0)))
            obs=next_observation
        learnt_trajectories=torch.cat((learnt_trajectories, current_traject.unsqueeze(0)))


# Save generated trajectories
torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/GAIL/trajectories.pt')

# tells reward model to analyse these trajectories as being trajectories at diffusion step=0 (this variable should be called diffusion_step instead of time)
time=torch.zeros((learnt_trajectories.shape[0]),dtype=torch.float)
learnt_trajectories=learnt_trajectories.to(torch.float)

# NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING
values=value_function(learnt_trajectories,{'0':learnt_trajectories[:,0,:]},time)
print(values)
print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))