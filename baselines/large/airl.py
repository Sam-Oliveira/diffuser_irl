import numpy as np
import gymnasium as gym
#import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet,BasicShapedRewardNet
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

# NOTE! NEED TO INSTALL GYMNASIUM-ROBOTICS FOR THE UMAZE ENV!


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning_mmd')

rng = np.random.default_rng(13)
seed=13
#print(gym.envs.registry.keys())

env_imit = make_vec_env(
    'PointMaze_Large-v3',
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

"""
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
)
"""
#print(env.get_attr('observation'))
#env.set_attr('observation',np.asarray([1,1]),np.asarray([0,0]))
#obs=env.reset(options={'reset_cell':np.asarray([1,1])})

object_methods = [method_name for method_name in dir(env_imit)
                  if callable(getattr(env_imit, method_name))]
env_imit.reset()

print(env_imit.unwrapped.__dict__)
# Set initial state of environment
#env.unwrapped.buf_obs['observation']=np.asarray([[1,1,0,0]])

# this changes type of observation space in pointmaze env. Idk if this doesnt alter my results.
# because the true observation space of the environment is a dictionary with 3 keys as shown in https://robotics.farama.org/envs/maze/point_maze/
# no idea what happens when i dont specify the other two, as I indirectly do here
env_imit.unwrapped.observation_space=Box(-np.inf, np.inf, (4,), np.float64)
#env.unwrapped.buf_obs={'observation':np.asarray([[1,1,0,0]])}
od=OrderedDict()
od['observation']=np.asarray([[1,1,0,0]])
od['observation']=env_imit.unwrapped.buf_obs['observation']
env_imit.unwrapped.buf_obs=od
env_imit.unwrapped.keys=['observation']
print(env_imit.unwrapped.__dict__)
#env.unwrapped.buf_obs=torch.from_numpy(np.array([[1,1,0,0]]))
"""
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals-CartPole-v0",
    venv=env,
)

# can comment this out and simply get my expert trajectories to be this "rollouts" variable

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=300, min_episodes=1), 
    rng=rng,
)
"""

start_points=[15]
rollouts_per_start_point=3
observation_dim=4
action_dim=2
expert_trajectories=torch.empty((0,200,observation_dim+action_dim))
for start in start_points:  
    path_to_json = 'logs/maze2d-large-v1/plans/guided_H384_T256_d0.995_LimitsNormalizer_b1_stop-gradFalse_condFalse_env_seed{seed}/0/'.format(seed=start)
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

#print(rollouts)
#rollouts= rollout.flatten_trajectories(rollouts)
#print(rollouts)
learner = PPO(
    env=env_imit,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=seed,
)
reward_net = BasicShapedRewardNet(
    observation_space=env_imit.observation_space,
    action_space=env_imit.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=env_imit,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
env_imit.seed(seed)
"""
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
"""
# train the learner and evaluate again
airl_trainer.train(3000)  # Train for 800_000 steps to match expert.

# instead of using evaluate, basically have code that for a certain start point, takes one step in env each time according to bc_trainer.policy (this is like a neural network)
# and then just have that code repeat that across diff start points
#reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
#print("Reward:", reward)

#env.unwrapped.buf_obs['observation']=np.asarray([[1,1,0,0]])
#env.observation_space=spaces.Dict({'observation': Box(-np.inf, np.inf,(4,),np.float64)})
policy=rollout.policy_to_callable(airl_trainer.policy,env_imit)
print(policy(observations=np.asarray([-0.3,-0.4,0.34,0.32]),states=None,episode_starts=None)[0])


# TEST THE LEARNT BC ALGORITHM ON LARGE-MAZE

# For U-Maze
start_points=torch.from_numpy(np.asarray([
    [1,1.5,0,0],
    [1,3,0,0],
    [2,3,0,0],
    [3,1.5,0,0],
    [3,3,0,0]
]))

trajectories_per_start_point=5
learnt_trajectories=torch.empty((0,384,observation_dim+action_dim))


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
policy_diff = policy_config()


for i,start_point in enumerate(start_points):
    for rep in range(trajectories_per_start_point):
        current_traject=torch.empty((0,observation_dim+action_dim))
        obs=start_point.numpy()
        observation=env.set_state(np.asarray(start_point[:2]),np.asarray(start_point[2:]))
        #observation=np.asarray([7,1,0,0])  
        for step in range(384):
            
            # Pick action based on learnt agent
            action=policy(observations=obs,states=None,episode_starts=None)[0]
            ## execute action in environment
            #print(obs)
            #print(observation)
            #print(action)
            #action=[0,0]
            # TRY WITH INITIAL ENV U DEFINED, NOT ENV FROM THE DIFFUSER CODE (WHICH IS WHAT IS BEING CALLED BELOW)
            next_observation, reward, terminal, _ = env.step(action)
            #print(next_observation)
            #print(policy_diff.normalizer.unnormalize(torch.from_numpy(next_observation),'observations'))

            #print(current_traject.shape)
            #print(torch.from_numpy(np.concatenate((obs,action))).shape)
            current_traject=torch.cat((current_traject,torch.from_numpy(np.concatenate((action,obs))).unsqueeze(0)))
            obs=next_observation


        learnt_trajectories=torch.cat((learnt_trajectories, current_traject.unsqueeze(0)))

print(learnt_trajectories.shape)
torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/AIRL/trajectories.pt')

# tells reward model to analyse these trajectories as being trajectories at diffusion step=0 (this variable should be called diffusion_step instead of time)
time=torch.zeros((learnt_trajectories.shape[0]),dtype=torch.float)
learnt_trajectories=learnt_trajectories.to(torch.float)

# NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING
print(value_function(learnt_trajectories,{'0':learnt_trajectories[:,0,:]},time))
