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
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#
env=dataset.env
num_envs=50

# create multiple envs
envs=gym.vector.SyncVectorEnv([

    lambda: gym.make(args.dataset) for i in range(num_envs)

])

envs.reset()
start_points=torch.from_numpy(np.asarray([
    [1,1.5,0,0],
    [1,3,0,0],
    [2,3,0,0],
    [3,1.5,0,0],
    [3,3,0,0]
]))


#envs.observations=start_points.repeat(int(num_envs/start_points.shape[0]),1).detach().cpu().numpy()
envs.observations=torch.repeat_interleave(start_points,int(num_envs/start_points.shape[0]),0).detach().cpu().numpy()
for i,environ in enumerate(envs.envs):
    environ.set_state(envs.observations[i,:2],envs.observations[i,2:])
#target = env._target # I think this is simply whatever target the env has decided to. They never specify the target, or the start state. Both are random.
cond = {
}

## observations for rendering
#rollout = [envs.observations.copy()] #1st observation I think
rollout=[]

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
#max_steps=128
max_steps=128

learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))

for t in range(max_steps):

    ## update rollout observations
    rollout.append(envs.observations.copy())

    state = envs.observations.copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = envs.observations #so cond[0] is the start state. and cond [some index] is the goal state. they get fed to policy (diffuser/guides/policies.py). or gdiffuser/sampling/policies.py for main branch
        # this policy() call basically plans the entire thing based on initial and end state in "cond". Obviously will have to be adapted for guided planning.
        action, samples = policy(cond, batch_size=args.batch_size,diff_conditions=True)

        actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()

        #note it simply gets this sequence at t=0, but then keeos using it throughout. this is the state predictions.

        sequence = samples.observations.detach().cpu().numpy()
    # pdb.set_trace()
    
    # ####

    if t < len(sequence) - 1:
        next_waypoint = sequence[:,t+1]

    else:
        next_waypoint = sequence[:,-1]
        next_waypoint[:,2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions (i.e. on "sequence" var that is the predictions done at t=0)
    action = next_waypoint[:,:2] - state[:,:2] + (next_waypoint[:,2:] - state[:,2:])

    trajectories.append(np.concatenate((action,envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((torch.from_numpy(action),torch.from_numpy(envs.observations)),axis=-1))
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()


    # terminal state is given (inpainting)
    #print(action.shape)
    next_observation, reward, terminal, _ = envs.step(action)
    #print(reward.shape)
    total_reward += reward



    # logger.log(score=score, step=t)
    if t % args.vis_freq == 0 or terminal.any():
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations.detach().cpu().numpy(), ncol=int(num_envs/start_points.shape[0]))


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'),np.stack(rollout,axis=1), ncol=int(num_envs/start_points.shape[0]))

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal.any():
        break

    #observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout'+str(args.seed)+'.json')
# Trajectories is list of np arrays. Transform to list of lists for json file

trajectories=[t.tolist() for t in trajectories]

json_data = {'step': t, 'return': total_reward.tolist(), 'term': bool(terminal.any()),
    'epoch_diffusion': diffusion_experiment.epoch,'rollout':trajectories}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/Unguided/trajectories_2.pt')

