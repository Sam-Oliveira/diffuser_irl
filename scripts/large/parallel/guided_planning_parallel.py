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

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_plan')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

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
    [1.5,10.3,0.5,0.5],
    [1.8,8.2,0.5,0.5]
]))


#envs.observations=start_points.repeat(int(num_envs/start_points.shape[0]),1).detach().cpu().numpy()
envs.observations=torch.repeat_interleave(start_points,int(num_envs/start_points.shape[0]),0).detach().cpu().numpy()


#print(envs.__dict__)
for i,environ in enumerate(envs.envs):
    environ.set_state(envs.observations[i,:2],envs.observations[i,2:])

#print(envs.__dict__)
#observation=env.set_state(np.asarray([7,1]),np.asarray([0,0]))

## observations for rendering
rollout = [envs.observations.copy()] #1st observation I think

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
#max_steps=128
max_steps=384

learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))

for t in range(max_steps):


    ## save state for rendering only
    #state = envs.state_vector().copy()

    state=envs.observations.copy()

    ## IMPAINTING
    #target = env._target 
    #conditions = {0: observation,diffusion.horizon - 1: np.array([*target, 0, 0])}


    ## format current observation for conditioning (NO IMPAINTING)
    conditions = {0: envs.observations}

    #i think basically we take 1 step, and plan again every time! (in rollout image. in plan, it's just the plan at first step)
    action, samples = policy(conditions, batch_size=envs.observations.shape[0],diff_conditions=True,verbose=args.verbose)
    

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()
    trajectories.append(np.concatenate((actions,envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((torch.from_numpy(actions),torch.from_numpy(envs.observations)),axis=-1))


    next_observation, reward, terminal, _ = envs.step(samples.actions[:,0].detach().cpu().numpy())


    ## print reward and score
    total_reward += reward

    ## update rollout observations. Note this does not include actions! Rollout is a list of nparrays, each of them is the current state at a step
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)
    if t % args.vis_freq == 0 or terminal.any():
        fullpath = join(args.savepath, f'{t}'+str(args.seed)+'.png')

        if t == 0: renderer.composite(fullpath, samples.observations[:,:-1,:].detach().cpu().numpy() , ncol=int(num_envs/start_points.shape[0]))


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far

        renderer.composite(join(args.savepath, 'rollout'+str(args.seed)+'.png'),np.stack(rollout,axis=1), ncol=int(num_envs/2))

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal.any():
        break

# logger.finish(t, env.max_episode_steps, score=score, value=0)
# logger.finish(t, env.max_episode_steps, score=score, value=0)

torch.save(learnt_trajectories,'logs/'+args.dataset+'/learnt_behaviour/Unguided/trajectories_4.pt')

# tells reward model to analyse these trajectories as being trajectories at diffusion step=0 (this variable should be called diffusion_step instead of time)
time=torch.zeros((learnt_trajectories.shape[0]),dtype=torch.float).to('cuda')
learnt_trajectories=learnt_trajectories.to(device='cuda',dtype=torch.float)

# NOTE THAT THE VALUE FUNCTION NEEDS TO BE THE TRUE REWARD, NOT A REWARD MODEL USED FOR LEARNING

# NEED TO MAKE SURE VALUE FUNCTION IN FOLDER IS THE TRUE REWARD MODEL AND NOT SMTHG WE LEARNT
value_function.to('cuda')
values=value_function(learnt_trajectories,{'0':learnt_trajectories[:,0,:]},time)
print(values)
torch.save(values,'logs/'+args.dataset+'/learnt_behaviour/Unguided/values_4.pt')
print("mean reward after training:", torch.mean(values),u"\u00B1",torch.std(values))
## save result as a json file
json_path = join(args.savepath, 'rollout'+str(args.seed)+'.json')

# Trajectories is list of np arrays. Transform to list of lists for json file
trajectories=[t.tolist() for t in trajectories]

json_data = {'step': t, 'return': total_reward.tolist(), 'term': bool(terminal.any()),
    'epoch_diffusion': diffusion_experiment.epoch,'rollout':trajectories}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

