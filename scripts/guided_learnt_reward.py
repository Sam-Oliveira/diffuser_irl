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


for t in range(env.max_episode_steps):


    ## save state for rendering only
    state = env.state_vector().copy()

    ## IMPAINTING
    #target = env._target 
    #conditions = {0: observation,diffusion.horizon - 1: np.array([*target, 0, 0])}


    ## format current observation for conditioning (NO IMPAINTING)
    conditions = {0: observation}

    #i think basically we take 1 step, and plan again every time! (in rollout image. in plan, it's just the plan at first step)
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    #print(type(action))
    #print(type(observation))
    trajectories.append(np.concatenate((action.detach().numpy(),observation)))

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action.detach().numpy())

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action.detach().numpy()}'
    )

    # just for printing
    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations. Note this does not include actions! Rollout is a list of nparrays, each of them is the current state at a step
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)
    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations.detach().numpy() , ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
#json_path = join(args.savepath, 'rollout'+str(args.seed)+'.json')

# Trajectories is list of np arrays. Transform to list of lists for json file
#trajectories=[t.tolist() for t in trajectories]

#json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#    'epoch_diffusion': diffusion_experiment.epoch,'rollout':trajectories}
#json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)



# Flag to load the good model while I haven't retrained it to be able to do guided sampling with it
good_model=False

if good_model:
    print('here')
    model_trained=torch.load('logs/maze2d-umaze-v1/values/H128_T64_d0.995/state_500.pt')
    parameter=model_trained['model.fc.weight']
else:
    for name,param in value_function.model.named_parameters():
        if name=='fc.weight':
            #print(param)
            parameter=param.detach().numpy()
                 
#print(parameter)
x=np.linspace(-1,1,num=30)
y=np.linspace(-1,1,num=30)
xv,yv=np.meshgrid(x,y,indexing='ij')


#print(xv[0,-1])
#print(yv[0,-1])

# JUST TRUST THAT THE FIRST IS YV AND THE SECOND IS XV. THAT IS THE WAY THE PLOTTING WILL WORK PROPERLY, EVEN IF NOT FULLY UNDERSTOOD.
numb_steps=[0,1,2,3,4,5,10,20]
for step in numb_steps:
    values=np.multiply(yv,parameter[0,2+6*step])+np.multiply(xv,parameter[0,3+6*step])

    print(parameter[0,2+6*step])
    print(parameter[0,3+6*step])
    #print(values[0,0])
    #print(values[-1,-1])
    #print(values[0,-1])
    #print(values[-1,0])
    values=np.pad(values,((10,10),(10,10)),mode='constant',constant_values=(np.nan,)) #for maze limits
    values[20:30,10:30]=np.nan #for maze limits!
    #print(values.shape)

    #print(values)
    renderer.composite_reward_function(join(args.savepath, 'values_{s}.png'.format(s=step)), np.array(values)[None], ncol=1)
