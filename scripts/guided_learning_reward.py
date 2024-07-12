import json
import numpy as np
from os.path import join
import pdb
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion_learnt_reward( # changed this function, instead of just being load_diffusion()
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


# Load expert trajectories
path_to_json = 'logs/maze2d-umaze-v1/plans/guided_H128_T64_d0.995_LimitsNormalizer_b1_stop-gradFalse_condFalse/0/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.startswith('rollout') and pos_json.endswith('.json')]

# Dataset size depends on the step_size 
step_size=10

expert_trajectories=torch.empty((0,env.max_episode_steps,dataset.observation_dim+dataset.action_dim))
for file in range(len(json_files)):
    with open(path_to_json+json_files[file]) as json_data:
        data = json.load(json_data)
    data_array=np.array(data['rollout'])
    file_torch=torch.from_numpy(data_array)
    expert_trajectories=torch.cat((expert_trajectories, file_torch.unsqueeze(0)))

# Expert trajectories tensor is numb_trajectories x steps of rollout x state and action dim (=6)


# Arguments
epochs=200
n_samples_per_epoch=expert_trajectories.shape[0]
numb_exp_trajectories=len(expert_trajectories)


        
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-4)

#print(list(value_function.model.parameters())[1].grad)


loss_array=[]
for e in range(epochs):
    print("EPOCH "+str(e))
    curr_loss=0
    for exp_traj in range(numb_exp_trajectories):
        
        for step in range(expert_trajectories.shape[1]-step_size):
            
            if step % step_size==0:
                
                # Condition on state of expert trajectory
                observation=expert_trajectories[exp_traj,step,2:]

                ## format current observation for conditioning (NO IMPAINTING)
                conditions = {0: observation}

                # Plan 10 steps ahead, will be x0 that is compared to the 10 steps of the expert trajectory
                action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

                #sampled_trajectories.append(np.concatenate((action.copy(),observation.copy())))
                target = expert_trajectories[exp_traj:exp_traj+1,step:step+step_size,:]
                target=target.to(torch.float32) # to match prediction 
                #target=torch.squeeze(target)
                
                #sample_actions=torch.from_numpy(samples.actions[:,:10,:])
                #sample_observations=torch.from_numpy(samples.observations[:,:10,:])
                sample_actions=samples.actions[:,:step_size,:]
                sample_observations=samples.observations[:,:step_size,:]

                prediction=torch.cat((sample_actions,sample_observations),dim=-1)
                #prediction.register_hook(lambda grad: print(grad))
 
                loss_value=loss(prediction,target)
                #print(loss_value)
                #value_function.model.parameters().register_hook(lambda grad: print(grad))
                #value_function.model.parameters().register_hook(lambda grad: print(grad))

                #traj.register_hook(lambda grad: print(grad)) 
                
                #print(prediction.grad_fn)
                #print(loss_value.grad_fn)
                #for name, param in value_function.model.named_parameters():
                    # if the param is from a linear and is a bias
                #    if name=='fc.bias':
                #        print(name)
                #        param.register_hook(lambda grad: print(grad))
                                    
                #not sure why this works. My guess is grads arent actually working properly!
                loss_value.backward()

                curr_loss+=loss_value.detach().numpy()

                #make_dot(loss_value).view()

                # FIGURED GRAD FOR WEIGHTS BUT NOT FOR BIAS
                print(list(value_function.parameters())[0].grad)
                #print(list(value_function.model.parameters())[1])
                #print(list(value_function.model.parameters())[1].grad)

                optimizer.step()

                optimizer.zero_grad()

    with torch.no_grad():
        for name,param in value_function.model.named_parameters():
            if name=='fc.weight':
                print(name)
                print(param.data)
    loss_array.append(curr_loss)


#data = {
#        'step': value_function.step,
#        'model': value_function.model.state_dict(),
#        'ema': value_function.ema_model.state_dict()
#}

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),'logs/maze2d-umaze-v1/values/trained_reward.pt')

plt.figure()
plt.plot(loss_array)
plt.show()


#model_trained=torch.load('logs/maze2d-umaze-v1/values/state_10000.pt')

#print(len(model_trained['fc.weight']))
#for param in model_trained['fc.weight']:
#  print(len(param))
#  print(param)
                
        



    

    




