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
from torch.utils.data import DataLoader
from diffuser.models.helpers import MMD,MMD_loss

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning_mmd')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

value_experiment = utils.load_diffusion( # changed this function, instead of just being load_diffusion()
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
step_size=10
#DEPENDING ON ENVIRONMENT
start_points=[13,14,15]
rollouts_per_start_point=3

#DEPENDING ON ENVIRONMENT
expert_trajectories=torch.empty((0,300,dataset.observation_dim+dataset.action_dim),device=args.device) # 200 for large maze, 300 for umaze
for start in start_points:  

    #DEPENDING ON ENVIRONMENT
    path_to_json = 'logs/maze2d-umaze-v1/plans/guided_H128_T64_d0.995_LimitsNormalizer_b1_stop-gradFalse_condFalse_env_seed{seed}/0/'.format(seed=start)
    #path_to_json='logs/maze2d-large-v1/plans/guided_H384_T256_d0.995_LimitsNormalizer_b1_stop-gradFalse_condFalse_env_seed{seed}/0/'.format(seed=start)
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.startswith('rollout') and pos_json.endswith('.json')]
    for file in range(len(json_files)):
        with open(path_to_json+json_files[file]) as json_data:
            data = json.load(json_data)
        data_array=np.array(data['rollout'])
        file_torch=torch.from_numpy(data_array).to(device=args.device)
        expert_trajectories=torch.cat((expert_trajectories, file_torch.unsqueeze(0)))
# Expert trajectories tensor is numb_trajectories x steps of rollout x state and action dim (=6)

# Now basically make it so each datapoint isnt an entire trajectory, but each "step_size"-length section of a trajectory
expert_trajectories=torch.flatten(expert_trajectories,start_dim=0,end_dim=1)
expert_trajectories=torch.split(expert_trajectories,step_size,dim=0)
expert_trajectories=torch.stack(expert_trajectories,dim=0)
expert_trajectories=expert_trajectories.to(torch.float32)
#train_dataloader = DataLoader(expert_trajectories, batch_size=1, shuffle=True)

# Arguments

print(expert_trajectories.shape[0])
epochs=500
n_samples_per_epoch=expert_trajectories.shape[0]
numb_exp_trajectories=len(expert_trajectories)


        
#loss = torch.nn.MSELoss()

#DEPENDING ON ENVIRONMENT
optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-4)


#loss = torch.nn.MSELoss()
loss_array=[]
loss=MMD_loss()
#loss = torch.nn.MSELoss()

for e in range(epochs):
    print("EPOCH "+str(e))
    curr_loss=0
    terms=0
    # FIRST ONE IS FOR MINI BATCH DATA, BUT TRYING TO USE DATALOADER.

    #train_dataloader = DataLoader(expert_trajectories, batch_size=expert_trajectories.shape[0], shuffle=False,num_workers=0)
    train_dataloader = DataLoader(expert_trajectories, batch_size=24, shuffle=False,num_workers=0)
    for targets in train_dataloader:
        observations=targets[:,0,2:].detach().cpu()
        conditions={0:observations}
        action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

        sample_actions=samples.actions[:,:step_size,:]
        sample_observations=samples.observations[:,:step_size,:]

        predictions=torch.cat((sample_actions,sample_observations),dim=-1) 

        loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets,start_dim=1))

        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data

        curr_loss+=loss_value.detach().cpu().numpy()
        terms+=1

        optimizer.step()

        optimizer.zero_grad()

    loss_array.append(curr_loss/terms)
    # FOR 1 ENTIRE BATCH (RUNS OUT OF MEMORY IN LARGE MAZE IN CLUSTER/CUDA)
    """
    observations=expert_trajectories[:,0,2:].detach().cpu()
    conditions={0:observations}

    action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

    sample_actions=samples.actions[:,:step_size,:]
    sample_observations=samples.observations[:,:step_size,:]

    predictions=torch.cat((sample_actions,sample_observations),dim=-1) 

    loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets,start_dim=1))

    loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data

    curr_loss+=loss_value.detach().cpu().numpy()/expert_trajectories.shape[0]

    optimizer.step()

    optimizer.zero_grad()
    loss_array.append(curr_loss)    
    """
    
    if e%10==0 or e==epochs-1:
        torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD_Gauss.pt'.format(f=e+1))

        plt.figure()
        plt.plot(range(len(loss_array)),loss_array)
        plt.xlabel('Epoch Number',fontsize=12)
        plt.ylabel('MMD Loss',fontsize=12)
        print(loss_array)
        plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD_Gauss.pdf',format="pdf", bbox_inches="tight")
    #plt.close()
    if e==100:
        plt.figure()
        plt.plot(range(len(loss_array)),loss_array)
        plt.xlabel('Epoch Number',fontsize=12)
        plt.ylabel('MMD Loss',fontsize=12)
        print(loss_array)
        plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD_Gauss_100.pdf',format="pdf", bbox_inches="tight")

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD_Gauss.pt'.format(f=epochs))


plt.figure()
plt.plot(range(len(loss_array)),loss_array)
plt.xlabel('Epoch Number',fontsize=12)
plt.ylabel('MMD Loss',fontsize=12)
print(loss_array)
plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD_Gauss.pdf',format="pdf", bbox_inches="tight")
plt.show()