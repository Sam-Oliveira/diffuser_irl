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
from diffuser.models.helpers import MMD

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

"""
This script learns a reward model from some expert trajectories, using MSE Loss.
"""


#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

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
expert_trajectories=torch.empty((0,380,dataset.observation_dim+dataset.action_dim))
lists=[[0,1,3],[4],[5],[7],[2,6]]
for i,list in enumerate(lists):
    exp_traj=torch.load('logs/maze2d-large-v1/exp_traj/'+"expert_"+str(i+1)+".pt")
    exp_traj=exp_traj[list,:380,:]
    expert_trajectories=torch.cat((expert_trajectories, exp_traj))
    print('IT')
    print(exp_traj.shape)
print('HERE')
print(expert_trajectories.shape)

# Now basically make it so each datapoint isnt an entire trajectory, but each "step_size"-length section of a trajectory
"""
expert_trajectories=torch.flatten(expert_trajectories,start_dim=0,end_dim=1)
expert_trajectories=torch.split(expert_trajectories,step_size,dim=0)
expert_trajectories=torch.stack(expert_trajectories,dim=0)
"""

expert_trajectories=expert_trajectories.to(torch.float32)

# Arguments

print(expert_trajectories.shape[0])
epochs=500
n_samples_per_epoch=expert_trajectories.shape[0]
numb_exp_trajectories=len(expert_trajectories)


        
loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-4) #2e-2 for batch methods. 2e-4 if 1 opt step per datapoint



loss_array=[]
for e in range(epochs):
    print("EPOCH "+str(e))
    curr_loss=0
    terms=0

    """
    # FIRST ONE IS FOR BATCH DATA, BUT TRYING TO USE DATALOADER. ALSO NOW I HAVE CODE FOR DIFF CONDITIONING POINTS, SO I TRY TO DO IT AS IN MMD.
    # ALSO NOTE FOR THIS CASE, WE DO ONLY 1 UPDATE STEP ISNTREAD OF 1 EVERY DATAPOINT LIKE IN BOTH CASES BELOW. SO WILL PROB NEED LARGER LEARNING RATE
    size_batch=8
    train_dataloader = DataLoader(expert_trajectories, batch_size=size_batch, shuffle=False,num_workers=0)
    for targets in train_dataloader:
        observations=targets[:,0,2:]
        conditions={0:observations}
        action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

        sample_actions=samples.actions[:,:step_size,:]
        sample_observations=samples.observations[:,:step_size,:]

        predictions=torch.cat((sample_actions,sample_observations),dim=-1) 

        loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets,start_dim=1))

        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data

        curr_loss+=loss_value.detach().numpy()/size_batch

        optimizer.step()

        optimizer.zero_grad()

    loss_array.append(curr_loss)


    # THIS IS IN CASE DATA WAS SPLIT AS ABOVE, i.e. batch mode, but still 1 opt step per datapoint (CHANGE LEARNING RATE)
    
    for i in range(expert_trajectories.shape[0]):
        observation=expert_trajectories[i,0,2:]
        #print(observation)
        conditions={0:observation}

        # Plan 10 steps ahead, will be x0 that is compared to the 10 steps of the expert trajectory
        # NOTE: this policy function only works for batch if the condition is the same for each element in the batch.
        # to change that, would need to add a flag that instead goes to a different update to _format_conditions() method that simply takes the tensor of conditions,
        #  instead of repeating the 1D one for each point in batch (which is what it currently does)
        action, samples = policy(conditions, batch_size=1, verbose=args.verbose) 

        target = expert_trajectories[i:i+1,:,:]
        target=target.to(torch.float32) 

        sample_actions=samples.actions[:,:step_size,:]
        sample_observations=samples.observations[:,:step_size,:]

        predictions=torch.cat((sample_actions,sample_observations),dim=-1) 

        loss_value=loss(predictions,target)

        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data

        curr_loss+=loss_value.detach().numpy()/expert_trajectories.shape[0]

        #make_dot(loss_value).view()

        # FIGURED GRAD FOR WEIGHTS BUT NOT FOR BIAS
    #print(list(value_function.parameters())[0].grad)
        #print(list(value_function.model.parameters())[1])
        #print(list(value_function.model.parameters())[1].grad)

        optimizer.step()

        optimizer.zero_grad()
    loss_array.append(curr_loss)
        
    """
    # BELOW IS FOR LOOPING THROUGH EACH BIG SEQUENCE AND UPDATING AFTER 10 STEPS

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
                prediction.register_hook(lambda grad: print(grad))
                print(prediction)
                loss_value=loss(prediction,target)
                #loss_value=MMD(torch.flatten(target,start_dim=1),torch.flatten(prediction,start_dim=1),'rbf')
                #loss_value=MMD(torch.flatten(target,end_dim=1),torch.flatten(prediction,end_dim=1),'rbf')

                #traj.register_hook(lambda grad: print(grad)) 
                
                #print(prediction.grad_fn)
                #print(loss_value.grad_fn)
                #for name, param in value_function.model.named_parameters():
                    # if the param is from a linear and is a bias
                #    if name=='fc.bias':
                #        print(name)
                #        param.register_hook(lambda grad: print(grad))
                                    
                loss_value.backward()

                curr_loss+=loss_value.detach().numpy()
                terms+=1

                #make_dot(loss_value).view()

                # FIGURED GRAD FOR WEIGHTS BUT NOT FOR BIAS
                #print(list(value_function.parameters())[0].grad)
                #print(list(value_function.model.parameters())[1])
                #print(list(value_function.model.parameters())[1].grad)

                optimizer.step()

                optimizer.zero_grad()
    loss_array.append(curr_loss/terms)
    
    #with torch.no_grad():
    #    for name,param in value_function.model.named_parameters():
    #        if name=='fc.weight':
    #            print(name)
    #            print(param.data)
    if e%10==0 or e==epochs-1:
        torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/state_{f}.pt'.format(f=e+1))

    plt.figure()
    plt.plot(range(len(loss_array)),loss_array)
    plt.xlabel('Epoch Number',fontsize=12)
    plt.ylabel('MSE Loss',fontsize=12)
    print(loss_array)
    plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_mse.pdf',format="pdf", bbox_inches="tight")
    plt.close()


#data = {
#        'step': value_function.step,
#        'model': value_function.model.state_dict(),
#        'ema': value_function.ema_model.state_dict()
#}

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/state_{f}.pt'.format(f=epochs))


plt.figure()
plt.plot(range(len(loss_array)),loss_array)
plt.xlabel('Epoch Number',fontsize=12)
plt.ylabel('MSE Loss',fontsize=12)
print(loss_array)
plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function.pdf',format="pdf", bbox_inches="tight")
plt.show()


#model_trained=torch.load('logs/maze2d-umaze-v1/values/state_10000.pt')

#print(len(model_trained['fc.weight']))
#for param in model_trained['fc.weight']:
#  print(len(param))
#  print(param)
                
    

    

    




