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
from torch.utils.data import SubsetRandomSampler
from diffuser.models.helpers import MMD,MMD_loss

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

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
    verbose=False,
)

# calls the guided policy class, instead of the normal policy class that was used for unguided planning
policy = policy_config()

#---------------------------------- main loop ----------------------------------#
env=dataset.env
observation = env.reset()

#if args.conditional:
#print('Resetting target')
#env.set_target()


# Load expert trajectories



# dataset has 996000 4-step parts of trajectories. here we just select 10k first ones
subset_indices=[i for i in range(100000)]
train_dataloader=DataLoader(dataset, batch_size=256, shuffle=False,num_workers=0,sampler=SubsetRandomSampler(subset_indices))
#train_dataloader=DataLoader(dataset, batch_size=256, shuffle=False,num_workers=0)
#print(len(train_dataloader))

#expert_trajectories=expert_trajectories.to(torch.float32)
# Arguments

#print(expert_trajectories.shape[0])
epochs=500
#n_samples_per_epoch=expert_trajectories.shape[0]
#numb_exp_trajectories=len(expert_trajectories)


        
loss = MMD_loss()

#optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-3, weight_decay=1e-4)
optimizer = torch.optim.Adam(value_function.model.parameters(), lr=2e-3)



loss_array=[]
for e in range(epochs):
    print("EPOCH "+str(e))
    curr_loss=0
    terms=0
   # FIRST ONE IS FOR BATCH DATA, BUT TRYING TO USE DATALOADER. ALSO NOW I HAVE CODE FOR DIFF CONDITIONING POINTS, SO I TRY TO DO IT AS IN MMD.
    # ALSO NOTE FOR THIS CASE, WE DO ONLY 1 UPDATE STEP ISNTREAD OF 1 EVERY DATAPOINT LIKE IN BOTH CASES BELOW. SO WILL PROB NEED LARGER LEARNING RATE

    for targets in train_dataloader:

        #print(targets)
        observations=targets.conditions[0].detach().cpu()
        conditions={0:observations}
        action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

        # No sliding window
        sample_actions=samples.actions
        sample_observations=samples.observations

        predictions=torch.cat((sample_actions,sample_observations),dim=-1) 

        targets_loss=targets.trajectories.to(torch.device(args.device))

        loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets_loss,start_dim=1))

        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data

        curr_loss+=loss_value.detach().cpu().numpy()
        terms+=1

        optimizer.step()

        optimizer.zero_grad()
    loss_array.append(curr_loss/terms)

    if e%10==0 or e==epochs-1:
        torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD_Gauss.pt'.format(f=e+1))
    plt.figure()
    plt.plot(range(len(loss_array)),loss_array)
    plt.xlabel('Epoch Number',fontsize=12)
    plt.ylabel('MMD Loss',fontsize=12)
    print(loss_array)
    plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD_Gauss.pdf',format="pdf", bbox_inches="tight")
    #plt.close()


#data = {
#        'step': value_function.step,
#        'model': value_function.model.state_dict(),
#        'ema': value_function.ema_model.state_dict()
#}

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/models/state_{f}_MMD_Gauss.pt'.format(f=epochs))


plt.figure()
plt.plot(range(len(loss_array)),loss_array)
plt.xlabel('Epoch Number',fontsize=12)
plt.ylabel('MMD Loss',fontsize=12)
print(loss_array)
plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MMD_Gauss.pdf',format="pdf", bbox_inches="tight")
plt.show()


#model_trained=torch.load('logs/maze2d-umaze-v1/values/state_10000.pt')

#print(len(model_trained['fc.weight']))
#for param in model_trained['fc.weight']:
#  print(len(param))
#  print(param)
                
    

    

    




