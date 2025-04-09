import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import diffuser.utils as utils
import diffuser.sampling as sampling
from torch.utils.data import DataLoader
from diffuser.models.helpers import MMD_loss
from torch.utils.data import SubsetRandomSampler

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('guided_learning')

#---------------------------------- loading ----------------------------------#
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.value_model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.value_diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
#print(batch)
#batch=batch[:-1]
# * unpacks arguments in batch
#loss, _ = diffusion.loss(*batch)

# havent managed to make this work in my super simple network. it was working with random linear network
# and then stopped, idk why, when I just did slicing of x coordinate
#loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# Just save untrained model checkpoint
trainer.save(0)

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

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=diffusion_experiment.dataset.normalizer,
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


# dataset has 996000 4-step parts of trajectories. here we just select 10k first ones
subset_indices=[i for i in range(968000//args.horizon)] #not full dataset
train_dataloader=DataLoader(dataset, batch_size=128,num_workers=0,sampler=SubsetRandomSampler(subset_indices))

#expert_trajectories=expert_trajectories.to(torch.float32)
# Arguments

epochs=200

loss = torch.nn.MSELoss(reduction='mean')
#loss = MMD_loss()
optimizer = torch.optim.Adam(value_function.model.parameters(), lr=1e-3, weight_decay=0.005)

loss_array=[]
for e in range(epochs):
    print(len(train_dataloader))
    print(len(dataset))
    print("EPOCH "+str(e))
    curr_loss=0
    terms=0
   # FIRST ONE IS FOR BATCH DATA, BUT TRYING TO USE DATALOADER. ALSO NOW I HAVE CODE FOR DIFF CONDITIONING POINTS, SO I TRY TO DO IT AS IN MMD.
    # ALSO NOTE FOR THIS CASE, WE DO ONLY 1 UPDATE STEP ISNTREAD OF 1 EVERY DATAPOINT LIKE IN BOTH CASES BELOW. SO WILL PROB NEED LARGER LEARNING RATE

    for targets in train_dataloader:

        observations=targets.conditions[0].detach().cpu()
        conditions={0:observations}
        action,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

        # No sliding window
        sample_actions=samples.actions
        sample_observations=samples.observations
        #sample_observations=dataset.normalizer.normalize(sample_observations, 'observations')
        #sample_actions=dataset.normalizer.normalize(sample_actions, 'actions')
        predictions=torch.cat((sample_actions,sample_observations),dim=-1)
        targets_loss=targets.trajectories.to(torch.device(args.device))

        loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets_loss,start_dim=1))
        #print(loss_value)
        #print(torch.flatten(predictions,start_dim=1)[:,720:])
        #print(torch.flatten(targets_loss,start_dim=1)[:,720:])
        #print(targets_loss.shape)
        #print(predictions.shape)
        loss_value.backward() # gradients will be accumulated across different datapoints, and then backprop once we have gone through entire data
        #torch.nn.utils.clip_grad_norm_(value_function.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_value_(value_function.model.parameters(), clip_value=0.05)
        #for name,param in value_function.named_parameters():
        #    print(name)
        #    print(param.grad)

        curr_loss+=loss_value.detach().cpu().numpy()
        terms+=1

        optimizer.step()

        optimizer.zero_grad()
    loss_array.append(curr_loss/terms)

    if e%10==0 or e==epochs-1:
        torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/state_{f}.pt'.format(f=e+1))
    plt.figure()
    plt.plot(range(len(loss_array)),loss_array)
    plt.xlabel('Epoch Number',fontsize=12)
    plt.ylabel('MSE Loss',fontsize=12)
    print(loss_array)
    plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MSE.pdf',format="pdf", bbox_inches="tight")
    #plt.close()

# NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
torch.save(value_function.state_dict(),args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/state_{f}.pt'.format(f=epochs))


plt.figure()
plt.plot(range(len(loss_array)),loss_array)
plt.xlabel('Epoch Number',fontsize=12)
plt.ylabel('MSE Loss',fontsize=12)
print(loss_array)
plt.savefig(args.logbase+'/'+args.dataset+'/'+args.value_loadpath+'/loss_function_MSE.pdf',format="pdf", bbox_inches="tight")
plt.show()


#model_trained=torch.load('logs/maze2d-umaze-v1/values/state_10000.pt')

#print(len(model_trained['fc.weight']))
#for param in model_trained['fc.weight']:
#  print(len(param))
#  print(param)

# EVALUATE    

value_experiment = utils.load_diffusion_learnt_reward(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.env_seed,
)
dataset = value_experiment.dataset
value_function = value_experiment.ema

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)
logger = logger_config()

env=dataset.env
num_envs=20

# create multiple envs
envs=gym.vector.SyncVectorEnv([

    lambda: gym.make(args.dataset) for i in range(num_envs)

])

observation=envs.reset()


## observations for rendering
rollout = [observation.copy()] #1st observation I think

total_reward = 0
trajectories=[]

max_steps=env.max_episode_steps
#max_steps=128
#max_steps=200
#max_steps=5
learnt_trajectories=torch.empty((num_envs,max_steps,dataset.observation_dim+dataset.action_dim))
for t in range(max_steps):

    if t % 10 == 0: print(args.savepath, flush=True)


    ## save state for rendering only
    state=envs.observations.copy()

    ## IMPAINTING
    #target = env._target 
    #conditions = {0: observation,diffusion.horizon - 1: np.array([*target, 0, 0])}


    ## format current observation for conditioning (NO IMPAINTING)
    conditions = {0: envs.observations}

    #i think basically we take 1 step, and plan again every time! (in rollout image. in plan, it's just the plan at first step)
    action, samples = policy(conditions, batch_size=args.batch_size,diff_conditions=True,verbose=args.verbose)
    

    actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()
    trajectories.append(np.concatenate((actions,envs.observations),axis=-1))
    learnt_trajectories[:,t,:]=(torch.cat((torch.from_numpy(actions),torch.from_numpy(envs.observations)),axis=-1))


    next_observation, reward, terminal, _ = envs.step(samples.actions[:,0].detach().cpu().numpy())

    ## print reward and score
    total_reward += reward

    ## update rollout observations. Note this does not include actions! Rollout is a list of nparrays, each of them is the current state at a step
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps. Just basically renders one of the items in batch.
    # can change render method if i want to render the entire batch. not worth it now
    #samples=samples._replace(observations=samples.observations[[0]])
    #samples=samples._replace(actions=samples.actions[[0]])
    #samples=samples._replace(values=samples.values[[0]])
    #logger.log(t, samples, state[[0]], np.stack(rollout,axis=1)[[0],:,:])


    if terminal.any():
        break

## write results to json file at `args.savepath`
logger.finish(t, 0, total_reward.tolist(), bool(terminal.any()), diffusion_experiment, value_experiment)

    

    




