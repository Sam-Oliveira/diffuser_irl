import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import diffuser.utils as utils
import diffuser.sampling as sampling
from torch.utils.data import DataLoader
from diffuser.models.helpers import MMD_loss
from torch.utils.data import SubsetRandomSampler
import wandb
from pathlib import Path

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-expert-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#
parser=Parser()
args = parser.parse_args('guided_learning')

def train(args,config=None):
    #---------------------------------- loading ----------------------------------#

    # create directory for new value function
    value_path=parser.mk_sweep_dir(args,config)

    dataset_config = utils.Config(
        args.loader,
        savepath=(value_path, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=config.horizon, #changed
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
    )

    render_config = utils.Config(
        args.renderer,
        savepath=(value_path, 'render_config.pkl'),
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
        savepath=(value_path, 'model_config.pkl'),
        horizon=config.horizon, #changed
        activation=config.activation_function,
        device=args.device,
    )

    diffusion_config = utils.Config(
        args.value_diffusion,
        savepath=(value_path, 'diffusion_config.pkl'),
        horizon=config.horizon, #changed
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
        savepath=(value_path, 'trainer_config.pkl'),
        train_batch_size=args.batch_size, 
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=value_path,
        bucket=args.bucket,
        n_reference=args.n_reference,
        n_samples=args.n_samples,
    )

    #-----------------------------------------------------------------------------#
    #-------------------------------- instantiate --------------------------------#
    #-----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(model, dataset, renderer)

    #-----------------------------------------------------------------------------#
    #------------------------ test forward & backward pass -----------------------#
    #-----------------------------------------------------------------------------#

    print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0])
    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    # Just save untrained model checkpoint
    #Path(args.logbase+'/'+args.dataset+'/'+args.value_loadpath).mkdir(parents=True, exist_ok=True)
    #torch.save(model.state_dict(),value_path+'/state_0.pt')
    #trainer.save(0)
    torch.save(model.state_dict(),value_path+'/state_0.pt')

    args.diffusion_loadpath='diffusion/H{horizon}_T{n_diffusion_steps}'.format(horizon=config.horizon,n_diffusion_steps=args.n_diffusion_steps)

    diffusion_experiment = utils.load_diffusion(args.logbase, 'halfcheetah-medium-replay-v2', args.diffusion_loadpath, epoch=args.diffusion_epoch,seed=args.env_seed)

    value_experiment = utils.load_diffusion_learnt_reward( # changed this function, instead of just being load_diffusion()
        '', '', value_path,
        epoch=args.value_epoch, seed=args.env_seed,
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(diffusion_experiment, value_experiment)

    diffusion = diffusion_experiment.ema
    #dataset = value_experiment.dataset
    renderer = diffusion_experiment.renderer

    ## initialize value guide
    value_function = value_experiment.model

    #ValueGuide (guiddes.py) takes ValueFunction (temporal.py) as its model
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()


    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=config.scale, #change
        diffusion_model=diffusion,
        normalizer=diffusion_experiment.dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs (idk what these mean)
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=config.t_stopgrad, #change
        scale_grad_by_std=config.scale_grad_by_std,#change
        verbose=False,
    )

    # calls the guided policy class, instead of the normal policy class that was used for unguided planning
    policy = policy_config()

    #---------------------------------- main loop ----------------------------------#
    env=dataset.env
    observation = env.reset()


    # dataset has 996000 4-step parts of trajectories. here we just select a subset
    #subset_indices=[i for i in range(968000//config.horizon)] # changed
    subset_indices=[i for i in range(50000)]
    #train_dataloader=DataLoader(dataset, batch_size=config.batch_size,num_workers=0,sampler=SubsetRandomSampler(subset_indices))# changed
    train_dataloader=DataLoader(dataset, batch_size=config.batch_size,num_workers=0,sampler=torch.utils.data.RandomSampler(dataset,num_samples=20000))
    # Arguments

    epochs=100

    # Loss function
    if config.loss=='MSE':
        loss = torch.nn.MSELoss(reduction='mean')
    elif config.loss=='MMD_Gauss':
        loss = MMD_loss()
    elif config.loss=='MMD_Matern':
        loss = MMD_loss(kernel='matern')
    else:
        raise Exception('Invalid Loss in W&B config file')
    
    # Optimizer
    if config.optimizer=='Adam':
        optimizer = torch.optim.Adam(value_function.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer=='RMSProp':
        optimizer = torch.optim.RMSprop(value_function.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer=='SGD':
        optimizer = torch.optim.SGD(value_function.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer=='Adagrad':
        optimizer = torch.optim.Adagrad(value_function.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise Exception('Invalid Opt in W&B config file')

    wandb.watch(value_function)
    for e in range(1,epochs+1):

        value_function.train()
        optimizer.zero_grad()
        print('EPOCH '+str(e))

        curr_loss=0

        # eval
        if (e-1)%10==0:
            value_function.eval()
            test(value_path,policy,diffusion_experiment,dataset,config,False,e)
        
        value_function.train()
        optimizer.zero_grad()
        
        for targets in train_dataloader:

            observations=targets.conditions[0].detach().cpu()
            conditions={0:observations}
            action,unnormed_samples,samples=policy(conditions,batch_size=observations.shape[0],diff_conditions=True,verbose=args.verbose)

            # No sliding window
            sample_actions=samples.actions
            sample_observations=samples.observations

            predictions=torch.cat((sample_actions,sample_observations),dim=-1)
            
            target_obs=policy.normalizer.normalize(targets.trajectories[:,:,6:], 'observations')
            target_act=policy.normalizer.normalize(targets.trajectories[:,:,:6], 'actions')
            targets_loss=torch.cat((target_act,target_obs),dim=-1).to(torch.device(args.device))
            
            #targets_loss=targets.trajectories.to(torch.device(args.device))

            loss_value=loss(torch.flatten(predictions,start_dim=1),torch.flatten(targets_loss,start_dim=1))

            loss_value.backward() 
            #torch.nn.utils.clip_grad_norm_(value_function.model.parameters(), max_norm=1.0)
            #print(predictions[0,-1,:])
            #print(targets_loss[0,-1,:])
            #print(predictions[0,0,6:])
            #print(targets_loss[0,0,6:])
            torch.nn.utils.clip_grad_value_(value_function.parameters(), clip_value=0.05)

            curr_loss+=loss_value.detach().cpu().numpy()

            optimizer.step()

            optimizer.zero_grad()
        
        if config.loss=='MSE':
            wandb.log({'mse_loss':curr_loss/len(train_dataloader)})
        elif config.loss=='MMD_Gauss':
            wandb.log({'mmd_gauss_loss':curr_loss/len(train_dataloader)})
        elif config.loss=='MMD_Matern':
            wandb.log({'mmd_matern_loss':curr_loss/len(train_dataloader)})
        else:
            raise Exception('Invalid Loss in W&B config file')

        # simply save as state_1 so we don't have too many files. need this to save all the time so that test uses recent value function
        torch.save(value_function.state_dict(),value_path+'/state_1.pt')

    # NOTE: SAVE WITHOUT .model. so that the parameters have name model.fc.weight instead of fc.weight, and thus match what load() function in training.py expects! 
    #torch.onnx.export(value_function, args=(),f="model.onnx")
    # simply save as state_1 so we don't have too many files. need this to save all the time so that test uses recent value function
    torch.save(value_function.state_dict(),value_path+'/state_2.pt')
    wandb.save(value_path+'/state_2.pt')
    value_function.eval()
    test(value_path,policy,diffusion_experiment,dataset,config,True,e)
    return value_path


def test(value_path,policy,diffusion_experiment,dataset,config=None,final=False,epoch=0):


    env=dataset.env
    num_envs=30

    # create multiple envs
    envs=gym.vector.SyncVectorEnv([

        lambda: gym.make(args.dataset) for i in range(num_envs)

    ])

    observation=envs.reset()


    ## observations for rendering
    rollout = [observation.copy()] #1st observation I think

    total_reward = 0

    max_steps=200

    for t in range(max_steps):

        ## format current observation for conditioning (NO IMPAINTING)
        conditions = {0: envs.observations}

        #i think basically we take 1 step, and plan again every time! (in rollout image. in plan, it's just the plan at first step)
        action, samples,_ = policy(conditions, batch_size=num_envs,diff_conditions=True,verbose=args.verbose)

        actions=torch.squeeze(samples.actions[:,0,:]).detach().cpu().numpy()
        next_observation, reward, terminal, _ = envs.step(samples.actions[:,0].detach().cpu().numpy())

        ## print reward and score
        total_reward += reward

        if terminal.any():
            break
    if final:
        wandb.log({'final_reward':np.mean(total_reward)})
    else:
        wandb.log({'reward':np.mean(total_reward)})


def learning_reward(config=None):
    with wandb.init(config=config):
        config=wandb.config
        args = Parser().parse_args('guided_learning')
        value_path=train(args,config)

def run_hyperparameter_sweep():
    sweep_config={'method':'random'}
    metric={'name':'final_reward','goal':'maximize'}
    sweep_config['metric']=metric
    
    # Hyperparameters to vary
    parameters_dict={
        'horizon':{
            'values':[4,32]
        },
        'scale':{
            'values':[0.1,0.01,0.001,0.0001]
        },
        'scale_grad_by_std':{
            'values':[True,False]
        },
        't_stopgrad':{
            'values':[0,2,4,8]
        },
        'lr':{
            'distribution':'log_uniform_values',
            'min':1e-4,
            'max':5e-2, #too large, need to change
        },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'values':[128,256,512]
        },
        'loss':{
            'values':['MSE','MMD_Gauss','MMD_Matern']
        },
        'optimizer':{
            'values':['Adam','RMSProp','SGD','Adagrad']
        },
        'weight_decay':{
            'values':[0,1e-5,1e-4,1e-3]
        },
        'activation_function':{
            'values':['ReLU','Tanh','LeakyReLU']
        },
    }
    sweep_config['parameters']=parameters_dict
    import pprint
    pprint.pprint(sweep_config)
    sweep_id=wandb.sweep(sweep_config,project='irl_halfcheetah')
    wandb.agent(sweep_id,function=learning_reward,count=75)

run_hyperparameter_sweep()