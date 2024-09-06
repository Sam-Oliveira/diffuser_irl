import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

value_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('discount','d'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
    ('env_seed','env_seed'),
]

learn_reward_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('discount','d'),
    ('env_seed','env_seed'),
]

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion_for_guide',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2,4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'n_samples': 2,
        'bucket': None,
        'device': 'cpu',
    },

    'init_values': {
        'model': 'models.ValueFunction_Mujoco',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'values/',
        'exp_name': watch(value_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cpu',
    },
    
    'unguided_plan':{
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cpu',
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment


        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/unguided',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',

        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },

    'guided_plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cpu',
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/guided',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
    'guided_learning': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cpu',
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
    'guided_learnt_reward': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cpu',
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/guided_learnt_reward',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },


    
}

#------------------------ overrides ------------------------#

## put environment-specific overrides here

hopper_medium_expert_v2 = {
    'guided_plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = halfcheetah_expert_v2= {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'unguided_plan': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'init_values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'guided_plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
    'guided_learning':{
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
    'guided_learnt_reward':{
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}

#halfcheetah_medium_expert_v2 = {
#    'diffusion': {
#        'horizon': 16,
#    },
#}
