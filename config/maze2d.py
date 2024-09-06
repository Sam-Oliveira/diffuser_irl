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
    ('stop_grad', 'stop-grad'),
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
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',    # I THINK I HAVE TO CHANGE THIS TO SEQUENCE_DATASET, THAT'S WHAT THEY HAVE FOR MAIN
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000, #changed this, as giovanni said he had changed it

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cpu',
    },

    'plan': {
        'batch_size': 10,
        'device': 'cpu',
        'seed':13, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for env

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10, # What is this?? this gets used to decide when to print in plan_maze2d??
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        # what does this do? I think it is for conditioning on certain observations (e.g. start/goal state)
        # It isn't about conditional sampling (i.e. setting h I think)
        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

    'init_values': {
        'model': 'models.ValueFunction_4Layer_UMaze',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 64,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.ValueDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000, #changed this, as giovanni said he had changed it

        ## serialization
        'discount':0.995,
        'logbase': 'logs',
        'prefix': 'values/',
        'exp_name': watch(value_args_to_watch),

        ## They use this in main branch, idk if i'll need them.
        #'discount': 1,
        #'normed': False,

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'value_l2',
        'n_train_steps': 2e6,
        'batch_size': 1,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cpu',
    },

    'guided_plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [], #think i might need the maze2d function from above. check what is does
        'device': 'cpu',

        ## sample_kwargs (Idk what these do)
        'n_guide_steps': 2, #the amount of steps actually taken in the environment based on each plan? just about how many steps of opt process we take in direction of guide gradient I think
        'scale': 0.1,
        't_stopgrad': 2, #no idea what this is supposed to do mathematically
        'stop_grad':False,
        'scale_grad_by_std': True,
        'conditional': False,
        'seed': 50, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':15, #seed for environment

        ## serialization
        'loadbase': None,
        'vis_freq': 10, # i think it's how often it renders
        'logbase': 'logs',
        'prefix': 'plans/guided',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        ## value function
        'discount': 0.995,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
    },

    'guided_learning': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'device': 'cpu',

        'termination_penalty': None,
        'preprocess_fns': [], #I think separates dataset into the different episodes based on timeouts
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000, #changed this, as giovanni said he had changed it

        ## sample_kwargs (Idk what these do)
        'n_guide_steps': 2, #the amount of steps actually taken in the environment based on each plan? just about how many steps of opt process we take in direction of guide gradient I think
        'scale': 0.1,
        't_stopgrad': 2, #no idea what this is supposed to do mathematically
        'stop_grad':False,
        'scale_grad_by_std': True,
        'conditional': False,
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## serialization
        'loadbase': None,
        'vis_freq': 10, # i think it's how often it renders
        'logbase': 'logs',
        'prefix': 'plans/guided',
        'exp_name': watch(learn_reward_args_to_watch),
        'suffix': '0',

        ## value function
        'discount': 0.995,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
    },

    'guided_learnt_reward': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size':1,
        'preprocess_fns': [], #think i might need the maze2d function from above. check what is does
        'device': 'cpu',

        ## sample_kwargs (Idk what these do)
        'n_guide_steps': 2, #the amount of steps actually taken in the environment based on each plan? just about how many steps of opt process we take in direction of guide gradient I think
        'scale': 0.1,
        't_stopgrad': 2, #no idea what this is supposed to do mathematically
        'stop_grad':False,
        'scale_grad_by_std': True,
        'conditional': False,
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## serialization
        'loadbase': None,
        'vis_freq': 10, # i think it's how often it renders
        'logbase': 'logs',
        'prefix': 'plans/guided_learnt_reward',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        ## value function
        'discount': 0.995,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
    },
    'guided_learning_mmd': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'device': 'cpu',

        'termination_penalty': None,
        'preprocess_fns': [], #I think separates dataset into the different episodes based on timeouts
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000, #changed this, as giovanni said he had changed it

        ## sample_kwargs (Idk what these do)
        'n_guide_steps': 2, #the amount of steps actually taken in the environment based on each plan? just about how many steps of opt process we take in direction of guide gradient I think
        'scale': 0.1,
        't_stopgrad': 2, #no idea what this is supposed to do mathematically
        'stop_grad':False,
        'scale_grad_by_std': True,
        'conditional': False,
        'seed': 40, #seed for diffusion (this gets used in diffuser/utils/setup.py to set the torch seed)
        'env_seed':13, #seed for environment

        ## serialization
        'loadbase': None,
        'vis_freq': 10, # i think it's how often it renders
        'logbase': 'logs',
        'prefix': 'plans/guided',
        'exp_name': watch(learn_reward_args_to_watch),
        'suffix': '0',

        ## value function
        'discount': 0.995,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'max_path_length': 4000,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'init_values': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'max_path_length': 4000,
    },
    'guided_plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'guided_learning': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'max_path_length': 4000,
    },
    'guided_learnt_reward': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'guided_learning_mmd': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        'max_path_length': 4000,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'init_values': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'guided_plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'guided_learning': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'guided_learnt_reward': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'guided_learning_mmd': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
