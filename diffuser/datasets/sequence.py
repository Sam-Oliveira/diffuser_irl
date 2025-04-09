from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,seed=0):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
    
class Dataset_medium_replay_norm():
    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,seed=0):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        # get medium-replay dataset
        itr_mr = sequence_dataset(load_environment('halfcheetah-medium-replay-v2'), self.preprocess_fn)

        fields_mr = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr_mr):
            fields_mr.add_path(episode)
        fields_mr.finalize()
        
        # changed this so normalizer is initialized with medium-replay dataset
        self.normalizer = DatasetNormalizer(fields_mr, normalizer, path_lengths=fields_mr['path_lengths'])


        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class Dataset_Expert_mixed_norm():
    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,seed=0,medium_replay_ratio=0.5):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, 'halfcheetah-medium-replay-v2')
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # Load both datasets
        env_mr = load_environment('halfcheetah-medium-replay-v2')
        env_mr.seed(seed)
        itr_mr = sequence_dataset(env_mr, self.preprocess_fn)

        env_exp = load_environment('halfcheetah-expert-v2')
        env_exp.seed(seed)
        itr_exp = sequence_dataset(env_exp, self.preprocess_fn)

        # First, collect all episodes
        mr_episodes = []
        exp_episodes = []
        for episode in itr_mr:
            mr_episodes.append(episode)
        for episode in itr_exp:
            exp_episodes.append(episode)

        # Create a temporary buffer for calculating normalizer
        normalizer_buffer = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        max_mr_episodes = len(mr_episodes)
        max_exp_episodes = len(exp_episodes)

        if medium_replay_ratio == 1:
            # Use only medium-replay data
            num_mr_episodes = min(max_n_episodes, max_mr_episodes)
            num_exp_episodes = 0
        elif medium_replay_ratio == 0:
            # Use only expert data
            num_mr_episodes = 0
            num_exp_episodes = min(max_n_episodes, max_exp_episodes)
        else:
            # Normal case with mixed data
            # diff calculation than the one in SplitDataset, but they are equivalent in final calculated
            # num_mr_episodes and num_exp_episodes
            desired_mr_episodes = int(max_n_episodes * medium_replay_ratio)
            desired_exp_episodes = max_n_episodes - desired_mr_episodes
            
            scale = min(max_mr_episodes / desired_mr_episodes,
                    max_exp_episodes / desired_exp_episodes)
            
            num_mr_episodes = int(desired_mr_episodes * scale)
            num_exp_episodes = int(desired_exp_episodes * scale)

        # Randomly select episodes for normalizer calculation
        rng = np.random.RandomState(seed)
        selected_mr_episodes = rng.choice(len(mr_episodes), num_mr_episodes, replace=False)
        selected_exp_episodes = rng.choice(len(exp_episodes), num_exp_episodes, replace=False)

        # Add selected episodes to normalizer buffer
        for idx in selected_mr_episodes:
            normalizer_buffer.add_path(mr_episodes[idx])
        for idx in selected_exp_episodes:
            normalizer_buffer.add_path(exp_episodes[idx])

        normalizer_buffer.finalize()

        # Calculate normalizer using the mixed dataset
        self.normalizer = DatasetNormalizer(normalizer_buffer, normalizer, 
                                          path_lengths=normalizer_buffer['path_lengths'])

        # Now create the actual dataset buffer with only expert data
        self.fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        # Add all expert episodes
        for episode in exp_episodes:
            self.fields.add_path(episode)
            
        self.fields.finalize()

        # Create indices for the expert-only dataset
        self.indices = self.make_indices(self.fields.path_lengths, horizon)

        self.observation_dim = self.fields.observations.shape[-1]
        self.action_dim = self.fields.actions.shape[-1]
        self.n_episodes = self.fields.n_episodes
        self.path_lengths = self.fields.path_lengths
        
        # Normalize the expert-only dataset using the mixed normalizer
        self.normalize()

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, env='hopper-medium-replay',horizon=64, normalizer='LimitsNormalizer', 
                 preprocess_fns=[], max_path_length=1000,
                 max_n_episodes=10000, termination_penalty=0, 
                 use_padding=True, seed=0, medium_replay_ratio=0.5):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, 'halfcheetah-medium-replay-v2')
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # Load both datasets
        env_mr = load_environment('halfcheetah-medium-replay-v2')
        env_mr.seed(seed)
        itr_mr = sequence_dataset(env_mr, self.preprocess_fn)

        env_exp = load_environment('halfcheetah-expert-v2')
        env_exp.seed(seed)
        itr_exp = sequence_dataset(env_exp, self.preprocess_fn)

        # First, collect all episodes
        mr_episodes = []
        exp_episodes = []
        for episode in itr_mr:
            mr_episodes.append(episode)
        for episode in itr_exp:
            exp_episodes.append(episode)

        # Create a single ReplayBuffer
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        # Calculate maximum possible episodes while maintaining ratio
        max_mr_episodes = len(mr_episodes)
        max_exp_episodes = len(exp_episodes)
        
        if medium_replay_ratio == 1:
            num_mr_episodes = min(max_n_episodes, max_mr_episodes)
            num_exp_episodes = 0
        elif medium_replay_ratio == 0:
            num_mr_episodes = 0
            num_exp_episodes = min(max_n_episodes, max_exp_episodes)
        else:
            # Calculate how many episodes we can include while maintaining the ratio
            # If we want ratio a:b, and have available A:B episodes,
            # we can include min(A/a, B/b) * a medium-replay episodes
            # and min(A/a, B/b) * b expert episodes
            scale = min(max_mr_episodes / medium_replay_ratio, 
                       max_exp_episodes / (1 - medium_replay_ratio))
            num_mr_episodes = int(scale * medium_replay_ratio)
            num_exp_episodes = int(scale * (1 - medium_replay_ratio))

        # Randomly select episodes
        rng = np.random.RandomState(seed)
        if num_mr_episodes > 0:
            selected_mr_episodes = rng.choice(len(mr_episodes), num_mr_episodes, replace=False)
            for idx in selected_mr_episodes:
                fields.add_path(mr_episodes[idx])
        if num_exp_episodes > 0:
            selected_exp_episodes = rng.choice(len(exp_episodes), num_exp_episodes, replace=False)
            for idx in selected_exp_episodes:
                fields.add_path(exp_episodes[idx])
            
        fields.finalize()

        # Create normalizer based on merged dataset
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        # Normalize the merged dataset
        self.normalize()

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
