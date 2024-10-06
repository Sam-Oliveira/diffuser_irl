from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')

class Policy:

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size,diff_conditions=False):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        if diff_conditions:
            return conditions
        else:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                'd -> repeat d', repeat=batch_size,
            )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1,diff_conditions=False):


        conditions = self._format_conditions(conditions, batch_size,diff_conditions)


        ## run reverse diffusion process
        sample = self.diffusion_model(conditions)
        sample = sample.trajectories.detach()

        ## extract action [ batch_size x horizon x transition_dim ]

        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')


        trajectories = Trajectories(actions, observations)
        return action, trajectories
