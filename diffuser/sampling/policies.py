from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process (I think it calls ValueDiffusion in diffuser/models/diffusion.py, but doesnt rly make sense cause that expects a t argument
        # IT SEEMS THIS CALLS FIRST THE BASE DIFFUSION AND ONLY THEN THE VALUEDIFFUSION GETS CALLED SOMEHOW. theory simply based on printing messages, no idea what is actually happening
        #print('HERE')
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)

        #print('THERE')
        trajectories = samples.trajectories

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        #actions.register_hook(lambda grad: print(grad))
        ## extract first action (of first element of batch i believe)
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        #observations.register_hook(lambda grad: print(grad))
        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
