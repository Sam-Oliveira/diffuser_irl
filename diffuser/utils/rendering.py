import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as clr
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment
import torch

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
}

class MazeRenderer:

    def __init__(self, env):
        if type(env) is str: env = load_environment(env)
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        norm_map=clr.Normalize(vmin=0, vmax=1)
        cbar=plt.colorbar(plt.cm.ScalarMappable(norm=norm_map, cmap='jet_r'),shrink=0.8)
        cbar.ax.get_yaxis().labelpad = 20
        #plt.gcf().axes[1].set(title='Beginning', xlabel='End')
        cbar.ax.set_title('Beginning')
        cbar.ax.set_xlabel('End',fontsize=14)
        cbar.set_ticks([])
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')
        #return images #ADDED FOR NOTEBOOK

class Maze2dRenderer(MazeRenderer):

    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def plot2img(fig, remove_margins=True):
        # https://stackoverflow.com/a/35362787/2912349
        # https://stackoverflow.com/a/54334430/2912349

        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if remove_margins:
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_as_string, (width, height) = canvas.print_to_buffer()
        return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

    def _render_field(self, field, title, **kwargs):
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        #print(self._background)
        #print(self._extent)
        ax.imshow(self._background,
                    extent=self._extent, alpha = 0.25, cmap=plt.cm.binary, vmin=0, vmax=1, zorder = 10)
        """
        cmap = plt.cm.viridis
        cmap.set_bad((0.66, 0.66, 0.66, 1))

        # Set the diagonal to NaN
        field[self._background] = np.nan

        img=ax.imshow(
            field, 
            cmap=cmap,
            extent = self._extent, 
            alpha = 1, zorder = 0, **kwargs
        ) 
        """
        field_interm=field.copy()
        field_interm[self._background]=np.nan
        #print(field_interm)
        #print(np.nanmax(field_interm))
        #print(field)
        """
        img=ax.imshow(
            field, 
            cmap=plt.cm.viridis,
            extent = self._extent, 
            alpha = 1, zorder = 0, interpolation='spline36',vmin=0,vmax=np.nanmax(field_interm),**kwargs
        ) 
        """
        img=ax.imshow(
            field, 
            cmap=plt.cm.viridis,
            extent = self._extent, 
            alpha = 1, zorder = 0, interpolation='spline36',vmin=np.nanmin(field_interm),vmax=np.nanmax(field_interm),**kwargs
        ) 

        #cmap = plt.cm.viridis
        #cmap.set_bad((0.66, 0.66, 0.66, 1))

        # Set the diagonal to NaN
        field[~self._background] = np.nan
        ax.imshow(
            field*0+0.43, 
            cmap=plt.cm.Greys,
            extent = self._extent, 
            alpha = 1, zorder = 0, vmin=0, vmax=1, **kwargs
        )

        plt.axis('off')
        cbar=plt.colorbar(img,ax=ax,shrink=0.8)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('State value',rotation=270,fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        #plt.title(title)
        plt.savefig("State_value.pdf", format="pdf", bbox_inches="tight")
        plt.savefig('HEATMAP.png')
        img = plot2img(fig, remove_margins=self._remove_margins)

        return img

    def render_reward_heatmap(self, trajectories, preds, shape_factor = 1, samples_thresh = 0):
        _, bx, _, by = MAZE_BOUNDS[self.env_name]
        bounds = torch.tensor([bx, by])
        shape_t = (bounds * shape_factor).int().reshape((1, 2))
        shape = shape_t.flatten().tolist()

        #horizon = trajectories.shape[1]
        trajectories_flat = trajectories
        preds_repeated = preds

        #------
        #breakpoint()
        trajectories = trajectories_flat.numpy()
        preds = preds_repeated.numpy()

        #preds = scipy.stats.mstats.winsorize(preds, limits = (0.005, 0.995))
        hist = np.histogram2d(
            trajectories[:, 0] + 0.5, 
            trajectories[:, 1] + 0.5, 
            weights = np.ravel(preds),
            bins = shape,
            range = [[0, bx], [0, by]]
        )

        hist_counts = np.histogram2d(
            trajectories[:, 0] + 0.5, 
            trajectories[:, 1] + 0.5, 
            bins = shape,
            range = [[0, bx], [0, by]]
        )

        mask = hist_counts[0] >= samples_thresh

        #counts_field = hist_counts[0] / trajectories.shape[0]
        reward_field = hist[0] / np.maximum(hist_counts[0], 1)

        reward_field *= mask
        #counts_field *= mask

        img_reward = self._render_field(reward_field, title="Reward heatmap")
        #img_counts = self._render_field(counts_field, title="Counts heatmap")

        return img_reward, reward_field

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)
        
    def composite_reward_function(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''

        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.render_reward(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

    def render_reward(self, values, conditions=None, title=None):
        bounds = MAZE_BOUNDS[self.env_name]

        if len(bounds) == 2:
            _, scale = bounds
            #observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            #observations[:, 0] /= iscale
            #observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        if conditions is not None:
            conditions /= scale

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        # Plot the overlay using a different colormap
        #plt.imshow(values, cmap='viridis_r', vmin=0, vmax=1)
        inner_extent=(0, 1,1, 0)
        plt.imshow(values, extent=inner_extent, interpolation='bilinear', cmap='viridis', alpha=0.8)
        plt.axis('off')
        plt.title(title)
        cbar=plt.colorbar(shrink=0.8)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Reward for state',rotation=270,fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img



#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
