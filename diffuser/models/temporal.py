import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import torch.nn.functional as F

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
    

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=6, 
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        #self.input_size = input_size
        #self.hidden_size = hidden_size
        #self.output_size = output_size
        #self.sin=SinusoidalPosEmb(dim),
        self.i2h = nn.Linear(768, 1536)
        self.h2h = nn.Linear(1536, 768)
        self.h2o = nn.Linear(768,1)
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        #x[:, :, 0] = 0
        #x = self.sin(x)

        # I THINK FIRST TWO ELEMENTS ARE THE ACTIONS
        #print(x[:,:,-1])
        #print(x[0,:,0])

        # So first two coordinates are actions. Then 3rd and 4th are coordinates, but I think y coordinate comes before x.
        # Also note when printed, I think y gets printed before x! and they start on top left corner.
        x=x[:,3,:]

        #x=torch.sum(x,dim=1)
        x=torch.sum(x,dim=1,keepdim=True)
        #print(x)
        #x=torch.where(x>1,(x-1)*10,x-1)
        #print(x)
        return 5*x
        #return x.reshape((1,x.shape[0]))

        x=torch.flatten(x,start_dim=1)
        x = F.relu(self.i2h(x))
        x = F.relu(self.h2h(x))
        x = F.relu(self.h2o(x))
        print(x.shape)
        return x
    

""" this class is used by them for guide in main branch. however, i think see class below for guide for maze2d! (they dont actually do it, but they include this code in maze2d branch?)
class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=6, 
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        #print(dims)
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out  
"""

# This was in maze2d original branch, but I haeve no idea why? it doesnt seem to be used anywhere.
# i think it's the equivalent of ValueFunction() but for maze2d
# note u need to call this in init_values in config/maze2d.py, and initiate values, before running guided planning
class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out


# class TemporalMixerUnet(nn.Module):

#     def __init__(
#         self,
#         horizon,
#         transition_dim,
#         cond_dim,
#         dim=32,
#         dim_mults=(1, 2, 4, 8),
#     ):
#         super().__init__()
#         # self.channels = channels

#         dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         time_dim = dim
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(dim),
#             nn.Linear(dim, dim * 4),
#             nn.Mish(),
#             nn.Linear(dim * 4, dim),
#         )
#         self.cond_mlp = nn.Sequential(
#             nn.Linear(cond_dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim),
#         )

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         print(in_out)
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(nn.ModuleList([
#                 ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 nn.Identity(),
#                 Downsample1d(dim_out) if not is_last else nn.Identity()
#             ]))

#             if not is_last:
#                 horizon = horizon // 2

#         mid_dim = dims[-1]
#         self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)
#         self.mid_attn = nn.Identity()
#         self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(nn.ModuleList([
#                 ResidualTemporalBlock(dim_out * 2, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 ResidualTemporalBlock(dim_in, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 nn.Identity(),
#                 Downsample1d(dim_in) if not is_last else nn.Identity()
#             ]))

#             if not is_last:
#                 horizon = horizon * 2

#         self.final_conv = nn.Sequential(
#             # TemporalHelper(dim, dim, kernel_size=5),
#             Conv1dBlock(dim, dim, kernel_size=5),
#             nn.Conv1d(dim, transition_dim, 1),
#         )


#     def forward(self, x, cond, time):
#         '''
#             x : [ batch x horizon x transition ]
#         '''
#         t = self.time_mlp(time)
#         # cond = self.cond_mlp(cond)
#         cond = None

#         h = []

#         # x = x[:,None]
#         # t = torch.cat([t, cond], dim=-1)

#         x = einops.rearrange(x, 'b h t -> b t h')

#         for resnet, resnet2, attn, downsample in self.downs:
#             # print('0', x.shape, t.shape)
#             x = resnet(x, t, cond)
#             # print('resnet', x.shape, t.shape)
#             x = resnet2(x, t, cond)
#             # print('resnet2', x.shape)
#             ##
#             x = einops.rearrange(x, 'b t h -> b t h 1')
#             x = attn(x)
#             x = einops.rearrange(x, 'b t h 1 -> b t h')
#             ##
#             # print('attn', x.shape)
#             h.append(x)
#             x = downsample(x)
#             # print('downsample', x.shape, '\n')

#         x = self.mid_block1(x, t, cond)
#         ##
#         x = einops.rearrange(x, 'b t h -> b t h 1')
#         x = self.mid_attn(x)
#         x = einops.rearrange(x, 'b t h 1 -> b t h')
#         ##
#         x = self.mid_block2(x, t, cond)
#         # print('mid done!', x.shape, '\n')

#         for resnet, resnet2, attn, upsample in self.ups:
#             # print('0', x.shape)
#             x = torch.cat((x, h.pop()), dim=1)
#             # print('cat', x.shape)
#             x = resnet(x, t, cond)
#             # print('resnet', x.shape)
#             x = resnet2(x, t, cond)
#             # print('resnet2', x.shape)
#             ##
#             x = einops.rearrange(x, 'b t h -> b t h 1')
#             x = attn(x)
#             x = einops.rearrange(x, 'b t h 1 -> b t h')
#             ##
#             # print('attn', x.shape)
#             x = upsample(x)
#             # print('upsample', x.shape)
#         # pdb.set_trace()
#         x = self.final_conv(x)

#         # x = x.squeeze(dim=1)

#         ##
#         x = einops.rearrange(x, 'b t h -> b h t')
#         ##
#         return x
