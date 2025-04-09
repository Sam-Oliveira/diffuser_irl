import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)

Activations = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

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
        attention=False,
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
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
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

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
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
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
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

        print(in_out)
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
    
class ValueFunction_1Layer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.fc = nn.Linear(128*6,1,bias=False) #dimensions for umaze
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        x=torch.flatten(x,start_dim=1) #changed this and the return a bit on 13th July
        x = self.fc(x)

        return x

class ValueFunction_4Layer_UMaze(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=8,  
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(128*6,384) #dimensions for umaze
        self.fc2 = nn.Linear(384,128) #dimensions for umaze
        self.fc3 = nn.Linear(128,64) #dimensions for umaze
        self.fc4 = nn.Linear(64,1) #dimensions for umaze
        self.non_lin=torch.nn.ReLU()
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        x=torch.flatten(x,start_dim=1) 
        x = self.non_lin(self.fc1(x))
        x = self.non_lin(self.fc2(x))
        x = self.non_lin(self.fc3(x))
        x = self.fc4(x)

        return x

# Value network to learn reward in large maze
class ValueFunction_4Layer_LargeMaze(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.fc1 = nn.Linear(384*6,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128) 
        self.fc4 = nn.Linear(128,1) 
        self.non_lin=torch.nn.ReLU()
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        x=torch.flatten(x,start_dim=1) 
        x = self.non_lin(self.fc1(x))
        x = self.non_lin(self.fc2(x))
        x = self.non_lin(self.fc3(x))
        x = self.fc4(x)

        return x
    
class ValueFunction_Mujoco_Horizon4(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=8,  # I THINK THIS MIGHT BE THE HORIZON?? since they had 32 in main branch, which is the base horizon for locomotion
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        #self.input_size = input_size
        #self.hidden_size = hidden_size
        #self.output_size = output_size
        #self.sin=SinusoidalPosEmb(dim),
        self.fc1 = nn.Linear(23*4,64) #dimensions for umaze
        self.fc2 = nn.Linear(64,32) #dimensions for umaze
        self.fc3 = nn.Linear(32,16) #dimensions for umaze
        self.fc4 = nn.Linear(16,1) #dimensions for umaze
        self.non_lin=torch.nn.ReLU()
        #self.fc = nn.Linear(384*6,1,bias=False) #dimensions for large maze
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        #x[:, :, 0] = 0
        #x = self.sin(x)

        # NN to learn reward of function below

        x=torch.flatten(x,start_dim=1) #changed this and the return a bit on 13th July
        x = self.non_lin(self.fc1(x))
        x = self.non_lin(self.fc2(x))
        x = self.non_lin(self.fc3(x))
        x = self.fc4(x)


        return x
    
class ValueFunction_Mujoco(nn.Module):
    def __init__(
        self,
        horizon=4,
        activation='ReLU'
    ):
        super().__init__()
        self.horizon=horizon
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.horizon),
            nn.Linear(self.horizon, self.horizon * 4),
            nn.Mish(),
            nn.Linear(self.horizon * self.horizon, 32),
        )
        if horizon==4:
            self.fc1 = nn.Linear(23*4+32,64) #dimensions for umaze
            self.fc2 = nn.Linear(64,32) #dimensions for umaze
            self.fc3 = nn.Linear(32,1) #dimensions for umaze
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            #self.ln1=nn.LayerNorm(64)
            #self.ln2=nn.LayerNorm(32)
        elif horizon==32:
            self.fc1 = nn.Linear(23*32+32,256) #dimensions for umaze
            self.fc2 = nn.Linear(256,128)
            self.fc3 = nn.Linear(128,1) #dimensions for umaze
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            #self.ln1=nn.LayerNorm(256)
            #self.ln2=nn.LayerNorm(128)
        else:
            self.fc1 = nn.Linear(23*horizon+32,horizon*8) #dimensions for umaze
            self.fc2 = nn.Linear(horizon*8,horizon*4) #dimensions for umaze
            self.fc3 = nn.Linear(horizon*4,1) #dimensions for umaze
            #self.fc4 = nn.Linear(horizon*2,1) #dimensions for umaze
        if activation=='Tanh':
            self.non_lin=torch.nn.Tanh()
        elif activation=='LeakyReLU':
            self.non_lin=torch.nn.LeakyReLU()
        else:
            self.non_lin=torch.nn.ReLU()
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''
        t=self.time_mlp(time)
        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        x[:, 6:, 0] = 0
        #x = self.sin(x)

        # NN to learn reward of function below

        x=torch.flatten(x,start_dim=1) #changed this and the return a bit on 13th July
        x=torch.cat((x,t),dim=-1)  
        x = self.non_lin(self.fc1(x))
        x = self.non_lin(self.fc2(x))
        x = self.fc3(x)
        #return torch.zeros
        return x

# True value model (5x x coordinate + 5x y coordinate) 
class TrueReward(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=8,  
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        self.fc = nn.Linear(128*6,1,bias=False) #dimensions for umaze
        
        
    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        x=x[:,2:4,:]
        x=torch.sum(x,dim=1)
        x=torch.sum(x,dim=1,keepdim=True)
        return 5*x
