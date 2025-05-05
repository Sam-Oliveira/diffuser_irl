import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb
import gpytorch
from torch_kdtree import build_kd_tree
import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)



#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    
    # Original code, fails when x requires_grad
    #for t, val in conditions.items():
    #    x[:, t, action_dim:] = val.clone()
    #return x

    # Code for when x requires_grad
    z=x.clone()
    for t, val in conditions.items():
        z[:, t, action_dim:] = val.clone()
    return z


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                # changed these so they dont use utils.to_np(). Not sure if would need to detach
                pred.squeeze(),
                targ.squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}


# Copied from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    # in case I change to gpu, need to somehow change this 
    #XX, YY, XY = (torch.zeros(xx.shape).to(args.device),
    #              torch.zeros(xx.shape).to(args.device),
     #             torch.zeros(xx.shape).to(args.device))
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)

# From https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
class MMD_loss(nn.Module):
    
    def __init__(self, kernel='gaussian',kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel=kernel

    def guassian_kernel(self, source, target, kernel_mul=2, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def matern_kernel(self, source, target,nu):
        total = torch.cat([source, target], dim=0)
        #print(source.device)
        #print(target.device)
        #print(total.shape)
        #covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu)).cuda()
        covar_module=gpytorch.kernels.MaternKernel(nu=nu).cuda()
        total=torch.nn.functional.normalize(total)
        n=covar_module(total)
        return n.to_dense()


    def forward(self, source, target):
        batch_size = int(source.size()[0])
        if self.kernel=='gaussian':
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            #print(XX)
            #print(YY)
            #print(XY)
            #print(YX)
            loss = torch.mean(XX + YY - XY -YX)
            return loss
        elif self.kernel=='matern':
            #kernels=self.matern_kernel(source,target,0.5)
            #XX = kernels[:batch_size, :batch_size]
            #YY = kernels[batch_size:, batch_size:]
            #XY = kernels[:batch_size, batch_size:]
            #YX = kernels[batch_size:, :batch_size]
            #loss = torch.mean(XX + YY - XY -YX)
            #return loss
            K=self.matern_kernel(source,target,1.5)
            N=source.shape[0]
            M=target.shape[0]
            #Kxx = K[:N,:N]
            #Kyy = K[N:,N:]
            #Kxy = K[:N,N:]
            #t1 = (1./(M*(M-1)))*torch.sum(Kxx - torch.diag(torch.diagonal(Kxx)))
            #t2 = (2./(M*N)) * torch.sum(Kxy)
            #t3 = (1./(N*(N-1)))* torch.sum(Kyy - torch.diag(torch.diagonal(Kyy)))
            #MMDsquared = (t1-t2+t3)
            #return MMDsquared
            X_size=source.shape[0]
            XX = torch.mean(K[:X_size, :X_size])
            XY = torch.mean(K[:X_size, X_size:])
            YY = torch.mean(K[X_size:, X_size:])
            return XX - 2 * XY + YY

class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()

    def forward(self, x, y):
        """Compute the Kullback-Leibler divergence between two multivariate samples.
        Parameters
        ----------
        x : 2D array (n,d)
            Samples from distribution P, which typically represents the true
            distribution.
        y : 2D array (m,d)
            Samples from distribution Q, which typically represents the approximate
            distribution.
        Returns
        -------
        out : float
            The estimated Kullback-Leibler divergence D(P||Q).
        References
        ----------
        Pérez-Cruz, F. Kullback-Leibler divergence estimation of
        continuous distributions IEEE International Symposium on Information
        Theory, 2008.
        """

        # Check the dimensions are consistent
        x = torch.atleast_2d(x)
        y = torch.atleast_2d(y)

        n,d = x.shape
        m,dy = y.shape

        assert(d == dy)


        # Build a KD tree representation of the samples and find the nearest neighbour
        # of each point in x.
        xtree = build_kd_tree(x)
        ytree = build_kd_tree(y)

        # Get the first two nearest neighbours for x, since the closest one is the
        # sample itself.
        #print(type(xtree.query(x, nr_nns_searches=2)))
        #print(xtree.query(x, nr_nns_searches=2))
        #r = torch.tensor(xtree.query(x, nr_nns_searches=2)[0], device=x.device, requires_grad=True)[:, 1]
        #s = torch.tensor(ytree.query(x, nr_nns_searches=1)[0], device=x.device, requires_grad=True)
        r=xtree.query(x, nr_nns_searches=2)[0].clone().requires_grad_()[:,1]
        s=ytree.query(x, nr_nns_searches=1)[0].clone().requires_grad_()[:,0]
        r = torch.clamp(r, min=1e-8,max=1e8)
        s = torch.clamp(s, min=1e-8,max=1e8)
        # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
        # on the first term of the right hand side.
        m= torch.tensor(m, device=x.device)
        n= torch.tensor(n, device=x.device)
        # removed multiplication by d in formula so as to reduce scale
        return -torch.log(r/s).sum() / n + torch.log(m / (n - 1.))
