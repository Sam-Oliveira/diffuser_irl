import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import gpytorch

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
    
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
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
        #print(total.shape)
        covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
        return covar_module(total).to_dense()

    def forward(self, source, target,kernel='gaussian'):
        batch_size = int(source.size()[0])
        if kernel=='gaussian':
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY -YX)
            return loss
        elif kernel=='matern':
            #kernels=self.matern_kernel(source,target,0.5)
            #XX = kernels[:batch_size, :batch_size]
            #YY = kernels[batch_size:, batch_size:]
            #XY = kernels[:batch_size, batch_size:]
            #YX = kernels[batch_size:, :batch_size]
            #loss = torch.mean(XX + YY - XY -YX)
            #return loss
            K=self.matern_kernel(source,target,0.5)
            N=source.shape[0]
            M=target.shape[0]
            Kxx = K[:N,:N]
            Kyy = K[N:,N:]
            Kxy = K[:N,N:]
            t1 = (1./(M*(M-1)))*torch.sum(Kxx - torch.diag(torch.diagonal(Kxx)))
            t2 = (2./(M*N)) * torch.sum(Kxy)
            t3 = (1./(N*(N-1)))* torch.sum(Kyy - torch.diag(torch.diagonal(Kyy)))
            #print(t1)
            #print(t2)
            #print(t3)
            MMDsquared = (t1-t2+t3)
            return MMDsquared
    

def K_ID(X,Y,gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    pdist = nn.PairwiseDistance(p=2)
    n_obs = X.shape[1]
    XY = torch.vstack((X,Y))
    dist_mat = (1/torch.sqrt(torch.tensor(n_obs, dtype=torch.int8)))*torch.cdist(XY,XY,p=2)
    if gamma == -1:
        gamma = torch.median(dist_mat[dist_mat > 0])
   
    K = torch.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K

def FPCA(X,n_comp = 0.95):
    """
    Computes principal components of given data up to a specified explained variance level
    
    Parameters:
    X - (n_samples,n_obs) array of function values
    n_comp - number of principal components to compute. If in (0,1) then it is the explained variance level
    
    Returns:
    Normalised eigenvalues and eigenfunctions
    """
    n_points = X.shape[1]
    pca = PCA(n_components = n_comp)
    pca.fit(X)
    return (1/n_points)*pca.explained_variance_,pca.components_

def K_COV(X,Y,gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the COV kernel
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - dummy variable noot used in function, is an input for ease of compatability with other kernels
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """    
    n_obs = X.shape[1]
    XY = torch.vstack((X,Y))
    return ((1/n_obs)*torch.dot(XY,XY.T))**2


def K_FPCA(X,Y,gamma = 1,n_comp = 0.95):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the FPCA decomposition operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    n_comp - number of principal components to compute. If in (0,1) then it is the explained variance level
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = torch.vstack((X,Y))
    e_vals,e_funcs = FPCA(XY,n_comp = n_comp)
    scaled_e_funcs = e_funcs*torch.sqrt(torch.tensor(e_vals[:,torch.newaxis], dtype=torch.int8))
    XY_e = (1/n_obs)*torch.dot(XY,scaled_e_funcs.T)
    dist_mat = pairwise_distances(XY_e,metric='euclidean')
    if gamma == -1:
        gamma = torch.median(dist_mat[dist_mat > 0])
    K = torch.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K

def K_SQR(X,Y,gamma = 1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the map which sends x -> (x,x^{2}) in the Cartesian product of L^{2} with itself.
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel to be used on the two norms, if -1 then median heuristic 
            is used to pick a different gamma for each norm, if gamma = 0 then median heuristic
            is used to pick a single gamma for each norm.
            
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = torch.vstack((X,Y))
    dist_mat_1 = (1/torch.sqrt(torch.tensor(n_obs, dtype=torch.int8)))**torch.cdist(XY,XY,p=2)
    dist_mat_2 = (1/torch.sqrt(torch.tensor(n_obs, dtype=torch.int8)))**torch.cdist(XY**2,XY**2,p=2)
    dist_mat = dist_mat_1 + dist_mat_2
    if gamma == 0:
        gamma = torch.median(dist_mat[dist_mat > 0])
        K = torch.exp(-0.5*(1/gamma**2)*dist_mat**2)
        return K
    if gamma == -1:
        gamma_1 = torch.median(dist_mat_1[dist_mat_1 > 0])
        gamma_2 = torch.median(dist_mat_2[dist_mat_2 > 0])
        K = torch.exp(-0.5*((1/gamma_1**2)*dist_mat_1**2 + (1/gamma_2**2)*dist_mat_2**2))
        return K
    K = torch.exp(-0.5*((1/gamma**2)*(dist_mat**2)))
    return K


def MMD_K(K,M,N):
    """
    Calculates the empirical MMD^{2} given a kernel matrix computed from the samples and the sample sizes of each distribution.
    
    Parameters:
    K - kernel matrix of all pairwise kernel values of the two distributions
    M - number of samples from first distribution
    N - number of samples from first distribution
    
    Returns:
    MMDsquared - empirical estimate of MMD^{2}
    """
    
    Kxx = K[:N,:N]
    Kyy = K[N:,N:]
    Kxy = K[:N,N:]
    
    t1 = (1./(M*(M-1)))*torch.sum(Kxx - torch.diag(torch.diagonal(Kxx)))
    t2 = (2./(M*N)) * torch.sum(Kxy)
    t3 = (1./(N*(N-1)))* torch.sum(Kyy - torch.diag(torch.diagonal(Kyy)))
    
    MMDsquared = (t1-t2+t3)
    
    return MMDsquared

def calculate_mmd(X,Y,n_perms=5,z_alpha = 0.05,make_K = K_ID,return_p = False):
    """
    Performs the two sample test and returns an accept or reject statement
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    return_p - option to return the p-value of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    
    Returns:
    rej - 1 if null rejected, 0 if null accepted
    p-value - p_value of test
    
    """
    
    # Number of samples of each distribution is identified and kernel matrix formed
    M = X.shape[0]
    N = Y.shape[0]

    # can add GAMMA AS ARGUMENT!
    K = make_K(X,Y)
    
    # Empirical MMD^{2} calculated
    MMD_test = MMD_K(K,M,N)

    # Added this cause i just want this to be returned (empirical MMD)
    return MMD_test

    # For n_perms repeats the kernel matrix is shuffled and empirical MMD^{2} recomputed
    # to simulate the null
    shuffled_tests = np.zeros(n_perms)
    for i in range(n_perms):
            idx = np.random.permutation(M+N)
            K = K[idx, idx[:, None]]
            shuffled_tests[i] = MMD_K(K,M,N)
    
    # Threshold of the null calculated and test is rejected if empirical MMD^{2} of the data
    # is larger than the threshold
    q = np.quantile(shuffled_tests, 1.0-z_alpha)
    rej = int(MMD_test > q)
    
    if return_p:
        p_value = 1-(percentileofscore(shuffled_tests,MMD_test)/100)
        return rej, p_value
    else:
        return rej