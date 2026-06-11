import sys
import os
import functools
import torch
import numpy as np
from ipdb import set_trace
from scipy import integrate
from utils.genpose_utils import get_pose_dim

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=90):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=90):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=90, T=1.0):
    _, sigma_max_prior = ve_marginal_prob(None, T, sigma_min=sigma_min, sigma_max=sigma_max)
    return torch.randn(*shape) * sigma_max_prior

def init_sde(sde_mode):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pytorch
    if sde_mode == 've':  # 当SDE模式为've'（Variance Exploding，方差扩展SDE）时
        sigma_min = 0.01  # 设置噪声标准差的最小值为0.01
        sigma_max = 50    # 设置噪声标准差的最大值为50
        eps = 1e-5        # 设置数值稳定性的小常数，用于避免数值计算问题
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)  # 创建边际概率函数的偏函数，固定sigma_min和sigma_max参数
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)  # 创建VE SDE函数的偏函数，固定sigma_min和sigma_max参数
        T = 1.0           # 设置时间终点T为1.0，表示SDE过程的结束时间
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)  # 创建先验分布函数的偏函数，固定sigma_min和sigma_max参数
        
    else:
        raise NotImplementedError
    
    return prior_fn, marginal_prob_fn, sde_fn, eps, T

