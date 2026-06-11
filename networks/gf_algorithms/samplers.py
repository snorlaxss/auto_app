import sys
import os
import torch
import numpy as np

from scipy import integrate
from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from utils.misc import normalize_rotation

def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz', 
        denoise=True,
        init_x=None,
        bs=None
    ):
    pose_dim = get_pose_dim(pose_mode)
    if bs is None:
        batch_size=data['pts'].shape[0] # bs * repeated_num
    else:
        batch_size = bs
    
    
    if init_x is None:
        init_x = prior((batch_size, pose_dim), T=T).to(device) # return torch.randn(*shape) * std (0.01*(50/0.01)**T)
        # std = 50           when T = 1.0
        # std = 1.082514     when T = 0.55
        # std = 0.035879     when T = 0.15

        # init_x = std * torch.randn(batch_size, pose_dim)
    else:
        init_x = init_x.to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t

        drift, diffusion = sde_coeff(torch.tensor(t)) 
        # drift = 0, diffusion = std(t) * sqrt(2*(log(sigma_max)-log(sigma_min)))
        # diffusion = sde_coeff(0.55) = 1.4959534928

        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        tmp = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # print(tmp.max(), tmp.min(), tmp.mean(), t)
        return tmp
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    # print(T, eps, num_steps)
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), 
                              rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    # xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    xs[:, :, -3:] = data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:]*=0 #### TESTING
    x[:, -3:] += data['pts_center']
    # print(x[0, -3:], x.shape)
    return xs.permute(1, 0, 2), x