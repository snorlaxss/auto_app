import math
import torch
from ipdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(
        model, 
        data,
        marginal_prob_func, 
        sde_fn, 
        eps=1e-5, 
        teacher_model=None,
        pts_feat_teacher=None
    ):
    pts = data['zero_mean_pts']
    gt_pose = data['zero_mean_gt_pose']
    
    ''' get std '''
    bs = pts.shape[0]
    random_t = torch.rand(bs, device=device) * (1. - eps) + eps         # [bs, ]
    random_t = random_t.unsqueeze(-1)                                   # [bs, 1]
    mu, std = marginal_prob_func(gt_pose, random_t)                     # [bs, pose_dim], [bs]
    std = std.view(-1, 1)                                               # [bs, 1]

    ''' perturb data and get estimated score '''
    z = torch.randn_like(gt_pose)                                       # [bs, pose_dim]
    perturbed_x = mu + z * std                                          # [bs, pose_dim]
    data['sampled_pose'] = perturbed_x
    data['t'] = random_t
    estimated_score = model(data) # PoseScoreNet.forward()              # [bs, pose_dim]

    ''' get target score '''
    if teacher_model is None:
        # theoretic estimation
        target_score = - z * std / (std ** 2)
    else:
        # distillation
        pts_feat_student = data['pts_feat'].clone()
        data['pts_feat'] = pts_feat_teacher
        target_score = teacher_model(data)
        data['pts_feat'] = pts_feat_student
        
    ''' loss weighting '''
    loss_weighting = std ** 2
    loss_ = torch.mean(torch.sum((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
    
    return loss_


