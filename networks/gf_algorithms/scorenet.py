import sys
import os
import torch
import numpy as np
import torch.nn as nn

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        # print(self.W)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PoseScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, dino_dim, pose_mode='quat_wxyz', regression_head='RT'):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
            regression_head (str, optional): _description_. Defaults to 'RT'.

        Raises:
            NotImplementedError: _description_
        """
        super(PoseScoreNet, self).__init__()
        self.regression_head = regression_head
        self.dino_dim = dino_dim
        self.act = nn.ReLU(True)
        pose_dim = get_pose_dim(pose_mode)
        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )

        ''' fusion tail '''
        if self.regression_head == 'Rx_Ry_and_T':  ## default setting
            if pose_mode != 'rot_matrix':
                raise NotImplementedError
            else:
                ''' rotation_x_axis regress head '''
                self.fusion_tail_rot_x = nn.Sequential(
                    nn.Linear(128+256+1024+dino_dim, 256),
                    self.act,  ## self.act = nn.ReLU(True)
                    zero_module(nn.Linear(256, 3)),
                )
                self.fusion_tail_rot_y = nn.Sequential(
                    nn.Linear(128+256+1024+dino_dim, 256),
                    self.act,
                    zero_module(nn.Linear(256, 3)),
                )
                
                ''' tranalation regress head '''
                self.fusion_tail_trans = nn.Sequential(
                    nn.Linear(128+256+1024+dino_dim, 256),
                    self.act,
                    zero_module(nn.Linear(256, 3)),
                ) 

                # print('rot_x layers:')
                # for name, param in self.fusion_tail_rot_x.named_parameters():
                #     print(f"Layer: {name}, Weight/Bias shape: {param.shape}")
                #     print(param)
                # print('rot_y layers:')
                # for name, param in self.fusion_tail_rot_y.named_parameters():
                #     print(f"Layer: {name}, Weight/Bias shape: {param.shape}")
                #     print(param)
                
                # quit()           
        else:
            raise NotImplementedError
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'rgb_feat': [bs, dino_dim] (optional)
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        pts_feat = data['pts_feat']  # 提取点云特征 [bs, 1024]
        if self.dino_dim:
            rgb_feat = data['rgb_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        
        # pts = pts.permute(0, 2, 1) # -> (bs, 3, 1024)
        # pts_feat = self.pts_encoder(pts) 
        
        t_feat = self.t_encoder(t.squeeze(1))        # 编码时间步特征 [bs, 128]
        pose_feat = self.pose_encoder(sampled_pose)  # 编码姿态特征 [bs, 256]

        if self.dino_dim:
            total_feat = torch.cat([pts_feat, t_feat, pose_feat, rgb_feat], dim=-1)
        else:
            total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=-1)

        _, std = self.marginal_prob_func(total_feat, t)
        
        if self.regression_head == 'Rx_Ry_and_T':  ## default setting
            rot_x = self.fusion_tail_rot_x(total_feat)   # 通过X轴旋转回归头输出X轴旋转向量 [bs, 3]
            rot_y = self.fusion_tail_rot_y(total_feat)   # 通过Y轴旋转回归头输出Y轴旋转向量 [bs, 3]
            trans = self.fusion_tail_trans(total_feat)   # 通过平移回归头输出平移向量 [bs, 3]
            out_score = torch.cat([rot_x, rot_y, trans], dim=-1) / (std+1e-7) # 拼接所有输出并归一化 [bs, 9]

            # print('rot_x layers:')
            # for name, param in self.fusion_tail_rot_x.named_parameters():
            #     print(f"Layer: {name}, Weight/Bias shape: {param.shape}")
            #     print(param)
            # print('rot_y layers:')
            # for name, param in self.fusion_tail_rot_y.named_parameters():
            #     print(f"Layer: {name}, Weight/Bias shape: {param.shape}")
            #     print(param)
        else:
            raise NotImplementedError
        return out_score