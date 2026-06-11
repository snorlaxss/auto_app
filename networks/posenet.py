import sys
import os
import torch
import torch.nn as nn
import contextlib

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
from networks.gf_algorithms.samplers import cond_ode_sampler
from networks.gf_algorithms.scorenet import PoseScoreNet

from networks.gf_algorithms.sde import init_sde
from configs.config import get_config



class GFObjectPose(nn.Module):
    dino_name = 'dinov2_vits14'
    dino_dim = 384
    embedding_dim = 60

    def __init__(self, cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T):
        super(GFObjectPose, self).__init__()
        
        self.cfg = cfg
        self.device = cfg.device
        self.is_testing = False
        
        ''' Load model, define SDE '''
        # init SDE config
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sampling_eps = sampling_eps
        self.T = T
        # self.T = 0.15
        # self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps = init_sde(cfg.sde_mode)
        
        ''' dino v2 '''
        if cfg.dino != 'none':
            path = '/home/snorlax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
            self.dino : nn.Module = torch.hub.load('/home/snorlax/.cache/torch/hub/facebookresearch_dinov2_main', GFObjectPose.dino_name, pretrained=False, source="local").to(cfg.device)
            self.dino.load_state_dict(torch.load(path, map_location=cfg.device))
            self.dino.requires_grad_(False)
            self.dino_dim = GFObjectPose.dino_dim
            self.embedding_dim = GFObjectPose.embedding_dim
        
        ''' encode pts '''
        if self.cfg.pts_encoder == 'pointnet2':
            if cfg.dino == 'pointwise':
                self.pts_encoder = Pointnet2ClsMSGFus(self.dino_dim)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        ''' score network'''
        if self.cfg.agent_type == 'score':
            self.pose_score_net = PoseScoreNet(
                self.marginal_prob_fn, 
                (0 if self.cfg.dino != 'global' else self.dino_dim + self.embedding_dim),
                self.cfg.pose_mode, 
                self.cfg.regression_head
            )


    def extract_pts_feature(self, data):
        """extract the input pointcloud feature
        Args:
            data (dict): batch example without pointcloud feature. {'pts': [bs, num_pts, 3], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        Returns:
            data (dict): batch example with pointcloud feature. {'pts': [bs, num_pts, 3], 'pts_feat': [bs, c], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        """
        pts = data['pts']
        if self.cfg.dino == 'pointwise':
            roi_rgb = data['roi_rgb']
            # 兼容 cfg.device 为字符串
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            use_cuda = device.type == 'cuda'
            autocast_ctx = torch.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=use_cuda
            ) if use_cuda else contextlib.nullcontext()
            with autocast_ctx:
                if use_cuda and roi_rgb.dtype == torch.float32:
                    roi_rgb = roi_rgb.to(torch.bfloat16)
                roi_rgb = roi_rgb.contiguous()
                feat = self.dino.get_intermediate_layers(roi_rgb)[0]
            xs = data['roi_xs'] // 14
            ys = data['roi_ys'] // 14
            pos = xs * 16 + ys
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
            rgb_feat = torch.gather(feat, 1, pos)
            rgb_feat.requires_grad_(False)
        if self.cfg.pts_encoder in ['pointnet2']:
            if self.cfg.dino == 'pointwise':
                pts_feat = self.pts_encoder(torch.concatenate([pts, rgb_feat], dim=-1))
            else:
                pts_feat = self.pts_encoder(pts)
        else:
            raise NotImplementedError
        return pts_feat
    
   
    def sample(self, data, sampler, atol=1e-5, rtol=1e-5, snr=0.16, denoise=True, init_x=None, T0=None):
            
        if sampler == 'ode':
            T0 = self.T if T0 is None else T0
            in_process_sample, res =  cond_ode_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                atol=atol,
                rtol=rtol,
                device=self.device,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.cfg.sampling_steps,
                pose_mode=self.cfg.pose_mode,
                denoise=denoise,
                init_x=init_x
            )
        
        else:
            raise NotImplementedError
        
        return in_process_sample, res ## in_process_sample doesn't increase computational cost, just memory overhead

    
    def forward(self, data, mode='score', init_x=None, T0=None):
        '''
        Args:
            data, dict {
                'pts': [bs, num_pts, 3]
                'pts_feat': [bs, c]
                'sampled_pose': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        if mode == 'score':
            out_score = self.pose_score_net(data) # normalisation
            return out_score
        elif mode == 'pts_feature':
            pts_feature = self.extract_pts_feature(data)
            return pts_feature
        elif mode == 'ode_sample':
            in_process_sample, res = self.sample(data, 'ode', init_x=init_x, T0=T0) ##
            return in_process_sample, res
        else:
            raise NotImplementedError



def test():
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    cfg = get_config()
    prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T = init_sde('ve')
    net = GFObjectPose(cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T)
    net_parameters_num= get_parameter_number(net)
    print(net_parameters_num['Total'], net_parameters_num['Trainable'])
if __name__ == '__main__':
    test()

