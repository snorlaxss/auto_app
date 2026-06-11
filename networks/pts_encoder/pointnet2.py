import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnet2_utils.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import networks.pts_encoder.pointnet2_utils.pointnet2.pytorch_utils as pt_utils
from ipdb import set_trace
from configs.config import get_config


cfg = get_config()


def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)

MSG_CFG = {
    'NPOINTS': [512, 256, 128, 64],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]], 
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'FP_MLPS': [[64, 64], [128, 128], [256, 256], [512, 512]],
    'CLS_FC': [128],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG = {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]], 
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]],
             [[512, 512], [512, 512]]],
    'DP_RATIO': 0.5,
}

ClsMSG_CFG_Dense = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[32, 64], [16, 32], [8, 16], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
}


########## Best before 29th April ###########
ClsMSG_CFG_Light = {
    'NPOINTS': [512, 256, 128, None],
    'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    'NSAMPLE': [[16, 32], [16, 32], [16, 32], [None, None]],
    'MLPS': [[[16, 16, 32], [32, 32, 64]],
             [[64, 64, 128], [64, 96, 128]],
             [[128, 196, 256], [128, 196, 256]], 
             [[256, 256, 512], [256, 384, 512]]],
    'DP_RATIO': 0.5,
    # 'NPOINTS': [128, 64, 32, None],
    # 'RADIUS': [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    # 'NSAMPLE': [[16, 32], [16, 32], [16, 32], [None, None]],
    # 'MLPS': [[[16, 16, 32], [32, 32, 64]],
    #          [[64, 64, 128], [64, 96, 128]],
    #          [[128, 196, 256], [128, 196, 256]], 
    #          [[256, 256, 512], [256, 384, 512]]],
    # 'DP_RATIO': 0.5,
}


ClsMSG_CFG_Lighter= {
    'NPOINTS': [512, 256, 128, 64, None],
    'RADIUS': [[0.01], [0.02], [0.04], [0.08], [None]],
    'NSAMPLE': [[64], [32], [16], [8], [None]],
    'MLPS': [[[32, 32, 64]],
             [[64, 64, 128]],
             [[128, 196, 256]],
             [[256, 256, 512]],
             [[512, 512, 1024]]],
    'DP_RATIO': 0.5,
}


if cfg.pointnet2_params == 'light':
    SELECTED_PARAMS = ClsMSG_CFG_Light
elif cfg.pointnet2_params == 'lighter':
    SELECTED_PARAMS = ClsMSG_CFG_Lighter
elif cfg.pointnet2_params == 'dense':
    SELECTED_PARAMS = ClsMSG_CFG_Dense
else:
    raise NotImplementedError


class Pointnet2ClsMSGFus(nn.Module):
    """
    Modified from Pointnet2ClsMSG: 
    concatenate input features with each layer, to enable full feature fusion
    """
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        for k in range(SELECTED_PARAMS['NPOINTS'].__len__()):
            mlps = SELECTED_PARAMS['MLPS'][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS['NPOINTS'][k],
                    radii=SELECTED_PARAMS['RADIUS'][k],
                    nsamples=SELECTED_PARAMS['NSAMPLE'][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            channel_in = channel_out + input_channels


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        # xyz:      bs * npoints * 3
        # features: bs * F * npoints

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i != 0:
                l_features[i] = torch.concatenate([l_features[i], features], dim=1) # concatenate
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], return_idx=True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            if idx != None:
                features = torch.gather(
                    features, 2, 
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(-1, features.shape[1], -1)
                ) # only keep features of remaining points
            else:
                assert i == len(self.SA_modules) - 1
        return l_features[-1].squeeze(-1)