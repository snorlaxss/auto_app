import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import random
import gc
import cv2
import open3d as o3d

import numpy as np
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ipdb import set_trace
from sklearn.cluster import DBSCAN
from networks.posenet_agent import PoseNet

from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from utils.so3_visualize import visualize_so3
from cutoop.eval_utils import DetectMatch
from configs.config import get_config
from datasets.datasets_infer import InferDataset
from runners.yomni_inference_helper import GenPose2
import argparse

from flask import Flask
flask_app = Flask(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    return p.parse_args()
def visualize_pose(data:InferDataset, all_final_pose, all_final_length, visualize_pts=False, visualize_image=False):
    color_img = cv2.cvtColor(data.color, cv2.COLOR_RGB2BGR)
    all_final_pose = all_final_pose[0].cpu().numpy()
    all_final_length = all_final_length[0].cpu().numpy()

    for index, (obj_pose, obj_length) in enumerate(zip(all_final_pose, all_final_length)):
        # if index != 0:
            if visualize_pts:
                pts = data.get_objects()['pts'].cpu().numpy()[index]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.visualization.draw_geometries([pcd])
            color_img = DetectMatch._draw_image(
                vis_img=color_img,
                pred_affine=obj_pose,
                pred_size=obj_length,
                gt_affine=None,
                gt_size=None,
                gt_sym_label=None,
                camera_intrinsics=data.cam_intrinsics,
                draw_pred=True,
                draw_gt=False,
                draw_label=False,
                draw_pred_axes_length=0.1,
                draw_gt_axes_length=None,
                thickness=True,
            )
    
    if visualize_image:
        cv2.namedWindow('rgb')
        cv2.imshow('rgb', color_img)
        cv2.waitKey() 
        cv2.destroyAllWindows()
    return color_img


def main():
    ######################################## PARAMETERS ########################################
    # DATA_PATH = 'data/test2/all'                 # Path to the data
    DATA_PATH = "/home/kewei/repo/FoundationStereo/output"
    TRACKING = True                                           # Tracking mode
    

    # Tracking parameter, if the relative pose between the current frame and the previous frame
    # is large, such as low video FPS or fast object motion, you can set a larger value. The default
    # TRACKING_TO is set to 0.15.
    TRACKING_T0 = 0.2

    SCORE_MODEL_PATH='/home/kewei/repo/GenPose2/results/ckpts/scorenet.pth'     # Path to the score model
    # SCORE_MODEL_PATH= "results/ckpts/ScoreNet/scorenet.pth"

    # SCALE_MODEL_PATH='results/ckpts/ScaleNet/1024_data_128_pointnet_w_teacher/ckpt_epoch100.pth'     # Path to the scale model
    SCALE_MODEL_PATH='/home/kewei/repo/GenPose2/results/ckpts/scalenet.pth'     # Path to the scale model

    PREV_POSE = None                                           # Previous pose
    ######################################## PARAMETERS ########################################

    ''' load data '''
    # Get data from image file
    color_images = sorted(glob.glob(DATA_PATH + '/*_color.png'))
    args = parse_args()
    args.T0 = 50
    args.Tp = 50
    args.eval_repeat_num = 50
    args.retain_ratio = 1
    args.score = SCORE_MODEL_PATH
    args.scale = SCALE_MODEL_PATH
    args.frame_gap_threshold = 10
    genpose2 = GenPose2(args=args)
    
    cv2.namedWindow('rgb')
    for index, color_image in enumerate(tqdm(color_images)):
        data_prefix = color_image.replace('color.png', '')
        
        data = InferDataset.alternetive_init(
            data_prefix, 
            img_size=genpose2.cfg.img_size, 
            device=genpose2.cfg.device, 
            n_pts=genpose2.cfg.num_points)
        pose, length = genpose2.inference(data, PREV_POSE, TRACKING, TRACKING_T0)
        print(pose)
        print(length)
        color_image_w_pose = visualize_pose(data, pose, length, visualize_image=False)
        PREV_POSE = pose
        cv2.imshow('rgb', color_image_w_pose)
        cv2.waitKey(0) 

    cv2.destroyAllWindows()    


if __name__ == '__main__':
    main()


