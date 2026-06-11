import os
import sys
import numpy as np
import torch
import random
import gc
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks.posenet_agent import PoseNet

from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch
from configs.config import get_config
from datasets.datasets_infer import InferDataset


class GenPose2:
    prev_pose_dict = None # public static variable to store previous poses for tracking
    FRAME_GAP_THRESHOLD = 20

    def __init__(self, args=None):
        ''' load config '''
        self.cfg = self._get_config(args)

        ''' set random seed '''
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        ''' load score model '''
        self.cfg.agent_type = 'score'
        self.score_agent = PoseNet(self.cfg)
        self.score_agent.load_ckpt(model_dir=self.cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        self.score_agent.eval()

        ''' load scale model '''
        self.cfg.agent_type = 'scale'
        self.scale_agent = PoseNet(self.cfg)
        self.scale_agent.load_ckpt(model_dir=self.cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
        self.scale_agent.eval()

        GenPose2.FRAME_GAP_THRESHOLD = args.frame_gap_threshold

    def _get_config(self, args=None):
        cfg = get_config()
        cfg.pretrained_score_model_path = args.score
        cfg.pretrained_scale_model_path = args.scale
        cfg.sampler_mode=['ode']
        cfg.T0 = args.T0
        cfg.Tp = args.Tp
        cfg.seed = 0
        cfg.eval_repeat_num = args.eval_repeat_num
        cfg.dino = 'pointwise'
        cfg.retain_ratio = args.retain_ratio
        return cfg

    def _validate_prev_poses(self, batch_sample, current_ids, frame_idx):
        tracking_poses = []
        valid_prev_label = []

        for i, box_id in enumerate(current_ids):
            # have previous pose
            if GenPose2.prev_pose_dict is not None and box_id in GenPose2.prev_pose_dict:
                # if within frame gap threshold
                if (frame_idx - GenPose2.prev_pose_dict[box_id]['last_seen_frame'] <= GenPose2.FRAME_GAP_THRESHOLD):
                    pose = GenPose2.prev_pose_dict[box_id]['pose'].clone().to(self.cfg.device)
                    # only add available previous poses to list
                    if i < len(batch_sample['pts_center']): 
                        pose[:, -3:] -= batch_sample['pts_center'][i:i+1]
                        tracking_poses.append(pose[-1])
                        valid_prev_label.append(box_id)
                    else:
                        tracking_poses.append(None)
                # treat as new pose
                else:
                    tracking_poses.append(None)
            else:
                tracking_poses.append(None)

        # remove stale
        if GenPose2.prev_pose_dict is not None:
            stale_ids = [box_id for box_id in GenPose2.prev_pose_dict 
                        if frame_idx - GenPose2.prev_pose_dict[box_id]['last_seen_frame'] > GenPose2.FRAME_GAP_THRESHOLD]
            for stale_id in stale_ids:
                del GenPose2.prev_pose_dict[stale_id]

        return tracking_poses, valid_prev_label

    def _inference_score(self, data:InferDataset, score_agent:PoseNet, T0=None, Tp=None, obj_ids=None, frame_idx=0):
        all_pred_pose = []
        all_score_feature = []

        batch_sample = data.get_objects() # where obj_idx = obj_idx[(obj_idx != 0) & (obj_idx != 255)]

        if batch_sample is None:
            print("Warning: InferDataset.get_objects() returned None — no valid objects to infer.")
            return [], []

        # current_ids = [mask_id for mask_id, box_id in obj_ids] # currentt_ids = box_ids in CURRENT frame

        # obj_ids = []
        # obj_ids.append([1,1])

        mapping_label_to_box = {mask_id: box_id for mask_id, box_id in obj_ids}
        if 'labels' in batch_sample:
            current_ids = [mapping_label_to_box.get(lbl, None) for lbl in batch_sample['labels']]

        tracking_poses, valid_prev_label = self._validate_prev_poses(batch_sample, current_ids, frame_idx)

        new_ids = [box_id for box_id in current_ids if box_id not in valid_prev_label]
        print(f"Tracking {len(valid_prev_label)} objects, {len(new_ids)} new objects.")

        # no pose / first frame
        if GenPose2.prev_pose_dict is None or all(p is None for p in tracking_poses):
            init_x = None
        else:
            box_to_mask = {v: k for k, v in mapping_label_to_box.items()}
            init_x = {}
            
            # Pair box_id with pose only when pose is NOT None
            for box_id, pose in zip(valid_prev_label, tracking_poses):
                if pose is not None:  # Only add non-None poses
                    mask_id = box_to_mask.get(box_id)
                    if mask_id is not None:
                        init_x[mask_id] = pose
            
            if not init_x:
                init_x = None

        valid_prev_label_mask = []
        if valid_prev_label:
            box_to_mask = {v: k for k, v in mapping_label_to_box.items()}
            valid_prev_label_mask = [box_to_mask.get(box_id) for box_id in valid_prev_label if box_to_mask.get(box_id) is not None]

        pred_results = score_agent.pred_func(
            data=batch_sample,
            repeat_num=self.cfg.eval_repeat_num,
            T0=self.cfg.T0 if T0 is None else T0,
            Tp=self.cfg.Tp if Tp is None else Tp,
            init_x=init_x,
            return_average_res=False,
            return_process=False,
            valid_prev_label=valid_prev_label_mask,
        )
        
        pred_pose, _ = pred_results
        all_pred_pose.append(pred_pose)

        # init after first frame
        if GenPose2.prev_pose_dict is None:
            GenPose2.prev_pose_dict = {}

        # idk
        for box_id, pose in zip(current_ids, pred_pose):    ## pred_pose.shape=(n,50,9)
            GenPose2.prev_pose_dict[box_id] = {'pose': pose, 'last_seen_frame': frame_idx}  # save frame


        all_score_feature.append({
            'pts_feat': batch_sample['pts_feat'].cpu(),
        })

        return all_pred_pose, all_score_feature


    def _aggregate_pose(self, all_pred_pose):
        all_aggregated_pose = []
        
        for i, pred_pose in enumerate(all_pred_pose):
            bs = pred_pose.shape[0]
            retain_num = int(self.cfg.eval_repeat_num * self.cfg.retain_ratio)
            # Simply take the first retain_num poses since we no longer have energy sorting
            good_pose = pred_pose[:, :retain_num, :]
            rot_matrix = get_rot_matrix(good_pose[:, :, :-3].reshape(bs * retain_num, -1), self.cfg.pose_mode)
            quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
            aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)
            if self.cfg.clustering:
                for j in range(bs):
                    # https://math.stackexchange.com/a/90098
                    # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
                    pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
                    dbscan = DBSCAN(eps=self.cfg.clustering_eps, min_samples=int(self.cfg.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().cpu().numpy())
                    labels = dbscan.labels_
                    if np.any(labels >= 0):
                        bins = np.bincount(labels[labels >= 0])
                        best_label = np.argmax(bins)
                        aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0))[0]
            aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
            aggregated_pose = torch.zeros(bs, 4, 4)
            aggregated_pose[:, 3, 3] = 1
            aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
            aggregated_pose[:, :3, 3] = aggregated_trans
            all_aggregated_pose.append(aggregated_pose)
            if i % 10 == 9:
                gc.collect()
        
        return all_aggregated_pose


    def _inference_scale(self, data:InferDataset, scale_agent:PoseNet, all_score_feature, all_aggregated_pose):
        if data.get_objects() is None:
                return [], []
        if self.cfg.pretrained_scale_model_path is None:
            all_final_length = []

            for i, test_batch in enumerate([data.get_objects()]):
                pcl: torch.Tensor = test_batch['pcl_in'] # [bs, 1024, 3]
                rotation: torch.Tensor = all_aggregated_pose[i][:, :3, :3] # [bs, 3, 3]
                rotation_t = torch.transpose(rotation, 1, 2) # [bs, 3, 3]
                translation: torch.Tensor = all_aggregated_pose[i][:, :3, 3] # [bs, 3]

                n_pts = pcl.shape[1]
                pcl = pcl - translation.unsqueeze(1) # [bs, 1024, 3]
                pcl = pcl.reshape(-1, 3, 1) # [bs * 1024, 3, 1]
                rotation_t = torch.repeat_interleave(rotation_t, n_pts, dim=0) # [bs * 1024, 3, 3]
                pcl = torch.bmm(rotation_t, pcl).reshape(-1, n_pts, 3) # [bs, 1024, 3]

                bbox_length, _ = torch.max(torch.abs(pcl), dim=1)
                bbox_length *= 2
                all_final_length.append(bbox_length.cpu())

                if i % 10 == 9:
                    gc.collect()

            return all_aggregated_pose, all_final_length
        
        all_final_pose = []
        all_final_length = []

        for i, batch_sample in enumerate([data.get_objects()]):
            batch_sample.update({key: (None if value is None else value.to(self.cfg.device)) 
                                for key, value in all_score_feature[i].items()})
            batch_sample['axes'] = all_aggregated_pose[i][:, :3, :3].to(self.cfg.device)
            cal_mat, length = scale_agent.pred_scale_func(batch_sample)
            final_pose = all_aggregated_pose[i].clone()
            final_pose[:, :3, :3] = cal_mat.cpu()
            all_final_pose.append(final_pose.cpu())
            all_final_length.append(length.cpu())
            if i % 4 == 3:
                gc.collect()
        
        return all_final_pose, all_final_length

    def inference(self, data:InferDataset, obj_ids=None, frame_idx=0, T0=None, Tp=None):
        
        all_pred_pose, all_score_feature = self._inference_score(data, self.score_agent, T0=T0, Tp=Tp, obj_ids=obj_ids, frame_idx=frame_idx)

        all_aggregated_pose = self._aggregate_pose(all_pred_pose)

        all_final_pose, all_final_length = self._inference_scale(data, self.scale_agent, all_score_feature, all_aggregated_pose)

        return all_final_pose, all_final_length