import os
import numpy as np
import cv2
import torch
import copy
import open3d as o3d

from cutoop.data_loader import Dataset, ImageMetaData
from utils.datasets_utils import aug_bbox_eval, get_2d_coord_np, crop_resize_by_warp_affine
from utils.sgpa_utils import get_bbox
from datasets.datasets_omni6dpose import Omni6DPoseDataSet
from cutoop.transform import pixel2xyz
from cutoop.image_meta import ViewInfo
from cutoop.data_types import CameraIntrinsicsBase

class InferDataset(object):
    def __init__(self, data: dict, img_size: int=224, device='cuda', n_pts=1024):
        """
        Args:
            data (dict): dictionary containing depth, color, mask, and meta data
                depth (np.ndarray): depth image
                color (np.ndarray): color image
                mask (np.ndarray): mask image
                meta (dict): camera intrinsics
            img_size (int): size of the image to be used for the network
            device (str): device to be used for the network
            n_pts (int): number of points to be used for the network
        """
        self._depth: np.ndarray = data['depth']
        self._color: np.ndarray = data['color']
        self._mask: np.ndarray = data['mask']
        # self.intrinsics= {
        #     "fx": 606.5540161132812,
        #     "fy": 606.3988647460938,
        #     "cx": 325.6007080078125,
        #     "cy": 252.87457275390625,
        #     "width": 640,
        #     "height": 480
        #     }
        # self.intrinsics= {
        #     "fx": 302.78513240759577,
        #     "fy": 302.8928223540444,
        #     "cx": 329.2054672046417,
        #     "cy": 193.75493228267774,
        #     "width": 640,
        #     "height": 480
        #     }
        # self.intrinsics= {
        #     "fx": 605.216, 
        #     "fy": 604.678,
        #     "cx": 320.218,
        #     "cy": 234.579,
        #     "width": 640,
        #     "height": 480,
        #     }

        self.intrinsics= {
            "fx": 606.965, 
            "fy": 606.133,
            "cx": 323.882,
            "cy": 257.623,
            "width": 640,
            "height": 480,
            }

        # if isinstance(data['meta'], dict):
            # camera_intrinsics = data['meta']['camera']['intrinsics']
        camera_intrinsics = self.intrinsics
        camera_intrinsics = CameraIntrinsicsBase(
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy'],
            width=camera_intrinsics['width'],
            height=camera_intrinsics['height']
        )
        camera = ViewInfo(None, None, camera_intrinsics, None, None, None, None, None)
        self._meta: ImageMetaData = ImageMetaData(None, camera, None, None, None, None, None, None, None, None)
        # else:
        #     self._meta: ImageMetaData = data['meta']

        self._img_size = img_size
        self._device = device
        self._n_pts = n_pts

    
    @classmethod
    def alternetive_init(cls, data, img_size: int=224, device='cuda', n_pts=1024):
        
        if isinstance(data, dict):
            # print(data.get("depth"))
            depth = Dataset.load_depth(data.get("depth"))/1000.
            color = Dataset.load_color(data.get("color"))
            mask = Dataset.load_mask(data.get("mask"))
        else:
            prefix = data if data.endswith(os.sep) else data + os.sep
            prefix = data.rstrip("/\\")
            depth_file = prefix + "depth.exr"
            color_file = prefix + "color.png"
            mask_file  = prefix + "mask.png"
            depth = Dataset.load_depth(depth_file)
            color = Dataset.load_color(color_file)
            mask = Dataset.load_mask(mask_file)
            # # magic
            # mask = mask-1
            # mask = 255 - mask
            # mask[mask == 255] = 0
            
            
        if depth is None:
            print("Warning: depth is None")
        if color is None:
            print("Warning: color is None")
        if mask is None:
            print("Warning: mask is None")
        print(f'mask shape: {mask.shape}, unique values: {np.unique(mask)[:10]}')  # 只显示前10个
        return cls({'depth': depth, 'color': color, 'mask': mask}, img_size=img_size, device=device, n_pts=n_pts)
    
    def get_per_object(self, obj_idx): ##
        object_mask = np.equal(self._mask, obj_idx)
        if not object_mask.any():
            assert False, f"Object {obj_idx} not found in mask"
        max_depth = 3 ###!!!
        self._depth[self._depth > max_depth] = 0
        if not (self._mask.shape[:2] == self._depth.shape[:2] == self._color.shape[:2]):
            assert False, "depth, mask, and rgb should have the same shape"
        intrinsics = self._meta.camera.intrinsics
        intrinsic_matrix = np.array([
            [intrinsics.fx, 0,             intrinsics.cx], 
            [0,             intrinsics.fy, intrinsics.cy], 
            [0,             0,             1]
            ], dtype=np.float32)
        
        img_width, img_height = self._color.shape[1], self._color.shape[0]
        scale_x = img_width / intrinsics.width
        scale_y = img_height / intrinsics.height
        intrinsic_matrix[0] *= scale_x
        intrinsic_matrix[1] *= scale_y

        coord_2d = get_2d_coord_np(img_width, img_height).transpose(1, 2, 0)

        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], img_height, img_width)

        # here resize and crop to a fixed size 224 x 224
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)

        # crop and resize
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        roi_rgb_ = crop_resize_by_warp_affine(
            self._color, bbox_center, scale, self._img_size, interpolation=cv2.INTER_LINEAR
        )
        roi_rgb = Omni6DPoseDataSet.rgb_transform(roi_rgb_)

        mask_target = self._mask.copy().astype(np.float32)
        mask_target[self._mask != obj_idx] = 0.0
        mask_target[self._mask == obj_idx] = 1.0

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            self._depth, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_depth = np.expand_dims(roi_depth, axis=0)

        valid = (np.squeeze(roi_depth, axis=0) > 0) * (np.squeeze(roi_mask, axis=0) > 0)
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        pcl_in = Omni6DPoseDataSet.depth_to_pcl(roi_depth, intrinsic_matrix, roi_coord_2d, valid)
        # print(pcl_in)
        # quit()

        if len(pcl_in) < 10:
            # assert False, f"Not enough points for pose estimation. {len(pcl_in)} points found"
            print(f"Warning: Not enough points for pose estimation. {len(pcl_in)} points found")
            return None
        ids, pcl_in = Omni6DPoseDataSet.sample_points(pcl_in, self._n_pts)
        xs, ys = xs[ids], ys[ids]

        data = {}
        data['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
        data['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
        data['roi_xs'] = torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous()
        data['roi_ys'] = torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous()
        data['roi_center_dir'] = torch.as_tensor(pixel2xyz(img_height, img_height, bbox_center, intrinsics), dtype=torch.float32).contiguous()

        return data
    

    def get_objects(self): ##
        obj_idx = np.unique(self._mask)
        obj_idx = obj_idx[(obj_idx != 0) & (obj_idx != 255)]  # remove background and unknown/invalid labels
        objects = {}
        labels = []

        for idx in obj_idx:
            obj = self.get_per_object(idx)
            if obj is None:
                continue
            
            labels.append(int(idx))

            for key, value in obj.items():
                if key not in objects:
                    objects[key] = []
                objects[key].append(value)

        for key, value in objects.items():
            objects[key] = torch.stack(value, dim=0)
            if 'pcl_in' not in objects:
                raise ValueError(f"No valid objects found / no pcl_in produced. Mask unique values: {np.unique(self._mask)}")
            
        try:
            PC_da = objects['pcl_in'].to(self._device)
        except:
            print(f"Warning: No valid pcl_in found for any object. Mask unique values: {np.unique(self._mask)}")
            return None
        
        data = {}
        data['labels'] = labels                     # list of object ids
        data['pts'] = PC_da                         # [bs, 1024, 3]
        data['pts_color'] = PC_da                   # [bs, 1024, 3]
        data['roi_rgb'] = objects['roi_rgb'].to(self._device)   # [bs, 3, imgsize, imgsize]
        assert data['roi_rgb'].shape[-1] == data['roi_rgb'].shape[-2]
        assert data['roi_rgb'].shape[-1] % 14 == 0

        data['roi_xs'] = objects['roi_xs'].to(self._device)     # [bs, 1024]
        data['roi_ys'] = objects['roi_ys'].to(self._device)     # [bs, 1024]
        data['roi_center_dir'] = objects['roi_center_dir'].to(self._device)     # [bs, 3]

        """ zero center """
        num_pts = data['pts'].shape[1]
        zero_mean = torch.mean(data['pts'][:, :, :3], dim=1)
        # data['zero_mean_pts'] = copy.deepcopy(data['pts'])
        data['zero_mean_pts'] = data['pts'].clone()
        data['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
        data['pts_center'] = zero_mean

        return data
    

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, depth):
        self._depth = depth

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def cam_intrinsics(self):
        return self._meta.camera.intrinsics
    
    @cam_intrinsics.setter
    def cam_intrinsics(self, intrinsics):
        self._meta.camera.intrinsics = intrinsics