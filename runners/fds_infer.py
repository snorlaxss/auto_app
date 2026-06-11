import os
# os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')
from ultralytics import YOLOE
from omegaconf import OmegaConf

# allow importing local modules
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Temporarily prevent imported modules from parsing CLI args (some project modules call argparse at import)
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from runners.yomni_inference_helper import GenPose2
sys.argv = _orig_argv
from datasets.datasets_infer import InferDataset
from cutoop.eval_utils import DetectMatch
from FoundationStereo.scripts.run import undistort_frames, load_calibration_params
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo', required=True, help='Path to YOLO weights .pt')
    p.add_argument('--score', required=True, help='Path to score model checkpoint')
    p.add_argument('--scale', default=None, help='Path to scale model checkpoint (optional)')
    p.add_argument('--realsense', required=True, action='store_true', help='Use Intel RealSense camera (aligned color+depth)')
    p.add_argument('--device', default='cuda', help='torch device')
    p.add_argument('--img-size', type=int, default=224, help='InferDataset image size')
    p.add_argument('--n-pts', type=int, default=1024, help='Number of points for point cloud sampling')
    p.add_argument('--show', action='store_true', help='Show window')
    p.add_argument('--frame_gap_threshold', type=int, default=20, help='Max frame gap to retain previous pose for tracking')
    p.add_argument('--T0', type=float, default=0.55, help='Initial sampling time T0')
    p.add_argument('--Tp', type=float, default=0.55, help='Denoising sampling time Tp')
    p.add_argument('--eval_repeat_num', type=int, default=25, help='Number of inference repetitions per frame for evaluation')
    p.add_argument('--retain_ratio', type=float, default=1, help='Outlier removal retain ratio')

    p.add_argument('--intrinsic_file', default='FoundationStereo/assets/K_self2.txt', type=str, help='camera intrinsic matrix and baseline file')
    p.add_argument('--fds', required=True, help='Path to FDS model checkpoint')
    p.add_argument('--depth_scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    p.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    p.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    p.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    p.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    p.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    p.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    p.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    p.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    return p.parse_args()

def make_combined_mask(h, w, masks, box_ids):
    obj_ids = []
    
    if masks is None or box_ids is None:
        return None, None

    obj_ids.append([0, 0])
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # len(mask.data) = number of detected objects
    for i in range(len(masks.data)):
        mask = masks.data[i].cpu().numpy()
        obj_ids.append([i+1, box_ids[i]]) # (mask label, box id)
        gray_value = i+1
        combined_mask[mask.squeeze() > 0] = gray_value
    return combined_mask, obj_ids

def draw_frame_info(vis, frame_idx, inference_time, num_detections):
    if inference_time is None:
        info = f'Frame {frame_idx}  Inference: SKIPPED (mask mismatch)  Detections: {num_detections}'
    else:
        info = f'Frame {frame_idx}  Inference: {inference_time:.2f}s  Detections: {num_detections}'
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis

def show_frame(vis, writer, args, frame_ids, save=False):
    if args.show:
        cv2.imshow('genpose2', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt()
    if writer is not None:
        writer.write(vis)
    return True

def process_frame(frame, fds, genpose2, yolo, args, writer, frame_idx, depth_scale, K, baseline):

    left_frame = frame[:, :640]
    right_frame = frame[:, 640:]

    left_undistorted, right_undistorted = undistort_frames(left_frame, right_frame)

    # run YOLO on original size
    results = yolo.track(left_undistorted, persist=True, tracker="bytetrack.yaml", verbose=False,conf= 0.8)
    boxes = results[0].boxes
    masks = results[0].masks
    # print(results[0].class_name)
    # quit()

    # Now resize for FDS processing
    img0 = cv2.resize(left_undistorted, fx=depth_scale, fy=depth_scale, dsize=None)
    img1 = cv2.resize(right_undistorted, fx=depth_scale, fy=depth_scale, dsize=None)
    H,W = img0.shape[:2]

    frame = img0.copy()

    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    t0 = time.time()
    with torch.cuda.amp.autocast(True):
      disp = fds.forward(img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True)
    t1 = time.time()
    print(f"FDS inference time: {t1 - t0:.3f} seconds")

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)

    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp
      invalid = us_right<0
      disp[invalid] = np.inf
    
    depth = K[0,0]*baseline/disp

    # show YOLO detections
    annotated_frame = results[0].plot()
    cv2.imshow('genpose', annotated_frame)

    # check for no detections
    if masks is None or len(masks.data) == 0:
        vis = frame.copy()
        num_detections = 0
        draw_frame_info(vis, frame_idx, 0.00, num_detections)
        show_frame(vis, writer, args, frame_idx, save=False)
        return None, None, 0.0

    # Extract track IDs for pose persistence
    box_ids = None
    if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
        box_ids = [int(id_val) for id_val in boxes.id.cpu().numpy()]

    # Use original undistorted frame dimensions for mask
    combined_mask, obj_ids = make_combined_mask(left_undistorted.shape[0], left_undistorted.shape[1], masks, box_ids)

    # Resize mask to match downscaled frame
    if combined_mask is not None:
        combined_mask = cv2.resize(combined_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    if obj_ids is None or (len(np.unique(combined_mask)) != len(obj_ids)):
        vis = frame.copy()
        num_detections = len(masks.data) if masks is not None else 0
        print('Mask and box ID mismatch, skipping frame')
        print('obj_ids:', obj_ids)
        draw_frame_info(vis, frame_idx, None, num_detections)
        show_frame(vis, writer, args, frame_idx, save=False)
        return None, None, 0.0

    func = InferDataset.alternetive_init
    frame_data = {
        "color": frame,
        "depth": (depth.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16) if depth is not None else None,
        "mask": combined_mask
    }

    try:
        data = func(frame_data, img_size=args.img_size, device=torch.device(args.device), n_pts=args.n_pts)
    except Exception:
        return None, None, 0.0

    obj_ids = list(filter(lambda row: row != [0,0] and row != [255,255], obj_ids))
    if len(obj_ids) == 0 or data is None:
        return None, None, 0.0

    t0 = time.time()
    if GenPose2.prev_pose_dict is None:
        pose, length = genpose2.inference(data, obj_ids=obj_ids, T0=args.T0, frame_idx=frame_idx)
    else:
        pose, length = genpose2.inference(data, obj_ids=obj_ids, T0=args.T0, Tp=args.Tp, frame_idx=frame_idx)
    t1 = time.time()
    # t3 = time.time()
    # print(f"All inference time: {t3 - t0:.3f} seconds")

    vis = frame.copy()
    valid_output = (
        isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple))
        and len(pose) > 0 and len(length) > 0
        and pose[0] is not None and length[0] is not None
    )

    if valid_output:
        all_final_pose = pose[0].to(torch.float32).cpu().numpy()
        all_final_length = length[0].to(torch.float32).cpu().numpy()
        for index, (obj_pose, obj_length) in enumerate(zip(all_final_pose, all_final_length)):
            vis = DetectMatch._draw_image(
                vis_img=vis,
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

    num_detections = len(masks.data) if masks is not None else 0
    draw_frame_info(vis, frame_idx, t1-t0, num_detections)
    show_frame(vis, writer, args, frame_idx, save=False)
    return pose, length, t1 - t0

def main():
    args = parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)

    from FoundationStereo.Utils import set_seed
    set_seed(0)

    torch.autograd.set_grad_enabled(False)
    ckpt_dir = args.fds
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg_smol.yaml')
    # cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg_big.yaml')

    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]

    args = OmegaConf.create(cfg)
    fds = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    fds.load_state_dict(ckpt['model'])
    fds.cuda()
    fds.eval()

    code_dir = os.path.dirname(os.path.realpath(__file__))

    depth_scale = args.depth_scale
    assert depth_scale<=1, "scale must be <=1"

    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    K[:2] *= depth_scale

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'

    print('Loading GenPose2 models...')
    _orig_argv_create = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    genpose2 = GenPose2(args=args)
    sys.argv = _orig_argv_create
    try:
        genpose2.cfg.device = device
    except Exception:
        print(Exception)

    print('Loading YOLO...')
    yoloe = YOLOE("yoloe-11l-seg-pf.pt")
    # yolo = YOLO(args.yolo)

    writer = None
    frame_idx = 0
    frames = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            pose, length, infer_time = process_frame(frame, fds, genpose2, yoloe, args, writer, frame_idx, depth_scale, K, baseline)
            frame_idx += 1

    except KeyboardInterrupt:
        print('Interrupted by user')

    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()