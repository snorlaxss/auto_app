import os
# os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, TransformStamped
from tf2_msgs.msg import TFMessage
import std_msgs.msg
import struct

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
from utils.transforms.rotation_conversions import matrix_to_quaternion
import pyrealsense2 as rs

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
    p.add_argument('--Tp', type=float, default=0.15, help='Denoising sampling time Tp')
    p.add_argument('--eval_repeat_num', type=int, default=25, help='Number of inference repetitions per frame for evaluation')
    p.add_argument('--retain_ratio', type=float, default=1, help='Outlier removal retain ratio')
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

def realsense_pipeline(w, h, fps):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f'RealSense initialized (depth scale = {depth_scale} m/unit)')
    return pipeline, align, depth_scale

def wait_frames(pipeline):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        return frames
    except Exception as e:
        print(f'RealSense frame timeout or error: {e}. Retrying...')
        pipeline.stop()
        time.sleep(1)

def release_realsense(pipeline=None, align=None):
    try:
        if pipeline is not None:
            pipeline.stop()
            
    except Exception as e:
        print(f'Error stopping RealSense pipeline: {e}')
    finally:
        del pipeline
        del align
        time.sleep(0.5)

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

def process_frame(frame, depth_raw, genpose2, yolo, args, writer, frame_idx, depth_scale, node=None):
    # run YOLO
    results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    masks = results[0].masks

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
    class_names = None
    if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
        box_ids = [int(id_val) for id_val in boxes.id.cpu().numpy()]
        if hasattr(boxes, 'cls') and boxes.cls is not None:
            class_indices = boxes.cls.cpu().numpy().astype(int)
            class_names = [yolo.names[idx] for idx in class_indices]

    combined_mask, obj_ids = make_combined_mask(frame.shape[0], frame.shape[1], masks, box_ids)

    if obj_ids is None or (len(np.unique(combined_mask)) != len(obj_ids)):
        vis = frame.copy()
        num_detections = len(masks.data) if masks is not None else 0
        draw_frame_info(vis, frame_idx, None, num_detections)
        show_frame(vis, writer, args, frame_idx, save=False)
        return None, None, 0.0

    func = InferDataset.alternetive_init
    frame_data = {
        "color": frame,
        "depth": (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16) if depth_raw is not None else None,
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

    if pose is not None:
        np_pose = list(pose[0].numpy())
        print("poses", np_pose)
        print("box_ids:", box_ids)
        pose_list = []
        pose_box_ids = []
        pose_class_names = []
        
        for i in range(len(np_pose)):
            xyz = np_pose[i][:3, 3]        # (3,)
            rot_matrix = np_pose[i][:3, :3]         # (3,3)
            rot_fix = np.array([[-1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])
            rot_matrix = rot_matrix @ rot_fix
            quat_wxyz = matrix_to_quaternion(torch.from_numpy(rot_matrix))  # [w, x, y, z]
            quat_wxyz = quat_wxyz.cpu().numpy()
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]        # [x, y, z, w]
            pose_vec = np.concatenate([xyz, quat_xyzw]).astype(float)
            pose_list.append(pose_vec)
            
            # 获取对应的box_id和class_name (obj_ids从1开始，索引0是背景)
            # if obj_ids is not None and i+1 < len(obj_ids):
                # mask_label, box_id = obj_ids[i+1]

            # if i == 0:
            #     pose_class_names.append("unknown")   
            #     pose_box_ids.append(i)
            # else:
            #     box_id = box_ids[i-1]
            #     pose_box_ids.append(box_id)
            #     box_idx = box_ids.index(box_id)
            #     pose_class_names.append(class_names[box_idx])
            box_id = box_ids[i]
            pose_box_ids.append(box_id)
            box_idx = box_ids.index(box_id)
            pose_class_names.append(class_names[box_idx])

        
        # 发布TF消息
        publish_poses(node, pose_list, pose_box_ids, pose_class_names)

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

def create_pointcloud2(rgb, depth, depth_scale, frame_id='camera_link', intrinsics=None):
    h, w = depth.shape
    points = []
    
    for i in range(h):
        for j in range(w):
            z = depth[i, j] * depth_scale  # mm
            if z == 0:
                continue
            x = np.float32((j - w / 2) * z / intrinsics["fx"])
            y = np.float32((i - h / 2) * z / intrinsics["fy"])
            z = np.float32(z)
            r, g, b = rgb[i, j]
            rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            points.append([x, y, z, rgb_packed])
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    header = std_msgs.msg.Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = frame_id
    pc2 = PointCloud2()
    pc2.header = header
    pc2.height = 1
    pc2.width = len(points)
    pc2.fields = fields
    pc2.is_bigendian = False
    pc2.point_step = 16
    pc2.row_step = pc2.point_step * len(points)
    pc2.is_dense = True
    pc2.data = b''.join([struct.pack('fffI', *p) for p in points])

    return pc2

# def publish_pose(node, pose, length, frame_id="camera_link"):
#     msg = PoseStamped()
#     msg.header.stamp = node.get_clock().now().to_msg()
#     msg.header.frame_id = frame_id
#     msg.pose.position.x = pose[0]
#     msg.pose.position.y = pose[1]
#     msg.pose.position.z = pose[2]
#     msg.pose.orientation.x = pose[3]
#     msg.pose.orientation.y = pose[4]
#     msg.pose.orientation.z = pose[5]
#     msg.pose.orientation.w = pose[6]
#     node.pose_pub.publish(msg)

def publish_poses(node, poses, box_ids, class_names, frame_id="camera_link"):
    """
    使用TFMessage发布多个pose，每个transform包含box_id和类别名称
    poses: 可迭代对象，每项为长度7的向量[x,y,z,qx,qy,qz,qw]
    box_ids: 对应的跟踪ID列表
    class_names: 对应的YOLO类别名称列表
    """
    tf_msg = TFMessage()
    stamp = node.get_clock().now().to_msg()
    
    for i, (p, box_id, cls_name) in enumerate(zip(poses, box_ids, class_names)):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame_id
        # 使用格式: object_{class_name}_{box_id}
        transform.child_frame_id = f"{cls_name}_{box_id}"
        
        transform.transform.translation.x = float(p[0])
        transform.transform.translation.y = float(p[1])
        transform.transform.translation.z = float(p[2])
        transform.transform.rotation.x = float(p[3])
        transform.transform.rotation.y = float(p[4])
        transform.transform.rotation.z = float(p[5])
        transform.transform.rotation.w = float(p[6])
        
        tf_msg.transforms.append(transform)
    
    node.tf_pub.publish(tf_msg)

def main():
    args = parse_args()
    VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
    VIDEO_FPS = 30
    rclpy.init()
    node = rclpy.create_node('pose_node')
    node.pc_pub = node.create_publisher(PointCloud2, 'pointcloud', 10)
    node.tf_pub = node.create_publisher(TFMessage, '/tf', 10)
    # node.pose_array_pub = node.create_publisher(PoseArray, 'object_poses', 10)

    # RealSense setup (initialized later if requested)
    rs_pipeline = None
    rs_align = None
    depth_scale = None

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
    yolo = YOLO(args.yolo)

    rs_pipeline, rs_align, depth_scale = realsense_pipeline(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)

    writer = None
    frame_idx = 0
    frames = None

    try:
        while True:
            frames = wait_frames(rs_pipeline)
            if frames is None:
                rs_pipeline, rs_align, depth_scale = realsense_pipeline(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
                continue
            aligned = rs_align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            pose, length, infer_time = process_frame(frame, depth_raw, genpose2, yolo, args, writer, frame_idx, depth_scale, node=node)
            frame_idx += 1

    except KeyboardInterrupt:
        print('Interrupted by user')

    finally:
        if args.realsense and rs_pipeline is not None:
            try:
                release_realsense(rs_pipeline, rs_align)
                rs_pipeline.stop()
            except Exception:
                pass
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()