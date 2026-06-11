import os
import sys
import gradio as gr
import numpy as np
import cv2
import torch
import time
import warnings

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.datasets_infer import InferDataset
from runners.yomni_inference_helper import GenPose2
from FoundationStereo.core.utils.utils import InputPadder
from fds_frontend_utils.scene_graph import get_relation

# Import utility modules
from fds_frontend_utils.visualization_utils import (
    visualize_pose, draw_detections, draw_filtered_detections,
    visualize_depth_map, draw_scene_graph
)
from fds_frontend_utils.detection_utils import (
    run_yolo_pipeline, make_combined_mask, filter_detections,
    compute_mask_centroids, build_object_metadata
)
from fds_frontend_utils.camera_utils import (
    capture_stereo_image, compute_depth_from_disparity,
    capture_realsense_image
)
from fds_frontend_utils.model_utils import create_default_config, setup_all_models
from utils.transforms.rotation_conversions import matrix_to_quaternion

import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

def publish_poses(node, poses, box_ids, class_names, frame_id="camera_link"):
    """
    Publish multiple poses using TFMessage, each transform includes box_id and class name
    poses: iterable of 7-element vectors [x,y,z,qx,qy,qz,qw]
    box_ids: corresponding tracking IDs
    class_names: corresponding YOLO class names
    """
    tf_msg = TFMessage()
    stamp = node.get_clock().now().to_msg()
    
    for i, (p, box_id, cls_name) in enumerate(zip(poses, box_ids, class_names)):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame_id
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

def process_stereo_image_and_cache(left_img, right_img, yolo):
    if left_img is None:
        return None, "Please capture an image first.", None
    
    raw_detections = run_yolo_pipeline(None, yolo, left_undistorted=left_img, right_undistorted=right_img)
    det = raw_detections[0]
    
    vis_image = draw_detections(left_img, det['boxes'], det['class_name'], det['id'])
    status_msg = f"{det['class_name']}"
    
    return raw_detections, status_msg, cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

def process_realsense_image_and_cache(input_image, yolo):
    if input_image is None:
        return None, "Please capture an image first.", None
    
    raw_detections = run_yolo_pipeline(input_image, yolo)
    det = raw_detections[0]

    vis_image = draw_detections(input_image, det['boxes'], det['class_name'], det['id'])
    status_msg = f"{det['class_name']}"

    return raw_detections, status_msg, cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

def update_view(left_image, rs_depth, raw_detections, keywords, fds, genpose2, args, K, baseline, depth_scale, frame_idx, node):
    if left_image is None or raw_detections is None:
        return left_image, None, None, None, None
    
    det = raw_detections[0]
    target_words = [k.strip().lower() for k in keywords.split(',') if k.strip()] if keywords.strip() else []
    
    # Get filtered data
    filter_result = filter_detections(det, target_words)
    if len(filter_result) == 3:
        filtered_masks, filtered_box_ids, filtered_indices = filter_result
    else:
        filtered_masks, filtered_box_ids = filter_result
        filtered_indices = list(range(len(det['class_name']))) if not target_words else []
    
    if rs_depth is None: # stereo
        left_undistorted, right_undistorted = det['left_undistorted'], det['right_undistorted']
        vis_image = draw_filtered_detections(left_undistorted, det, filtered_indices, target_words)

        if filtered_masks is None or len(filtered_masks.data) == 0:
            return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

        # Resize and process for FDS
        img0 = cv2.resize(left_undistorted, fx=depth_scale, fy=depth_scale, dsize=None)
        img1 = cv2.resize(right_undistorted, fx=depth_scale, fy=depth_scale, dsize=None)
        H, W = img0.shape[:2]

        # FDS depth estimation
        img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
        img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

        t0 = time.time()
        with torch.cuda.amp.autocast(True):
            disp = fds.forward(img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True)
        print(f"FDS inference time: {time.time() - t0:.3f} seconds")

        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H,W)

        depth = compute_depth_from_disparity(disp, K, baseline, remove_invalid=args.remove_invisible)
        
        combined_mask, obj_ids = make_combined_mask(left_undistorted.shape[0], left_undistorted.shape[1], filtered_masks, filtered_box_ids)

        if combined_mask is not None:
            combined_mask = cv2.resize(combined_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            centers = compute_mask_centroids(combined_mask)

        if obj_ids is None or len(np.unique(combined_mask)) != len(obj_ids):
            print('Mask and box ID mismatch, skipping frame')
            return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

        frame_data = {
            "color": img0.copy(),
            "depth": (depth.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16) if depth is not None else None,
            "mask": combined_mask
        }

    else: # rs
        vis_image = draw_filtered_detections(left_image, det, filtered_indices, target_words)

        if filtered_masks is None or len(filtered_masks.data) == 0:
            return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None
        
        # Convert RealSense depth from uint16 (mm) to float32 (meters)
        depth = rs_depth.astype(np.float32) / 1000.0
        H, W = left_image.shape[:2]
        combined_mask, obj_ids = make_combined_mask(H, W, filtered_masks, filtered_box_ids)
        
        if combined_mask is not None:
            combined_mask = cv2.resize(combined_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            centers = compute_mask_centroids(combined_mask)

        if obj_ids is None or len(np.unique(combined_mask)) != len(obj_ids):
            print('Mask and box ID mismatch, skipping frame')
            return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

        frame_data = {
            "color": left_image.copy(),
            "depth": rs_depth.copy(),  # Keep as uint16 mm for InferDataset
            "mask": combined_mask
        }

    depth_color_rgb = visualize_depth_map(depth, z_far=getattr(args, 'z_far', 10.0))

    try:
        data = InferDataset.alternetive_init(frame_data, img_size=args.img_size, 
                                           device=torch.device(args.device), n_pts=args.n_pts)
    except Exception:
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

    obj_ids = [row for row in obj_ids if row not in [[0,0], [255,255]]]
    if not obj_ids or data is None:
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

    # Build object metadata
    obj_meta = build_object_metadata(obj_ids, det, filtered_indices)

    # GenPose2 inference
    pose_fn = genpose2.inference
    pose_args = {'data': data, 'obj_ids': obj_ids, 'frame_idx': frame_idx}
    
    if GenPose2.prev_pose_dict is None:
        pose, length = pose_fn(T0=args.T0, **pose_args)
    else:
        pose, length = pose_fn(T0=args.T0, Tp=args.Tp, **pose_args)
    
    # Validate output and visualize
    if not (isinstance(pose, (list, tuple)) and isinstance(length, (list, tuple)) 
            and len(pose) > 0 and len(length) > 0 
            and pose[0] is not None and length[0] is not None):
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), None, None, None, None

    color_image_w_pose = visualize_pose(data, pose, length, visualize_image=False)
    color_image_w_pose = cv2.cvtColor(color_image_w_pose, cv2.COLOR_BGR2RGB)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    scene_vis = cv2.cvtColor(left_undistorted.copy() if left_image is None else left_image.copy(), cv2.COLOR_BGR2RGB)

    if pose[0] is not None:
        scene = get_relation(pose[0])
        scene_vis = draw_scene_graph(scene_vis, scene, obj_meta, centers)

    pose_data = None
    if pose is not None and len(pose) > 0 and pose[0] is not None:
        np_pose = list(pose[0].numpy())
        
        # Get valid labels to correctly map poses to box_ids
        batch_sample = data.get_objects()
        if batch_sample is None or 'labels' not in batch_sample:
            print('No valid objects found in data, skipping pose save')
            return vis_image, color_image_w_pose, depth_color_rgb, scene_vis, None
        
        valid_mask_labels = batch_sample['labels']
        mapping_mask_to_box = {mask_id: box_id for mask_id, box_id in obj_ids}
        
        pose_list = []
        pose_box_ids = []
        pose_class_names = []
        
        for i, mask_label in enumerate(valid_mask_labels):
            if i >= len(np_pose):
                break
            
            box_id = mapping_mask_to_box.get(mask_label)
            if box_id is None:
                continue
            
            # Find the box_id in filtered_box_ids to get class name
            try:
                if isinstance(filtered_box_ids, np.ndarray):
                    box_idx_arr = np.where(filtered_box_ids == box_id)[0]
                    if len(box_idx_arr) == 0:
                        continue
                    box_idx = box_idx_arr[0]
                else:
                    box_idx = filtered_box_ids.index(box_id)
                class_name = det['class_name'][filtered_indices[box_idx]]
            except (ValueError, IndexError):
                continue
            
            xyz = np_pose[i][:3, 3]
            rot_matrix = np_pose[i][:3, :3]
            rot_fix = np.array([[-1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]])
            rot_matrix = rot_matrix @ rot_fix
            quat_wxyz = matrix_to_quaternion(torch.from_numpy(rot_matrix))
            quat_wxyz = quat_wxyz.cpu().numpy()
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            pose_vec = np.concatenate([xyz, quat_xyzw]).astype(float)
            pose_list.append(pose_vec)
            
            pose_box_ids.append(box_id)
            pose_class_names.append(class_name)

        if pose_list:
            pose_data = {
                'poses': pose_list,
                'box_ids': pose_box_ids,
                'class_names': pose_class_names
            }

    return vis_image, color_image_w_pose, depth_color_rgb, scene_vis, pose_data


def publish_saved_poses(pose_data, node):
    """Publish poses from saved state"""
    if pose_data is None:
        return "No poses to publish"
    
    publish_poses(node, pose_data['poses'], pose_data['box_ids'], pose_data['class_names'])
    return f"Published {len(pose_data['poses'])} poses"


def main():
    args = create_default_config()
    # args.cam_type = 'stereo'
    args.cam_type = 'realsense'

    fds, genpose2, yoloe, K, baseline = setup_all_models(args)
    frame_idx = 0

    rclpy.init()
    node = rclpy.create_node('pose_node')
    node.tf_pub = node.create_publisher(TFMessage, '/tf', 10)
    
    with gr.Blocks(title="YOLO-Pose Interactive Filter") as demo:
        gr.Markdown("## wtf is this")
        
        with gr.Row():
            with gr.Column(scale=0.5):
                input_image = gr.Image(label="Input Image", type="numpy", sources=[], interactive=False)
                capture_btn = gr.Button("Capture from Camera", variant="secondary")
                
                yolo_state = gr.State()
                left_image_state = gr.State()
                right_image_state = gr.State()
                pose_state = gr.State()
                rs_depth = gr.State()
                rs_depth_scale = gr.State()
                rs_image_bgr = gr.State()  # Add state for BGR image

                analyze_btn = gr.Button("1. Run Detection", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
                
                gr.HTML("<hr>")
                
                filter_text = gr.Textbox(label="Filter Keywords", placeholder="e.g., person, cup (leave empty to see all)")
                filter_btn = gr.Button("2. Apply Filter & Generate Pose")
                
                gr.HTML("<hr>")
                
                publish_btn = gr.Button("3. Publish Poses", variant="primary")
                publish_status = gr.Textbox(label="Publish Status", interactive=False)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Filtered Result")
                vis_pose = gr.Image(label="Pose Visualization")
            
            with gr.Column(scale=1):
                depth_image = gr.Image(label="Depth Map", interactive=False)
                scene_graph = gr.Image(label="Scene Graph", interactive=False)
                

        # Event handlers
        if args.cam_type == 'stereo':
            capture_btn.click(capture_stereo_image, outputs=[input_image, left_image_state, right_image_state])
        elif args.cam_type == 'realsense':
            capture_btn.click(capture_realsense_image, outputs=[input_image, rs_image_bgr, rs_depth, rs_depth_scale])
        
        if args.cam_type == 'stereo':
            analyze_btn.click(
                lambda l, r: process_stereo_image_and_cache(l, r, yoloe),
                inputs=[left_image_state, right_image_state],
                outputs=[yolo_state, status_text, output_image]
            )
        elif args.cam_type == 'realsense':
            analyze_btn.click(
                lambda inp: process_realsense_image_and_cache(inp, yoloe),
                inputs=[rs_image_bgr],
                outputs=[yolo_state, status_text, output_image]
            )

        if args.cam_type == 'stereo':
            filter_btn.click(
                lambda l, r, y, k: update_view(l, None, y, k, fds, genpose2, args, K, baseline, args.depth_scale, frame_idx, node),
                inputs=[left_image_state, right_image_state, yolo_state, filter_text],
                outputs=[output_image, vis_pose, depth_image, scene_graph, pose_state]
            )
        elif args.cam_type == 'realsense':
            filter_btn.click(
                lambda i, r, y, k, ds: update_view(i, r, y, k, None, genpose2, args, K, None, ds, frame_idx, node),
                inputs=[rs_image_bgr, rs_depth, yolo_state, filter_text, rs_depth_scale],
                outputs=[output_image, vis_pose, depth_image, scene_graph, pose_state]
            )

        publish_btn.click(
            lambda pd: publish_saved_poses(pd, node),
            inputs=[pose_state],
            outputs=[publish_status]
        )

    demo.launch()

if __name__ == "__main__":
    main()