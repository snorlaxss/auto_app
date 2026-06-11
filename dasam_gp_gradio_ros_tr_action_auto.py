import os
import sys
import argparse
import importlib.machinery
import types
import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr
import threading
import yaml
import time
import json
import contextlib  # 补齐
import copy
from datetime import datetime
from types import SimpleNamespace
# ROS2 imports
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16MultiArray, Float32
# Import custom ROS2 messages
try:
    from fake_interface_pkg.msg import KeypointPose, KeypointPoseArray
    HAS_KEYPOINT_MSG = True
except ImportError:
    print("Warning: fake_interface_pkg not found. Action publishing will be disabled.")
    HAS_KEYPOINT_MSG = False

AUTO_APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3_DOCKER_DIR = os.path.join(AUTO_APP_DIR, "Sam3Docker")
SAM3_SOURCE_DIR = os.path.join(SAM3_DOCKER_DIR, "sam3-main")
if os.path.isdir(SAM3_SOURCE_DIR) and SAM3_SOURCE_DIR not in sys.path:
    sys.path.insert(0, SAM3_SOURCE_DIR)

FLOWPOSE_DOCKER_DIR = os.path.join(AUTO_APP_DIR, "FlowPoseDocker")
FLOWPOSE_SOURCE_DIR = os.path.join(FLOWPOSE_DOCKER_DIR, "FlowPose")
FLOWPOSE_PY_RUNNER_DIR = os.path.join(FLOWPOSE_SOURCE_DIR, "py_runners")
for _path in (FLOWPOSE_PY_RUNNER_DIR, FLOWPOSE_SOURCE_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_IMPORT_ERROR = None
except Exception as e:
    build_sam3_image_model = None
    Sam3Processor = None
    SAM3_IMPORT_ERROR = e

class ExecutionRecorder:
    """
    执行记录器：
    1. 执行开始时记录目标 TF（一帧快照）
    2. 持续记录 control/target_poseL & poseR（10Hz）
    3. 监听 /control/task_percentage：由非零变为零后停止
    4. 保存两个 yaml 到 /home/snorlax/work/gradio_sam_genpose/json/<timestamp>/
    """

    SAVE_DIR = "/home/snorlax/work/gradio_sam_genpose/json"

    def __init__(self, ros_node: "ROS2ImageSubscriber"):
        self._node = ros_node
        self._lock = threading.Lock()

        self._active_arm = None  # "left" / "right"

        self._pose_L_records = []
        self._pose_R_records = []
        self._gripper_L_records = []
        self._gripper_R_records = []

        # task_percentage 状态
        self._seen_active = False
        self._seen_zero = False
        self._last_msg_time: float | None = None
        self._percentage_value: float = 0.0

        self._recording = False
        self._record_thread: threading.Thread | None = None

        # ROS 订阅
        self._pose_L_sub = ros_node.create_subscription(
            PoseStamped, "/control/target_poseL", self._on_pose_L, 10
        )
        self._pose_R_sub = ros_node.create_subscription(
            PoseStamped, "/control/target_poseR", self._on_pose_R, 10
        )

        # 新增：夹爪值订阅
        self._gripper_L_sub = ros_node.create_subscription(
            Float32, "/control/gripperValueL", self._on_gripper_L, 10
        )
        self._gripper_R_sub = ros_node.create_subscription(
            Float32, "/control/gripperValueR", self._on_gripper_R, 10
        )

        self._pct_sub = ros_node.create_subscription(
            Float32, "/control/task_percentage", self._on_percentage, 10
        )

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #
    def _on_pose_L(self, msg: PoseStamped):
        if not self._recording or self._active_arm != "left":
            return
        with self._lock:
            self._pose_L_records.append(self._pose_stamped_to_dict(msg))

    def _on_pose_R(self, msg: PoseStamped):
        if not self._recording or self._active_arm != "right":
            return
        with self._lock:
            self._pose_R_records.append(self._pose_stamped_to_dict(msg))

    # 新增：夹爪回调
    def _on_gripper_L(self, msg: Float32):
        if not self._recording or self._active_arm != "left":
            return
        with self._lock:
            self._gripper_L_records.append({
                "t": time.time(),
                "value": float(msg.data),
            })

    def _on_gripper_R(self, msg: Float32):
        if not self._recording or self._active_arm != "right":
            return
        with self._lock:
            self._gripper_R_records.append({
                "t": time.time(),
                "value": float(msg.data),
            })

    def _on_percentage(self, msg: Float32):
        """监听 /control/task_percentage：先出现>0，再出现==0 视为任务进入完成阶段"""
        with self._lock:
            val = float(msg.data)
            self._percentage_value = val
            self._last_msg_time = time.time()

            if val > 0.0:
                self._seen_active = True

            if self._seen_active and val == 0.0:
                self._seen_zero = True

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pose_stamped_to_dict(msg: PoseStamped) -> dict:
        t = msg.pose.position
        r = msg.pose.orientation
        return {
            "stamp": {
                "sec": msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec,
            },
            "frame_id": msg.header.frame_id,
            "position": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
            "orientation": {
                "x": float(r.x), "y": float(r.y),
                "z": float(r.z), "w": float(r.w),
            },
        }

    @staticmethod
    def _transform_to_dict(ts) -> dict:
        t = ts.transform.translation
        r = ts.transform.rotation
        return {
            "parent_frame": ts.header.frame_id,
            "child_frame": ts.child_frame_id,
            "stamp": {
                "sec": ts.header.stamp.sec,
                "nanosec": ts.header.stamp.nanosec,
            },
            "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
            "rotation": {
                "x": float(r.x), "y": float(r.y),
                "z": float(r.z), "w": float(r.w),
            },
        }

    def _is_finished(self, silence_after_zero: float = 0.5) -> bool:
        now = time.time()
        with self._lock:
            seen_active = self._seen_active
            seen_zero = self._seen_zero
            last_msg_time = self._last_msg_time
        return bool(
            seen_active
            and seen_zero
            and last_msg_time is not None
            and (now - last_msg_time) >= silence_after_zero
        )

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def start(self, target_frame: str, labels: list, arm: str):
        """
        执行前调用：
        - 快照目标 TF
        - 重置计数器
        - 启动后台线程等待完成
        """
        # 重置状态
        with self._lock:
            self._active_arm = arm
            self._pose_L_records = []
            self._pose_R_records = []
            self._gripper_L_records = []
            self._gripper_R_records = []
            self._seen_active = False
            self._seen_zero = False
            self._last_msg_time = None
            self._percentage_value = 0.0

        # 快照所有相关子 frame 的 TF（只取最新一帧）
        snapshot = {}
        try:
            for line in self._node.tf_buffer.all_frames_as_string().split("\n"):
                parts = line.split()
                if len(parts) < 6:
                    continue
                child = parts[1].strip()
                parent = parts[5].rstrip(".")
                if parent != "camera_rgb_link":
                    continue
                # 过滤：只记录和 target 相关的 frame
                matched = any(
                    lbl.lower() in child.lower() or child.lower().startswith(lbl.lower())
                    for lbl in labels
                )
                if not matched and target_frame.lower() not in child.lower():
                    continue
                try:
                    ts = self._node.tf_buffer.lookup_transform(
                        "camera_rgb_link",
                        child,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.05),
                    )
                    snapshot[child] = self._transform_to_dict(ts)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Recorder] TF snapshot error: {e}")

        # 同时也记录 target_frame 本身（如果还没记录）
        if target_frame and target_frame not in snapshot:
            try:
                ts = self._node.tf_buffer.lookup_transform(
                    "camera_rgb_link",
                    target_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
                snapshot[target_frame] = self._transform_to_dict(ts)
            except Exception as e:
                print(f"[Recorder] target TF lookup failed: {e}")

        with self._lock:
            self._tf_snapshot = snapshot
            self._recording = True

        print(f"[Recorder] 开始记录, TF快照 {len(snapshot)} 个 frame")

        # 后台等待完成并保存
        self._record_thread = threading.Thread(
            target=self._wait_and_save,
            args=(target_frame,),
            daemon=True,
        )
        self._record_thread.start()

    def _wait_and_save(self, target_frame: str, timeout: float = 30.0):
        """后台线程：等待任务完成后保存"""
        hz = 10.0
        interval = 1.0 / hz
        start = time.time()

        while not self._is_finished():
            if time.time() - start > timeout:
                print("[Recorder] ⚠️ 等待超时，强制保存")
                break
            time.sleep(interval)

        with self._lock:
            self._recording = False

        self._save(target_frame)

    def _save(self, target_frame: str):
        """保存两个 yaml 文件"""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.SAVE_DIR, ts_str)
        os.makedirs(save_dir, exist_ok=True)

        with self._lock:
            active_arm = self._active_arm
            pose_L = list(self._pose_L_records)
            pose_R = list(self._pose_R_records)
            grip_L = list(self._gripper_L_records)
            grip_R = list(self._gripper_R_records)
            tf_data = dict(self._tf_snapshot)

        # yaml 1: TF 快照
        tf_yaml_path = os.path.join(save_dir, "tf_snapshot.yaml")
        with open(tf_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "target_frame": target_frame,
                    "record_time": ts_str,
                    "tf_snapshot": tf_data,
                },
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

        # yaml 2: 控制 pose 记录
        ctrl_yaml_path = os.path.join(save_dir, "control_pose_records.yaml")
        with open(ctrl_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "target_frame": target_frame,
                    "active_arm": active_arm,
                    "control_target_poseL": pose_L,
                    "control_target_poseR": pose_R,
                    "control_gripperValueL": grip_L,
                    "control_gripperValueR": grip_R,
                },
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

        print(
            f"[Recorder] ✅ 已保存到 {save_dir}\n"
            f"  tf_snapshot.yaml       ({len(tf_data)} frames)\n"
            f"  control_pose_records.yaml  (L={len(pose_L)}, R={len(pose_R)})"
        )

# Allow importing local modules
sys.path.append(os.path.dirname(__file__))

LINGBOT_DEPTH_DIR = os.path.join(os.path.dirname(__file__), "lingbot-depth")
if os.path.isdir(LINGBOT_DEPTH_DIR) and LINGBOT_DEPTH_DIR not in sys.path:
    sys.path.append(LINGBOT_DEPTH_DIR)

def _is_path_under(path, root):
    if not path or not root:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except Exception:
        return False


def _clear_non_flowpose_cached_modules():
    """Avoid collisions with auto_app top-level utils/dataset/network packages."""
    prefixes = ("utils", "dataset", "inference", "networks")
    for name in list(sys.modules):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", "") if module is not None else ""
        if not _is_path_under(module_file, FLOWPOSE_SOURCE_DIR):
            sys.modules.pop(name, None)


def _register_flowpose_namespace_package(name, package_dir):
    if not os.path.isdir(package_dir):
        return
    module = types.ModuleType(name)
    module.__path__ = [package_dir]
    module.__package__ = name
    module.__file__ = os.path.join(package_dir, "__init__.py")
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [package_dir]
    module.__spec__ = spec
    sys.modules[name] = module


def _register_flowpose_namespace_packages():
    # FlowPose uses top-level imports like "utils.transforms.mask" but its
    # package directories are namespace-style. Register them explicitly so
    # auto_app/utils cannot shadow them.
    for name in ("utils", "dataset", "inference", "networks"):
        _register_flowpose_namespace_package(
            name,
            os.path.join(FLOWPOSE_SOURCE_DIR, name),
        )
    _register_flowpose_namespace_package(
        "utils.transforms",
        os.path.join(FLOWPOSE_SOURCE_DIR, "utils", "transforms"),
    )


# Prevent argument parsing at import
_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
try:
    for _path in (FLOWPOSE_PY_RUNNER_DIR, FLOWPOSE_SOURCE_DIR):
        if os.path.isdir(_path):
            with contextlib.suppress(ValueError):
                sys.path.remove(_path)
    for _path in (FLOWPOSE_SOURCE_DIR, FLOWPOSE_PY_RUNNER_DIR):
        if os.path.isdir(_path):
            sys.path.insert(0, _path)
    _clear_non_flowpose_cached_modules()
    _register_flowpose_namespace_packages()

    from api_runner import PoseInferenceSession
    from inference.inference_helper import Flow
    from networks.dino.dino import DinoLoader
    from utils.yomni_vis import visualize_detections as flowpose_visualize_detections
    FLOWPOSE_IMPORT_ERROR = None
except Exception as e:
    PoseInferenceSession = None
    Flow = None
    DinoLoader = None
    flowpose_visualize_detections = None
    FLOWPOSE_IMPORT_ERROR = e
finally:
    sys.argv = _orig_argv


def _load_flowpose_config():
    config_path = os.path.join(FLOWPOSE_DOCKER_DIR, "config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load FlowPose config {config_path}: {e}")
        return {}


def _resolve_flowpose_path(raw_path=""):
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if os.path.isfile(candidate) or os.path.isdir(candidate):
        return candidate
    if candidate.startswith("/workspace/"):
        mapped = os.path.join(FLOWPOSE_DOCKER_DIR, candidate[len("/workspace/"):])
        if os.path.exists(mapped):
            return mapped
    mapped = os.path.join(FLOWPOSE_DOCKER_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    return candidate


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _unpack_flowpose_output(pose_out, length_out):
    if pose_out is None or length_out is None:
        return None, None
    pose_all = pose_out[0] if isinstance(pose_out, (list, tuple)) and pose_out else pose_out
    length_all = length_out[0] if isinstance(length_out, (list, tuple)) and length_out else length_out
    if pose_all is None or length_all is None:
        return None, None
    return _to_jsonable(pose_all), _to_jsonable(length_all)


def _build_flowpose_objects(pose_all, length_all, obj_ids, class_names=None, instance_names=None):
    if pose_all is None or length_all is None or obj_ids is None:
        return []
    class_names = class_names or []
    instance_names = instance_names or []
    objects = []
    n = min(len(pose_all), len(length_all), len(obj_ids))
    for i in range(n):
        obj_id = obj_ids[i]
        box_id = None
        if isinstance(obj_id, (list, tuple)) and len(obj_id) >= 2:
            try:
                box_id = int(obj_id[1])
            except Exception:
                box_id = None
        if i < len(instance_names) and instance_names[i]:
            name = str(instance_names[i])
        elif i < len(class_names) and class_names[i]:
            name = str(class_names[i])
        elif box_id is not None:
            name = f"obj_{box_id}"
        else:
            name = f"obj_{i + 1}"
        objects.append({
            "name": name,
            "pose": pose_all[i],
            "length": length_all[i],
            "obj_id": obj_id,
            "box_id": box_id,
        })
    return objects


class FlowPoseEstimator:
    def __init__(self, device):
        if PoseInferenceSession is None or Flow is None or DinoLoader is None:
            raise RuntimeError(f"FlowPose import failed: {FLOWPOSE_IMPORT_ERROR}")

        cfg = _load_flowpose_config()
        paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        vis_cfg = cfg.get("visualization", {}) if isinstance(cfg, dict) else {}

        py_runner_path = _resolve_flowpose_path(paths_cfg.get("py_runner_path", ""))
        flowpose_root = os.path.dirname(py_runner_path) if py_runner_path else FLOWPOSE_SOURCE_DIR
        for p in (py_runner_path, flowpose_root):
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        self.fx = float(vis_cfg.get("fx", 606.9654541015625))
        self.fy = float(vis_cfg.get("fy", 606.133056640625))
        self.cx = float(vis_cfg.get("cx", 323.8822937011719))
        self.cy = float(vis_cfg.get("cy", 257.6228332519531))
        self.width = int(vis_cfg.get("width", 640))
        self.height = int(vis_cfg.get("height", 480))
        self.axis_len = float(vis_cfg.get("axis_len", 0.08))
        self.device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"

        args = argparse.Namespace(
            pretrained_flow_model_path=_resolve_flowpose_path(paths_cfg.get("pretrained_flow_model_path", "")),
            pretrained_scale_model_path=_resolve_flowpose_path(paths_cfg.get("pretrained_scale_model_path", "")),
            device=self.device,
            img_size=224,
            n_pts=1024,
            frame_gap_threshold=10,
            eval_repeat_num=25,
            retain_ratio=0.4,
            enable_tracking=True,
            seed=0,
            dropout=0,
            use_edm_aug=False,
            log_dir="debug",
            use_pretrain=False,
            is_train=False,
            pose_mode="rot_matrix",
            optimizer="Adam",
            lr=1e-2,
            lr_decay=0.98,
            num_points=1024,
            scale_embedding=180,
            ema_rate=0.999,
            repeat_num=20,
            clustering=1,
            clustering_eps=0.05,
            clustering_minpts=0.1667,
        )

        print(f"[FlowPose] Loading Flow on {self.device}...")
        self.flow = Flow(args)
        local_dino_repo = _resolve_flowpose_path("/workspace/model/facebookresearch_dinov2_main")
        print(f"[FlowPose] Loading DINO from {local_dino_repo}...")
        self.dino_loader = DinoLoader(
            model_name="dinov2_vits14",
            device=args.device,
            local_repo_path=local_dino_repo,
        )
        print("[FlowPose] Creating PoseInferenceSession...")
        self.inferencer = PoseInferenceSession(
            self.flow,
            args,
            intrinsics={
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.width,
                "height": self.height,
            },
        )

    def build_cam_intrinsics(self):
        data = SimpleNamespace()
        data.fx = self.fx
        data.fy = self.fy
        data.cx = self.cx
        data.cy = self.cy
        data.width = self.width
        data.height = self.height
        data.intrinsic_matrix = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        data.K = data.intrinsic_matrix
        return data

    def infer(self, rgb, depth, combined_mask, obj_ids, class_names=None, instance_names=None, depth_scale=0.001):
        t0 = time.time()
        obj_ids_filtered = [row for row in (obj_ids or []) if row != [0, 0] and row != [255, 255]]
        if len(obj_ids_filtered) == 0:
            return {
                "status": "ok",
                "objects": [],
                "pose_all": None,
                "length_all": None,
                "elapsed_sec": 0.0,
            }

        pose_out, length_out = self.inferencer.infer(
            dino_loader=self.dino_loader,
            rgb=rgb.astype(np.uint8),
            depth=depth.astype(np.float32),
            mask=combined_mask.astype(np.uint8),
            obj_ids=obj_ids_filtered,
            depth_scale=float(depth_scale or 0.001),
        )
        pose_all, length_all = _unpack_flowpose_output(pose_out, length_out)
        objects = _build_flowpose_objects(
            pose_all,
            length_all,
            obj_ids_filtered,
            class_names=class_names,
            instance_names=instance_names,
        )
        return {
            "status": "ok",
            "objects": objects,
            "pose_all": pose_all,
            "length_all": length_all,
            "elapsed_sec": round(time.time() - t0, 4),
        }

    def visualize(self, rgb, pose_all, length_all, labels):
        vis = rgb.copy()
        if pose_all is not None and length_all is not None:
            poses = np.asarray(pose_all, dtype=np.float32)
            lengths = np.asarray(length_all, dtype=np.float32)
            if flowpose_visualize_detections is not None and len(poses) > 0 and len(lengths) > 0:
                vis = flowpose_visualize_detections(
                    vis,
                    poses,
                    lengths,
                    self.build_cam_intrinsics(),
                    color=(0, 255, 0),
                    thickness=2,
                    axes_length=self.axis_len,
                )
            for i, pose_mat in enumerate(poses):
                if i >= len(labels):
                    break
                center = pose_mat[:3, 3]
                if abs(float(center[2])) < 1e-8:
                    continue
                u = int((self.fx * center[0] / center[2]) + self.cx)
                v = int((self.fy * center[1] / center[2]) + self.cy)
                cv2.putText(vis, str(labels[i]), (u - 30, v - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return vis


def matrix_to_quaternion(rotation_matrix):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # Returns [x, y, z, w]
    return quat

def load_yaml_file(yaml_path):
    """Load YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {yaml_path}: {e}")
        return None

def correct_orientation(pos, quat, mode="none", arm=None,
                       z_up=np.array([0., 0., 1.], dtype=float),
                       x_forward=np.array([1., 0., 0.], dtype=float),
                       z_thresh_deg=20,
                       x_thresh_deg=90):
    """
    统一的姿态校正函数
    
    mode 选项:
    - "reverse": 反向模式(Z轴朝下,X轴朝后)
    - "condition_forward": 条件前向模式
    - "forward": 前向模式（返回单位四元数）
    - "none": 不进行校正
    - "disorder": 无序模式,自动识别最接近XY平面的轴作为X轴
    """
    q = np.array(quat, dtype=float)
    if np.linalg.norm(q) < 1e-8:
        return pos, np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    
    # mode == "none": 直接返回原始姿态
    if mode == "none":
        return pos, q
    
    # mode == "forward": 返回单位四元数
    if mode == "forward":
        R_mat = R.from_quat(q).as_matrix()
        z_axis = R_mat[:, 2]
        z_axis /= np.linalg.norm(z_axis)
        z_up_norm = z_up / np.linalg.norm(z_up)
        cos_z = np.clip(np.dot(z_axis, z_up_norm), -1.0, 1.0)
        angle_z_deg = np.degrees(np.arccos(cos_z))
        
        if angle_z_deg > z_thresh_deg:
            new_z = -z_up_norm
            old_x = R_mat[:, 0]
            new_x = old_x - np.dot(old_x, new_z) * new_z
            if np.linalg.norm(new_x) < 1e-6:
                new_x = np.array([1., 0., 0.]) - np.dot([1., 0., 0.], new_z) * new_z
            new_x /= np.linalg.norm(new_x)
            new_y = np.cross(new_z, new_x)
            new_y /= np.linalg.norm(new_y)
            R_mat = np.column_stack([new_x, new_y, new_z])
        
        x_axis = R_mat[:, 0]
        x_axis /= np.linalg.norm(x_axis)
        x_forward_norm = x_forward / np.linalg.norm(x_forward)
        cos_x = np.clip(np.dot(x_axis, x_forward_norm), -1.0, 1.0)
        angle_x_deg = np.degrees(np.arccos(cos_x))
        
        if angle_x_deg > x_thresh_deg:
            new_x = x_forward.copy()
            new_z = R_mat[:, 2]
            new_y = np.cross(new_z, new_x)
            new_y /= np.linalg.norm(new_y)
            new_z = np.cross(new_x, new_y)
            new_z /= np.linalg.norm(new_z)
        
        return pos, np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    # mode == "disorder": 无序模式 - 自动识别最接近XY平面的轴作为X轴
    if mode == "disorder":
        R_mat = R.from_quat(q).as_matrix()
        
        # 获取当前的三个轴
        axis_x = R_mat[:, 0]
        axis_y = R_mat[:, 1]
        axis_z = R_mat[:, 2]
        
        # 归一化
        axis_x /= np.linalg.norm(axis_x)
        axis_y /= np.linalg.norm(axis_y)
        axis_z /= np.linalg.norm(axis_z)
        
        # 计算每个轴与Z轴([0,0,1])的夹角（范围 [0, 180]）
        z_standard = np.array([0., 0., 1.])
        
        angle_x_to_z = np.degrees(np.arccos(np.clip(np.dot(axis_x, z_standard), -1.0, 1.0)))
        angle_y_to_z = np.degrees(np.arccos(np.clip(np.dot(axis_y, z_standard), -1.0, 1.0)))
        angle_z_to_z = np.degrees(np.arccos(np.clip(np.dot(axis_z, z_standard), -1.0, 1.0)))
        
        
        # 找出最接近XY平面的轴（与Z轴夹角最接近90°）
        angles = [angle_x_to_z, angle_y_to_z, angle_z_to_z]
        distances_to_90 = [abs(a - 90.0) for a in angles]
        min_dist = min(distances_to_90)
        tol_deg = 10.0  # 容差：视为并列接近90°
        candidates = [i for i, d in enumerate(distances_to_90) if d - min_dist <= tol_deg]
        axes = [axis_x, axis_y, axis_z]

        if len(candidates) == 1:
            selected_idx = candidates[0]
        else:
            # 并列时，选与 [1,0,0] 夹角最小的作为 X 轴
            x_forward_norm = np.array([1., 0., 0.])
            angle_to_x_forward = []
            for idx in candidates:
                a = axes[idx]
                a = a / np.linalg.norm(a)
                cos_val = np.clip(np.dot(a, x_forward_norm), -1.0, 1.0)
                angle_to_x_forward.append(np.degrees(np.arccos(cos_val)))
            selected_idx = candidates[int(np.argmin(angle_to_x_forward))]
        
        selected_axis = axes[selected_idx]
        
        
        # 步骤1：直接设置Z轴为[0,0,1]
        new_z = np.array([0., 0., 1.])
        
        # 步骤2：投影选中的轴到XY平面作为新X轴
        new_x = selected_axis - np.dot(selected_axis, new_z) * new_z
        new_x_norm = np.linalg.norm(new_x)
        
        if new_x_norm < 1e-6:
            # 选中的轴平行于Z轴，选择默认方向[1,0,0]
            new_x = np.array([1., 0., 0.])
        else:
            new_x /= new_x_norm
        
        # 步骤3：用右手法则计算Y轴：Y = Z × X
        new_y = np.cross(new_z, new_x)
        new_y /= np.linalg.norm(new_y)
        
        # 组合旋转矩阵
        R_mat = np.column_stack([new_x, new_y, new_z])
        
        # 步骤4：检测X轴方向，如果与[1,0,0]夹角过大则反向
        x_axis_final = R_mat[:, 0]
        x_forward_norm = np.array([1., 0., 0.])
        cos_x = np.clip(np.dot(x_axis_final, x_forward_norm), -1.0, 1.0)
        angle_x_deg = np.degrees(np.arccos(cos_x))
        if angle_x_deg > x_thresh_deg:
            # 绕Z轴旋转180度
            rotation_matrix_180 = np.array([
                [-1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.]
            ])
            R_mat = R_mat @ rotation_matrix_180
        
        quat_corrected = R.from_matrix(R_mat).as_quat()
        return pos, quat_corrected

    # mode == "reverse": Z轴朝上，X轴根据手臂方向调整
    if mode == "reverse":
        R_mat = R.from_quat(q).as_matrix()
        z_axis = R_mat[:, 2]
        z_axis /= np.linalg.norm(z_axis)
        z_up_norm = z_up / np.linalg.norm(z_up)
        cos_z = np.clip(np.dot(z_axis, z_up_norm), -1.0, 1.0)
        angle_z_deg = np.degrees(np.arccos(cos_z))
        if angle_z_deg > z_thresh_deg:
            new_z = z_up.copy()
            old_x = R_mat[:, 0]
            new_x = old_x - np.dot(old_x, new_z) * new_z
            new_x /= np.linalg.norm(new_x)
            new_y = np.cross(new_z, new_x)
            new_y /= np.linalg.norm(new_y)
            R_mat = np.column_stack([new_x, new_y, new_z])
        x_axis = R_mat[:, 0]
        x_axis /= np.linalg.norm(x_axis)
        x_forward_norm = x_forward / np.linalg.norm(x_forward)
        cos_x = np.clip(np.dot(x_axis, x_forward_norm), -1.0, 1.0)
        angle_x_deg = np.degrees(np.arccos(cos_x)) 
        if angle_x_deg > x_thresh_deg:
            new_x = -x_axis
            new_z = R_mat[:, 2]
            new_y = np.cross(new_z, new_x)
            new_y /= np.linalg.norm(new_y)
            new_z = np.cross(new_x, new_y)
            new_z /= np.linalg.norm(new_z)
            R_mat = np.column_stack([new_x, new_y, new_z])
        x_axis_final = R_mat[:, 0]
        x_axis_final /= np.linalg.norm(x_axis_final)
        x_forward_norm = x_forward / np.linalg.norm(x_forward)
        cross_prod = np.cross(x_forward_norm, x_axis_final)
        rotation_direction = cross_prod[2]  # 正数表示逆时针，负数表示顺时针
        
        if arm == "right":
            if rotation_direction < 0: 
                # 创建绕 z 轴逆时针 90 度的旋转矩阵
                angle_rad = np.pi / 2  # 90 度
                rotation_matrix = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 1]
                ])
                # 将旋转矩阵应用到 R_mat
                R_mat = R_mat @ rotation_matrix
                # 提取旋转后的坐标轴
                new_x = R_mat[:, 0]
                new_y = R_mat[:, 1]
                new_z = R_mat[:, 2]
        elif arm == "left":
            if rotation_direction > 0:
                # 创建绕 z 轴顺时针 90 度的旋转矩阵
                angle_rad = -np.pi / 2  # -90 度（顺时针）
                rotation_matrix = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 1]
                ])
                # 将旋转矩阵应用到 R_mat
                R_mat = R_mat @ rotation_matrix
                # 提取旋转后的坐标轴
                new_x = R_mat[:, 0]
                new_y = R_mat[:, 1]
                new_z = R_mat[:, 2]

        quat_corrected = R.from_matrix(R_mat).as_quat()

        return pos, quat_corrected

    return pos, quat_corrected


class ROS2ImageSubscriber(Node):
    def __init__(self):
        super().__init__('sam_genpose_subscriber')
        self.color_image = None
        self.depth_image = None
        self.depth_scale = 0.001
        self.frame_received = threading.Event()

        # RealSense is used directly for RGB-D capture. ROS2 is still used for TF,
        # action publication, and execution recording.
        self.rs_width = int(os.getenv("REALSENSE_WIDTH", "640"))
        self.rs_height = int(os.getenv("REALSENSE_HEIGHT", "480"))
        self.rs_fps = int(os.getenv("REALSENSE_FPS", "30"))
        self.rs_serial = os.getenv("REALSENSE_SERIAL", "").strip()
        self.rs_pipeline = None
        self.rs_profile = None
        self.rs_align = None
        self.rs_lock = threading.Lock()

        # TF publisher
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        # TF buffer and listener for reading transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Keypoint pose publisher for action execution
        if HAS_KEYPOINT_MSG:
            self.pose_pub = self.create_publisher(KeypointPoseArray, '/fusion_pose', 10)
            self.get_logger().info('KeypointPoseArray publisher created on /fusion_pose')
        else:
            self.pose_pub = None

        # Track last arm used
        self.last_arm = None

        # 执行记录器（延迟初始化，避免订阅在节点完全创建前触发）
        self.recorder: ExecutionRecorder | None = None

        self.get_logger().info(
            'ROS2 node initialized; RGB-D capture will use pyrealsense2 directly'
        )

    def init_recorder(self):
        """在节点完全初始化后调用"""
        if self.recorder is None:
            self.recorder = ExecutionRecorder(self)
            self.get_logger().info('[Recorder] ExecutionRecorder 已初始化')

    def start_realsense(self):
        """Start the RealSense RGB-D pipeline on first capture."""
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not available. Please install pyrealsense2 "
                "and run on a machine with an Intel RealSense camera."
            )

        with self.rs_lock:
            if self.rs_pipeline is not None:
                return

            pipeline = rs.pipeline()
            config = rs.config()
            if self.rs_serial:
                config.enable_device(self.rs_serial)
            config.enable_stream(
                rs.stream.color,
                self.rs_width,
                self.rs_height,
                rs.format.bgr8,
                self.rs_fps,
            )
            config.enable_stream(
                rs.stream.depth,
                self.rs_width,
                self.rs_height,
                rs.format.z16,
                self.rs_fps,
            )

            try:
                profile = pipeline.start(config)
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = float(depth_sensor.get_depth_scale())
                self.rs_pipeline = pipeline
                self.rs_profile = profile
                self.rs_align = rs.align(rs.stream.color)
            except Exception:
                with contextlib.suppress(Exception):
                    pipeline.stop()
                raise

            self.get_logger().info(
                f"RealSense pipeline started: {self.rs_width}x{self.rs_height}@{self.rs_fps}, "
                f"depth_scale={self.depth_scale:.6f}, serial={self.rs_serial or '<auto>'}"
            )

            # Drop a few startup frames so auto-exposure/depth sync can settle.
            for _ in range(5):
                with contextlib.suppress(Exception):
                    pipeline.wait_for_frames(1000)

    def stop_realsense(self):
        """Stop the RealSense pipeline if it was started."""
        with self.rs_lock:
            pipeline = self.rs_pipeline
            self.rs_pipeline = None
            self.rs_profile = None
            self.rs_align = None
        if pipeline is not None:
            with contextlib.suppress(Exception):
                pipeline.stop()
            self.get_logger().info('RealSense pipeline stopped')

    def wait_for_frame(self, timeout=5.0):
        """Capture one aligned RGB-D frame directly from RealSense."""
        self.start_realsense()
        deadline = time.time() + float(timeout)
        last_error = None

        with self.rs_lock:
            pipeline = self.rs_pipeline
            align = self.rs_align
            if pipeline is None:
                return None, None, None

            while time.time() < deadline:
                remaining_ms = max(1, int((deadline - time.time()) * 1000))
                wait_ms = min(1000, remaining_ms)
                try:
                    frames = pipeline.wait_for_frames(wait_ms)
                    aligned_frames = align.process(frames) if align is not None else frames
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data()).copy()
                    depth_image = np.asanyarray(depth_frame.get_data()).copy()
                    if depth_image.dtype != np.uint16:
                        depth_image = depth_image.astype(np.uint16)

                    self.color_image = color_image
                    self.depth_image = depth_image
                    self.frame_received.set()
                    return color_image.copy(), depth_image.copy(), self.depth_scale
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            self.get_logger().warning(f'RealSense frame capture failed: {last_error}')
        return None, None, None

    def destroy_node(self):
        self.stop_realsense()
        return super().destroy_node()

    def publish_poses_as_tf(self, poses, labels, frame_id="camera_rgb_link"):
        """Publish poses as TF transforms"""
        tf_msg = TFMessage()
        stamp = self.get_clock().now().to_msg()

        for i, (pose_matrix, label) in enumerate(zip(poses, labels)):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = frame_id
            transform.child_frame_id = f"{label}_{i+1}"

            transform.transform.translation.x = float(pose_matrix[0, 3])
            transform.transform.translation.y = float(pose_matrix[1, 3])
            transform.transform.translation.z = float(pose_matrix[2, 3])

            rot_matrix = pose_matrix[:3, :3]
            quat = matrix_to_quaternion(rot_matrix)

            transform.transform.rotation.x = float(quat[0])
            transform.transform.rotation.y = float(quat[1])
            transform.transform.rotation.z = float(quat[2])
            transform.transform.rotation.w = float(quat[3])

            tf_msg.transforms.append(transform)

        self.tf_pub.publish(tf_msg)
        self.get_logger().info(f"Published {len(poses)} TF transforms")

    def lookup_transform(self, target_frame, source_frame, timeout=0.2):
        """Lookup transform between frames"""
        try:
            ts = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout)
            )
            return ts
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform {source_frame} -> {target_frame}: {e}")
            return None

    def publish_kp_separately(self, sequence):
        """Publish keypoint poses one by one"""
        if not HAS_KEYPOINT_MSG or self.pose_pub is None:
            self.get_logger().warning("KeypointPoseArray message not available, skipping publish")
            return False

        for info in sequence:
            kp_array = KeypointPoseArray()
            kp_array.header.stamp = self.get_clock().now().to_msg()
            kp_array.header.frame_id = "base_link"

            kp = KeypointPose()
            kp.name = info["name"]
            kp.arm = info["arm"]
            kp.poses = info.get("poses", [])
            kp.constraints = [float(x) for x in info.get("constraints", [1, 1, 1])]
            kp.speed = float(info.get("speed", 1.0))
            kp.gripper_value = [float(x) for x in info.get("gripper_value", [0])]
            kp.time = [float(x) for x in info.get("time", [0])]

            kp_array.poses.append(kp)
            try:
                self.pose_pub.publish(kp_array)
                self.get_logger().info(f"[Fusion] Published step '{kp.name}' for arm '{kp.arm}'")
            except Exception as e:
                self.get_logger().warning(f"Failed to publish step '{kp.name}': {e}")
            time.sleep(0.5)

        return True

    def publish_home_both(self):
        """Publish home position for both arms"""
        if not HAS_KEYPOINT_MSG or self.pose_pub is None:
            self.get_logger().warning("KeypointPoseArray message not available, skipping home publish")
            return False

        for arm_name in ["right", "left"]:
            home_seq = [{
                "name": "home",
                "arm": arm_name,
                "poses": [],
                "constraints": [100, 100, 100],
                "speed": 0.3,
                "gripper_value": [0.0, 0.0],
                "time": [0, 0.1]
            }]
            self.publish_kp_separately(home_seq)
            time.sleep(2)

        return True

    def publish_home_separately(self, arm):
        """Publish home position for a specific arm"""
        if not HAS_KEYPOINT_MSG or self.pose_pub is None:
            self.get_logger().warning("KeypointPoseArray message not available, skipping home publish")
            return False

        home_seq = [{
            "name": "home",
            "arm": arm,
            "poses": [],
            "constraints": [100, 100, 100],
            "speed": 0.3,
            "gripper_value": [0.0, 0.0],
            "time": [0, 0.1]
        }]
        self.publish_kp_separately(home_seq)
        return True


# Global ROS2 node instance
_ros2_node = None
_ros2_executor = None
_ros2_thread = None


def init_ros2():
    """Initialize ROS2 and create subscriber node"""
    global _ros2_node, _ros2_executor, _ros2_thread
    
    if _ros2_node is not None:
        return _ros2_node
    
    if not rclpy.ok():
        rclpy.init()
    
    _ros2_node = ROS2ImageSubscriber()
    _ros2_executor = rclpy.executors.SingleThreadedExecutor()
    _ros2_executor.add_node(_ros2_node)
    
    # Spin in a separate thread
    def spin_thread():
        while rclpy.ok():
            _ros2_executor.spin_once(timeout_sec=0.1)
    
    _ros2_thread = threading.Thread(target=spin_thread, daemon=True)
    _ros2_thread.start()

    # 节点启动后初始化 recorder
    time.sleep(0.3)
    _ros2_node.init_recorder()
    return _ros2_node


def shutdown_ros2():
    """Shutdown ROS2"""
    global _ros2_node, _ros2_executor, _ros2_thread
    
    if _ros2_executor is not None:
        _ros2_executor.shutdown()
    if _ros2_node is not None:
        _ros2_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    
    _ros2_node = None
    _ros2_executor = None
    _ros2_thread = None


def capture_realsense_frame():
    """Capture single aligned RGB-D frame directly from pyrealsense2."""
    global _ros2_node

    if _ros2_node is None:
        init_ros2()

    print("Waiting for RealSense RGB-D frames...")

    color_image, depth_image, depth_scale = _ros2_node.wait_for_frame(timeout=10.0)

    if color_image is None or depth_image is None:
        raise RuntimeError(
            "Failed to receive frames from RealSense camera. "
            "Check camera connection, permissions, and pyrealsense2 installation."
        )

    print(f"Frame captured from RealSense. Depth scale: {depth_scale}")

    return color_image, depth_image, depth_scale


def _load_sam3_config():
    config_path = os.path.join(SAM3_DOCKER_DIR, "config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load SAM3 config {config_path}: {e}")
        return {}


def _resolve_sam3_checkpoint(raw_checkpoint=""):
    env_checkpoint = os.getenv("SAM3_CHECKPOINT", "").strip()
    candidate = env_checkpoint or str(raw_checkpoint or "").strip()

    if candidate and os.path.isfile(candidate):
        return candidate

    if candidate.startswith("/workspace/"):
        mapped = os.path.join(SAM3_DOCKER_DIR, candidate[len("/workspace/"):])
        if os.path.isfile(mapped):
            return mapped

    if candidate:
        mapped = os.path.join(SAM3_DOCKER_DIR, candidate)
        if os.path.isfile(mapped):
            return mapped

    fallback = os.path.join(SAM3_DOCKER_DIR, "model", "sam3.pt")
    if os.path.isfile(fallback):
        return fallback

    raise FileNotFoundError(
        "SAM3 checkpoint not found. Set SAM3_CHECKPOINT or place sam3.pt under "
        f"{os.path.join(SAM3_DOCKER_DIR, 'model')}"
    )


def _parse_semantic_prompts(prompts_text):
    if prompts_text is None:
        return []
    if isinstance(prompts_text, (list, tuple)):
        raw_items = prompts_text
    else:
        raw = str(prompts_text).replace("，", ",").replace(";", ",").replace("\n", ",")
        raw_items = raw.split(",")
    prompts = []
    seen = set()
    for item in raw_items:
        prompt = str(item).strip()
        if not prompt:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
    return prompts


class Sam3SemanticSegmenter:
    def __init__(self, device, checkpoint_path=None, score_threshold=None):
        if build_sam3_image_model is None or Sam3Processor is None:
            raise RuntimeError(f"SAM3 import failed: {SAM3_IMPORT_ERROR}")

        cfg = _load_sam3_config()
        sam3_cfg = cfg.get("sam3", {}) if isinstance(cfg, dict) else {}
        raw_checkpoint = checkpoint_path or sam3_cfg.get("checkpoint_path", "")
        self.checkpoint_path = _resolve_sam3_checkpoint(raw_checkpoint)
        self.score_threshold = float(
            os.getenv(
                "SAM3_SCORE_THRESHOLD",
                str(score_threshold if score_threshold is not None else sam3_cfg.get("score_threshold", 0.4)),
            )
        )
        self.device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"

        print(f"Loading SAM3 model from {self.checkpoint_path} on {self.device}...")
        t0 = time.time()
        self.model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
            device=self.device,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.score_threshold,
        )
        print(f"SAM3 model loaded in {time.time() - t0:.2f}s, score_threshold={self.score_threshold}")

    @torch.inference_mode()
    def segment(self, color_image_bgr, prompts):
        prompts = _parse_semantic_prompts(prompts)
        if not prompts:
            return [], [], [], []

        image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        t0 = time.time()
        inference_state = self.processor.set_image(image)
        embed_sec = time.time() - t0

        masks = []
        labels = []
        detections = []
        prompt_timings = []
        det_id = 1

        for prompt in prompts:
            p0 = time.time()
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            prompt_timings.append({"prompt": prompt, "elapsed_sec": round(time.time() - p0, 4)})

            current_masks = output.get("masks")
            current_boxes = output.get("boxes")
            current_scores = output.get("scores")
            if current_masks is None or current_scores is None:
                continue

            current_masks = current_masks.detach().cpu().numpy()
            current_scores = current_scores.detach().cpu().numpy()
            if current_boxes is not None:
                current_boxes = current_boxes.detach().cpu().numpy()

            for i, score_raw in enumerate(current_scores):
                score = float(score_raw)
                if score <= self.score_threshold:
                    continue

                mask = np.squeeze(current_masks[i])
                binary_mask = (mask > 0.5)
                if np.count_nonzero(binary_mask) == 0:
                    continue

                if current_boxes is not None and i < len(current_boxes):
                    bbox = [float(v) for v in current_boxes[i].tolist()]
                else:
                    y_coords, x_coords = np.where(binary_mask)
                    bbox = [
                        float(x_coords.min()),
                        float(y_coords.min()),
                        float(x_coords.max()),
                        float(y_coords.max()),
                    ]

                masks.append(binary_mask)
                labels.append(prompt)
                detections.append(
                    {
                        "id": det_id,
                        "label": prompt,
                        "score": score,
                        "bbox": bbox,
                    }
                )
                det_id += 1

        print(
            f"SAM3 semantic segmentation: prompts={prompts}, detections={len(detections)}, "
            f"embed={embed_sec:.3f}s"
        )
        return masks, labels, detections, prompt_timings


def initialize_sam3(device):
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return Sam3SemanticSegmenter(device)


def create_combined_mask(masks):
    """Convert SAM3 masks to FlowPose combined-mask format with unique IDs"""
    if not masks:
        return None, None
    
    h, w = masks[0].shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    obj_ids = []
    
    for i, mask in enumerate(masks):
        obj_id = i + 1
        combined_mask[mask > 0] = obj_id
        obj_ids.append([obj_id, obj_id])  # [mask_label, box_id]
    
    return combined_mask, obj_ids


def visualize_sam_results(color_image, masks, labels):
    """Visualize SAM segmentation results"""
    vis = color_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = colors[i % len(colors)]
        # Create colored overlay
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = color
        vis = cv2.addWeighted(vis, 1, overlay, 0.4, 0)
        
        # Find centroid for label
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            cy, cx = int(y_coords.mean()), int(x_coords.mean())
            cv2.putText(vis, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
    
    return vis


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)

def auto_reuse_points_with_check(state_dict, iou_min_th=0.4, iou_mean_th=0.6):
    """SAM3语义模式不再使用上一帧点提示复用。"""
    return None


def create_gradio_interface():
    """创建Gradio界面"""
    
    # 全局状态
    state_dict = {
        "sam3_segmenter": None,
        "sam3_detections": [],
        "flowpose_estimator": None,
        "color_image": None,
        "depth_image": None,
        "depth_scale": None,
        "masks": None,
        "labels": None,
        "device": None,
        "all_labels": [],  # 存储所有对象的标签
        "temp_image": None,  # 用于显示标注的临时图像
        "label_history": [],
        "prev_masks": None,
        "prev_labels": None,
        "prev_points": None,  # 上一帧点提示
        # Action execution state
        "pose_results": None,
        "length_results": None,
        "data_results": None,
        "task_yaml": None,
        "object_yaml": None,
        "current_task_idx": 0,
        "action_state": "idle",  # idle, running, waiting
        "template_handlers": {},
        "last_executed_kps": None,  # Store last executed keypoints for redo
        "use_lingbot_depth": False,   # 新增：是否启用 lingbot 深度优化
        "trackers": [],
        "tracking_enabled": False,
    }
    
    # Default YAML paths
    DEFAULT_TASK_YAML = "/home/snorlax/work/gradio_sam_genpose/translator_1/data/0603/box/task.yaml"
    DEFAULT_OBJECT_YAML = "/home/snorlax/work/gradio_sam_genpose/translator_1/data/0603/box/test.yaml"
    DEFAULT_SAM3_PROMPTS = os.getenv(
        "SAM3_PROMPTS",
        "basket, remote control, grey screwdriver, yellow screwdriver",
    )

    def init_models():
        """初始化模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict["device"] = device
        
        # 确保ROS2已初始化
        init_ros2()
        
        # 初始化SAM3语义分割
        state_dict["sam3_segmenter"] = initialize_sam3(device)
        
        # 初始化FlowPose姿态估计
        state_dict["flowpose_estimator"] = FlowPoseEstimator(device)
        
        msg_status = "✅" if HAS_KEYPOINT_MSG else "⚠️ (无KeypointPose消息)"
        return f"✅ 模型初始化完成 (设备: {device})\n🧠 SAM3语义分割已就绪\n🎯 FlowPose姿态估计已就绪\n📡 ROS2节点已启动 {msg_status}"
    
    def capture_frame():
        """捕获RealSense帧，后续由SAM3语义prompt分割。"""
        color_image, depth_image, depth_scale = capture_realsense_frame()
        state_dict["color_image"] = color_image
        state_dict["depth_image"] = depth_image
        state_dict["depth_scale"] = depth_scale
        state_dict["temp_image"] = color_image.copy()
        state_dict["masks"] = None
        state_dict["labels"] = None
        state_dict["all_labels"] = []
        state_dict["sam3_detections"] = []
        state_dict["tracking_enabled"] = False
        state_dict["trackers"] = []

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return rgb_image, f"✅ 已捕获RealSense图像 (深度缩放: {depth_scale:.6f})"
    
    def run_sam_segmentation(prompts_text):
        """使用SAM3语义prompt直接分割目标。"""
        if state_dict["sam3_segmenter"] is None:
            return None, "❌ 请先初始化模型"
        if state_dict["color_image"] is None:
            return None, "❌ 请先捕获图像"

        prompts = _parse_semantic_prompts(prompts_text)
        if not prompts:
            return None, "❌ 请输入至少一个语义目标，例如: basket, remote control"

        try:
            masks, labels, detections, prompt_timings = state_dict["sam3_segmenter"].segment(
                state_dict["color_image"],
                prompts,
            )
        except Exception as e:
            return None, f"❌ SAM3分割失败: {e}"

        state_dict["masks"] = masks if masks else None
        state_dict["labels"] = labels if labels else None
        state_dict["all_labels"] = list(labels)
        state_dict["sam3_detections"] = detections
        state_dict["prev_masks"] = None
        state_dict["prev_labels"] = None
        state_dict["prev_points"] = None
        state_dict["trackers"] = []
        state_dict["tracking_enabled"] = False
        for label in labels:
            if label not in state_dict["label_history"]:
                state_dict["label_history"].append(label)

        if not masks:
            rgb_image = cv2.cvtColor(state_dict["color_image"], cv2.COLOR_BGR2RGB)
            return rgb_image, f"⚠️ SAM3未检测到目标: {', '.join(prompts)}"

        vis = visualize_sam_results(state_dict["color_image"], masks, labels)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        timing_text = ", ".join(
            f"{item['prompt']}={item['elapsed_sec']:.3f}s" for item in prompt_timings
        )
        return (
            vis_rgb,
            f"✅ SAM3语义分割完成: prompts={len(prompts)}, detections={len(detections)}\n"
            f"目标: {', '.join(labels)}\n"
            f"耗时: {timing_text}"
        )
    
    def run_pose_estimation():
        """使用FlowPose运行姿态估计并发布TF。"""
        global _ros2_node
        
        if state_dict["masks"] is None:
            return None, "❌ 请先运行SAM3分割"
        if state_dict["flowpose_estimator"] is None:
            return None, "❌ 请先初始化模型"
        if state_dict["color_image"] is None or state_dict["depth_image"] is None:
            return None, "❌ 请先捕获RealSense图像"
        
        combined_mask, obj_ids = create_combined_mask(state_dict["masks"])
        if combined_mask is None or not obj_ids:
            return None, "❌ 当前mask为空，无法运行FlowPose"

        labels = list(state_dict.get("labels") or [])
        try:
            flowpose_result = state_dict["flowpose_estimator"].infer(
                rgb=state_dict["color_image"],
                depth=state_dict["depth_image"],
                combined_mask=combined_mask,
                obj_ids=obj_ids,
                class_names=labels,
                instance_names=labels,
                depth_scale=state_dict.get("depth_scale") or 0.001,
            )
        except Exception as e:
            return None, f"❌ FlowPose姿态估计失败: {e}"

        pose_all = flowpose_result.get("pose_all")
        length_all = flowpose_result.get("length_all")
        objects = flowpose_result.get("objects", [])

        state_dict["pose_results"] = pose_all
        state_dict["length_results"] = length_all
        state_dict["data_results"] = flowpose_result

        if pose_all is not None and _ros2_node is not None:
            all_final_pose = np.asarray(pose_all, dtype=np.float32)
            _ros2_node.publish_poses_as_tf(all_final_pose, labels)

        vis = state_dict["flowpose_estimator"].visualize(
            state_dict["color_image"],
            pose_all,
            length_all,
            labels,
        )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        return (
            vis_rgb,
            f"✅ FlowPose姿态估计完成, objects={len(objects)}, elapsed={flowpose_result.get('elapsed_sec', 0.0)}s\n"
            f"📡 已发布TF到 /tf 话题"
        )
    
    # ==================== Action Execution Functions ====================
    
    def load_yaml_configs(task_yaml_path, object_yaml_path):
        """加载YAML配置文件"""
        task_data = load_yaml_file(task_yaml_path)
        object_data = load_yaml_file(object_yaml_path)
        
        if task_data is None:
            return "❌ 无法加载任务YAML文件", "", gr.update(choices=[])
        if object_data is None:
            return "❌ 无法加载对象YAML文件", "", gr.update(choices=[])
        
        # Parse task yaml
        if isinstance(task_data, dict) and "tasks" in task_data:
            state_dict["task_yaml"] = task_data["tasks"]
        else:
            state_dict["task_yaml"] = task_data if isinstance(task_data, list) else []
        # 记录原始任务，便于展开/重置
        state_dict["task_yaml_base"] = copy.deepcopy(state_dict["task_yaml"])
        
        # Parse object yaml and create handlers
        state_dict["object_yaml"] = object_data
        templates = object_data.get("templates", {})
        state_dict["template_handlers"] = {k: ActionHandler(v, node=_ros2_node) for k, v in templates.items()}
        
        state_dict["current_task_idx"] = 0
        
        # Get task list for dropdown
        task_choices = []
        for i, task in enumerate(state_dict["task_yaml"]):
            task_str = f"{i}: {task.get('target', 'unknown')} - {task.get('action_name', 'unknown')}"
            task_choices.append(task_str)
        
        return (f"✅ 已加载 {len(state_dict['task_yaml'])} 个任务", 
                get_current_task_info(),
                gr.update(choices=task_choices, value=task_choices[0] if task_choices else None))
    
    def select_task(selected_str):
        """根据下拉框选择切换当前任务索引"""
        if not selected_str or not state_dict.get("task_yaml"):
            return get_current_task_info()
        try:
            idx = int(str(selected_str).split(":")[0].strip())
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(state_dict["task_yaml"]) - 1))
        state_dict["current_task_idx"] = idx
        return get_current_task_info()

    def get_current_task_info():
        """获取当前任务信息"""
        if state_dict["task_yaml"] is None or len(state_dict["task_yaml"]) == 0:
            return "未加载任务"
        
        idx = state_dict["current_task_idx"]
        if idx >= len(state_dict["task_yaml"]):
            return "✅ 所有任务已完成"
        
        task = state_dict["task_yaml"][idx]
        info = f"""📋 **任务 {idx + 1}/{len(state_dict["task_yaml"])}**
- **Target**: {task.get('target', 'N/A')}
- **Action**: {task.get('action_name', 'N/A')}
- **Arm**: {task.get('arm', 'N/A')}
- **Speed**: {task.get('speed', 'N/A')}
- **Correction Mode**: {task.get('correction_mode', 'N/A')}"""
        return info
    
    # === 工具函数：匹配 TF 列表 ===
    def find_tf_frames_for_target(labels, target):
        frames = []
        for i, label in enumerate(labels):
            if target and (target.lower() in label.lower() or label.lower() in target.lower()):
                frames.append(f"{label}_{i+1}")
        if not frames:
            frames.append(f"{target}_1")
        return frames

    # === 新增：展开多实例成对任务到列表 ===
    def expand_tasks_multi_instances():
        """按实例展开成对的 pick→place，生成新的任务列表"""
        base_tasks = state_dict.get("task_yaml_base") or state_dict.get("task_yaml") or []
        labels = state_dict.get("labels", [])
        if not base_tasks:
            return "❌ 请先加载任务YAML", get_current_task_info(), gr.update()
        if not labels:
            return "❌ 请先完成姿态估计以获得TF实例", get_current_task_info(), gr.update()

        expanded = []
        i = 0
        while i < len(base_tasks):
            t = base_tasks[i]
            # 成对 pick → place
            if t.get("action_name") == "pick" and (i + 1) < len(base_tasks) and base_tasks[i + 1].get("pair_type") == "pair":
                place = base_tasks[i + 1]
                pick_target = t.get("target")
                place_target = place.get("target")

                pick_frames = find_tf_frames_for_target(labels, pick_target)
                place_frames = find_tf_frames_for_target(labels, place_target)
                place_frame = place_frames[0] if place_frames else f"{place_target}_1"

                for pf in pick_frames:
                    t_copy = copy.deepcopy(t)
                    t_copy["template_target"] = pick_target  # 原模板名
                    t_copy["target"] = pf                    # 实例名
                    expanded.append(t_copy)

                    p_copy = copy.deepcopy(place)
                    p_copy["template_target"] = place_target
                    p_copy["target"] = place_frame            # 复用同一容器实例
                    expanded.append(p_copy)
                i += 2
            else:
                expanded.append(copy.deepcopy(t))
                i += 1

        state_dict["task_yaml"] = expanded
        state_dict["current_task_idx"] = 0

        task_choices = [f"{idx}: {tsk.get('target','unknown')} - {tsk.get('action_name','unknown')}"
                        for idx, tsk in enumerate(expanded)]
        return (
            f"✅ 已展开多实例任务，共 {len(expanded)} 步",
            get_current_task_info(),
            gr.update(choices=task_choices, value=task_choices[0] if task_choices else None)
        )

    # === 工具函数：执行单个动作 ===
    def execute_action_on_tf(task, tf_frame, handler, ts, last_arm_hint=None):
        action_name = task.get("action_name")
        arm = task.get("arm")
        speed = task.get("speed", 1.0)
        correction_mode = task.get("correction_mode", "none")

        task_arm = arm
        if task_arm == "copy last one":
            task_arm = last_arm_hint if last_arm_hint else (_ros2_node.last_arm if _ros2_node and _ros2_node.last_arm else "right")
        elif task_arm == "correct":
            y_value = ts.transform.translation.y
            task_arm = "left" if y_value > 0 else "right"

        if handler is None:
            return False, f"⚠️ 未找到模板: {task.get('target')}"
        kps = handler.process_for_object(ts.transform, action_name, task_arm=task_arm, task_speed=speed, correction_mode=correction_mode)
        if not kps:
            return False, f"⚠️ 未生成关键点: {action_name} ({tf_frame})"
        ok = _ros2_node.publish_kp_separately(kps) if _ros2_node else False
        if ok:
            _ros2_node.last_arm = task_arm
        return ok, f"{'✅' if ok else '⚠️'} {action_name} -> {tf_frame} (arm={task_arm}, speed={speed})"

    def execute_current_task():
        """执行当前任务并发布ROS2消息"""
        global _ros2_node

        if state_dict["task_yaml"] is None:
            return "❌ 请先加载任务配置", get_current_task_info()
        if _ros2_node is None:
            return "❌ ROS2节点未初始化", get_current_task_info()

        idx = state_dict["current_task_idx"]
        if idx >= len(state_dict["task_yaml"]):
            return "✅ 所有任务已完成", get_current_task_info()

        task = state_dict["task_yaml"][idx]
        target = task.get("target")
        base_target = task.get("template_target", target)  # 用于匹配模板
        action_name = task.get("action_name")
        labels = state_dict.get("labels", [])

        # -------- 移除原先自动成对循环，改为逐步执行列表 --------
        # 解析 TF：若 target 本身是具体实例名，直接用它
        tf_frame = None
        # 直接命中
        for lbl in labels:
           
            if tf_frame:
                break
            for j in range(1, 10):
                if target == f"{lbl}_{j}":
                    tf_frame = target
                    break
        # 模糊匹配
        if tf_frame is None:
            frames = find_tf_frames_for_target(labels, target)
            tf_frame = frames[0] if frames else f"{target}_1"

        ts = _ros2_node.lookup_transform('base_link', tf_frame, timeout=1.0)
        if ts is None:
            return f"❌ 无法找到目标 '{target}' 的TF变换 ({tf_frame})\n请确保已运行姿态估计", get_current_task_info()

        task_arm = task.get("arm")
        speed = task.get("speed", 1.0)
        correction_mode = task.get("correction_mode", "none")
        if task_arm == "copy last one":
            task_arm = _ros2_node.last_arm if _ros2_node and _ros2_node.last_arm else "right"
        elif task_arm == "correct":
            y_value = ts.transform.translation.y
            task_arm = "left" if y_value > 0 else "right"

        handler = state_dict["template_handlers"].get(base_target)
        if handler is None:
            return f"⚠️ 未找到目标 '{base_target}' 的动作模板", get_current_task_info()

        kps = handler.process_for_object(ts.transform, action_name, task_arm=task_arm, task_speed=speed, correction_mode=correction_mode)
        if not kps:
            return f"⚠️ 动作 '{action_name}' 未生成任何关键点", get_current_task_info()

        state_dict["last_executed_kps"] = kps
        publish_success = _ros2_node.publish_kp_separately(kps)

        # ---- 执行记录器：发布成功后立即开始记录 ----
        if publish_success and _ros2_node.recorder is not None:
            labels = state_dict.get("labels", [])
            _ros2_node.recorder.start(
                target_frame=tf_frame,
                labels=labels,
                arm=task_arm
            )

        if publish_success and task.get("finish_home", False):
            if kps and len(kps) > 0:
                actual_arm = kps[0].get("arm")
                threading.Thread(
                    target=lambda: (time.sleep(6), _ros2_node.publish_home_separately(actual_arm)),
                    daemon=True
                ).start()

        if publish_success:
            if state_dict["current_task_idx"] < len(state_dict["task_yaml"]):
                state_dict["current_task_idx"] += 1
            log_msg = f"✅ 执行任务 {idx + 1} 并已发布:\n- Target: {target} ({tf_frame})\n- Action: {action_name}\n- Arm: {task_arm}\n- Speed: {speed}\n- 发布 {len(kps)} 条到 /fusion_pose"
        else:
            log_msg = f"⚠️ 执行任务 {idx + 1} (发布失败):\n- Target: {target} ({tf_frame})\n- Action: {action_name}"
        return log_msg, get_current_task_info()

    def action_next():
        """执行当前任务并移到下一步 (对应 'c')"""
        global _ros2_node
        
        if state_dict["task_yaml"] is None:
            return "❌ 请先加载任务配置", get_current_task_info()

        
        # 直接执行当前任务，索引自增在 execute_current_task 内完成
        exec_result, _ = execute_current_task()
        
        # 如果已经是最后一个，保持提示
        if state_dict["current_task_idx"] >= len(state_dict["task_yaml"]):
            return f"{exec_result}\n\n✅ 已是最后一个任务", get_current_task_info()
        else:
            return f"{exec_result}\n\n⏭️ 已移至任务 {state_dict['current_task_idx'] + 1}", get_current_task_info()
    
    def action_redo():
        """重做上一任务 (对应 'r')"""
        if state_dict["last_executed_kps"] and _ros2_node:
            ok = _ros2_node.publish_kp_separately(state_dict["last_executed_kps"])
            msg = "✅ 已重做上一任务" if ok else "⚠️ 重做失败（发布失败）"
        else:
            msg = "⚠️ 无可重做的任务"
        return msg, get_current_task_info()

    def clear_inference_cache():
        """清理一次 GenPose 推理缓存并释放显存"""
        state_dict["pose_results"] = None
        state_dict["length_results"] = None
        state_dict["data_results"] = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"clear cache warn: {e}")
        return "🧹 已清理GenPose缓存", get_current_task_info()

    def action_stop():
        """停止并回 Home (对应 's')"""
        if _ros2_node:
            _ros2_node.publish_home_both()
        state_dict["current_task_idx"] = 0
        msg_cache, info = clear_inference_cache()
        return f"🛑 已停止并发布 Home\n{msg_cache}", info
    
    # ==================== Gradio Interface ====================
    
    with gr.Blocks(title="SAM3 + FlowPose 6D姿态估计 + 动作执行") as demo:
        gr.Markdown("# 🎯 SAM3 + FlowPose 6D姿态估计 + 动作执行系统")
        gr.Markdown("### 步骤: 1️⃣ 初始化 → 2️⃣ 捕获RealSense → 3️⃣ 输入语义目标 → 4️⃣ SAM3分割 → 5️⃣ FlowPose姿态估计 → 6️⃣ 动作执行")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 模型初始化")
                init_btn = gr.Button("🔧 初始化模型", variant="primary", size="lg")
                init_status = gr.Textbox(label="初始化状态", interactive=False)
                
                gr.Markdown("### 2. 图像捕获")
                capture_btn = gr.Button("📷 捕获RealSense图像", variant="primary")
                capture_status = gr.Textbox(label="捕获状态", interactive=False, lines=2)
            
            with gr.Column(scale=2):
                gr.Markdown("### 3. SAM3语义目标")
                semantic_prompts_input = gr.Textbox(
                    label="语义目标（逗号或换行分隔）",
                    value=DEFAULT_SAM3_PROMPTS,
                    placeholder="basket, remote control, grey screwdriver",
                    lines=3,
                )
                sam_btn = gr.Button("🎨 运行SAM3语义分割", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_display = gr.Image(
                    label="RealSense图像",
                    type="numpy",
                    interactive=False,
                )
            with gr.Column(scale=1):
                sam_output = gr.Image(label="SAM3分割结果", type="numpy")
                sam_status = gr.Textbox(label="分割状态", interactive=False, lines=4)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 5. FlowPose姿态估计")
                pose_btn = gr.Button("🎯 运行FlowPose姿态估计", variant="primary", size="lg")
                pose_output = gr.Image(label="姿态估计结果", type="numpy")
                pose_status = gr.Textbox(label="估计状态", interactive=False)
        
        # ==================== Action Execution Section ====================
        gr.Markdown("---")
        gr.Markdown("## 🤖 6. 动作执行")
        gr.Markdown("💡 **按钮说明**: Execute=执行当前任务 | Next=执行并跳转下一任务 | Redo=重做上一任务 | Stop=停止并回Home")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 配置加载")
                task_yaml_input = gr.Textbox(
                    label="Task YAML路径", 
                    value=DEFAULT_TASK_YAML,
                    placeholder="translator_1/data/task_plate.yaml"
                )
                object_yaml_input = gr.Textbox(
                    label="Object YAML路径", 
                    value=DEFAULT_OBJECT_YAML,
                    placeholder="translator_1/data/test_plate.yaml"
                )
                load_yaml_btn = gr.Button("📂 加载YAML配置", variant="primary")
                expand_btn = gr.Button("🔀 展开多实例", variant="secondary")
                yaml_status = gr.Textbox(label="加载状态", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 任务选择")
                task_dropdown = gr.Dropdown(
                    label="选择任务",
                    choices=[],
                    interactive=True
                )
                current_task_display = gr.Markdown("未加载任务")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 执行控制")
                with gr.Row():
                    execute_btn = gr.Button("▶️ 执行 (Execute)", variant="primary", size="lg")
                    next_btn = gr.Button("⏭️ 下一步 (Next/c)", variant="primary", size="lg")
                with gr.Row():
                    redo_btn = gr.Button("🔄 重做 (Redo/r)", variant="secondary", size="lg")
                    stop_btn = gr.Button("🛑 停止 (Stop/s)", variant="stop", size="lg")
                    clear_btn = gr.Button("🧹 清理缓存", variant="secondary", size="md")
                action_log = gr.Textbox(label="执行日志", interactive=False, lines=8)
        
        # ==================== 事件绑定 ====================
        init_btn.click(fn=init_models, outputs=init_status)
        capture_btn.click(fn=capture_frame, outputs=[image_display, capture_status])
        sam_btn.click(fn=run_sam_segmentation, inputs=semantic_prompts_input, outputs=[sam_output, sam_status])
        pose_btn.click(fn=run_pose_estimation, outputs=[pose_output, pose_status])
        
        # Action execution bindings
        load_yaml_btn.click(
            fn=load_yaml_configs, 
            inputs=[task_yaml_input, object_yaml_input],
            outputs=[yaml_status, current_task_display, task_dropdown]
        )
        task_dropdown.change(
            fn=select_task,
            inputs=[task_dropdown],
            outputs=[current_task_display]
        )
        execute_btn.click(
            fn=execute_current_task,
            outputs=[action_log, current_task_display]
        )
        stop_btn.click(
            fn=action_stop,
            outputs=[action_log, current_task_display]
        )
        redo_btn.click(
            fn=action_redo,
            outputs=[action_log, current_task_display]
        )
        next_btn.click(
            fn=action_next,
            outputs=[action_log, current_task_display]
        )
        expand_btn.click(
            fn=expand_tasks_multi_instances,
            outputs=[yaml_status, current_task_display, task_dropdown]
        )
        clear_btn.click(fn=clear_inference_cache, outputs=[action_log, current_task_display])
    
    return demo


# filepath: 
try:
    from sam_gp_gradio_ros_tr_action_dino import ActionHandler, load_yaml_file
except ImportError:
    ActionHandler = None
    def load_yaml_file(*args, **kwargs):
        return None


def main():
    try:
        # Initialize ROS2 at startup
        init_ros2()
        print("ROS2 initialized. RealSense capture uses pyrealsense2 directly.")
        
        demo = create_gradio_interface()
        demo.launch()
    finally:
        shutdown_ros2()


if __name__ == '__main__':
    main()
