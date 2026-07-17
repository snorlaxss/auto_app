import os
import sys
import argparse
import ast
import re
import importlib.machinery
import types
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import gradio as gr
import threading
import yaml
import time
import json
import ctypes
import contextlib  # 补齐
import copy
import subprocess
from collections import deque
from types import SimpleNamespace
try:
    from torchvision.transforms import functional as TVF, InterpolationMode
    SIGLIP_TORCHVISION_IMPORT_ERROR = None
except Exception as e:
    TVF = None
    InterpolationMode = None
    SIGLIP_TORCHVISION_IMPORT_ERROR = e
try:
    from transformers import AutoImageProcessor, AutoModel
    SIGLIP_TRANSFORMERS_IMPORT_ERROR = None
except Exception as e:
    AutoImageProcessor = None
    AutoModel = None
    SIGLIP_TRANSFORMERS_IMPORT_ERROR = e
AUTO_APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_APP_CONFIG_PATH = os.getenv(
    "AUTO_APP_CONFIG",
    os.path.join(AUTO_APP_DIR, "config.yaml"),
)


def _ensure_localhost_no_proxy():
    localhost_hosts = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        changed = False
        for host in localhost_hosts:
            if host not in parts:
                parts.append(host)
                changed = True
        if changed or not existing:
            os.environ[key] = ",".join(parts)


_ensure_localhost_no_proxy()
MARVIN_DOCKER_DIR = os.path.join(AUTO_APP_DIR, "MarvinDocker")
ROBOTACTION_DIR = os.path.join(MARVIN_DOCKER_DIR, "robotaction")
ROBOTACTION_INSTALL_DIR = os.path.join(ROBOTACTION_DIR, "install")
SIGLIP_DOCKER_DIR = os.path.join(AUTO_APP_DIR, "SiglipDocker")
SIGLIP_CONFIG_PATH = os.path.join(SIGLIP_DOCKER_DIR, "config.yaml")

# Let the in-repo Marvin robotaction install provide fake_interface_pkg when
# the user's shell has not sourced that ROS2 overlay yet.
for _lib_dir in Path(ROBOTACTION_INSTALL_DIR).glob("*/lib"):
    for _so_name in (
        "libfake_interface_pkg__rosidl_generator_c.so",
        "libfake_interface_pkg__rosidl_generator_py.so",
        "libfake_interface_pkg__rosidl_typesupport_c.so",
        "libfake_interface_pkg__rosidl_typesupport_fastrtps_c.so",
        "libfake_interface_pkg__rosidl_typesupport_introspection_c.so",
    ):
        _so_path = _lib_dir / _so_name
        if _so_path.is_file():
            with contextlib.suppress(Exception):
                ctypes.CDLL(str(_so_path), mode=ctypes.RTLD_GLOBAL)
for _site_packages in Path(ROBOTACTION_INSTALL_DIR).glob("*/lib/python*/site-packages"):
    _site_path = str(_site_packages)
    if _site_path not in sys.path:
        sys.path.insert(0, _site_path)
if os.path.isdir(MARVIN_DOCKER_DIR) and MARVIN_DOCKER_DIR not in sys.path:
    sys.path.insert(0, MARVIN_DOCKER_DIR)

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
from std_msgs.msg import Int16MultiArray, String
# Import custom ROS2 messages
try:
    from fake_interface_pkg.msg import KeypointPose, KeypointPoseArray
    HAS_KEYPOINT_MSG = True
except ImportError:
    print("Warning: fake_interface_pkg not found. Action publishing will be disabled.")
    HAS_KEYPOINT_MSG = False

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

try:
    from rclpy.callback_groups import ReentrantCallbackGroup
    from robotaction.robot_action import FusionNode as MarvinFusionNode
    from robotaction.common import (
        ActionHandler,
        ArmResolver,
        FixedEndpointManager,
        Step,
        TaskStatus,
        extract_siglip_payload,
        normalize_frame_id,
        normalize_obj_name,
        normalize_state_text,
        transform_to_matrix,
    )
    from robotaction.execution import FusionExecutionMixin
    from robotaction.tf_helpers import FusionTfMixin
    from robotaction.watchers import ObjectTfWatcher, TaskProgressWatcher
    ROBOTACTION_IMPORT_ERROR = None
except Exception as e:
    MarvinFusionNode = None
    ActionHandler = None
    ArmResolver = None
    FixedEndpointManager = None
    Step = None
    FusionExecutionMixin = object
    FusionTfMixin = object
    ObjectTfWatcher = None
    TaskProgressWatcher = None
    normalize_frame_id = lambda value: str(value or "").strip().lstrip("/")
    normalize_obj_name = lambda value: str(value or "").strip().lower().replace("_", " ")
    normalize_state_text = lambda value: str(value or "").strip().lower()
    extract_siglip_payload = lambda payload: payload if isinstance(payload, dict) else None
    transform_to_matrix = None
    TaskStatus = SimpleNamespace(STOP="stop")
    ROBOTACTION_IMPORT_ERROR = e

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


_AUTO_APP_CONFIG_CACHE = None


def _load_yaml_config_file(config_path):
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load config {config_path}: {e}")
        return {}


def _load_auto_app_config():
    global _AUTO_APP_CONFIG_CACHE
    if _AUTO_APP_CONFIG_CACHE is None:
        _AUTO_APP_CONFIG_CACHE = _load_yaml_config_file(AUTO_APP_CONFIG_PATH)
    return _AUTO_APP_CONFIG_CACHE


def _resolve_path_from_roots(raw_path="", roots=()):
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if os.path.exists(candidate):
        return candidate
    for root in roots:
        if not root:
            continue
        mapped = os.path.join(root, candidate)
        if os.path.exists(mapped):
            return mapped
    return candidate


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
    auto_cfg = _load_auto_app_config()
    flowpose_cfg = auto_cfg.get("flowpose", {}) if isinstance(auto_cfg, dict) else {}
    if flowpose_cfg:
        return flowpose_cfg

    config_path = os.path.join(FLOWPOSE_DOCKER_DIR, "config.yaml")
    return _load_yaml_config_file(config_path)


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
    mapped = os.path.join(AUTO_APP_DIR, candidate)
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
        self.axis_len = float(vis_cfg.get("axis_len", 0.035))
        self.device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.last_segment_stats = {}

        args = argparse.Namespace(
            pretrained_flow_model_path=_resolve_flowpose_path(paths_cfg.get("pretrained_flow_model_path", "")),
            pretrained_scale_model_path=_resolve_flowpose_path(paths_cfg.get("pretrained_scale_model_path", "")),
            device=self.device,
            img_size=224,
            n_pts=1024,
            frame_gap_threshold=10,
            eval_repeat_num=25,
            retain_ratio=0.4,
            enable_tracking=False,
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
            repeat_num=25,
            clustering=1,
            clustering_eps=0.01,
            clustering_minpts=0.1667,
        )

        print(f"[FlowPose] Loading Flow on {self.device}...")
        self.flow = Flow(args)
        local_dino_repo = _resolve_flowpose_path(
            paths_cfg.get("dino_repo_path", "/workspace/model/facebookresearch_dinov2_main")
        )
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

    def visualize(self, rgb, pose_all, length_all, labels=None, draw_labels=False):
        vis = rgb.copy()
        labels = labels or []
        if pose_all is not None and length_all is not None:
            poses = np.asarray(pose_all, dtype=np.float32)
            lengths = np.asarray(length_all, dtype=np.float32)
            if flowpose_visualize_detections is not None and len(poses) > 0 and len(lengths) > 0:
                raw_pose = getattr(self.inferencer, "last_raw_pose", None)
                bbox_poses = None
                if raw_pose is not None:
                    try:
                        if hasattr(raw_pose, "detach"):
                            raw_pose = raw_pose.detach().cpu().numpy()
                        bbox_poses = np.asarray(raw_pose, dtype=np.float32)
                        if bbox_poses.shape[0] != poses.shape[0]:
                            bbox_poses = None
                    except Exception:
                        bbox_poses = None

                vis = flowpose_visualize_detections(
                    vis,
                    poses,
                    lengths,
                    self.build_cam_intrinsics(),
                    color=None,
                    thickness=1,
                    axes_length=self.axis_len,
                    bbox_poses=bbox_poses,
                )
            if draw_labels:
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



class SiglipQuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SiglipResidualAttentionBlock(torch.nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=4.0):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, int(d_model * mlp_ratio)),
            SiglipQuickGELU(),
            torch.nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.ln_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class SiglipLASTViTVisionEncoder(torch.nn.Module):
    """LAST-ViT vision encoder copied from SiglipDocker ZeroMQServer_lastvit."""

    def __init__(self, embed_dim=512, image_size=224, patch_size=16,
                 width=768, layers=12, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.width = width
        scale = width ** -0.5

        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        num_patches = self.grid_size * self.grid_size
        self.class_embedding = torch.nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.Sequential(*[
            SiglipResidualAttentionBlock(width, heads, mlp_ratio) for _ in range(layers)
        ])
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(scale * torch.randn(width, embed_dim))
        self.register_buffer('_cached_gaussian', None, persistent=False)

    def _build_gaussian_kernel(self, device):
        w = self.width
        kernel = torch.exp(-0.5 * (torch.arange(-w // 2 + 1, w // 2 + 1, device=device).float() / (w ** 0.5)) ** 2)
        return (kernel / kernel.max()).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = self.conv1(x)
        bsz, channels, height, width = x.shape
        x = x.reshape(bsz, channels, height * width).permute(0, 2, 1).contiguous()
        cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.positional_embedding.unsqueeze(0)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self._cached_gaussian is None or self._cached_gaussian.device != x.device:
            self._cached_gaussian = self._build_gaussian_kernel(x.device)
        x_detach = x[:, 1:]
        x_f = torch.fft.fftshift(torch.fft.fft(x_detach, dim=-1), dim=-1)
        x_smooth = torch.fft.ifft(torch.fft.ifftshift(x_f * self._cached_gaussian, dim=-1), dim=-1).real
        diff = x_detach / torch.abs(x_smooth - x_detach).clamp(min=1e-8)
        _, idx = torch.topk(diff, k=1, dim=1, largest=True)
        x = torch.mean(torch.gather(x_detach, 1, idx), dim=1)
        x = self.ln_post(x)
        return x @ self.proj if self.proj is not None else x


def _load_siglip_config():
    config_path = os.getenv("SIGLIP_CONFIG", "").strip() or SIGLIP_CONFIG_PATH
    auto_cfg = _load_auto_app_config()
    siglip_cfg = auto_cfg.get("siglip", {}) if isinstance(auto_cfg, dict) else {}
    if siglip_cfg and not os.getenv("SIGLIP_CONFIG", "").strip():
        return siglip_cfg

    config_path = _resolve_siglip_path(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Siglip config not found: {config_path}")
    return _load_yaml_config_file(config_path)


def _resolve_siglip_path(raw_path=""):
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if os.path.exists(candidate):
        return candidate
    if candidate == "/workspace/config.yaml":
        return SIGLIP_CONFIG_PATH
    if candidate.startswith("/workspace/"):
        mapped = os.path.join(SIGLIP_DOCKER_DIR, candidate[len("/workspace/"):])
        if os.path.exists(mapped):
            return mapped
    mapped = os.path.join(AUTO_APP_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    mapped = os.path.join(SIGLIP_DOCKER_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    return candidate


def _siglip_parse_center_feature(value):
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except Exception:
            value = ast.literal_eval(text)

    arr = np.array(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"center feature must be 1D, got shape={arr.shape}")

    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def _siglip_jsonable_result(result):
    if isinstance(result, dict):
        return {str(k): _siglip_jsonable_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_siglip_jsonable_result(v) for v in result]
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, np.integer):
        return int(result)
    if isinstance(result, np.floating):
        return float(result)
    return result


class CrossViewAttentionPooler(torch.nn.Module):
    """Siglip V2 pooler from SiglipDocker ZeroMQServerema."""

    def __init__(self, embed_dim=1152, num_queries=8, num_heads=8,
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.query_tokens = torch.nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(torch.nn.ModuleDict({
                "cross_attn": torch.nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm1": torch.nn.LayerNorm(embed_dim),
                "norm2": torch.nn.LayerNorm(embed_dim),
                "ffn": torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, embed_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(embed_dim * 4, embed_dim),
                    torch.nn.Dropout(dropout),
                ),
            }))
        self.output_ln = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            residual = queries
            queries = layer["norm1"](queries)
            queries = layer["cross_attn"](
                query=queries,
                key=x,
                value=x,
            )[0] + residual

            residual = queries
            queries = layer["norm2"](queries)
            queries = layer["ffn"](queries) + residual

        output = queries.mean(dim=1)
        return self.output_ln(output)


class SingleViewSigLIPModel(torch.nn.Module):
    """Siglip V2 single-view model: base SigLIP patch tokens + trained pooler."""

    def __init__(self, base_model, pooler):
        super().__init__()
        self.base_model = base_model
        self.pooler = pooler
        self.config = base_model.config

    def encode(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state
        feature = self.pooler(patch_tokens)
        return feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-12)


class SiglipStateEstimator:
    def __init__(self, device=None):
        if TVF is None or InterpolationMode is None:
            raise RuntimeError(f"torchvision import failed: {SIGLIP_TORCHVISION_IMPORT_ERROR}")

        cfg = _load_siglip_config()
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        self.checkpoint_path = _resolve_siglip_path(
            os.getenv("SIGLIP_CHECKPOINT", "").strip() or model_cfg.get("checkpoint", "")
        )
        self.base_model_path = _resolve_siglip_path(
            os.getenv("SIGLIP_BASE_MODEL", "").strip()
            or model_cfg.get("path", "")
            or os.path.join(SIGLIP_DOCKER_DIR, "model", "siglip2-so400m-patch14-224")
        )
        self.graph_info_path = _resolve_siglip_path(
            os.getenv("SIGLIP_GRAPH_INFO", "").strip()
            or model_cfg.get("graph_info_file", "")
            or model_cfg.get("cache_file", "")
        )
        self.device = "cuda" if str(device or "").startswith("cuda") and torch.cuda.is_available() else "cpu"
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        inference_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
        self.topk = int(os.getenv("SIGLIP_TOPK", str(inference_cfg.get("topk", 5))))
        self.image_size = int(os.getenv("SIGLIP_IMAGE_SIZE", str(inference_cfg.get("image_size", 224))))
        self.pad_head_ratio = float(
            os.getenv("SIGLIP_PAD_HEAD_RATIO", str(inference_cfg.get("pad_head_ratio", 0.25)))
        )
        self.lock = threading.Lock()
        self.model_type = "unknown"
        self.processor = None

        self.model = self._load_model()
        self.centers, self.state_list = self._load_centers(self.graph_info_path)
        print(
            f"[Siglip] {self.model_type} ready on {self.device}, states={len(self.state_list)}, "
            f"checkpoint={self.checkpoint_path}, graph={self.graph_info_path}"
        )

    def _load_model(self):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Siglip checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        has_siglip2_state = any(str(key).startswith("base_model.") for key in state_dict)
        if has_siglip2_state:
            return self._load_siglip2_model(state_dict)

        model = SiglipLASTViTVisionEncoder(embed_dim=512)
        vision_state = {}
        for key, value in state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state[key[len("vision_encoder."):]] = value

        if not vision_state:
            keys = list(state_dict.keys())[:8]
            raise RuntimeError(
                f"Unsupported Siglip checkpoint format: no vision_encoder.* or base_model.* "
                f"weights found in {self.checkpoint_path}; first keys={keys}"
            )

        missing, unexpected = model.load_state_dict(vision_state, strict=False)
        missing = [k for k in missing if "cached" not in k]
        if missing:
            print(f"[Siglip] missing keys: {missing[:5]}")
        if unexpected:
            print(f"[Siglip] unexpected keys: {unexpected[:5]}")

        model = model.to(self.device)
        model.eval()
        self.model_type = "LAST-ViT"
        return model

    def _load_siglip2_model(self, state_dict):
        if AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError(f"transformers import failed: {SIGLIP_TRANSFORMERS_IMPORT_ERROR}")
        if not os.path.isdir(self.base_model_path):
            raise FileNotFoundError(f"Siglip2 base model path not found: {self.base_model_path}")

        print(f"[Siglip] loading Siglip2 base model: {self.base_model_path}")
        base_model = AutoModel.from_pretrained(self.base_model_path)
        self.processor = AutoImageProcessor.from_pretrained(
            self.base_model_path,
            use_fast=False,
        )

        has_pooler_state = any(str(key).startswith("pooler.") for key in state_dict)
        if has_pooler_state:
            embed_dim = int(base_model.config.vision_config.hidden_size)
            num_queries = int(state_dict["pooler.query_tokens"].shape[1])
            num_layers = sum(
                1 for key in state_dict
                if str(key).endswith(".cross_attn.in_proj_weight")
            )
            pooler = CrossViewAttentionPooler(
                embed_dim=embed_dim,
                num_queries=num_queries,
                num_heads=8,
                num_layers=num_layers,
                dropout=0.0,
            )
            model = SingleViewSigLIPModel(base_model, pooler)
            model.load_state_dict(state_dict)
            self.model_type = "Siglip2-EMA"
            print(
                f"[Siglip] Siglip2 EMA model loaded "
                f"(queries={num_queries}, layers={num_layers})"
            )
        else:
            model_state = {
                key[len("base_model."):]: value
                for key, value in state_dict.items()
                if str(key).startswith("base_model.")
            }
            model = base_model
            incompatible = model.load_state_dict(model_state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing:
                print(f"[Siglip] Siglip2 missing keys: {missing[:5]}")
            if unexpected:
                print(f"[Siglip] Siglip2 unexpected keys: {unexpected[:5]}")
            self.model_type = "Siglip2"

        model = model.to(self.device)
        model.eval()
        return model

    def _load_centers(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Siglip graph info not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])

        def _node_key(node):
            try:
                return int(node.get("node_id", 0))
            except Exception:
                return 10 ** 9

        centers = {}
        state_list = []
        idx = 0
        for node in sorted(nodes, key=_node_key):
            desc = str(node.get("state_description", "")).strip()
            center_raw = (
                node.get("center_feature_lastvit", None)
                or node.get("center_feature_siglip2", None)
                or node.get("center_feature", None)
            )
            node_id = str(node.get("node_id", "")).strip()
            if not desc or center_raw is None:
                continue

            idx += 1
            cid = f"C{idx}"
            category = f"{cid}: {desc}"
            centers[category] = _siglip_parse_center_feature(center_raw)
            state_list.append({
                "id": cid,
                "node_id": node_id,
                "name": desc,
                "category": category,
            })

        if not centers:
            raise RuntimeError(f"No valid Siglip centers loaded from {path}")
        return centers, state_list

    def _apply_padding_head(self, image):
        if self.pad_head_ratio <= 0:
            return image
        image = image.copy()
        width, height = image.size
        cutoff = int(height * self.pad_head_ratio)
        if cutoff <= 0:
            return image
        pixels = image.load()
        for y in range(cutoff):
            for x in range(width):
                pixels[x, y] = (0, 0, 0)
        return image

    def _preprocess_image(self, image):
        img = TVF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC)
        img = TVF.to_tensor(img)
        img = TVF.normalize(
            img,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return img.unsqueeze(0)

    @torch.inference_mode()
    def _encode_image(self, image):
        if self.model_type == "Siglip2":
            inputs = self.processor(images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model.get_image_features(pixel_values=pixel_values)
            feature = outputs.pooler_output
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feature[0].detach().cpu().numpy().astype(np.float32)
        if self.model_type == "Siglip2-EMA":
            inputs = self.processor(images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            feature = self.model.encode(pixel_values)
            return feature[0].detach().cpu().numpy().astype(np.float32)

        pixel_values = self._preprocess_image(image).to(self.device)
        feature = self.model(pixel_values)
        feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feature[0].detach().cpu().numpy().astype(np.float32)

    def _state_for_category(self, category):
        category = str(category or "").strip()
        for state in self.state_list:
            if str(state.get("category", "")).strip() == category:
                return state
        if ":" in category:
            state_id = category.split(":", 1)[0].strip()
            for state in self.state_list:
                if str(state.get("id", "")).strip() == state_id:
                    return state
        for state in self.state_list:
            if str(state.get("name", "")).strip() == category:
                return state
        return None

    def predict(self, image_bgr):
        start_time = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb).convert("RGB")
        image = self._apply_padding_head(image)

        with self.lock:
            feat_np = self._encode_image(image)
            sims = {k: float(np.dot(feat_np, v)) for k, v in self.centers.items()}

        best_category, best_similarity = max(sims.items(), key=lambda item: item[1])
        topk = sorted(sims.items(), key=lambda item: item[1], reverse=True)[:self.topk]
        best_state = self._state_for_category(best_category) or {}
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "ok": True,
            "module": f"SiglipDocker-{self.model_type}",
            "best_category": best_category,
            "best_similarity": float(best_similarity),
            "best_state_id": str(best_state.get("id", "")),
            "best_node_id": str(best_state.get("node_id", "")),
            "best_state_name": str(best_state.get("name", "")),
            "topk": [
                {"category": category, "similarity": float(similarity)}
                for category, similarity in topk
            ],
            "elapsed_ms": float(elapsed_ms),
        }

def matrix_to_quaternion(rotation_matrix):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # Returns [x, y, z, w]
    return quat

def _safe_tf_label(label):
    text = str(label or "object").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "object"


def _make_object_child_frame_id(label, index):
    return f"{_safe_tf_label(label)}_{int(index)}"


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

        # RealSense is used directly for RGB-D capture. ROS2 is still used for TF
        # and action publication.
        auto_cfg = _load_auto_app_config()
        realsense_cfg = auto_cfg.get("realsense", {}) if isinstance(auto_cfg, dict) else {}
        fresh_tf_cfg = auto_cfg.get("fresh_tf", {}) if isinstance(auto_cfg, dict) else {}
        self.rs_width = int(os.getenv("REALSENSE_WIDTH", str(realsense_cfg.get("width", 640))))
        self.rs_height = int(os.getenv("REALSENSE_HEIGHT", str(realsense_cfg.get("height", 480))))
        self.rs_fps = int(os.getenv("REALSENSE_FPS", str(realsense_cfg.get("fps", 30))))
        self.rs_serial = os.getenv("REALSENSE_SERIAL", "").strip()
        self.rs_camera_serials = self._load_realsense_camera_serials(realsense_cfg)
        self.perception_camera_name = os.getenv(
            "PERCEPTION_CAMERA",
            str(
                fresh_tf_cfg.get(
                    "perception_camera",
                    realsense_cfg.get("perception_camera", "front"),
                )
            ),
        ).strip() or "front"
        self.perception_camera_name = self._normalize_realsense_camera_name(self.perception_camera_name)
        self.rs_pipelines = {}
        self.rs_profiles = {}
        self.rs_aligns = {}
        self.rs_depth_scales = {}
        self.rs_lock = threading.Lock()

        self.object_tf_lock = threading.Lock()
        self.latest_object_tf_cache = None
        marvin_cfg = auto_cfg.get("marvin", {}) if isinstance(auto_cfg, dict) else {}
        self.object_tf_keepalive_hz = float(
            os.getenv("MARVIN_TF_KEEPALIVE_HZ", str(marvin_cfg.get("tf_keepalive_hz", 10.0)))
        )
        self.object_tf_timer = None

        # TF publisher
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        if self.object_tf_keepalive_hz > 0:
            self.object_tf_timer = self.create_timer(
                1.0 / self.object_tf_keepalive_hz,
                self.republish_latest_object_tfs,
            )

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

        self.get_logger().info(
            f"ROS2 node initialized; RGB-D capture will use pyrealsense2 directly, "
            f"perception_camera={self.perception_camera_name}, cameras={self.rs_camera_serials}"
        )

    @staticmethod
    def _clean_realsense_serial(serial):
        text = str(serial or "").strip()
        return text[1:] if text.startswith("_") else text

    def _normalize_realsense_camera_name(self, camera_name):
        text = str(camera_name or "").strip().lower().replace("_", " ")
        aliases = {
            "front": "front",
            "head": "head",
            "camera head": "head",
            "head camera": "head",
            "left": "left",
            "left wrist": "left",
            "camera left wrist": "left",
            "right": "right",
            "right wrist": "right",
            "camera right wrist": "right",
        }
        return aliases.get(text, text or "front")

    def _load_realsense_camera_serials(self, realsense_cfg):
        serials = {
            "front": "347622076256",
            "head": "347622076256",
            "left": "427622273358",
            "right": "427622271647",
        }
        cameras_cfg = realsense_cfg.get("cameras", {}) if isinstance(realsense_cfg, dict) else {}
        if isinstance(cameras_cfg, dict):
            for key, value in cameras_cfg.items():
                name = self._normalize_realsense_camera_name(key)
                raw_serial = value.get("serial_no", value.get("serial", "")) if isinstance(value, dict) else value
                serial = self._clean_realsense_serial(raw_serial)
                if serial:
                    serials[name] = serial
                    if name == "head":
                        serials["front"] = serial
        env_serials = {
            "front": os.getenv("REALSENSE_FRONT_SERIAL", ""),
            "head": os.getenv("REALSENSE_HEAD_SERIAL", ""),
            "left": os.getenv("REALSENSE_LEFT_SERIAL", ""),
            "right": os.getenv("REALSENSE_RIGHT_SERIAL", ""),
        }
        if self.rs_serial:
            env_serials["front"] = self.rs_serial
            env_serials["head"] = self.rs_serial
        for name, raw_serial in env_serials.items():
            serial = self._clean_realsense_serial(raw_serial)
            if serial:
                serials[name] = serial
                if name == "head":
                    serials["front"] = serial
        return serials

    def start_realsense(self, camera_name=None):
        """Start the selected RealSense RGB-D pipeline on first capture."""
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not available. Please install pyrealsense2 "
                "and run on a machine with an Intel RealSense camera."
            )

        camera_name = self._normalize_realsense_camera_name(camera_name or self.perception_camera_name)
        serial = self.rs_camera_serials.get(camera_name, "")
        if camera_name == "front" and not serial:
            serial = self.rs_camera_serials.get("head", "")

        with self.rs_lock:
            if camera_name in self.rs_pipelines:
                return camera_name

            pipeline = rs.pipeline()
            config = rs.config()
            if serial:
                config.enable_device(serial)
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
                depth_scale = float(depth_sensor.get_depth_scale())
                self.depth_scale = depth_scale
                self.rs_pipelines[camera_name] = pipeline
                self.rs_profiles[camera_name] = profile
                self.rs_aligns[camera_name] = rs.align(rs.stream.color)
                self.rs_depth_scales[camera_name] = depth_scale
            except Exception:
                with contextlib.suppress(Exception):
                    pipeline.stop()
                raise

            self.get_logger().info(
                f"RealSense pipeline started: camera={camera_name}, "
                f"{self.rs_width}x{self.rs_height}@{self.rs_fps}, "
                f"depth_scale={self.depth_scale:.6f}, serial={serial or '<auto>'}"
            )

            # Drop a few startup frames so auto-exposure/depth sync can settle.
            for _ in range(5):
                with contextlib.suppress(Exception):
                    pipeline.wait_for_frames(1000)
            return camera_name

    def stop_realsense(self):
        """Stop RealSense pipelines if they were started."""
        with self.rs_lock:
            pipelines = dict(self.rs_pipelines)
            self.rs_pipelines.clear()
            self.rs_profiles.clear()
            self.rs_aligns.clear()
            self.rs_depth_scales.clear()
        for camera_name, pipeline in pipelines.items():
            with contextlib.suppress(Exception):
                pipeline.stop()
            self.get_logger().info(f"RealSense pipeline stopped: camera={camera_name}")

    def wait_for_frame(self, timeout=5.0, camera_name=None):
        """Capture one aligned RGB-D frame directly from RealSense."""
        camera_name = self._normalize_realsense_camera_name(camera_name or self.perception_camera_name or "front")
        self.start_realsense(camera_name)
        deadline = time.time() + float(timeout)
        last_error = None

        with self.rs_lock:
            pipeline = self.rs_pipelines.get(camera_name)
            align = self.rs_aligns.get(camera_name)
            depth_scale = self.rs_depth_scales.get(camera_name, self.depth_scale)
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
                    self.depth_scale = depth_scale
                    self.frame_received.set()
                    return color_image.copy(), depth_image.copy(), depth_scale
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            self.get_logger().warning(f"RealSense frame capture failed: camera={camera_name}, error={last_error}")
        return None, None, None

    def wait_for_siglip_frame(self, camera_arm="head", timeout=2.0):
        """Capture one RGB frame from the selected RealSense camera for Siglip."""
        color_image, _, _ = self.wait_for_frame(timeout=timeout, camera_name=camera_arm)
        return color_image

    def destroy_node(self):
        self.stop_realsense()
        return super().destroy_node()

    def _build_object_tf_message(self, poses, labels, frame_id, stamp):
        tf_msg = TFMessage()
        for i, (pose_matrix, label) in enumerate(zip(poses, labels)):
            pose_matrix = np.asarray(pose_matrix, dtype=np.float32)
            if pose_matrix.shape != (4, 4):
                continue

            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = frame_id
            transform.child_frame_id = _make_object_child_frame_id(label, i + 1)

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
        return tf_msg

    def clear_object_tf_cache(self):
        """Clear cached object poses so stale FlowPose results are not kept alive."""
        with self.object_tf_lock:
            self.latest_object_tf_cache = None

    def current_object_child_frames(self):
        """Return child frame ids from the latest FlowPose publish only."""
        with self.object_tf_lock:
            cache = self.latest_object_tf_cache
            if cache is None:
                return set()
            labels = list(cache.get("labels") or [])
        return {_make_object_child_frame_id(label, i + 1) for i, label in enumerate(labels)}

    def publish_poses_as_tf(self, poses, labels, frame_id="camera_rgb_link"):
        """Publish poses as TF transforms and cache them for Marvin TF keepalive."""
        poses_arr = np.asarray(poses, dtype=np.float32)
        labels_list = list(labels or [])
        if poses_arr.ndim != 3 or poses_arr.shape[1:] != (4, 4):
            self.get_logger().warning(f"Invalid pose array shape for TF publish: {poses_arr.shape}")
            return

        with self.object_tf_lock:
            self.latest_object_tf_cache = {
                "poses": poses_arr.copy(),
                "labels": labels_list,
                "frame_id": frame_id,
                "updated_at": time.time(),
            }

        tf_msg = self._build_object_tf_message(
            poses_arr,
            labels_list,
            frame_id,
            self.get_clock().now().to_msg(),
        )
        self.tf_pub.publish(tf_msg)
        self.get_logger().info(
            f"Published {len(tf_msg.transforms)} TF transforms; "
            f"keepalive_hz={self.object_tf_keepalive_hz:.2f}"
        )

    def republish_latest_object_tfs(self):
        """Republish latest FlowPose object TFs with fresh stamps for robotaction lookup."""
        with self.object_tf_lock:
            cache = self.latest_object_tf_cache
            if cache is None:
                return
            poses = cache["poses"].copy()
            labels = list(cache["labels"])
            frame_id = cache["frame_id"]

        tf_msg = self._build_object_tf_message(
            poses,
            labels,
            frame_id,
            self.get_clock().now().to_msg(),
        )
        if tf_msg.transforms:
            self.tf_pub.publish(tf_msg)

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
_robotaction_node = None
_robotaction_config = None
_robotaction_lock = threading.Lock()
_task_loop_thread = None
_task_loop_stop_event = threading.Event()
_fresh_tf_callback = None
_fresh_tf_lock = threading.RLock()
_fresh_tf_last_run = 0.0
_fresh_tf_state = threading.local()
_task_siglip_estimator = None
_task_siglip_lock = threading.Lock()


def set_fresh_tf_callback(callback):
    global _fresh_tf_callback
    _fresh_tf_callback = callback


def _trigger_fresh_tf_perception(reason="", target=None, force=False):
    """Run one perception pass before Marvin queries object TF."""
    global _fresh_tf_last_run

    callback = _fresh_tf_callback
    if callback is None:
        return False
    if getattr(_fresh_tf_state, "active", False):
        return False

    auto_cfg = _load_auto_app_config()
    fresh_tf_cfg = auto_cfg.get("fresh_tf", {}) if isinstance(auto_cfg, dict) else {}
    min_interval = float(
        os.getenv("FRESH_TF_MIN_INTERVAL_SEC", str(fresh_tf_cfg.get("min_interval_sec", 1.0)))
    )
    settle_sec = float(os.getenv("FRESH_TF_SETTLE_SEC", str(fresh_tf_cfg.get("settle_sec", 0.15))))
    now = time.time()
    with _fresh_tf_lock:
        if (
            not force
            and min_interval > 0
            and (now - _fresh_tf_last_run) < min_interval
        ):
            return False

        _fresh_tf_last_run = now
        _fresh_tf_state.active = True
        try:
            ok, message = callback(reason=reason, target=target)
            status = "ok" if ok else "failed"
            print(f"[FreshTF] perception {status}: reason={reason}, target={target}, {message}")
            if ok and settle_sec > 0:
                time.sleep(settle_sec)
            return bool(ok)
        except Exception as e:
            print(f"[FreshTF] perception failed: reason={reason}, target={target}, error={e}")
            return False
        finally:
            _fresh_tf_state.active = False


if MarvinFusionNode is not None:
    class FreshTfMarvinFusionNode(MarvinFusionNode):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fresh_tf_destroying = False
            self._fresh_tf_paused = False

        def prepare_for_destroy(self, timeout=1.0):
            self._fresh_tf_destroying = True
            with contextlib.suppress(Exception):
                self.timer.cancel()

            lock = getattr(self, "_control_lock", None)
            if lock is None:
                return
            acquired = False
            try:
                acquired = lock.acquire(timeout=timeout)
                if not acquired:
                    self.get_logger().warning(
                        f"[FreshTF] robotaction timer still running after {timeout:.1f}s; destroying anyway"
                    )
            finally:
                if acquired:
                    lock.release()

        def pause_for_stop(self, timeout=1.0):
            self._fresh_tf_paused = True
            self._fresh_tf_destroying = True
            with contextlib.suppress(Exception):
                self.timer.cancel()
            self.active_state = None
            self.active_steps = []
            self.active_step_idx = 0

            lock = getattr(self, "_control_lock", None)
            if lock is None:
                return
            acquired = False
            try:
                acquired = lock.acquire(timeout=timeout)
                if not acquired:
                    self.get_logger().warning(
                        f"[FreshTF] robotaction pause requested while timer is still running after {timeout:.1f}s"
                    )
            finally:
                if acquired:
                    lock.release()

        def resume_after_stop(self):
            if not getattr(self, "_fresh_tf_paused", False):
                return False
            self._fresh_tf_destroying = False
            self._fresh_tf_paused = False
            self.timer = self.create_timer(
                0.2,
                self.on_timer,
                callback_group=self.timer_group,
            )
            self.get_logger().info("[FreshTF] robotaction resumed from paused state")
            return True

        def on_timer(self):
            if getattr(self, "_fresh_tf_destroying", False) or getattr(self, "_fresh_tf_paused", False):
                return
            try:
                return super().on_timer()
            except Exception as e:
                self.get_logger().warning(f"[FreshTF] robotaction timer callback skipped after error: {e}")
                return None

        def _wait_until_finished_or_preempt(self, *args, **kwargs):
            if getattr(self, "_fresh_tf_destroying", False):
                return TaskStatus.STOP, None
            return super()._wait_until_finished_or_preempt(*args, **kwargs)

        def _run_home_and_wait(self, *args, **kwargs):
            if getattr(self, "_fresh_tf_destroying", False):
                return TaskStatus.STOP, None
            return super()._run_home_and_wait(*args, **kwargs)

        def _publish_current_action(self, step, phase="run"):
            if getattr(self, "_fresh_tf_destroying", False):
                return
            try:
                return super()._publish_current_action(step, phase=phase)
            except Exception as e:
                self.get_logger().warning(f"[FreshTF] current action publish skipped: {e}")

        def _publish_home_action(self, arm):
            if getattr(self, "_fresh_tf_destroying", False):
                return
            try:
                return super()._publish_home_action(arm)
            except Exception as e:
                self.get_logger().warning(f"[FreshTF] home action publish skipped: {e}")

        def publish_kp_separately(self, sequence):
            if getattr(self, "_fresh_tf_destroying", False):
                return False
            try:
                return super().publish_kp_separately(sequence)
            except Exception as e:
                self.get_logger().warning(f"[FreshTF] keypoint publish skipped: {e}")
                return False

        def _latest_object_child_frames(self):
            if _ros2_node is None:
                return set()
            getter = getattr(_ros2_node, "current_object_child_frames", None)
            if not callable(getter):
                return set()
            return set(getter())

        def _filter_latest_object_frames(self, frames):
            latest_children = self._latest_object_child_frames()
            if not latest_children:
                return {}
            return {
                child: ts
                for child, ts in (frames or {}).items()
                if str(child).strip().lstrip("/") in latest_children
            }

        def get_camera_link_frames(self, target=None, wait_fresh=False, timeout=0.3, poll_interval=0.02):
            _trigger_fresh_tf_perception(
                reason="get_camera_link_frames",
                target=target,
            )
            frames = super().get_camera_link_frames(
                target=target,
                wait_fresh=wait_fresh,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            filtered = self._filter_latest_object_frames(frames)
            if frames and not filtered:
                self.get_logger().warning(
                    f"[FreshTF] filtered stale object TFs; latest={sorted(self._latest_object_child_frames())}"
                )
            return filtered

        def _lookup_transform_base_to_instance(self, inst):
            if _env_bool("FRESH_TF_REFRESH_ON_LOOKUP", False):
                _trigger_fresh_tf_perception(
                    reason="_lookup_transform_base_to_instance",
                    target=inst,
                )
            latest_children = self._latest_object_child_frames()
            inst_name = str(inst or "").strip().lstrip("/")
            if latest_children and inst_name not in latest_children:
                self.get_logger().warning(
                    f"[FreshTF] skip stale TF '{inst_name}', latest={sorted(latest_children)}"
                )
                return None
            return super()._lookup_transform_base_to_instance(inst)
else:
    FreshTfMarvinFusionNode = None


def _parse_task_target_value(value):
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [part.strip() for part in inner.split(",") if part.strip()]
    return text


def _parse_step_verification(task):
    verify = task.get("verify")
    legacy_siglip = task.get("siglip")
    if verify is None and legacy_siglip is None:
        return None

    if verify is None:
        verify = legacy_siglip
    if isinstance(verify, str):
        verify = {
            "source": "siglip",
            "expected": verify,
        }
    elif not isinstance(verify, dict):
        raise ValueError(f"Invalid task verify config: {verify!r}")

    source = str(verify.get("source", "siglip")).strip().lower()
    if source != "siglip":
        raise ValueError(f"Unsupported verification source: {source}")
    expected = normalize_state_text(
        verify.get("expected", verify.get("state", legacy_siglip))
    )
    if not expected:
        raise ValueError("Siglip verification requires a non-empty expected state")
    return {
        "source": source,
        "expected": expected,
        "timeout": max(0.1, float(verify.get("timeout", 3.0))),
        "stable_frames": max(1, int(verify.get("stable_frames", 5))),
        "delay_before": max(
            0.0,
            float(
                verify.get(
                    "delay_before",
                    verify.get(
                        "delay_before_sec",
                        os.getenv("TASK_SIGLIP_VERIFY_DELAY_SEC", "0"),
                    ),
                )
            ),
        ),
        "on_failure": str(verify.get("on_failure", "home")).strip().lower(),
        "arm": _parse_verify_arm_value(verify.get("arm", "copy last one")),
    }


def _parse_verify_arm_value(value):
    if value is None:
        return ["copy last one"]
    if isinstance(value, str):
        text = value.strip().lower().replace("_", " ")
        return [text] if text else ["copy last one"]
    if isinstance(value, (list, tuple, set)):
        arms = []
        for item in value:
            text = str(item or "").strip().lower().replace("_", " ")
            if text:
                arms.append(text)
        return arms or ["copy last one"]
    text = str(value or "").strip().lower().replace("_", " ")
    return [text] if text else ["copy last one"]


def _normalize_siglip_arm(value):
    text = str(value or "").strip().lower().replace("_", " ")
    compact = text.replace("/", " ")
    if "left" in compact:
        return "left"
    if "right" in compact:
        return "right"
    if "head" in compact or "front" in compact:
        return "head"
    aliases = {
        "l": "left",
        "left arm": "left",
        "left wrist": "left",
        "camera left wrist": "left",
        "r": "right",
        "right arm": "right",
        "right wrist": "right",
        "camera right wrist": "right",
        "front": "head",
        "head camera": "head",
        "camera head": "head",
    }
    return aliases.get(text, text)


def _payload_siglip_arm(payload):
    if not isinstance(payload, dict):
        return ""
    candidates = [
        payload.get("arm"),
        payload.get("camera"),
        payload.get("camera_name"),
        payload.get("camera_id"),
        payload.get("source_camera"),
        payload.get("view"),
        payload.get("frame_id"),
    ]
    for value in candidates:
        arm = _normalize_siglip_arm(value)
        if arm in {"left", "right", "head"}:
            return arm
    return ""


def _load_task_yaml_steps(task_yaml_path):
    if Step is None:
        raise RuntimeError(f"robotaction import failed: {ROBOTACTION_IMPORT_ERROR}")
    with open(task_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Task YAML has no tasks: {task_yaml_path}")

    steps = []
    for i, task in enumerate(tasks):
        task = task or {}
        step = Step(
            action_name=task.get("action_name", f"step_{i}"),
            target=_parse_task_target_value(task.get("targets", task.get("objects", task.get("target", "")))),
            arm=task.get("arm"),
            speed=task.get("speed"),
            correction_mode=task.get("correction_mode", "org"),
            finish_home=bool(task.get("finish_home", task.get("finfish_home", False))),
            fixed_endpoint=task.get("fixed_endpoint", False),
            approach_count=int(task.get("approach_count", 0) or 0),
            status_timeout=float(task.get("status_timeout", 2.0)),
            task_finish_timeout=float(task.get("task_finish_timeout", 60.0)),
            policy=str(task.get("policy", "first")).strip().lower() or "first",
        )
        step.min_action_duration = max(
            0.0,
            float(
                task.get(
                    "min_action_duration",
                    task.get(
                        "min_action_duration_sec",
                        os.getenv("TASK_LOOP_MIN_ACTION_DURATION_SEC", "0"),
                    ),
                )
            ),
        )
        correction_mode_overrides = task.get("correction_mode_overrides", {}) or {}
        generic_overrides = task.get("overrides", {}) or {}
        normalized_correction_overrides = {}
        if isinstance(correction_mode_overrides, dict):
            for key, value in correction_mode_overrides.items():
                name = normalize_obj_name(str(key))
                if name and value is not None:
                    normalized_correction_overrides[name] = str(value).strip()
        if isinstance(generic_overrides, dict):
            for key, value in generic_overrides.items():
                if not isinstance(value, dict):
                    continue
                mode = value.get("correction_mode")
                name = normalize_obj_name(str(key))
                if name and mode is not None:
                    normalized_correction_overrides[name] = str(mode).strip()
        step.correction_mode_overrides = normalized_correction_overrides
        step.verification = _parse_step_verification(task)
        steps.append(step)
    return steps


class TaskSiglipWatcher:
    def __init__(self, node, topic, name="default", callback_group=None):
        self.node = node
        self.topic = topic
        self.name = name
        self.lock = threading.Lock()
        self.states = deque(maxlen=32)
        self.generation = 0
        self.subscription = node.create_subscription(
            String,
            topic,
            self._callback,
            20,
            callback_group=callback_group,
        )

    def _callback(self, msg):
        try:
            payload = json.loads(msg.data)
            payload = extract_siglip_payload(payload) or payload
            if not isinstance(payload, dict):
                return
            if "ok" in payload and not payload.get("ok", False):
                return
            state = normalize_state_text(
                payload.get("best_state_name") or payload.get("best_category")
            )
            if not state:
                return
            arm = _payload_siglip_arm(payload)
            with self.lock:
                self.states.append((self.generation, state, arm, time.time()))
        except Exception as exc:
            self.node.get_logger().warning(f"[TaskSiglip] parse failed: {exc}")

    def reset(self):
        with self.lock:
            self.generation += 1
            self.states.clear()
            return self.generation

    def values_for_generation(self, generation, accepted_arms=None):
        accepted = {
            _normalize_siglip_arm(arm)
            for arm in (accepted_arms or [])
            if _normalize_siglip_arm(arm) in {"left", "right", "head"}
        }
        with self.lock:
            return [
                state
                for item_generation, state, arm, _ in self.states
                if item_generation == generation
                and (not accepted or not arm or arm in accepted)
            ]

    def wait_stable(self, expected, timeout, stable_frames, accepted_arms=None):
        expected = normalize_state_text(expected)
        generation = self.reset()
        deadline = time.time() + float(timeout)
        latest_stable = None
        while rclpy.ok() and not _task_loop_stop_event.is_set():
            values = self.values_for_generation(generation, accepted_arms)
            if len(values) >= stable_frames:
                recent = values[-stable_frames:]
                if len(set(recent)) == 1:
                    actual = recent[-1]
                    latest_stable = actual
                    if actual == expected:
                        return True, actual
            if time.time() >= deadline:
                return False, latest_stable or (values[-1] if values else None)
            time.sleep(0.05)
        return False, None


def _get_task_siglip_estimator(device=None):
    global _task_siglip_estimator
    with _task_siglip_lock:
        if _task_siglip_estimator is None:
            _task_siglip_estimator = SiglipStateEstimator(device=device)
        return _task_siglip_estimator


class TaskSiglipVerifier:
    def __init__(self, node, device=None):
        self.node = node
        self.device = device

    def predict_once(self, camera_arm, timeout):
        camera_arm = _normalize_siglip_arm(camera_arm) or "head"
        if camera_arm not in {"left", "right", "head"}:
            camera_arm = "head"
        estimator = _get_task_siglip_estimator(self.device)
        if _ros2_node is None:
            raise RuntimeError("ROS2 image node is not initialized")
        capture = getattr(_ros2_node, "wait_for_siglip_frame", None)
        if not callable(capture):
            raise RuntimeError("ROS2 image node cannot provide Siglip camera frames")

        image_bgr = capture(camera_arm=camera_arm, timeout=timeout)
        if image_bgr is None:
            raise RuntimeError(f"Siglip camera frame timeout: camera={camera_arm}")

        result = estimator.predict(image_bgr)
        result["camera_arm"] = camera_arm
        result["stamp"] = time.time()
        return result

    def wait_expected(self, expected, camera_arm, timeout, stable_frames):
        expected = normalize_state_text(expected)
        camera_arm = _normalize_siglip_arm(camera_arm) or "head"
        _get_task_siglip_estimator(self.device)
        deadline = time.time() + float(timeout)
        history = []
        latest = None
        while rclpy.ok() and not _task_loop_stop_event.is_set():
            remaining = max(0.1, deadline - time.time())
            result = self.predict_once(camera_arm, timeout=min(remaining, 2.0))
            actual = normalize_state_text(
                result.get("best_state_name") or result.get("best_category")
            )
            latest = actual
            history.append(actual)
            history = history[-stable_frames:]
            self.node.get_logger().info(
                f"[TaskSiglip] camera='{camera_arm}' predicted='{actual}', "
                f"expected='{expected}', sim={float(result.get('best_similarity', 0.0)):.4f}"
            )
            if len(history) >= stable_frames and len(set(history)) == 1:
                return history[-1] == expected, actual
            if time.time() >= deadline:
                break
        return False, latest


class TaskLoopActionNode(FusionExecutionMixin, FusionTfMixin, Node):
    def __init__(
        self,
        object_yaml_path,
        task_yaml_path,
        progress_topic,
        object_tf_topic=None,
        base_frame="base_link",
        camera_frames=None,
    ):
        if ActionHandler is None:
            raise RuntimeError(f"robotaction import failed: {ROBOTACTION_IMPORT_ERROR}")
        super().__init__("robotaction_task_loop")
        self.timer_group = ReentrantCallbackGroup()
        self.progress_group = ReentrantCallbackGroup()
        self.object_tf_group = ReentrantCallbackGroup()

        with open(object_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            self.object_templates = data.get("templates", {})

        self.task_steps = _load_task_yaml_steps(task_yaml_path)
        self.last_arm = None
        self.base_frame = normalize_frame_id(base_frame)
        self.camera_frames = {
            normalize_frame_id(v)
            for v in (camera_frames or ["camera_rgb_link", "camera_link_rgb"])
        }
        self.object_tf_topic = object_tf_topic
        self.template_handlers = {
            k: ActionHandler(v, node=self)
            for k, v in self.object_templates.items()
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.object_tf_watcher = None
        if object_tf_topic and ObjectTfWatcher is not None:
            self.object_tf_watcher = ObjectTfWatcher(
                self,
                topic=object_tf_topic,
                callback_group=self.object_tf_group,
            )

        self.pose_pub = self.create_publisher(KeypointPoseArray, "/fusion_pose", 10)
        self.action_pub = self.create_publisher(String, "/fusion/current_action", 10)
        self.arm_resolver = ArmResolver(lambda: self.last_arm)
        self.fixed_ep_mgr = FixedEndpointManager()
        self.task_progress_watcher = TaskProgressWatcher(
            self,
            topic=progress_topic,
            callback_group=self.progress_group,
        )
        self.siglip_verifier = TaskSiglipVerifier(self)
        self.siglip_verification_failed = False
        self.active_state = "task_loop"
        self.active_steps = []
        self.active_step_idx = 0
        self.allow_home_preempt = False
        self._last_tf_stamp_ns = {}

        self.get_logger().info(
            f"[TaskLoop] object_yaml={object_yaml_path}, task_yaml={task_yaml_path}, "
            f"progress_topic={progress_topic}, siglip=on-demand-camera, base_frame={self.base_frame}"
        )

    def _is_home_step(self, step: Step):
        return str(step.action_name).strip().lower() == "home"

    def _is_absolute_home_step(self, step: Step):
        return normalize_obj_name(step.primary_target()) == "home" and not self._is_home_step(step)

    def _get_preempt_state(self):
        return None

    def _publish_arm_home_three_times(self, arm):
        """Publish one arm's home command three times without waiting for progress."""
        arm_name = str(arm or "").strip().lower()
        if arm_name not in {"left", "right"}:
            return False
        home_sequence = [{
            "name": "home",
            "arm": arm_name,
            "poses": [],
            "constraints": [100, 100, 100],
            "speed": 1.0,
            "gripper_value": [0.0, 0.0],
            "time": [0.0, 0.1],
        }]
        for publish_index in range(2):
            self.publish_kp_separately(home_sequence)
            self.get_logger().info(
                f"[TaskLoop] arm switch: publish previous arm home "
                f"{publish_index + 1}/2 arm='{arm_name}'"
            )
            if publish_index < 2:
                time.sleep(0.1)
        self.get_logger().info(
            f"[TaskLoop] arm switch: previous arm home published arm='{arm_name}'"
        )
        return True

    def _instance_base_name(self, inst):
        text = str(inst or "").strip()
        text = re.sub(r"_\d+$", "", text)
        return normalize_obj_name(text)

    def _step_with_target_overrides(self, step: Step, inst: str):
        overrides = getattr(step, "correction_mode_overrides", {}) or {}
        if not isinstance(overrides, dict) or not overrides:
            return step
        inst_name = self._instance_base_name(inst)
        mode = overrides.get(inst_name)
        if not mode:
            return step

        effective_step = copy.copy(step)
        effective_step.correction_mode = mode
        self.get_logger().info(
            f"[TaskLoop] correction_mode override: target='{inst_name}', "
            f"default='{step.correction_mode}', override='{mode}'"
        )
        return effective_step

    def _execute_step_once(self, step: Step, inst: str):
        step = self._step_with_target_overrides(step, inst)
        get_last_arm = getattr(self.arm_resolver, "_get_last_arm", None)
        previous_arm = get_last_arm() if callable(get_last_arm) else self.last_arm

        # The parent resolves the current arm and builds keypoints, but does not
        # publish them yet. This leaves a safe point to home the previous arm.
        kps = super()._execute_step_once(step, inst)
        current_arm = self.last_arm

        previous_name = str(previous_arm or "").strip().lower()
        current_name = str(current_arm or "").strip().lower()
        if (
            kps is not None
            and previous_name in {"left", "right"}
            and current_name in {"left", "right"}
            and previous_name != current_name
        ):
            self._publish_arm_home_three_times(previous_name)
        return kps

    def _wait_step_finished(self, step: Step):
        start = time.time()
        finish_percentage = 1.0
        min_action_duration = max(0.0, float(getattr(step, "min_action_duration", 0.0) or 0.0))
        observed_new_cycle = False
        early_finish_logged = False
        while rclpy.ok() and not _task_loop_stop_event.is_set():
            watcher = self.task_progress_watcher
            with watcher._lock:
                latest_value = watcher._latest_value
                last_msg_time = watcher._last_msg_time

            # The controller may briefly keep publishing the previous command's
            # terminal value after reset. Do not accept that stale 1.0 as this
            # step's completion; first require a post-reset non-terminal value.
            if (
                last_msg_time is not None
                and latest_value is not None
                and latest_value < finish_percentage
            ):
                observed_new_cycle = True

            elapsed = time.time() - start
            finish_seen = observed_new_cycle and watcher.is_finished(
                silence_after_zero=0.5,
                finish_percentage=finish_percentage,
            )
            if finish_seen and elapsed >= min_action_duration:
                self.get_logger().info("[TaskLoop] 当前动作完成")
                return TaskStatus.SUCCESS
            if finish_seen and not early_finish_logged:
                early_finish_logged = True
                self.get_logger().info(
                    f"[TaskLoop] progress reported complete after {elapsed:.2f}s; "
                    f"waiting until min_action_duration={min_action_duration:.2f}s "
                    f"before verification"
                )
            if elapsed >= step.task_finish_timeout:
                self.get_logger().warning(
                    f"[TaskLoop] 等待动作完成超时 action='{step.action_name}', "
                    f"observed_new_cycle={observed_new_cycle}, latest={latest_value}"
                )
                return TaskStatus.TIMEOUT
            time.sleep(0.2)
        return TaskStatus.STOP

    def _resolve_verify_arms(self, verification):
        raw_arms = verification.get("arm") or ["copy last one"]
        if isinstance(raw_arms, str):
            raw_arms = [raw_arms]
        resolved = []
        for raw in raw_arms:
            arm = _normalize_siglip_arm(raw)
            if arm in {"copy last one", "copy last", "last"}:
                arm = _normalize_siglip_arm(self.last_arm)
            if arm in {"left", "right", "head"} and arm not in resolved:
                resolved.append(arm)
        return resolved

    def _siglip_watchers_for_arms(self, arms):
        if not arms:
            return ["head"]
        return list(arms)

    def _verify_step_with_siglip(self, step):
        verification = getattr(step, "verification", None)
        if not verification:
            return TaskStatus.SUCCESS

        expected = verification["expected"]
        timeout = verification["timeout"]
        stable_frames = verification["stable_frames"]
        delay_before = max(0.0, float(verification.get("delay_before", 0.0) or 0.0))
        arms = self._resolve_verify_arms(verification)
        camera_arms = self._siglip_watchers_for_arms(arms)
        arm_desc = ",".join(camera_arms)
        self.get_logger().info(
            f"[TaskSiglip] running on-demand inference expected='{expected}', camera='{arm_desc}', "
            f"stable_frames={stable_frames}, timeout={timeout:.1f}s"
        )
        if delay_before > 0:
            self.get_logger().info(
                f"[TaskSiglip] waiting {delay_before:.2f}s before verification"
            )
            deadline = time.time() + delay_before
            while time.time() < deadline and not _task_loop_stop_event.is_set():
                remaining = max(0.0, deadline - time.time())
                time.sleep(min(0.05, remaining))

        latest_actual = None
        for camera_arm in camera_arms:
            try:
                matched, actual = self.siglip_verifier.wait_expected(
                    expected=expected,
                    camera_arm=camera_arm,
                    timeout=timeout,
                    stable_frames=stable_frames,
                )
            except Exception as exc:
                actual = f"error: {exc}"
                matched = False
                self.get_logger().warning(
                    f"[TaskSiglip] camera='{camera_arm}' inference failed: {exc}"
                )
            latest_actual = actual or latest_actual
            if matched:
                self.siglip_verification_failed = False
                self.get_logger().info(
                    f"[TaskSiglip] verification passed: '{actual}', camera='{camera_arm}'"
                )
                return TaskStatus.SUCCESS

        self.siglip_verification_failed = True
        self.get_logger().warning(
            f"[TaskSiglip] verification failed: expected='{expected}', "
            f"camera='{arm_desc}', actual='{latest_actual or '<timeout>'}'"
        )
        if verification.get("on_failure") == "home":
            arm = str(self.last_arm or "").strip().lower()
            if arm in {"left", "right"}:
                self.get_logger().warning(
                    f"[TaskSiglip] publishing home without progress wait: arm='{arm}'"
                )
                self.publish_home_separately(arm)
            else:
                self.get_logger().warning(
                    "[TaskSiglip] cannot resolve current arm for failure home"
                )
        return TaskStatus.FAIL

    def select_target_instance(self):
        if not self.task_steps:
            return None
        first_step = self.task_steps[0]
        if _ros2_node is not None:
            getter = getattr(_ros2_node, "current_object_child_frames", None)
            if callable(getter):
                latest_frames = set(getter())
                if not latest_frames:
                    self.get_logger().info(
                        "[TaskLoop] latest perception contains no object TF; "
                        "skip historical TF buffer fallback"
                    )
                    return None
                frames = {frame: None for frame in latest_frames}
                return self._select_instance(first_step, frames)

        # Compatibility fallback for deployments without the local perception
        # cache. In the integrated task-loop app, the cache above is authoritative
        # and prevents stale transforms from an earlier frame being selected.
        frames = self._wait_target_instances_ready(
            first_step.target_names(),
            timeout=2.0,
            settle_time=0.1,
            poll_interval=0.1,
        )
        return self._select_instance(first_step, frames)

    def _refresh_retry_instance(self, step, current_inst):
        if self._is_absolute_home_step(step):
            return step.primary_target()
        target = current_inst or step.target_names()
        empty_confirmations = max(
            1,
            int(os.getenv("TASK_LOOP_EMPTY_CONFIRMATIONS", "3")),
        )
        empty_retry_interval = max(
            0.0,
            float(os.getenv("TASK_LOOP_EMPTY_RETRY_INTERVAL_SEC", "0.5")),
        )
        for attempt in range(1, empty_confirmations + 1):
            if _task_loop_stop_event.is_set():
                return None
            ok = _trigger_fresh_tf_perception(
                reason="task_siglip_retry",
                target=target,
                force=True,
            )
            if ok:
                refreshed_inst = self.select_target_instance()
                if refreshed_inst:
                    self.get_logger().info(
                        f"[TaskSiglip] retry perception selected='{refreshed_inst}' "
                        f"(previous='{current_inst}', attempt={attempt}/{empty_confirmations})"
                    )
                    return refreshed_inst
                self.get_logger().warning(
                    f"[TaskSiglip] retry perception found no target: "
                    f"{attempt}/{empty_confirmations}; skip stale TF reuse"
                )
            else:
                self.get_logger().warning(
                    f"[TaskSiglip] retry perception failed: "
                    f"{attempt}/{empty_confirmations}, target='{target}'"
                )
            if attempt < empty_confirmations and empty_retry_interval > 0:
                time.sleep(empty_retry_interval)

        self.get_logger().warning(
            f"[TaskSiglip] retry perception confirmed no target after "
            f"{empty_confirmations} attempts; stop retry"
        )
        return None

    def run_task_for_instance(self, inst):
        self.active_steps = list(self.task_steps)
        self.active_step_idx = 0
        self.siglip_verification_failed = False
        for idx, step in enumerate(self.active_steps):
            if _task_loop_stop_event.is_set():
                return TaskStatus.STOP
            self.active_step_idx = idx
            retry_count = 0
            while rclpy.ok() and not _task_loop_stop_event.is_set():
                self._publish_current_action(step, phase="run")
                self.task_progress_watcher.reset()
                run_inst = inst
                if self._is_absolute_home_step(step):
                    run_inst = step.primary_target()
                retry_suffix = f", retry={retry_count}" if retry_count else ""
                self.get_logger().info(
                    f"[TaskLoop] step={idx + 1}/{len(self.active_steps)} "
                    f"target='{step.display_target()}' selected='{run_inst}' "
                    f"action='{step.action_name}'{retry_suffix}"
                )
                status = self._run_step(step, run_inst)
                if status != TaskStatus.SUCCESS:
                    return status
                status = self._wait_step_finished(step)
                if status != TaskStatus.SUCCESS:
                    return status
                status = self._verify_step_with_siglip(step)
                if status == TaskStatus.SUCCESS:
                    break
                if getattr(step, "verification", None):
                    retry_count += 1
                    self.get_logger().warning(
                        f"[TaskSiglip] verification failed; retry current step "
                        f"action='{step.action_name}', selected='{run_inst}', retry={retry_count}"
                    )
                    refreshed_inst = self._refresh_retry_instance(step, run_inst)
                    if not refreshed_inst:
                        return TaskStatus.FAIL
                    inst = refreshed_inst
                    time.sleep(float(os.getenv("TASK_SIGLIP_RETRY_SETTLE_SEC", "0.2")))
                    continue
                return status
            if _task_loop_stop_event.is_set():
                return TaskStatus.STOP
        return TaskStatus.SUCCESS


def init_ros2():
    """Initialize ROS2 and create subscriber node"""
    global _ros2_node, _ros2_executor, _ros2_thread

    if _ros2_node is not None:
        return _ros2_node

    if not rclpy.ok():
        rclpy.init()

    _ros2_node = ROS2ImageSubscriber()
    _ros2_executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    _ros2_executor.add_node(_ros2_node)

    # Spin in a separate thread
    def spin_thread():
        while rclpy.ok():
            _ros2_executor.spin_once(timeout_sec=0.1)

    _ros2_thread = threading.Thread(target=spin_thread, daemon=True)
    _ros2_thread.start()

    return _ros2_node


def shutdown_ros2():
    """Shutdown ROS2"""
    global _ros2_node, _ros2_executor, _ros2_thread

    set_fresh_tf_callback(None)
    stop_task_loop(publish_home=False)
    stop_marvin_action_node(publish_home=False, destroy=True)

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

    camera_name = getattr(_ros2_node, "perception_camera_name", "front")
    print(f"Waiting for RealSense RGB-D frames: camera={camera_name}...")

    color_image, depth_image, depth_scale = _ros2_node.wait_for_frame(
        timeout=10.0,
        camera_name=camera_name,
    )

    if color_image is None or depth_image is None:
        raise RuntimeError(
            f"Failed to receive frames from RealSense camera '{camera_name}'. "
            "Check camera connection, permissions, and pyrealsense2 installation."
        )

    print(f"Frame captured from RealSense camera '{camera_name}'. Depth scale: {depth_scale}")

    return color_image, depth_image, depth_scale



def _resolve_marvin_path(raw_path=""):
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if os.path.exists(candidate):
        return candidate
    if candidate.startswith("/workspace/"):
        mapped = os.path.join(MARVIN_DOCKER_DIR, candidate[len("/workspace/"):])
        if os.path.exists(mapped):
            return mapped
    mapped = os.path.join(AUTO_APP_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    mapped = os.path.join(ROBOTACTION_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    mapped = os.path.join(MARVIN_DOCKER_DIR, candidate)
    if os.path.exists(mapped):
        return mapped
    return candidate


def _default_marvin_object_yaml():
    auto_cfg = _load_auto_app_config()
    marvin_cfg = auto_cfg.get("marvin", {}) if isinstance(auto_cfg, dict) else {}
    configured = marvin_cfg.get("object_yaml_path", "") if isinstance(marvin_cfg, dict) else ""
    return os.getenv(
        "MARVIN_OBJECT_YAML",
        _resolve_marvin_path(configured) if configured else os.path.join(ROBOTACTION_DIR, "data", "tool", "tool.yaml"),
    )


def _default_marvin_task_yaml():
    auto_cfg = _load_auto_app_config()
    marvin_cfg = auto_cfg.get("marvin", {}) if isinstance(auto_cfg, dict) else {}
    configured = marvin_cfg.get("task_yaml_path", "") if isinstance(marvin_cfg, dict) else ""
    return os.getenv(
        "MARVIN_TASK_YAML",
        _resolve_marvin_path(configured) if configured else os.path.join(ROBOTACTION_DIR, "data", "tool", "task_siglip.yaml"),
    )


def _default_marvin_config_value(key, env_name, fallback):
    auto_cfg = _load_auto_app_config()
    marvin_cfg = auto_cfg.get("marvin", {}) if isinstance(auto_cfg, dict) else {}
    configured = marvin_cfg.get(key, fallback) if isinstance(marvin_cfg, dict) else fallback
    return os.getenv(env_name, str(configured))


def _format_marvin_action_status():
    node = _robotaction_node
    if node is None:
        return "TaskLoop动作端未启动"
    if getattr(node, "_fresh_tf_paused", False):
        return "TaskLoop动作端已暂停"

    active_state = getattr(node, "active_state", None)
    active_steps = getattr(node, "active_steps", []) or []
    active_step_idx = int(getattr(node, "active_step_idx", 0) or 0)
    if active_state and active_step_idx < len(active_steps):
        step = active_steps[active_step_idx]
        step_text = (
            f"{active_step_idx + 1}/{len(active_steps)} "
            f"target={step.display_target()} action={step.action_name} arm={step.arm}"
        )
    elif active_state:
        step_text = f"{active_step_idx + 1}/{len(active_steps)} finishing"
    else:
        step_text = "idle"

    return (
        "TaskLoop动作端运行中\n"
        f"- active_state: {active_state or '<none>'}\n"
        f"- step: {step_text}\n"
        f"- last_arm: {getattr(node, 'last_arm', None) or '<none>'}"
    )


def _stop_marvin_action_node_locked(publish_home=False, destroy=False):
    global _robotaction_node, _robotaction_config
    node = _robotaction_node
    if node is None:
        return False

    if not destroy:
        pause = getattr(node, "pause_for_stop", None)
        if callable(pause):
            pause()
            return True

    _robotaction_node = None
    _robotaction_config = None
    if publish_home:
        with contextlib.suppress(Exception):
            node.publish_home_both()
    with contextlib.suppress(Exception):
        prepare = getattr(node, "prepare_for_destroy", None)
        if callable(prepare):
            prepare()
    if _ros2_executor is not None:
        with contextlib.suppress(Exception):
            _ros2_executor.remove_node(node)
    with contextlib.suppress(Exception):
        node.destroy_node()
    return True


def stop_marvin_action_node(publish_home=False, destroy=False):
    with _robotaction_lock:
        return _stop_marvin_action_node_locked(publish_home=publish_home, destroy=destroy)


def start_task_loop_node(
    object_yaml_path=None,
    task_yaml_path=None,
    progress_topic=None,
    object_tf_topic="",
    base_frame=None,
    camera_frames=None,
):
    global _robotaction_node, _robotaction_config

    if TaskLoopActionNode is None:
        raise RuntimeError(f"TaskLoop robotaction import failed: {ROBOTACTION_IMPORT_ERROR}")

    ros_node = init_ros2()
    object_yaml_path = _resolve_marvin_path(object_yaml_path or _default_marvin_object_yaml())
    task_yaml_path = _resolve_marvin_path(task_yaml_path or _default_marvin_task_yaml())
    progress_topic = str(
        progress_topic
        or _default_marvin_config_value("progress_topic", "MARVIN_PROGRESS_TOPIC", "/control/task_percentage")
    ).strip() or "/control/task_percentage"
    object_tf_topic = str(
        object_tf_topic or _default_marvin_config_value("object_tf_topic", "MARVIN_OBJECT_TF_TOPIC", "")
    ).strip()
    base_frame = str(
        base_frame or _default_marvin_config_value("base_frame", "MARVIN_BASE_FRAME", "base_link")
    ).strip() or "base_link"
    camera_frames = camera_frames or ["camera_rgb_link", "camera_link_rgb"]

    if not os.path.isfile(object_yaml_path):
        raise FileNotFoundError(f"Marvin object YAML not found: {object_yaml_path}")
    if not os.path.isfile(task_yaml_path):
        raise FileNotFoundError(f"Marvin task YAML not found: {task_yaml_path}")

    config = {
        "mode": "task_loop",
        "object_yaml_path": object_yaml_path,
        "task_yaml_path": task_yaml_path,
        "progress_topic": progress_topic,
        "object_tf_topic": object_tf_topic,
        "base_frame": base_frame,
        "camera_frames": tuple(camera_frames),
    }

    with _robotaction_lock:
        if _robotaction_node is not None:
            if _robotaction_config == config:
                return _robotaction_node, "TaskLoop动作端已在运行"
            _stop_marvin_action_node_locked(publish_home=False, destroy=True)

        node = TaskLoopActionNode(
            object_yaml_path=object_yaml_path,
            task_yaml_path=task_yaml_path,
            progress_topic=progress_topic,
            object_tf_topic=object_tf_topic or None,
            base_frame=base_frame,
            camera_frames=list(camera_frames),
        )
        _ros2_executor.add_node(node)
        _robotaction_node = node
        _robotaction_config = config
        ros_node.get_logger().info(
            f"TaskLoop robotaction started: object_yaml={object_yaml_path}, task_yaml={task_yaml_path}"
        )
        return node, "TaskLoop动作端已启动，按 task.yaml 循环发布 /fusion_pose"


def stop_task_loop(publish_home=False):
    global _task_loop_thread
    _task_loop_stop_event.set()
    if publish_home and _robotaction_node is not None:
        with contextlib.suppress(Exception):
            _robotaction_node.publish_home_both()
    thread = _task_loop_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    _task_loop_thread = None
    return True



def _load_sam3_config():
    auto_cfg = _load_auto_app_config()
    sam3_cfg = auto_cfg.get("sam3", {}) if isinstance(auto_cfg, dict) else {}
    if sam3_cfg:
        return {"sam3": sam3_cfg}

    config_path = os.path.join(SAM3_DOCKER_DIR, "config.yaml")
    return _load_yaml_config_file(config_path)


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
        mapped = os.path.join(AUTO_APP_DIR, candidate)
        if os.path.isfile(mapped):
            return mapped
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


def _env_bool(name, default=False):
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)


def _bbox_containment(inner_bbox, outer_bbox):
    inner_area = _bbox_area(inner_bbox)
    if inner_area <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in inner_bbox[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in outer_bbox[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1 + 1.0) * max(0.0, iy2 - iy1 + 1.0)
    return float(inter) / float(inner_area)


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
        self.dedup_iou_threshold = float(
            os.getenv(
                "SAM3_DEDUP_IOU_THRESHOLD",
                str(sam3_cfg.get("dedup_iou_threshold", 0.55)),
            )
        )
        self.suppress_contained_masks = _env_bool(
            "SAM3_SUPPRESS_CONTAINED_MASKS",
            bool(sam3_cfg.get("suppress_contained_masks", True)),
        )
        self.containment_threshold = float(
            os.getenv(
                "SAM3_CONTAINMENT_THRESHOLD",
                str(sam3_cfg.get("containment_threshold", 0.85)),
            )
        )
        self.bbox_containment_threshold = float(
            os.getenv(
                "SAM3_BBOX_CONTAINMENT_THRESHOLD",
                str(sam3_cfg.get("bbox_containment_threshold", 0.85)),
            )
        )
        self.containment_min_area_ratio = float(
            os.getenv(
                "SAM3_CONTAINMENT_MIN_AREA_RATIO",
                str(sam3_cfg.get("containment_min_area_ratio", 1.2)),
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
        print(
            f"SAM3 model loaded in {time.time() - t0:.2f}s, "
            f"score_threshold={self.score_threshold}, dedup_iou_threshold={self.dedup_iou_threshold}, "
            f"suppress_contained_masks={self.suppress_contained_masks}, "
            f"containment_threshold={self.containment_threshold}, "
            f"bbox_containment_threshold={self.bbox_containment_threshold}, "
            f"containment_min_area_ratio={self.containment_min_area_ratio}"
        )

    @torch.inference_mode()
    def segment(self, color_image_bgr, prompts, suppress_contained_masks=None):
        prompts = _parse_semantic_prompts(prompts)
        if not prompts:
            return [], [], [], []

        image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        t0 = time.time()
        inference_state = self.processor.set_image(image)
        embed_sec = time.time() - t0

        candidates = []
        prompt_timings = []

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

                candidates.append(
                    {
                        "label": prompt,
                        "score": score,
                        "bbox": bbox,
                        "mask": binary_mask,
                        "area": int(np.count_nonzero(binary_mask)),
                    }
                )

        kept = []
        suppressed = 0
        dedup_iou = max(0.0, float(self.dedup_iou_threshold))
        for candidate in sorted(candidates, key=lambda det: (det["score"], det["area"]), reverse=True):
            duplicate_of_kept = False
            if dedup_iou > 0:
                for selected in kept:
                    if mask_iou(candidate["mask"], selected["mask"]) >= dedup_iou:
                        duplicate_of_kept = True
                        suppressed += 1
                        break
            if not duplicate_of_kept:
                kept.append(candidate)

        suppress_contained = (
            self.suppress_contained_masks
            if suppress_contained_masks is None
            else bool(suppress_contained_masks)
        )
        suppressed_contained = 0
        if suppress_contained and kept:
            containment_threshold = max(0.0, min(1.0, float(self.containment_threshold)))
            bbox_containment_threshold = max(0.0, min(1.0, float(self.bbox_containment_threshold)))
            min_area_ratio = max(1.0, float(self.containment_min_area_ratio))
            area_sorted = sorted(kept, key=lambda det: det["area"], reverse=True)
            filtered = []
            for candidate in area_sorted:
                candidate_area = max(1, int(candidate["area"]))
                covered_by_larger = False
                for selected in filtered:
                    selected_area = int(selected["area"])
                    if selected_area < candidate_area * min_area_ratio:
                        continue
                    overlap = np.logical_and(candidate["mask"], selected["mask"]).sum()
                    mask_containment = float(overlap) / float(candidate_area)
                    bbox_containment = _bbox_containment(candidate.get("bbox"), selected.get("bbox"))
                    if (
                        mask_containment >= containment_threshold
                        or bbox_containment >= bbox_containment_threshold
                    ):
                        covered_by_larger = True
                        suppressed_contained += 1
                        break
                if not covered_by_larger:
                    filtered.append(candidate)
            kept = sorted(filtered, key=lambda det: (det["score"], det["area"]), reverse=True)

        masks = []
        labels = []
        detections = []
        for det_id, det in enumerate(kept, start=1):
            masks.append(det["mask"])
            labels.append(det["label"])
            detections.append(
                {
                    "id": det_id,
                    "label": det["label"],
                    "score": det["score"],
                    "bbox": det["bbox"],
                }
            )

        print(
            f"SAM3 semantic segmentation: prompts={prompts}, candidates={len(candidates)}, "
            f"detections={len(detections)}, suppressed={suppressed}, "
            f"suppressed_contained={suppressed_contained}, "
            f"embed={embed_sec:.3f}s"
        )
        self.last_segment_stats = {
            "candidates": len(candidates),
            "detections": len(detections),
            "suppressed_dedup": suppressed,
            "suppressed_contained": suppressed_contained,
            "suppress_contained_masks": suppress_contained,
        }
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


def _load_work_area():
    cfg = _load_auto_app_config()
    area = cfg.get("work_area", {}) if isinstance(cfg, dict) else {}
    try:
        parsed = {
            "x_min": float(area["x_min"]),
            "x_max": float(area["x_max"]),
            "y_min": float(area["y_min"]),
            "y_max": float(area["y_max"]),
        }
        if "z_min" in area and "z_max" in area:
            parsed["z_min"] = float(area["z_min"])
            parsed["z_max"] = float(area["z_max"])
        return parsed
    except (KeyError, TypeError, ValueError):
        return None


def _filter_flowpose_by_work_area(
    pose_all,
    length_all,
    labels,
    objects=None,
    camera_frame="camera_rgb_link",
    base_frame="base_link",
):
    poses = np.asarray(pose_all, dtype=np.float32) if pose_all is not None else np.empty((0, 4, 4), dtype=np.float32)
    lengths = np.asarray(length_all, dtype=np.float32) if length_all is not None else np.empty((0,), dtype=np.float32)
    labels = list(labels or [])
    objects = list(objects or [])
    area = _load_work_area()
    count = min(len(poses), len(lengths), len(labels))

    if count == 0:
        return {
            "pose_all": None,
            "length_all": None,
            "labels": [],
            "objects": [],
            "statuses": [],
            "base_poses": [],
            "out_of_area_count": 0,
            "area_check_available": area is not None,
        }

    base_to_camera = None
    if area is not None and _ros2_node is not None and transform_to_matrix is not None:
        ts = _ros2_node.lookup_transform(base_frame, camera_frame, timeout=0.5)
        if ts is not None:
            base_to_camera = transform_to_matrix(ts.transform)

    keep_indices = []
    statuses = []
    base_poses = []
    for i in range(count):
        base_pose = None
        in_area = True
        reason = "in area"
        if area is None:
            reason = "area disabled"
        elif base_to_camera is None:
            # Work-area filtering must fail closed. Without the camera->base
            # transform, the pose cannot be safely compared with x/y/z limits.
            in_area = False
            reason = "area unknown"
        else:
            base_pose = base_to_camera @ poses[i]
            x, y, z = [float(v) for v in base_pose[:3, 3]]
            in_area = (
                area["x_min"] <= x <= area["x_max"]
                and area["y_min"] <= y <= area["y_max"]
            )
            if "z_min" in area and "z_max" in area:
                in_area = (
                    in_area
                    and area["z_min"] <= z <= area["z_max"]
                )
            reason = "in area" if in_area else "out of area"
            base_poses.append(base_pose.tolist())

        statuses.append({
            "label": labels[i],
            "in_area": bool(in_area),
            "status": reason,
            "base_xyz": (
                [float(v) for v in base_pose[:3, 3]]
                if base_pose is not None
                else None
            ),
        })
        xyz_text = (
            "None"
            if base_pose is None
            else "[%.3f, %.3f, %.3f]" % tuple(base_pose[:3, 3])
        )
        print(
            f"[WorkArea] label={labels[i]!r}, base_xyz={xyz_text}, "
            f"status={reason}"
        )
        if in_area:
            keep_indices.append(i)

    filtered_objects = []
    for i in keep_indices:
        obj = dict(objects[i]) if i < len(objects) and isinstance(objects[i], dict) else {}
        obj["work_area"] = statuses[i]
        filtered_objects.append(obj)

    return {
        "pose_all": poses[keep_indices].tolist() if keep_indices else None,
        "length_all": lengths[keep_indices].tolist() if keep_indices else None,
        "labels": [labels[i] for i in keep_indices],
        "objects": filtered_objects,
        "statuses": statuses,
        "base_poses": base_poses,
        "out_of_area_count": sum(1 for item in statuses if not item["in_area"]),
        "area_check_available": area is None or base_to_camera is not None,
    }


VIS_PALETTE_BGR = [
    (75, 62, 20),
    (25, 72, 80),
    (78, 35, 60),
    (38, 78, 45),
    (78, 52, 25),
    (50, 42, 78),
    (25, 52, 80),
    (62, 78, 28),
]


def _stylize_frame_for_overlay(color_image):
    """Make the camera frame a little smoother and more animation-like."""
    base = color_image.copy()
    smooth = cv2.bilateralFilter(base, 7, 55, 55)
    vis = cv2.addWeighted(base, 0.55, smooth, 0.45, 0)
    vis = cv2.convertScaleAbs(vis, alpha=1.03, beta=4)

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edge_tint = np.zeros_like(vis)
    edge_tint[edges > 0] = (18, 22, 28)
    vis = cv2.addWeighted(vis, 1.0, edge_tint, 0.18, 0)
    return vis


def _draw_translucent_rect(img, x1, y1, x2, y2, color, alpha):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def _draw_label_chip(img, label, anchor, color, index):
    text = str(label or "object").strip().replace("_", " ")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 7
    x, y = int(anchor[0]), int(anchor[1])
    x1 = max(8, min(img.shape[1] - tw - pad_x * 2 - 8, x))
    y1 = max(10, min(img.shape[0] - th - baseline - pad_y * 2 - 8, y))
    x2 = x1 + tw + pad_x * 2
    y2 = y1 + th + baseline + pad_y * 2

    shadow = img.copy()
    cv2.rectangle(shadow, (x1 + 3, y1 + 3), (x2 + 3, y2 + 3), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.22, img, 0.78, 0, dst=img)

    _draw_translucent_rect(img, x1, y1, x2, y2, (18, 22, 32), 0.72)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x1 + 10, y1 + (y2 - y1) // 2), 4, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (x1 + pad_x + 10, y2 - pad_y - baseline),
        font,
        scale,
        (248, 250, 255),
        thickness,
        cv2.LINE_AA,
    )


def _apply_animated_mask_overlay(vis, mask, color, alpha=0.45):
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return None

    color_arr = np.array(color, dtype=np.float32)
    pixels = vis[mask_bool].astype(np.float32)
    vis[mask_bool] = np.clip(pixels * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        glow = np.zeros_like(vis)
        cv2.drawContours(glow, contours, -1, color, 5, lineType=cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 4)
        cv2.addWeighted(glow, 0.10, vis, 1.0, 0, dst=vis)
        cv2.drawContours(vis, contours, -1, color, 1, lineType=cv2.LINE_AA)

    ys, xs = np.where(mask_bool)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def visualize_sam_results(color_image, masks, labels, draw_labels=False):
    """Visualize SAM segmentation results with a single optional label style."""
    vis = _stylize_frame_for_overlay(color_image)
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = VIS_PALETTE_BGR[i % len(VIS_PALETTE_BGR)]
        bbox = _apply_animated_mask_overlay(vis, mask, color)
        if draw_labels and bbox is not None:
            x1, y1, _, _ = bbox
            _draw_label_chip(vis, label, (x1, max(8, y1 - 34)), color, i + 1)
    return vis


def visualize_mask_pose_results(
    color_image,
    masks,
    labels,
    pose_all,
    length_all,
    estimator,
    work_area_statuses=None,
):
    """Compose mask and pose into one polished frame with one label per object."""
    vis = _stylize_frame_for_overlay(color_image)
    label_anchors = []
    work_area_statuses = list(work_area_statuses or [])

    for i, (mask, label) in enumerate(zip(masks or [], labels or [])):
        area_status = work_area_statuses[i] if i < len(work_area_statuses) else {}
        out_of_area = area_status.get("status") == "out of area"
        color = (40, 40, 235) if out_of_area else VIS_PALETTE_BGR[i % len(VIS_PALETTE_BGR)]
        bbox = _apply_animated_mask_overlay(vis, mask, color)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        display_label = f"{label} | out of area" if out_of_area else label
        label_anchors.append((display_label, (x1, max(8, y1 - 34)), color, i + 1))

        # A subtle target mark at the mask center gives the still image a motion-graphic feel.
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        cv2.circle(vis, (cx, cy), 5, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    if estimator is not None:
        vis = estimator.visualize(vis, pose_all, length_all, labels=[], draw_labels=False)

    for label, anchor, color, index in label_anchors:
        _draw_label_chip(vis, label, anchor, color, index)

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
        "siglip_estimator": None,
        "marvin_action_node": None,
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
        "perception_lock": threading.RLock(),
        "ui_lock": threading.Lock(),
        "latest_task_loop_image": None,
        "latest_task_loop_status": "",
        "latest_task_loop_iteration": 0,
        "suppress_contained_masks": _env_bool("SAM3_SUPPRESS_CONTAINED_MASKS", True),
    }

    # Default YAML paths
    DEFAULT_MARVIN_OBJECT_YAML = _default_marvin_object_yaml()
    DEFAULT_MARVIN_TASK_YAML = _default_marvin_task_yaml()
    auto_app_cfg = _load_auto_app_config()
    sam3_app_cfg = auto_app_cfg.get("sam3", {}) if isinstance(auto_app_cfg, dict) else {}
    DEFAULT_SAM3_PROMPTS = os.getenv(
        "SAM3_PROMPTS",
        str(
            sam3_app_cfg.get(
                "prompts",
                "basket, control, grey screwdriver, yellow screwdriver, black screwdriver, saw, knife, pliers, graver, glass scraper",
            )
        ),
    )

    def init_models():
        """初始化模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict["device"] = device

        # 确保ROS2已初始化
        init_ros2()

        if state_dict["sam3_segmenter"] is None:
            state_dict["sam3_segmenter"] = initialize_sam3(device)

        if state_dict["flowpose_estimator"] is None:
            state_dict["flowpose_estimator"] = FlowPoseEstimator(device)

        if state_dict["siglip_estimator"] is None:
            state_dict["siglip_estimator"] = _get_task_siglip_estimator(device)

        msg_status = "✅" if HAS_KEYPOINT_MSG else "⚠️ (无KeypointPose消息)"
        return (
            f"✅ 模型初始化完成 (设备: {device})\n"
            f"🧠 SAM3语义分割已就绪\n"
            f"🎯 FlowPose姿态估计已就绪\n"
            f"🔎 Siglip状态验证已就绪\n"
            f"📡 ROS2节点已启动 {msg_status}"
        )

    def capture_frame():
        """捕获RealSense帧，后续由SAM3语义prompt分割。"""
        color_image, depth_image, depth_scale = capture_realsense_frame()
        if _ros2_node is not None:
            _ros2_node.clear_object_tf_cache()
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
        return rgb_image, f"✅ 已捕获front相机RealSense图像 (深度缩放: {depth_scale:.6f})"

    def run_sam_segmentation(prompts_text, suppress_contained_masks=None):
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
                suppress_contained_masks=suppress_contained_masks,
            )
        except Exception as e:
            return None, f"❌ SAM3分割失败: {e}"

        state_dict["masks"] = masks if masks else None
        state_dict["labels"] = labels if labels else None
        state_dict["all_labels"] = list(labels)
        state_dict["sam3_detections"] = detections
        segment_stats = getattr(state_dict["sam3_segmenter"], "last_segment_stats", {}) or {}
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

        all_pose = flowpose_result.get("pose_all")
        all_length = flowpose_result.get("length_all")
        all_objects = flowpose_result.get("objects", [])
        area_result = _filter_flowpose_by_work_area(
            all_pose,
            all_length,
            labels,
            all_objects,
        )
        pose_all = area_result["pose_all"]
        length_all = area_result["length_all"]
        filtered_labels = area_result["labels"]
        objects = area_result["objects"]

        flowpose_result["pose_all"] = pose_all
        flowpose_result["length_all"] = length_all
        flowpose_result["objects"] = objects
        flowpose_result["work_area"] = area_result
        state_dict["pose_results"] = pose_all
        state_dict["length_results"] = length_all
        state_dict["labels"] = filtered_labels
        state_dict["data_results"] = flowpose_result

        if pose_all is not None and _ros2_node is not None:
            _ros2_node.publish_poses_as_tf(
                np.asarray(pose_all, dtype=np.float32),
                filtered_labels,
            )

        vis = visualize_mask_pose_results(
            state_dict["color_image"],
            state_dict.get("masks") or [],
            labels,
            all_pose,
            all_length,
            state_dict["flowpose_estimator"],
            work_area_statuses=area_result["statuses"],
        )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        return (
            vis_rgb,
            f"✅ FlowPose姿态估计完成, objects={len(objects)}, "
            f"out_of_area={area_result['out_of_area_count']}, "
            f"elapsed={flowpose_result.get('elapsed_sec', 0.0)}s\n"
            f"📡 已发布TF到 /tf 话题"
        )

    def ensure_models_ready():
        """Ensure SAM3, FlowPose, Siglip and ROS2 are initialized."""
        if (
            state_dict["sam3_segmenter"] is not None
            and state_dict["flowpose_estimator"] is not None
            and state_dict["siglip_estimator"] is not None
        ):
            return "✅ 模型已就绪"
        return init_models()

    def _capture_segment_pose_impl(prompts_text, suppress_contained_masks=None):
        """Capture one RealSense frame, run SAM3 semantic segmentation, FlowPose, and publish TF."""
        global _ros2_node

        try:
            init_msg = ensure_models_ready()
        except Exception as e:
            return None, f"❌ 模型自动初始化失败: {e}"

        prompts = _parse_semantic_prompts(prompts_text)
        if not prompts:
            return None, f"{init_msg}\n❌ 请输入至少一个语义目标，例如: basket, control"
        suppress_contained_masks = state_dict["suppress_contained_masks"] if suppress_contained_masks is None else bool(suppress_contained_masks)

        try:
            color_image, depth_image, depth_scale = capture_realsense_frame()
        except Exception as e:
            return None, f"{init_msg}\n❌ front相机RealSense捕获失败: {e}"

        if _ros2_node is not None:
            _ros2_node.clear_object_tf_cache()

        state_dict["color_image"] = color_image
        state_dict["depth_image"] = depth_image
        state_dict["depth_scale"] = depth_scale
        state_dict["temp_image"] = color_image.copy()
        state_dict["masks"] = None
        state_dict["labels"] = None
        state_dict["all_labels"] = []
        state_dict["sam3_detections"] = []
        state_dict["pose_results"] = None
        state_dict["length_results"] = None
        state_dict["data_results"] = None
        state_dict["tracking_enabled"] = False
        state_dict["trackers"] = []

        try:
            masks, labels, detections, prompt_timings = state_dict["sam3_segmenter"].segment(
                color_image,
                prompts,
                suppress_contained_masks=suppress_contained_masks,
            )
        except Exception as e:
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            return rgb_image, f"{init_msg}\n✅ 已捕获front相机RealSense图像\n❌ SAM3分割失败: {e}"

        state_dict["masks"] = masks if masks else None
        state_dict["labels"] = labels if labels else None
        state_dict["all_labels"] = list(labels)
        state_dict["sam3_detections"] = detections
        segment_stats = getattr(state_dict["sam3_segmenter"], "last_segment_stats", {}) or {}
        state_dict["prev_masks"] = None
        state_dict["prev_labels"] = None
        state_dict["prev_points"] = None
        for label in labels:
            if label not in state_dict["label_history"]:
                state_dict["label_history"].append(label)

        if not masks:
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            timing_text = ", ".join(
                f"{item['prompt']}={item['elapsed_sec']:.3f}s" for item in prompt_timings
            )
            return (
                rgb_image,
                f"{init_msg}\n"
                f"✅ 已捕获front相机RealSense图像 (depth_scale={depth_scale:.6f})\n"
                f"⚠️ SAM3未检测到目标: {', '.join(prompts)}\n"
                f"耗时: {timing_text}"
            )

        mask_vis = visualize_sam_results(color_image, masks, labels, draw_labels=True)
        combined_mask, obj_ids = create_combined_mask(masks)
        if combined_mask is None or not obj_ids:
            vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
            return vis_rgb, f"{init_msg}\n⚠️ 当前mask为空，无法运行FlowPose"

        try:
            flowpose_result = state_dict["flowpose_estimator"].infer(
                rgb=color_image,
                depth=depth_image,
                combined_mask=combined_mask,
                obj_ids=obj_ids,
                class_names=list(labels),
                instance_names=list(labels),
                depth_scale=depth_scale or 0.001,
            )
        except Exception as e:
            vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
            return vis_rgb, f"{init_msg}\n✅ SAM3完成: detections={len(detections)}\n❌ FlowPose姿态估计失败: {e}"

        all_pose = flowpose_result.get("pose_all")
        all_length = flowpose_result.get("length_all")
        all_objects = flowpose_result.get("objects", [])
        area_result = _filter_flowpose_by_work_area(
            all_pose,
            all_length,
            labels,
            all_objects,
        )
        pose_all = area_result["pose_all"]
        length_all = area_result["length_all"]
        filtered_labels = area_result["labels"]
        objects = area_result["objects"]

        flowpose_result["pose_all"] = pose_all
        flowpose_result["length_all"] = length_all
        flowpose_result["objects"] = objects
        flowpose_result["work_area"] = area_result
        state_dict["pose_results"] = pose_all
        state_dict["length_results"] = length_all
        state_dict["labels"] = filtered_labels
        state_dict["all_labels"] = list(filtered_labels)
        state_dict["data_results"] = flowpose_result

        if pose_all is not None and _ros2_node is not None:
            _ros2_node.publish_poses_as_tf(
                np.asarray(pose_all, dtype=np.float32),
                filtered_labels,
            )

        vis = visualize_mask_pose_results(
            color_image,
            masks,
            labels,
            all_pose,
            all_length,
            state_dict["flowpose_estimator"],
            work_area_statuses=area_result["statuses"],
        )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        timing_text = ", ".join(
            f"{item['prompt']}={item['elapsed_sec']:.3f}s" for item in prompt_timings
        )
        return (
            vis_rgb,
            f"{init_msg}\n"
            f"✅ 已捕获front相机RealSense图像 (depth_scale={depth_scale:.6f})\n"
            f"✅ SAM3完成: prompts={len(prompts)}, detections={len(detections)}, labels={', '.join(labels)}\n"
            f"🧺 包含过滤: {'开启' if suppress_contained_masks else '关闭'}, "
            f"suppressed_contained={segment_stats.get('suppressed_contained', 0)}\n"
            f"✅ FlowPose完成: objects={len(objects)}, "
            f"out_of_area={area_result['out_of_area_count']}, "
            f"elapsed={flowpose_result.get('elapsed_sec', 0.0)}s\n"
            f"📡 已发布并保活目标TF到 /tf\n"
            f"SAM3耗时: {timing_text}"
        )

    def capture_segment_pose(prompts_text, suppress_contained_masks):
        with state_dict["perception_lock"]:
            state_dict["suppress_contained_masks"] = bool(suppress_contained_masks)
            return _capture_segment_pose_impl(prompts_text, suppress_contained_masks)

    def capture_segment_pose_default(suppress_contained_masks):
        image, _status = capture_segment_pose(DEFAULT_SAM3_PROMPTS, suppress_contained_masks)
        with state_dict["ui_lock"]:
            state_dict["latest_task_loop_image"] = image
            state_dict["latest_task_loop_status"] = str(_status or "")
            state_dict["latest_task_loop_iteration"] = 0
        return image

    def latest_task_loop_image():
        with state_dict["ui_lock"]:
            image = state_dict.get("latest_task_loop_image")
        if image is None:
            return gr.update()
        return image

    def _fresh_tf_prompts_text(target=None):
        configured = os.getenv("FRESH_TF_PROMPTS", "").strip()
        if configured:
            return configured

        mode = os.getenv("FRESH_TF_PROMPT_MODE", "default").strip().lower()
        if mode == "target" and target:
            values = target if isinstance(target, (list, tuple, set)) else [target]
            prompts = []
            for value in values:
                text = str(value or "").strip().lower()
                text = re.sub(r"_\d+$", "", text)
                text = text.replace("_", " ")
                if text and text not in prompts:
                    prompts.append(text)
            if prompts:
                return ", ".join(prompts)

        return DEFAULT_SAM3_PROMPTS

    def fresh_tf_capture(reason="", target=None):
        prompts_text = _fresh_tf_prompts_text(target)
        suppress_contained = _env_bool(
            "FRESH_TF_SUPPRESS_CONTAINED_MASKS",
            state_dict["suppress_contained_masks"],
        )
        with state_dict["perception_lock"]:
            _, status = _capture_segment_pose_impl(prompts_text, suppress_contained)
        status_text = str(status or "")
        ok = "✅ FlowPose完成" in status_text and "📡 已发布" in status_text
        compact_status = " | ".join(line.strip() for line in status_text.splitlines() if line.strip())
        return ok, compact_status

    set_fresh_tf_callback(fresh_tf_capture)

    # ==================== Marvin Action Execution Functions ====================

    def _task_prompts_text(task_yaml_path):
        steps = _load_task_yaml_steps(task_yaml_path)
        prompts = []
        for step in steps:
            for target in step.target_names():
                if normalize_obj_name(target) == "home":
                    continue
                text = str(target).strip().replace("_", " ")
                if text and text not in prompts:
                    prompts.append(text)
        return ", ".join(prompts) if prompts else DEFAULT_SAM3_PROMPTS

    def _task_loop_worker(task_yaml_path):
        max_iterations = int(os.getenv("TASK_LOOP_MAX_ITERATIONS", "100"))
        empty_confirmations = max(
            1,
            int(os.getenv("TASK_LOOP_EMPTY_CONFIRMATIONS", "3")),
        )
        empty_retry_interval = max(
            0.0,
            float(os.getenv("TASK_LOOP_EMPTY_RETRY_INTERVAL_SEC", "0.5")),
        )
        consecutive_empty = 0
        task_steps = _load_task_yaml_steps(task_yaml_path)
        first_step = task_steps[0] if task_steps else None
        target_norms = {
            normalize_obj_name(target)
            for target in (first_step.target_names() if first_step is not None else [])
        }
        prompts_text = _task_prompts_text(task_yaml_path)
        suppress_contained = state_dict["suppress_contained_masks"]
        auto_cfg = _load_auto_app_config()
        fresh_tf_cfg = auto_cfg.get("fresh_tf", {}) if isinstance(auto_cfg, dict) else {}
        reuse_manual_first_frame = _env_bool(
            "TASK_LOOP_REUSE_MANUAL_FIRST_FRAME",
            bool(fresh_tf_cfg.get("reuse_manual_first_frame", False)),
        )
        for iteration in range(1, max_iterations + 1):
            if _task_loop_stop_event.is_set():
                print("[TaskLoop] stop requested")
                break

            latest_frames = set()
            if _ros2_node is not None:
                getter = getattr(_ros2_node, "current_object_child_frames", None)
                if callable(getter):
                    latest_frames = set(getter())
            manual_has_task_target = any(
                normalize_obj_name(frame) in target_norms
                for frame in latest_frames
            )
            use_manual_first_frame = (
                reuse_manual_first_frame
                and
                iteration == 1
                and state_dict.get("pose_results") is not None
                and state_dict.get("labels")
                and _ros2_node is not None
                and getattr(_ros2_node, "latest_object_tf_cache", None) is not None
                and manual_has_task_target
            )
            if use_manual_first_frame:
                status_text = "✅ 使用手动捕获的上一帧FlowPose结果作为第1轮TF"
                if _ros2_node is not None:
                    with contextlib.suppress(Exception):
                        _ros2_node.republish_latest_object_tfs()
                    time.sleep(float(os.getenv("TASK_LOOP_MANUAL_TF_SETTLE_SEC", "0.2")))
                print(f"[TaskLoop] perception iteration={iteration}: {status_text}")
            else:
                if iteration == 1 and latest_frames:
                    debug_frames = ", ".join(
                        f"{frame}->{normalize_obj_name(frame)}"
                        for frame in sorted(latest_frames)
                    )
                    print(
                        "[TaskLoop] manual frame has no task target; recapturing. "
                        f"available={debug_frames}, targets={sorted(target_norms)}"
                    )
                with state_dict["perception_lock"]:
                    image, status = _capture_segment_pose_impl(prompts_text, suppress_contained)
                status_text = str(status or "")
                with state_dict["ui_lock"]:
                    state_dict["latest_task_loop_image"] = image
                    state_dict["latest_task_loop_status"] = status_text
                    state_dict["latest_task_loop_iteration"] = iteration
                print(f"[TaskLoop] perception iteration={iteration}: {status_text}")
                if "✅ FlowPose完成" not in status_text:
                    consecutive_empty += 1
                    print(
                        f"[TaskLoop] no FlowPose result: "
                        f"{consecutive_empty}/{empty_confirmations}"
                    )
                    if consecutive_empty >= empty_confirmations:
                        print(
                            "[TaskLoop] no target confirmed in consecutive frames; "
                            "stop loop"
                        )
                        break
                    if empty_retry_interval > 0:
                        time.sleep(empty_retry_interval)
                    continue

            node = _robotaction_node
            if node is None:
                print("[TaskLoop] action node stopped")
                break

            inst = node.select_target_instance()
            if not inst:
                consecutive_empty += 1
                print(
                    f"[TaskLoop] no remaining task target detected: "
                    f"{consecutive_empty}/{empty_confirmations}"
                )
                if consecutive_empty >= empty_confirmations:
                    print(
                        "[TaskLoop] no target confirmed in consecutive frames; "
                        "stop loop"
                    )
                    break
                if empty_retry_interval > 0:
                    time.sleep(empty_retry_interval)
                continue

            consecutive_empty = 0
            result = node.run_task_for_instance(inst)
            print(f"[TaskLoop] selected={inst}, result={result}")
            if result != TaskStatus.SUCCESS:
                break

        siglip_failed = bool(
            _robotaction_node is not None
            and getattr(_robotaction_node, "siglip_verification_failed", False)
        )
        if not _task_loop_stop_event.is_set() and not siglip_failed:
            node = _robotaction_node
            if node is not None:
                try:
                    print("[TaskLoop] loop ended; publishing both arms home")
                    node.publish_home_both()
                    print("[TaskLoop] both arms home published")
                except Exception as e:
                    print(f"[TaskLoop] failed to publish both arms home: {e}")
        elif siglip_failed:
            print(
                "[TaskLoop] Siglip verification failed; current arm home already "
                "published, skip both-arms home"
            )
        print("[TaskLoop] loop finished")

    def start_task_loop_actions_default():
        """启动 task.yaml 驱动的循环动作端。"""
        global _task_loop_thread
        try:
            ensure_models_ready()
            node, msg = start_task_loop_node(
                object_yaml_path=DEFAULT_MARVIN_OBJECT_YAML,
                task_yaml_path=DEFAULT_MARVIN_TASK_YAML,
                progress_topic=_default_marvin_config_value("progress_topic", "MARVIN_PROGRESS_TOPIC", "/control/task_percentage"),
                object_tf_topic=_default_marvin_config_value("object_tf_topic", "MARVIN_OBJECT_TF_TOPIC", ""),
                base_frame=_default_marvin_config_value("base_frame", "MARVIN_BASE_FRAME", "base_link"),
            )
            state_dict["marvin_action_node"] = node
            _task_loop_stop_event.clear()
            if _task_loop_thread is None or not _task_loop_thread.is_alive():
                _task_loop_thread = threading.Thread(
                    target=_task_loop_worker,
                    args=(DEFAULT_MARVIN_TASK_YAML,),
                    name="task-loop-worker",
                    daemon=True,
                )
                _task_loop_thread.start()
            return f"✅ {msg}\n{_format_marvin_action_status()}", _format_marvin_action_status()
        except Exception as e:
            state_dict["marvin_action_node"] = None
            return f"❌ TaskLoop动作端启动失败: {e}", _format_marvin_action_status()

    def stop_marvin_actions():
        """停止 task loop，不额外发布home。"""
        stop_task_loop(publish_home=False)
        stopped = stop_marvin_action_node(publish_home=False, destroy=True)
        state_dict["marvin_action_node"] = _robotaction_node
        msg = "✅ TaskLoop动作端已停止" if stopped else "⚠️ TaskLoop动作端未运行"
        return msg, _format_marvin_action_status()

    def marvin_home_both():
        """通过Marvin动作端发布双臂home；若未启动则使用主ROS节点兜底。"""
        node = _robotaction_node
        try:
            if node is not None:
                node.publish_home_both()
                return "✅ 已通过Marvin动作端发布双臂 Home", _format_marvin_action_status()
            if _ros2_node is not None:
                ok = _ros2_node.publish_home_both()
                return ("✅ 已通过主ROS节点发布双臂 Home" if ok else "⚠️ Home发布失败"), _format_marvin_action_status()
            return "❌ ROS2节点未初始化", _format_marvin_action_status()
        except Exception as e:
            return f"❌ Home发布失败: {e}", _format_marvin_action_status()

    MARVIN_TMUX_SCRIPT = os.getenv(
        "MARVIN_TMUX_SCRIPT",
        "/home/snorlax/work/distri_0112_org/ros2_ws/start_marvin_tmux.sh",
    )

    def _run_marvin_tmux_command(*args, timeout=8.0):
        script_path = os.path.abspath(MARVIN_TMUX_SCRIPT)
        if not os.path.isfile(script_path):
            return f"❌ Marvin启动脚本不存在: {script_path}", _format_marvin_action_status()
        workdir = os.path.dirname(script_path)
        env = os.environ.copy()
        env.setdefault("TMUX_ATTACH", "0")
        env.setdefault("ROS_WS", workdir)
        cmd = ["bash", script_path, *[str(arg) for arg in args if str(arg)]]
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "").strip()
            return (
                f"⚠️ Marvin tmux命令超时，但可能已经执行\n"
                f"命令: {' '.join(cmd)}\n{output}",
                _format_marvin_action_status(),
            )
        output = (result.stdout or "").strip()
        if result.returncode == 0:
            return (
                f"✅ Marvin tmux命令已执行: {' '.join(args)}\n{output}",
                _format_marvin_action_status(),
            )
        return (
            f"❌ Marvin tmux命令失败: {' '.join(args)}\n"
            f"returncode={result.returncode}\n{output}",
            _format_marvin_action_status(),
        )

    def start_marvin_bringup():
        return _run_marvin_tmux_command("start", timeout=12.0)

    def stop_marvin_bringup():
        return _run_marvin_tmux_command("stop", timeout=8.0)

    def restart_marvin_planner():
        return _run_marvin_tmux_command("restart", "planner", timeout=8.0)

    def restart_marvin_control():
        return _run_marvin_tmux_command("restart", "control", timeout=8.0)

    def restart_marvin_gripper():
        return _run_marvin_tmux_command("restart", "gripper", timeout=8.0)

    def restart_marvin_task_manager():
        return _run_marvin_tmux_command("restart", "task_manager", timeout=8.0)

    try:
        initial_model_status = init_models()
    except Exception as e:
        initial_model_status = f"❌ 自动初始化模型失败: {e}"
        print(initial_model_status)

    # ==================== Gradio Interface ====================

    with gr.Blocks(title="SAM3 + FlowPose Task Loop") as demo:
        gr.Markdown("# 🎯 SAM3 + FlowPose Task Loop")
        gr.Markdown("### 步骤: 1️⃣ 程序自动初始化 → 2️⃣ 捕获front相机并完成SAM3+FlowPose → 3️⃣ 按 task.yaml 循环执行")
        task_loop_image_timer = gr.Timer(
            value=float(os.getenv("TASK_LOOP_UI_REFRESH_SEC", "0.5")),
            active=True,
        )

        with gr.Row():
            with gr.Column(scale=1):
                suppress_contained_input = gr.Checkbox(
                    label="过滤被大mask覆盖的小目标",
                    value=state_dict["suppress_contained_masks"],
                )
                perception_btn = gr.Button("📷 捕获并运行SAM3+FlowPose", variant="primary", size="lg")
            with gr.Column(scale=2):
                perception_output = gr.Image(
                    label="SAM3 Mask + FlowPose Pose结果",
                    type="numpy",
                    interactive=False,
                )

        # ==================== Action Execution Section ====================
        gr.Markdown("---")
        gr.Markdown("## 🤖 TaskLoop动作端")
        gr.Markdown("💡 **流程**: SAM3+FlowPose 当前目标列表 → task.yaml 顺序动作 → /fusion_pose → 下一轮感知")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 运行状态")
                current_task_display = gr.Textbox(
                    label="TaskLoop状态",
                    value="TaskLoop动作端未启动",
                    interactive=False,
                    lines=8,
                )
                action_log = gr.Textbox(label="动作日志", interactive=False, lines=8)

        with gr.Row():
            start_bringup_btn = gr.Button("🚀 启动机器人Bringup", variant="primary", size="lg")
            restart_control_btn = gr.Button("🔁 重发Control初始化", variant="secondary", size="lg")

        with gr.Row():
            restart_planner_btn = gr.Button("🔁 重启Planner", variant="secondary", size="sm")

        with gr.Row():
            start_marvin_btn = gr.Button("▶️ 启动TaskLoop动作端", variant="primary", size="lg")
            home_btn = gr.Button("🏠 Home", variant="secondary", size="lg")
            stop_btn = gr.Button("🛑 停止Marvin", variant="stop", size="lg")

        # ==================== 事件绑定 ====================
        perception_btn.click(
            fn=capture_segment_pose_default,
            inputs=[suppress_contained_input],
            outputs=[perception_output],
        )
        task_loop_image_timer.tick(
            fn=latest_task_loop_image,
            inputs=[],
            outputs=[perception_output],
        )

        # Marvin action bindings
        start_bringup_btn.click(fn=start_marvin_bringup, outputs=[action_log, current_task_display])
        restart_control_btn.click(fn=restart_marvin_control, outputs=[action_log, current_task_display])
        restart_planner_btn.click(fn=restart_marvin_planner, outputs=[action_log, current_task_display])
        start_marvin_btn.click(
            fn=start_task_loop_actions_default,
            inputs=[],
            outputs=[action_log, current_task_display],
        )
        home_btn.click(fn=marvin_home_both, outputs=[action_log, current_task_display])
        stop_btn.click(fn=stop_marvin_actions, outputs=[action_log, current_task_display])

    return demo


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
