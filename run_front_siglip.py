import asyncio
import base64
import contextlib
import copy
import os
import re
import sys
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

AUTO_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if AUTO_APP_DIR not in sys.path:
    sys.path.insert(0, AUTO_APP_DIR)

import run_siglip_pyrealsense as core

try:
    from sensor_msgs.msg import JointState
except Exception:
    JointState = None


MARVIN_JOINT_NAMES = [
    "Joint1_L", "Joint2_L", "Joint3_L", "Joint4_L", "Joint5_L", "Joint6_L", "Joint7_L",
    "Joint1_R", "Joint2_R", "Joint3_R", "Joint4_R", "Joint5_R", "Joint6_R", "Joint7_R",
]


class ApiError(Exception):
    def __init__(self, code, message, http_status=400):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.http_status = int(http_status)


class CaptureRequest(BaseModel):
    camera_id: str = "cam_01"
    width: Optional[int] = Field(default=1280, ge=1)
    height: Optional[int] = Field(default=720, ge=1)
    format: str = "jpeg"
    quality: int = Field(default=85, ge=1, le=100)
    return_mode: str = "base64"


class GraspRequest(BaseModel):
    target_object: Optional[str] = None
    approach_speed: float = Field(default=0.5, ge=0.1, le=1.0)
    grasp_force: float = Field(default=0.8, ge=0.1, le=1.0)
    trajectory_mode: str = "linear"


class StopRequest(BaseModel):
    mode: str = "reset"
    emergency: bool = False


def _api_envelope(data=None, message="success", code=0, http_status=200):
    return JSONResponse(
        status_code=http_status,
        content={"code": int(code), "message": str(message), "data": data},
    )


def _api_error_response(exc):
    if isinstance(exc, ApiError):
        return _api_envelope(None, exc.message, exc.code, exc.http_status)
    return _api_envelope(None, str(exc), 2001, 500)


def _operation_id(prefix="op"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _image_id():
    return f"img_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _env_bool(name, default=False):
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _normalize_label(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"_\d+$", "", text)
    return text.replace("_", " ")


class JointStateMonitor(core.Node):
    def __init__(self):
        super().__init__("run_task_joint_state_monitor")
        self.lock = threading.RLock()
        self.latest = {}
        self.updated_at = 0.0
        topic = os.getenv("MARVIN_JOINT_STATE_TOPIC", "/joint_states")
        if JointState is None:
            self.get_logger().warning("sensor_msgs/JointState unavailable; websocket will publish zero joints")
            return
        self.create_subscription(JointState, topic, self._on_joint_state, 20)
        self.get_logger().info(f"JointState websocket monitor subscribed: {topic}")

    def _on_joint_state(self, msg):
        snapshot = {}
        names = list(getattr(msg, "name", []) or [])
        positions = list(getattr(msg, "position", []) or [])
        velocities = list(getattr(msg, "velocity", []) or [])
        efforts = list(getattr(msg, "effort", []) or [])
        for i, name in enumerate(names):
            if not name:
                continue
            snapshot[str(name)] = {
                "angle_rad": float(positions[i]) if i < len(positions) else 0.0,
                "velocity_rad_s": float(velocities[i]) if i < len(velocities) else 0.0,
                "torque_nm": float(efforts[i]) if i < len(efforts) else 0.0,
            }
        if snapshot:
            with self.lock:
                self.latest.update(snapshot)
                self.updated_at = time.time()

    def message(self, seq):
        now = time.time()
        with self.lock:
            latest = copy.deepcopy(self.latest)
            updated_at = float(self.updated_at or 0.0)
        stale = updated_at <= 0.0 or (now - updated_at) > 1.0
        joints = []
        for name in MARVIN_JOINT_NAMES:
            data = latest.get(name)
            if data is None:
                joints.append({
                    "name": name,
                    "angle_rad": 0.0,
                    "velocity_rad_s": 0.0,
                    "torque_nm": 0.0,
                    "temperature_c": 0.0,
                    "status": "missing" if updated_at > 0.0 else "unknown",
                })
                continue
            joints.append({
                "name": name,
                "angle_rad": float(data.get("angle_rad", 0.0)),
                "velocity_rad_s": float(data.get("velocity_rad_s", 0.0)),
                "torque_nm": float(data.get("torque_nm", 0.0)),
                "temperature_c": 0.0,
                "status": "stale" if stale else "ok",
            })
        return {
            "type": "joint_state",
            "seq": int(seq),
            "timestamp": now,
            "robot_id": "marvin_01",
            "joints": joints,
            "end_effector": {
                "left": {"frame": "Joint7_L", "pose": None},
                "right": {"frame": "Joint7_R", "pose": None},
            },
        }


class RobotTaskApiService:
    def __init__(self):
        auto_cfg = core._load_auto_app_config()
        sam3_cfg = auto_cfg.get("sam3", {}) if isinstance(auto_cfg, dict) else {}
        self.default_sam3_prompts = os.getenv(
            "SAM3_PROMPTS",
            str(sam3_cfg.get(
                "prompts",
                "control, grey screwdriver, yellow screwdriver, saw, knife, pliers, graver, black screwdriver, red screwdriver",
            )),
        )
        self.object_yaml = core._default_marvin_object_yaml()
        self.task_yaml = core._default_marvin_task_yaml()
        self.progress_topic = core._default_marvin_config_value(
            "progress_topic", "MARVIN_PROGRESS_TOPIC", "/control/task_percentage"
        )
        self.object_tf_topic = core._default_marvin_config_value(
            "object_tf_topic", "MARVIN_OBJECT_TF_TOPIC", ""
        )
        self.base_frame = core._default_marvin_config_value(
            "base_frame", "MARVIN_BASE_FRAME", "base_link"
        )
        self.state = {
            "perception_lock": threading.RLock(),
            "sam3_segmenter": None,
            "flowpose_estimator": None,
            "siglip_estimator": None,
            "model_device": None,
            "suppress_contained_masks": _env_bool("SAM3_SUPPRESS_CONTAINED_MASKS", True),
            "last_capture": None,
            "labels": [],
            "pose_results": None,
            "data_results": None,
        }
        self.operations = {}
        self.operation_lock = threading.RLock()
        self.worker_thread = None
        self.worker_stop = threading.Event()
        self.joint_monitor = None

    def startup(self):
        core.init_ros2()
        self._ensure_joint_monitor()
        if _env_bool("API_INIT_MODELS", True):
            self.ensure_models_ready()
        core.set_fresh_tf_callback(self.fresh_tf_capture)

    def shutdown(self):
        self.worker_stop.set()
        core._task_loop_stop_event.set()
        core.set_fresh_tf_callback(None)
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            core.stop_marvin_action_node(publish_home=False, destroy=True)
        if self.joint_monitor is not None and core._ros2_executor is not None:
            with contextlib.suppress(Exception):
                core._ros2_executor.remove_node(self.joint_monitor)
            with contextlib.suppress(Exception):
                self.joint_monitor.destroy_node()
        core.shutdown_ros2()

    def _ensure_joint_monitor(self):
        if self.joint_monitor is not None:
            return
        self.joint_monitor = JointStateMonitor()
        if core._ros2_executor is not None:
            core._ros2_executor.add_node(self.joint_monitor)

    def ensure_models_ready(self):
        if (
            self.state["sam3_segmenter"] is not None
            and self.state["flowpose_estimator"] is not None
            and self.state["siglip_estimator"] is not None
        ):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.state["sam3_segmenter"] is None:
            self.state["sam3_segmenter"] = core.initialize_sam3(device)
        if self.state["flowpose_estimator"] is None:
            self.state["flowpose_estimator"] = core.FlowPoseEstimator(device)
        if self.state["siglip_estimator"] is None:
            self.state["siglip_estimator"] = core._get_task_siglip_estimator(device)
        self.state["model_device"] = str(device)

    def _task_prompts_text(self):
        try:
            steps = core._load_task_yaml_steps(self.task_yaml)
        except Exception:
            return self.default_sam3_prompts
        prompts = []
        for step in steps:
            targets = step.target_names()
            for target in targets:
                text = _normalize_label(target)
                if text and text != "home" and text not in prompts:
                    prompts.append(text)
        return ", ".join(prompts) if prompts else self.default_sam3_prompts

    def _fresh_tf_prompts_text(self, target=None):
        configured = os.getenv("FRESH_TF_PROMPTS", "").strip()
        if configured:
            return configured

        mode = os.getenv("FRESH_TF_PROMPT_MODE", "default").strip().lower()
        if mode == "target" and target:
            values = target if isinstance(target, (list, tuple, set)) else [target]
            prompts = []
            for value in values:
                text = _normalize_label(value)
                if text and text != "home" and text not in prompts:
                    prompts.append(text)
            if prompts:
                return ", ".join(prompts)

        return self._task_prompts_text()

    def fresh_tf_capture(self, reason="", target=None):
        prompts_text = self._fresh_tf_prompts_text(target)
        suppress_contained = _env_bool(
            "FRESH_TF_SUPPRESS_CONTAINED_MASKS",
            self.state["suppress_contained_masks"],
        )
        capture = self.capture_perception(prompts_text, suppress_contained)
        status = str(capture.get("status", ""))
        labels = capture.get("labels", []) or []
        message = (
            f"reason={reason}, status={status}, labels={labels}, "
            f"message={capture.get('message', '')}"
        )
        return status == "ok", message

    def capture_perception(self, prompts_text=None, suppress_contained_masks=None):
        with self.state["perception_lock"]:
            return self._capture_perception_impl(
                prompts_text or self.default_sam3_prompts,
                self.state["suppress_contained_masks"] if suppress_contained_masks is None else bool(suppress_contained_masks),
            )

    def _capture_perception_impl(self, prompts_text, suppress_contained_masks):
        self.ensure_models_ready()
        prompts = core._parse_semantic_prompts(prompts_text)
        if not prompts:
            raise ApiError(1001, "语义目标为空", 400)
        try:
            color_image, depth_image, depth_scale = core.capture_realsense_frame()
        except Exception as exc:
            raise ApiError(1004, f"相机未连接 / 捕获失败: {exc}", 409)

        if core._ros2_node is not None:
            core._ros2_node.clear_object_tf_cache()

        try:
            masks, labels, detections, prompt_timings = self.state["sam3_segmenter"].segment(
                color_image, prompts, suppress_contained_masks=suppress_contained_masks
            )
        except Exception as exc:
            raise ApiError(2001, f"SAM3分割失败: {exc}", 500)

        segment_stats = getattr(self.state["sam3_segmenter"], "last_segment_stats", {}) or {}
        result = {
            "labels": list(labels or []),
            "detections": detections or [],
            "objects": [],
            "status": "no_detection",
            "message": f"SAM3未检测到目标: {', '.join(prompts)}",
            "depth_scale": float(depth_scale or 0.001),
            "segment_stats": segment_stats,
            "prompt_timings": prompt_timings,
        }
        if not masks:
            result["image_rgb"] = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self._save_perception_state(result, None, None)
            return result

        combined_mask, obj_ids = core.create_combined_mask(masks)
        if combined_mask is None or not obj_ids:
            vis = core.visualize_sam_results(color_image, masks, labels, draw_labels=True)
            result.update({
                "image_rgb": cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                "status": "mask_empty",
                "message": "当前mask为空，无法运行FlowPose",
            })
            self._save_perception_state(result, None, None)
            return result

        try:
            flowpose_result = self.state["flowpose_estimator"].infer(
                rgb=color_image,
                depth=depth_image,
                combined_mask=combined_mask,
                obj_ids=obj_ids,
                class_names=list(labels),
                instance_names=list(labels),
                depth_scale=depth_scale or 0.001,
            )
        except Exception as exc:
            raise ApiError(2001, f"FlowPose姿态估计失败: {exc}", 500)

        all_pose = flowpose_result.get("pose_all")
        all_length = flowpose_result.get("length_all")
        all_objects = flowpose_result.get("objects", [])
        area_result = core._filter_flowpose_by_work_area(
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
        if pose_all is not None and core._ros2_node is not None:
            core._ros2_node.publish_poses_as_tf(
                np.asarray(pose_all, dtype=np.float32),
                filtered_labels,
            )

        vis = core.visualize_mask_pose_results(
            color_image,
            masks,
            labels,
            all_pose,
            all_length,
            self.state["flowpose_estimator"],
            work_area_statuses=area_result["statuses"],
        )
        result.update({
            "image_rgb": cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            "labels": filtered_labels,
            "objects": objects,
            "status": "ok",
            "message": (
                f"success; objects={len(objects)}, "
                f"out_of_area={area_result['out_of_area_count']}"
            ),
            "flowpose": flowpose_result,
            "work_area": area_result,
        })
        self._save_perception_state(result, pose_all, flowpose_result)
        return result

    def _save_perception_state(self, result, pose_all, flowpose_result):
        self.state["last_capture"] = result
        self.state["labels"] = list(result.get("labels", []) or [])
        self.state["pose_results"] = pose_all
        self.state["data_results"] = flowpose_result

    def encode_capture(self, capture_result, request):
        image_rgb = np.asarray(capture_result["image_rgb"], dtype=np.uint8)
        fmt = str(request.format or "jpeg").strip().lower()
        if fmt in {"jpg", "jpeg"}:
            fmt, ext, params = "jpeg", ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(request.quality)]
        elif fmt == "png":
            ext, params = ".png", []
        else:
            raise ApiError(1001, "format 只支持 jpeg / png", 400)

        width = int(request.width or image_rgb.shape[1])
        height = int(request.height or image_rgb.shape[0])
        if width != image_rgb.shape[1] or height != image_rgb.shape[0]:
            image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(ext, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), params)
        if not ok:
            raise ApiError(2001, "图像编码失败", 500)
        image_bytes = encoded.tobytes()
        return {
            "image_id": _image_id(),
            "timestamp": time.time(),
            "width": width,
            "height": height,
            "format": fmt,
            "size_bytes": len(image_bytes),
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "image_url": None,
            "perception": {
                "status": capture_result.get("status"),
                "message": capture_result.get("message"),
                "labels": capture_result.get("labels", []),
                "detections": capture_result.get("detections", []),
                "objects": capture_result.get("objects", []),
                "depth_scale": capture_result.get("depth_scale"),
                "segment_stats": capture_result.get("segment_stats", {}),
            },
        }

    def capture(self, request):
        return self.encode_capture(self.capture_perception(), request)

    def _store_operation(self, operation_type, status="running", **extra):
        now = time.time()
        op = {
            "operation_id": _operation_id(),
            "operation_type": operation_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        op.update(extra)
        self.operations[op["operation_id"]] = op
        return op

    def _has_running_grasp_locked(self):
        for op in self.operations.values():
            if op.get("operation_type") == "grasp" and op.get("status") == "running":
                return op
        return None

    def _task_target_norms(self):
        steps = core._load_task_yaml_steps(self.task_yaml)
        if not steps:
            return set()
        return {_normalize_label(v) for v in steps[0].target_names() if _normalize_label(v)}

    def _manual_frame_has_task_target(self):
        if not self.state.get("pose_results") is not None:
            return False
        if core._ros2_node is None:
            return False
        frames = []
        getter = getattr(core._ros2_node, "current_object_child_frames", None)
        if callable(getter):
            frames = list(getter() or [])
        available = {_normalize_label(frame) for frame in frames}
        return bool(available & self._task_target_norms())

    def _set_operation_done(self, operation_id, status, result, **extra):
        with self.operation_lock:
            op = self.operations.get(operation_id)
            if not op:
                return
            now = time.time()
            op["status"] = status
            op["result"] = result
            op["updated_at"] = now
            op["duration_ms"] = int((now - float(op.get("created_at", now))) * 1000)
            op.update(extra)

    def _task_loop_worker(self, operation_id):
        prompts_text = self._task_prompts_text()
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
        suppress_contained = _env_bool("TASK_LOOP_SUPPRESS_CONTAINED_MASKS", self.state["suppress_contained_masks"])
        try:
            for iteration in range(1, max_iterations + 1):
                if self.worker_stop.is_set() or core._task_loop_stop_event.is_set():
                    self._set_operation_done(operation_id, "aborted", "aborted", iteration=iteration)
                    return

                use_manual = iteration == 1 and self._manual_frame_has_task_target()
                if use_manual:
                    print(f"[TaskLoopAPI] perception iteration={iteration}: use previous manual capture")
                    if core._ros2_node is not None:
                        core._ros2_node.republish_latest_object_tfs()
                    time.sleep(float(os.getenv("TASK_LOOP_MANUAL_TF_SETTLE_SEC", "0.2")))
                else:
                    cap = self.capture_perception(prompts_text, suppress_contained)
                    print(
                        f"[TaskLoopAPI] perception iteration={iteration}: "
                        f"status={cap.get('status')} labels={cap.get('labels', [])}"
                    )
                    if cap.get("status") != "ok":
                        consecutive_empty += 1
                        print(
                            f"[TaskLoopAPI] no FlowPose result: "
                            f"{consecutive_empty}/{empty_confirmations}"
                        )
                        if consecutive_empty >= empty_confirmations:
                            self._set_operation_done(
                                operation_id,
                                "completed",
                                "success",
                                final_reason="no_target_confirmed",
                                iteration=iteration,
                            )
                            print("[TaskLoopAPI] no target confirmed; loop complete")
                            return
                        if empty_retry_interval > 0:
                            self.worker_stop.wait(timeout=empty_retry_interval)
                        continue

                node = core._robotaction_node
                if node is None:
                    raise RuntimeError("TaskLoop动作端未启动")

                inst = node.select_target_instance()
                if not inst:
                    consecutive_empty += 1
                    print(
                        f"[TaskLoopAPI] no remaining task target detected: "
                        f"{consecutive_empty}/{empty_confirmations}"
                    )
                    if consecutive_empty >= empty_confirmations:
                        self._set_operation_done(
                            operation_id,
                            "completed",
                            "success",
                            final_reason="no_remaining_task_target",
                            iteration=iteration,
                        )
                        print("[TaskLoopAPI] no remaining task target detected; loop complete")
                        return
                    if empty_retry_interval > 0:
                        self.worker_stop.wait(timeout=empty_retry_interval)
                    continue

                consecutive_empty = 0
                status = node.run_task_for_instance(inst)
                if status != core.TaskStatus.SUCCESS:
                    if status == core.TaskStatus.STOP:
                        self._set_operation_done(operation_id, "aborted", "aborted", iteration=iteration, selected=inst)
                    else:
                        self._set_operation_done(
                            operation_id,
                            "failed",
                            str(status),
                            iteration=iteration,
                            selected=inst,
                        )
                    return

            self._set_operation_done(operation_id, "completed", "success", final_reason="max_iterations")
        except Exception as exc:
            self._set_operation_done(operation_id, "failed", str(exc))
        finally:
            siglip_failed = bool(
                core._robotaction_node is not None
                and getattr(core._robotaction_node, "siglip_verification_failed", False)
            )
            should_home = (
                not self.worker_stop.is_set()
                and not core._task_loop_stop_event.is_set()
                and not siglip_failed
            )
            if should_home:
                node = core._robotaction_node
                if node is not None:
                    try:
                        print("[TaskLoopAPI] loop ended; publishing both arms home")
                        node.publish_home_both()
                        print("[TaskLoopAPI] both arms home published")
                    except Exception as exc:
                        print(f"[TaskLoopAPI] failed to publish both arms home: {exc}")
            elif siglip_failed:
                print(
                    "[TaskLoopAPI] Siglip verification failed; current arm home "
                    "already published, skip both-arms home"
                )

    def start_grasp(self, request):
        if request.trajectory_mode not in {"linear", "arc"}:
            raise ApiError(1001, "trajectory_mode 只支持 linear / arc", 400)
        with self.operation_lock:
            if self._has_running_grasp_locked() is not None:
                raise ApiError(1002, "机器人正在执行操作中，请先停止或等待完成", 409)
            try:
                core._task_loop_stop_event.clear()
                self.worker_stop.clear()
                node, message = core.start_task_loop_node(
                    object_yaml_path=self.object_yaml,
                    task_yaml_path=self.task_yaml,
                    progress_topic=self.progress_topic,
                    object_tf_topic=self.object_tf_topic,
                    base_frame=self.base_frame,
                )
            except Exception as exc:
                raise ApiError(1003, f"机器人未连接 / TaskLoop启动失败: {exc}", 409)

            op = self._store_operation(
                "grasp",
                status="running",
                target_object=request.target_object,
                approach_speed=float(request.approach_speed),
                grasp_force=float(request.grasp_force),
                trajectory_mode=request.trajectory_mode,
                marvin_message=message,
                object_yaml=self.object_yaml,
                task_yaml=self.task_yaml,
            )
            self.worker_thread = threading.Thread(
                target=self._task_loop_worker,
                args=(op["operation_id"],),
                daemon=True,
            )
            self.worker_thread.start()
            return {"operation_id": op["operation_id"], "estimated_duration_ms": 4000}

    def stop(self, request):
        if request.mode not in {"reset", "hold"}:
            raise ApiError(1001, "mode 只支持 reset / hold", 400)
        self.worker_stop.set()
        core._task_loop_stop_event.set()
        with contextlib.suppress(Exception):
            core.stop_marvin_action_node(publish_home=True, destroy=False)
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=float(os.getenv("TASK_LOOP_STOP_JOIN_TIMEOUT", "1.0")))
        with self.operation_lock:
            now = time.time()
            for op in self.operations.values():
                if op.get("operation_type") == "grasp" and op.get("status") == "running":
                    op["status"] = "aborted"
                    op["result"] = "aborted"
                    op["updated_at"] = now
                    op["duration_ms"] = int((now - float(op.get("created_at", now))) * 1000)
            stop_op = self._store_operation(
                "stop",
                status="completed",
                result="success",
                mode=request.mode,
                emergency=bool(request.emergency),
            )
            return {"operation_id": stop_op["operation_id"], "mode": request.mode}

    def get_operation(self, operation_id):
        with self.operation_lock:
            op = self.operations.get(operation_id)
            if not op:
                raise ApiError(1001, f"operation_id 不存在: {operation_id}", 404)
            return dict(op)

    def status(self):
        with self.operation_lock:
            running = self._has_running_grasp_locked()
        if core._ros2_node is None or not rclpy.ok():
            robot_state = "offline"
        elif running is not None:
            robot_state = "moving"
        else:
            robot_state = "idle"
        camera_streaming = bool(core._ros2_node is not None and getattr(core._ros2_node, "rs_pipeline", None) is not None)
        camera_fps = int(getattr(core._ros2_node, "rs_fps", 0) or 0) if core._ros2_node is not None else 0
        return {
            "robot": {
                "connected": bool(core._ros2_node is not None and rclpy.ok()),
                "state": robot_state,
                "joint_count": len(MARVIN_JOINT_NAMES),
                "current_operation": running.get("operation_id") if running else None,
                "mode": "task_loop",
            },
            "camera": {
                "connected": bool(core.rs is not None),
                "streaming": camera_streaming,
                "fps": camera_fps,
            },
            "safety": {
                "e_stop_active": False,
                "collision_detected": False,
                "in_safe_zone": True,
            },
            "last_updated": time.time(),
        }

    def joint_state_message(self, seq):
        if self.joint_monitor is not None:
            return self.joint_monitor.message(seq)
        return {
            "type": "joint_state",
            "seq": int(seq),
            "timestamp": time.time(),
            "robot_id": "marvin_01",
            "joints": [
                {
                    "name": name,
                    "angle_rad": 0.0,
                    "velocity_rad_s": 0.0,
                    "torque_nm": 0.0,
                    "temperature_c": 0.0,
                    "status": "unknown",
                }
                for name in MARVIN_JOINT_NAMES
            ],
            "end_effector": {
                "left": {"frame": "Joint7_L", "pose": None},
                "right": {"frame": "Joint7_R", "pose": None},
            },
        }


def create_api_app():
    service = RobotTaskApiService()
    api = FastAPI(title="Robot Task Loop API", version="1.0.0")
    cors_origins = [
        origin.strip()
        for origin in os.getenv("API_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.on_event("startup")
    def _startup():
        service.startup()

    @api.on_event("shutdown")
    def _shutdown():
        service.shutdown()

    @api.get("/")
    def root():
        return {"name": "Robot Task Loop API", "docs": "/docs", "base_url": "/api/v1"}

    @api.post("/api/v1/capture")
    def capture(request: CaptureRequest):
        try:
            return _api_envelope(data=service.capture(request))
        except Exception as exc:
            return _api_error_response(exc)

    @api.post("/api/v1/grasp")
    def grasp(request: GraspRequest):
        try:
            return _api_envelope(data=service.start_grasp(request), message="抓取操作已启动")
        except Exception as exc:
            return _api_error_response(exc)

    @api.post("/api/v1/stop")
    def stop(request: StopRequest):
        try:
            return _api_envelope(data=service.stop(request), message="停止指令已发送")
        except Exception as exc:
            return _api_error_response(exc)

    @api.get("/api/v1/status")
    def status():
        try:
            return _api_envelope(data=service.status())
        except Exception as exc:
            return _api_error_response(exc)

    @api.get("/api/v1/operation/{operation_id}")
    def operation(operation_id: str):
        try:
            return _api_envelope(data=service.get_operation(operation_id))
        except Exception as exc:
            return _api_error_response(exc)

    @api.websocket("/ws/robot/state")
    async def robot_state_ws(
        websocket: WebSocket,
        interval_ms: int = Query(default=20, ge=10),
        compression: int = Query(default=1),
        token: Optional[str] = Query(default=None),
    ):
        del compression, token
        await websocket.accept()
        seq = 0
        try:
            while True:
                seq += 1
                await websocket.send_json(service.joint_state_message(seq))
                await asyncio.sleep(max(10, int(interval_ms)) / 1000.0)
        except WebSocketDisconnect:
            return

    return api


app = create_api_app()


def main():
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
