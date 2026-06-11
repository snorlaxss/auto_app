import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from fake_interface_pkg.msg import KeypointPose, KeypointPoseArray
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import yaml
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0

# listen tf
# 

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
                print(f"[DEBUG] Left arm: rotating x CW 90 degrees")
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


def tf_to_template(tf_name: str) -> str:
    return tf_name.rsplit("_", 1)[0]


class ActionHandler:
    def __init__(self, template_yaml, node=None):
        self.template = template_yaml
        self.node = node  # 新增：保存对 node 的引用
        # build top_level_actions
        self.top_level_actions = []
        for k, v in template_yaml.items():
            if k in ("obj_name", "action_type"):
                continue
            self.top_level_actions.append((k, v))

    def process_for_object(self, ts_transform, action_name=None, task_arm=None, task_speed=None, correction_mode="org"):
        all_kps = []
        for key, block in self.top_level_actions:
            if isinstance(block, list):
                for item in block:
                    act = item.get("action_name", None)
                    if act != action_name:
                        continue
                    sub_block = item.get(act, {})
                    all_kps.extend(
                        self._process_single_block(
                            sub_block, act, ts_transform,
                            task_arm=task_arm, task_speed=task_speed, correction_mode=correction_mode
                        )
                    )
            elif isinstance(block, dict):
                act = block.get("action_name", key)
                if act != action_name:
                    continue
                all_kps.extend(
                    self._process_single_block(
                        block, act, ts_transform,
                        task_arm=task_arm, task_speed=task_speed, correction_mode=correction_mode
                    )
                )
        return all_kps

    def _process_single_block(self, block, action_name, ts_transform,
                            task_arm=None, task_speed=None, correction_mode="none"):
        if block is None:
            return []

        rotation_constraint = block.get("rotation_constraint", [1, 1, 1])
        pose_rel = block.get("pose_relative")
        gripper_states = block.get("gripper_state", [])
        time_list = block.get("time", [])

        if pose_rel is None:
            self._warn("Warning: pose_relative is None in block: %s" % (block,))
            return []
        if isinstance(pose_rel[0], (float, int)):
            pose_rel_list = [pose_rel]
        else:
            pose_rel_list = pose_rel

        arm = task_arm
        speed = task_speed



        if not isinstance(gripper_states, list):
            gripper_states = [gripper_states]
        if not isinstance(time_list, list):
            time_list = [time_list]

        final_arm = arm  # 保存初始 arm 值
        poses = []

        for rel in pose_rel_list:
            pos_world, quat_world = self.apply_relative(ts_transform, rel, correction_mode, final_arm)
            pose_msg = self.to_pose_msg(pos_world, quat_world)
            poses.append(pose_msg)
        
        # 保存当前 arm 到 node 供下一个动作使用
        if final_arm not in ("correct", "copy last one") and final_arm is not None:
            if self.node:
                self.node.last_arm = final_arm
            print(f"[DEBUG] Saved last_arm = {final_arm}")
        else:
            print(f"[DEBUG] Not saving arm: final_arm={final_arm}")

        return [{
            "name": action_name,
            "arm": final_arm,
            "poses": poses,
            "constraints": [
                float(rotation_constraint[0]),
                float(rotation_constraint[1]),
                float(rotation_constraint[2])
            ],
            "speed": float(speed if speed is not None else 1.0),
            "gripper_value": gripper_states,
            "time": time_list,
        }]

    def to_pose_msg(self, pos, quat):
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return p

    def apply_relative(self, ts_transform, rel, correction_mode="org", arm=None):
        # 支持 ts_transform 为 None（例如 base_link），此时使用单位位姿
        if ts_transform is None:
            t_world_org = np.zeros(3, dtype=float)
            q_world_org = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            t_world_org = np.array([ts_transform.translation.x,
                                    ts_transform.translation.y,
                                    ts_transform.translation.z], dtype=float)
            q_world_org = np.array([ts_transform.rotation.x,
                                    ts_transform.rotation.y,
                                    ts_transform.rotation.z,
                                    ts_transform.rotation.w], dtype=float)
        t_world, q_world = correct_orientation(t_world_org, q_world_org, mode=correction_mode, arm=arm)
        
        t_relative = np.array(rel[:3], dtype=float)
        q_relative = np.array(rel[3:], dtype=float)

        # normalize quats
        if np.linalg.norm(q_world) > 1e-8:
            q_world = q_world / np.linalg.norm(q_world)
        else:
            q_world = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(q_relative) > 1e-8:
            q_relative = q_relative / np.linalg.norm(q_relative)
        else:
            q_relative = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        r_world_rot = R.from_quat(q_world)
        r_relative_rot = R.from_quat(q_relative)

        pos_final = r_world_rot.apply(t_relative) + t_world
        r_final = r_world_rot * r_relative_rot
        q_final = r_final.as_quat()
        return pos_final.tolist(), q_final.tolist()


class FusionNode(Node):
    def __init__(self, object_yaml_path, task_yaml_path):
        super().__init__('fusion_node_revised')
        
        # load YAMLs
        with open(object_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.object_templates = data.get("templates", {})
        with open(task_yaml_path, 'r') as f:
            data2 = yaml.safe_load(f)
            if isinstance(data2, dict) and "tasks" in data2:
                self.tasks = data2["tasks"]
            else:
                self.tasks = data2 if isinstance(data2, list) else []

        # 新增：在 FusionNode 中保存 last_arm
        self.last_arm = None

        # 创建 handlers 时传入 self（node 实例）
        self.template_handlers = {k: ActionHandler(v, node=self) for k, v in self.object_templates.items()}

        # tf2 buffer/listener and broadcaster
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)


        # publishers
        self.pose_pub = self.create_publisher(KeypointPoseArray, '/fusion_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        # timer / scheduling
        self.timer = self.create_timer(0.2, self.on_timer)

        # state
        self.tf_cache = {}         
        self._control_lock = threading.Lock()
        self.all_sent = False
        self.active_task_idx = 0
        self.instance_queue = {}    
        self._pair_queue = []        
        self._redo_to_index = None
        self.warmup_seconds = 5.0
        self.warmup_start_time = None
        self.warmup_done = False
        self._pair_none_count = 0
        self._pair_none_threshold = 10
        self.current_pick_idx = 0
        self.pick_instances_list = []  
        self.place_instance = None  
        self.current_processing_pick = None
        self.current_pair_total = 0 
        self.allowed_pick_ids = set()  # 记录本任务对允许的 pick 实例集合
        self.completed_picks = set()
        self.get_logger().info("FusionNode initialized, waiting for /tf messages...")


    def tf_callback(self, msg: TFMessage):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.warmup_start_time is None:
            self.warmup_start_time = now
        with self._control_lock:
            for t in msg.transforms:
                # 保存 TransformStamped 和最后更新时间
                self.tf_cache[t.child_frame_id] = (t, time.time())
        if not self.warmup_done and now - self.warmup_start_time >= self.warmup_seconds:
            self.get_logger().info(f"[Warmup] Finished. TF frames: {list(self.tf_cache.keys())}")
            self.warmup_done = True

    def publish_home_both(self):
        for arm_name in ["right", "left"]:
            home_seq = [{
                "name": "home",
                "arm": arm_name,
                "poses": [],
                "constraints": [100,100,100],
                "speed": 0.3,
                "gripper_value": [0.0, 0.0],
                "time": [0, 0.1]
            }]
            self.publish_kp_separately(home_seq)
            time.sleep(2)
            # self._pump_ros_for(duration=1.5)

    def publish_home_separately(self, arm):
        # for arm_name in ["left", "right"]:
        home_seq = [{
            "name": "home",
            "arm": arm,
            "poses": [],
            "constraints": [100,100,100],
            "speed": 0.3,
            "gripper_value": [0.0, 0.0],
            "time": [0, 0.1]
        }]
        self.publish_kp_separately(home_seq)
        # time.sleep(1.5)
        # self._pump_ros_for(duration=1.5)

    def publish_kp_separately(self, sequence):
        for info in sequence:
            kp_array = KeypointPoseArray()
            kp_array.header.stamp = self.get_clock().now().to_msg()
            kp_array.header.frame_id = "base_link"

            kp = KeypointPose()
            kp.name = info["name"]
            kp.arm = info["arm"]
            kp.poses = info.get("poses", [])
            kp.constraints = [float(x) for x in info.get("constraints", [1,1,1])]
            kp.speed = float(info.get("speed",1.0))
            kp.gripper_value = [float(x) for x in info.get("gripper_value",[0])]
            kp.time = [float(x) for x in info.get("time",[0])]

            kp_array.poses.append(kp)
            try:
                self.pose_pub.publish(kp_array)
                self.get_logger().info(f"[Fusion] Published step '{kp.name}' for arm '{kp.arm}'")
            except Exception as e:
                self.get_logger().warning(f"Failed to publish step '{kp.name}': {e}")
            # self._broadcast_transforms(sequence)
            time.sleep(0.5)
            # rclpy.spin_once(self, timeout_sec=0.01)

    # def _broadcast_transforms(self, final_sequence, axis_scale=0.05):
    #     """
    #     将 final_sequence 中的每个 pose 以 TransformStamped 广播到 /tf，
    #     child_frame_id 使用 fusion_{step}_{poseidx} 形式，frame_id 与 Marker 相同（camera_link）
    #     """
    #     now = self.get_clock().now().to_msg()
    #     sent = 0
    #     for step_idx, info in enumerate(final_sequence):
    #         for pose_idx, pose in enumerate(info.get("poses", [])):
    #             ts = TransformStamped()
    #             ts.header.stamp = now
    #             ts.header.frame_id = "base_link"
    #             ts.child_frame_id = f"fusion_{step_idx}_{pose_idx}"
    #             ts.transform.translation.x = float(pose.position.x)
    #             ts.transform.translation.y = float(pose.position.y)
    #             ts.transform.translation.z = float(pose.position.z)
    #             ts.transform.rotation.x = float(pose.orientation.x)
    #             ts.transform.rotation.y = float(pose.orientation.y)
    #             ts.transform.rotation.z = float(pose.orientation.z)
    #             ts.transform.rotation.w = float(pose.orientation.w)
    #             try:
    #                 self.tf_broadcaster.sendTransform(ts)
    #                 sent += 1
    #             except Exception as e:
    #                 self.get_logger().warn(f"Failed to broadcast transform {ts.child_frame_id}: {e}")
    #     self.get_logger().info(f"[Fusion] Broadcasted {sent} transforms to /tf")



    # -------------------------------
    # 多实例配对
    # -------------------------------

    def match_instances(self, tf_dict, template_name):
        """
        Find instances in tf_dict whose template name matches template_name (strip trailing _\d+).
        Sort by numeric suffix ascending (so index order).
        """
        # print("list of tf_dict", list(tf_dict.keys()))
        matches = [name for name in tf_dict if tf_to_template(name) == template_name]
        # sort by numeric suffix if present
        def key_fn(x):
            m = re.match(r".+_(\d+)$", x)
            return int(m.group(1)) if m else 0
        matches.sort(key=key_fn)
        return matches
    
    def generate_instance_pair(self, pick_idx, place_idx, camera_link_frames):
        """生成所有 pick 实例与唯一 place 实例的配对"""
        pick_task = self.tasks[pick_idx]
        place_task = self.tasks[place_idx]
        
        pick_instances = self.match_instances(camera_link_frames, pick_task["target"])

        place_instances = self.match_instances(camera_link_frames, place_task["target"])
        place_instance = place_instances[-1] if place_instances else None
        
        # 返回 (pick_instances_list, place_instance)
        if pick_instances and place_instance:
            return (pick_instances, place_instance)
        else:
            return None

    def select_active_pair(self, camera_link_frames):
        """选择当前要处理的配对,并处理物体编号变化"""
        pair_data = self.generate_instance_pair(
            self.active_task_idx, 
            self.active_task_idx + 1, 
            camera_link_frames
        )
        
        if pair_data is None:
            return None
        
        new_pick_instances, new_place_instance = pair_data
        if not new_pick_instances:
            return None

        # 情况1: 首次进入这个任务对
        if not self.pick_instances_list:
            self.pick_instances_list = new_pick_instances
            self.allowed_pick_ids = set(new_pick_instances)
            self.place_instance = new_place_instance
            self.current_pick_idx = 0
            self.current_processing_pick = None
            self.current_pair_total = len(new_pick_instances)
            self.get_logger().info(f"New task pair started with {len(new_pick_instances)} pick instances")
        # 情况1.1: 仅 place 实例变化，不重置 pick 列表/索引
        elif new_place_instance != self.place_instance:
            self.place_instance = new_place_instance
            self.get_logger().info(f"Place instance changed -> {self.place_instance}, keep remaining picks")
        else:
            # 情况2: 检测物体编号是否发生变化
            # 只看当前索引位置的物体（正在处理的那个）
            if self.current_pick_idx < len(self.pick_instances_list):
                current_expected_id = self.pick_instances_list[self.current_pick_idx]

                # 当前期望 ID 不在本帧，进行替换
                if current_expected_id not in new_pick_instances:
                    # 优先选择未完成的、且当前帧可见的实例
                    candidates = [pid for pid in new_pick_instances if pid not in self.completed_picks]
                    if not candidates:
                        # 若都已完成，退而选当前帧任意同类实例
                        candidates = new_pick_instances
                    if candidates:
                        new_id = candidates[min(self.current_pick_idx, len(candidates)-1)]
                        self.get_logger().info(f"Object ID changed: {current_expected_id} -> {new_id}")
                        # 替换当前索引的 ID
                        self.pick_instances_list[self.current_pick_idx] = new_id
                        # 更新白名单
                        self.allowed_pick_ids.add(new_id)
                        self.allowed_pick_ids.discard(current_expected_id)
                        self.current_processing_pick = new_id
                    else:
                        # 当前无可用实例，返回 None 等待下一帧
                        return None
                else:
                    # 当前 ID 仍存在，继续使用
                    self.current_processing_pick = current_expected_id

        if self.current_pick_idx >= len(self.pick_instances_list):
            return None
        
        self.current_processing_pick = self.pick_instances_list[self.current_pick_idx]
        return (self.current_processing_pick, self.place_instance)

    def pop_active_pair(self):
        if self._pair_queue:
            self._pair_queue.pop(0)

    def process_one_instance(self, idx, curr_task, inst):
        handler = self.template_handlers.get(curr_task["target"])
        if handler is None:
            self.get_logger().warning(f"No handler for {curr_task['target']}")
            return "ok"
        
        # 从 task 中读取校正模式，默认为 "org"
        correction_mode = curr_task.get("correction_mode", "org")
        
        while rclpy.ok():
            print("task now is", inst)
            ts = None
            ts = self.tf_buffer.lookup_transform('base_link', inst, rclpy.time.Time(),
                                                timeout=Duration(seconds=0.2))
            if ts is None:
                self.get_logger().warning(f"No transform for {inst}, skip instance.")
                return "ok"
            t = ts.transform.translation
            r = ts.transform.rotation
            self.get_logger().info(
                f"[TF] base_link->{inst}: t=({t.x:.3f},{t.y:.3f},{t.z:.3f}), "
                f"q=({r.x:.3f},{r.y:.3f},{r.z:.3f},{r.w:.3f})"
            )
            
            task_arm = curr_task.get("arm")
            # 处理 "copy last one" - 复制上一个动作的 arm
            if task_arm == "copy last one":
                # 从 self 获取 last_arm（因为 self 就是 FusionNode）
                last_arm = self.last_arm
                task_arm = last_arm
                print(f"[DEBUG] Copying last_arm: {last_arm}")
                if task_arm is None:
                    print(f"[WARNING] 'copy last one' but last_arm is None, using default 'right'")
                    task_arm = "right"
                else:
                    print(f"[DEBUG] Copied last_arm: {task_arm}")            
            # 如果 arm == "correct"，从物体的世界坐标系 y 值判断
            if task_arm == "correct":
                y_value = t.y  # 物体在世界坐标系下的 y 值
                if y_value > 0:
                    task_arm = "left"
                else:
                    task_arm = "right"
                print(f"[DEBUG] 'correct' mode: y={y_value:.3f}, determined arm: {task_arm}")
            
            kps = handler.process_for_object(ts.transform if ts else None,
                                            curr_task["action_name"],
                                            task_arm=task_arm,
                                            task_speed=curr_task.get("speed"),
                                            correction_mode=correction_mode)
        
            self.publish_kp_separately(kps)
            if curr_task.get("finish_home", False):
                # 从 kps 中获取实际的 arm 值
                if kps and len(kps) > 0:
                    actual_arm = kps[0].get("arm")
                    self.get_logger().info(f"finish_home is True, publishing home for arm '{actual_arm}'")
                    time.sleep(6.0)
                    self.publish_home_separately(actual_arm)

            decision = input(f"Task[{idx}] ({curr_task['action_name']}) waiting (c/r/f/s): ").strip().lower()
            if decision == "s":
                self.publish_home_both()
                rclpy.shutdown()
                return "stop"
            if decision == "f":
                return "refresh"
            if decision == "r":
                self._redo_to_index = max(0, idx-1)
                return "redo"
            if decision in ("c",""):
                break
        return "ok"

    # -------------------------------
    # pick+place 对处理
    # -------------------------------
    def process_one_pair(self):
        # 获取最新的 camera_link_frames
        camera_link_frames = self.get_camera_link_frames()
        
        if not camera_link_frames:
            self._pair_none_count += 1
            if self._pair_none_count >= self._pair_none_threshold:
                self.get_logger().info("No camera_link frames found repeatedly — publish home")
                self.publish_home_both()
                self._pair_none_count = 0
            return
        
        print("camera_link_frames:", list(camera_link_frames.keys()))
        pair = self.select_active_pair(camera_link_frames)
        
        if pair is None:
            # 当前任务对的所有 pick 实例都已处理完
            if self.pick_instances_list and self.current_pick_idx >= len(self.pick_instances_list):
                self.get_logger().info(f"All {len(self.pick_instances_list)} pick instances completed, moving to next task pair")
                self.active_task_idx += 2
                self.current_pick_idx = 0
                self.pick_instances_list = []
                self.place_instance = None
                self.current_processing_pick = None
                self.current_pair_total = 0 
                self.allowed_pick_ids.clear()
                self.completed_picks.clear()
                return
            
            # 没有找到配对
            self._pair_none_count += 1
            if self._pair_none_count >= self._pair_none_threshold:
                self.get_logger().info("pair not found repeatedly — publish home")
                pick_task = self.tasks[self.active_task_idx]
                place_task = self.tasks[self.active_task_idx + 1]
                if pick_task.get("arm"):
                    self.publish_home_separately(pick_task.get("arm"))
                self._pair_none_count = 0
                if place_task.get("arm"):
                    self.publish_home_separately(place_task.get("arm"))
                self._pair_none_count = 0
            return
        
        if self._pair_none_count:
            self._pair_none_count = 0
        
        pick_inst, place_inst = pair
        pick_task = self.tasks[self.active_task_idx]
        place_task = self.tasks[self.active_task_idx + 1]
        
        print(f"Processing pair {self.current_pick_idx + 1}/{len(self.pick_instances_list)}: {pick_inst} -> {place_inst}")

        # 外层循环：允许 place 失败后重新执行 pick
        while True:
            # pick 阶段，处理重复
            while True:
                res_pick = self.process_one_instance(self.active_task_idx, pick_task, pick_inst)
                if res_pick == "redo":
                    return "redo"
                if res_pick == "refresh":
                    # 重新获取最新的 camera_link_frames
                    camera_link_frames = self.get_camera_link_frames()

                    # 优先继续使用当前实例（若仍存在且目标匹配）
                    if pick_inst in camera_link_frames and tf_to_template(pick_inst) == pick_task["target"]:
                        # 同时更新 place 实例为最新
                        place_list = self.match_instances(camera_link_frames, place_task["target"])
                        if place_list:
                            place_inst = place_list[-1]
                        continue
                    # 否则回退到最新配对
                    new_pair = self.select_active_pair(camera_link_frames)
                    if new_pair is None:
                        return
                    pick_inst, place_inst = new_pair
                    continue
                # 返回 "ok" 则 pick 完成，跳出循环
                break

            # place 阶段，处理重复
            place_redo = False
            while True:
                res_place = self.process_one_instance(self.active_task_idx + 1, place_task, place_inst)
                if res_place == "redo":
                    # place 失败，需要重新执行 pick
                    # 先检查当前 pick_inst 是否还存在
                    camera_link_frames = self.get_camera_link_frames()
                    
                    # 优先查找当前 pick_inst
                    if pick_inst in camera_link_frames and tf_to_template(pick_inst) == pick_task["target"]:
                        self.get_logger().info(f"Place redo: reusing existing pick instance {pick_inst}")
                        place_redo = True
                        break
                    
                    # 如果当前 pick_inst 不存在，查找新出现的实例
                    all_pick_instances = self.match_instances(camera_link_frames, pick_task["target"])
                    new_ids = set(all_pick_instances) - self.allowed_pick_ids
                    
                    if new_ids:
                        # 使用新出现的 ID
                        new_pick_inst = list(new_ids)[0]
                        self.get_logger().info(f"Place redo: pick instance changed {pick_inst} -> {new_pick_inst}")
                        # 更新当前索引位置的 ID
                        self.pick_instances_list[self.current_pick_idx] = new_pick_inst
                        self.allowed_pick_ids.add(new_pick_inst)
                        self.allowed_pick_ids.discard(pick_inst)
                        pick_inst = new_pick_inst
                        self.current_processing_pick = new_pick_inst
                        place_redo = True
                        break
                    else:
                        # 如果没有新 ID，尝试使用最新的 pick 实例
                        if all_pick_instances:
                            new_pick_inst = all_pick_instances[-1]
                            self.get_logger().info(f"Place redo: using latest pick instance {new_pick_inst}")
                            self.pick_instances_list[self.current_pick_idx] = new_pick_inst
                            self.allowed_pick_ids.add(new_pick_inst)
                            self.allowed_pick_ids.discard(pick_inst)
                            pick_inst = new_pick_inst
                            self.current_processing_pick = new_pick_inst
                            place_redo = True
                            break
                        else:
                            self.get_logger().warning("Place redo: no pick instances available")
                            return
                    
                if res_place == "refresh":
                    # 重新获取最新的 camera_link_frames
                    camera_link_frames = self.get_camera_link_frames()
                    # 重新选择配对
                    new_pair = self.select_active_pair(camera_link_frames)
                    if new_pair is None:
                        return
                    pick_inst, place_inst = new_pair
                    continue
                # 返回 "ok" 则 place 完成，跳出循环
                break
            
            # 如果 place 需要 redo，继续外层循环重新执行 pick
            if place_redo:
                # 同时更新 place 实例为最新
                camera_link_frames = self.get_camera_link_frames()
                place_list = self.match_instances(camera_link_frames, place_task["target"])
                if place_list:
                    place_inst = place_list[-1]
                continue
            
            # 否则，成功完成当前配对
            break

        # 成功完成当前配对
        self.completed_picks.add(pick_inst)
        self.current_pick_idx += 1
        self.current_processing_pick = None
        total = self.current_pair_total or len(self.pick_instances_list)
        self.get_logger().info(f"Completed {pick_inst}, total completed: {self.current_pick_idx}/{total}")

    def get_camera_link_frames(self):
        """从 tf_buffer 中查找所有 camera_link 下的 frames"""
        camera_link_frames = {}
        try:
            all_frames = self.tf_buffer.all_frames_as_string()
            frame_list = all_frames.split('\n')
            now = self.get_clock().now()
            for frame_line in frame_list:
                if not frame_line.strip():
                    continue
                parts = frame_line.split()
                if len(parts) >= 6:
                    child_frame = parts[1].strip()
                    parent_frame = parts[5].rstrip('.')
                    if parent_frame == 'camera_link':
                        try:
                            ts = self.tf_buffer.lookup_transform('camera_link', child_frame, 
                                                                rclpy.time.Time(),
                                                                timeout=Duration(seconds=0.05))
                            
                            # 检查时间戳是否在最近 2 秒内
                            tf_time_ns = ts.header.stamp.sec * 1e9 + ts.header.stamp.nanosec
                            now_ns = now.nanoseconds
                            age_sec = (now_ns - tf_time_ns) / 1e9
                            
                            if age_sec <= 2.0:
                                camera_link_frames[child_frame] = ts
                            else:
                                self.get_logger().debug(f"Transform {child_frame} is {age_sec:.1f}s old, skipping")
                        except Exception as e:
                            self.get_logger().debug(f"Failed to lookup {child_frame}: {e}")
                            continue
        except Exception as e:
            self.get_logger().warning(f"Failed to get frames: {e}")
        
        return camera_link_frames

    # -------------------------------
    # 定时器回调
    # -------------------------------
    def on_timer(self):
        if self.all_sent: return
        if self._redo_to_index is not None:
            self.active_task_idx = self._redo_to_index
            self._redo_to_index = None
            return
        if self.active_task_idx >= len(self.tasks):
            self.get_logger().info("All tasks processed")
            self.publish_home_both()
            try: self.timer.cancel()
            except Exception: pass
            self.all_sent = True
            return

        # 判断是否是成对任务
        if (self.active_task_idx + 1 < len(self.tasks) and 
            "pair" in (self.tasks[self.active_task_idx].get("pair_type",""), 
                    self.tasks[self.active_task_idx + 1].get("pair_type",""))):
            res = self.process_one_pair()
            if res == "redo":
                self.active_task_idx = max(0, self.active_task_idx - 1)
                return
            if res == "stop": 
                return
            if res == "refresh":
                return
        else:
            # 单任务处理
            camera_link_frames = self.get_camera_link_frames()
            
            if not camera_link_frames:
                self._pair_none_count += 1
                if self._pair_none_count >= self._pair_none_threshold:
                    self.get_logger().info("No camera_link frames found — publish home")
                    self.publish_home_both()
                    self._pair_none_count = 0
                return
            
            curr_task = self.tasks[self.active_task_idx]
            instances = [frame for frame in camera_link_frames.keys() 
                        if tf_to_template(frame) == curr_task["target"]]
            instances = sorted(instances, key=lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 0)
            
            if not instances:
                self._pair_none_count += 1
                if self._pair_none_count >= self._pair_none_threshold:
                    self.get_logger().info("target not found repeatedly — publish home")
                    self.publish_home_separately(curr_task.get("arm"))
                    self._pair_none_count = 0
                return
            
            if self._pair_none_count:
                self._pair_none_count = 0
            
            inst = instances[-1]
            while True:
                res = self.process_one_instance(self.active_task_idx, curr_task, inst)
                if res == "redo":
                    self.active_task_idx = max(0, self.active_task_idx - 1)
                    return
                if res == "refresh":
                    camera_link_frames = self.get_camera_link_frames()
                    instances = [frame for frame in camera_link_frames.keys() 
                                if tf_to_template(frame) == curr_task["target"]]
                    instances = sorted(instances, key=lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 0)
                    
                    if not instances:
                        return
                    
                    inst = instances[-1]
                    continue            
                self.active_task_idx += 1
                return



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_yaml_path',default="translator_1/data/test_coffee.yaml", required=True)
    parser.add_argument('--task_yaml_path',default="translator_1/data/task_coffee.yaml", required=True)
    args = parser.parse_args()

    rclpy.init()
    node = FusionNode(args.object_yaml_path, args.task_yaml_path)
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()