import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


class DualMarkerPublisher(Node):
    def __init__(self):
        super().__init__('dual_6dof_marker_publisher')

        # 分别创建两个话题的publisher
        self.pub_left = self.create_publisher(PoseStamped, 'control/target_poseL', 10)
        self.pub_right = self.create_publisher(PoseStamped, 'control/target_poseR', 10)

        self.left_marker_ready = False
        self.right_marker_ready = False
        self.eef_left_sub = self.create_subscription(
            PoseStamped,
            'info/eef_left',
            self.eef_left_callback,
            10,
        )
        self.eef_right_sub = self.create_subscription(
            PoseStamped,
            'info/eef_right',
            self.eef_right_callback,
            10,
        )

        # 创建交互标记服务器
        self.server = InteractiveMarkerServer(self, 'dual_6dof_marker')

        self.get_logger().info(
            'Waiting for info/eef_left and info/eef_right to initialize markers.'
        )

    def eef_left_callback(self, msg: PoseStamped):
        if self.left_marker_ready:
            return
        self.make_marker(
            'left_target',
            'Left Target',
            msg,
            color=(0.0, 0.8, 0.2),
        )
        self.left_marker_ready = True
        self.get_logger().info(
            f"Left marker initialized at current EEF: "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )

    def eef_right_callback(self, msg: PoseStamped):
        if self.right_marker_ready:
            return
        self.make_marker(
            'right_target',
            'Right Target',
            msg,
            color=(0.8, 0.2, 0.2),
        )
        self.right_marker_ready = True
        self.get_logger().info(
            f"Right marker initialized at current EEF: "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )

    # === marker feedback 处理 ===
    def process_feedback(self, feedback: InteractiveMarkerFeedback):
        msg = PoseStamped()
        msg.header = feedback.header
        msg.pose = feedback.pose

        if feedback.marker_name == 'left_target':
            self.pub_left.publish(msg)
            self.get_logger().info(
                f"🟩 Left target updated: "
                f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )

        elif feedback.marker_name == 'right_target':
            self.pub_right.publish(msg)
            self.get_logger().info(
                f"🟥 Right target updated: "
                f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )

    # === 构造单个marker ===
    def make_marker(self, name, desc, pose_msg: PoseStamped, color=(0.0, 0.5, 1.0)):
        marker = InteractiveMarker()
        marker.header.frame_id = 'base_link'
        marker.name = name
        marker.description = desc
        marker.scale = 0.3
        marker.pose = pose_msg.pose

        # --- 中间立方体 ---
        cube = Marker()
        cube.type = Marker.CUBE
        cube.scale.x = cube.scale.y = cube.scale.z = 0.1
        cube.color.r, cube.color.g, cube.color.b = color
        cube.color.a = 0.9

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(cube)
        marker.controls.append(control)

        # --- 添加旋转和平移控制轴 ---
        def add_control(orientation, name, mode):
            c = InteractiveMarkerControl()
            c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z = orientation
            c.name = name
            c.interaction_mode = mode
            marker.controls.append(c)

        add_control((1.0, 1.0, 0.0, 0.0), 'rotate_x', InteractiveMarkerControl.ROTATE_AXIS)
        add_control((1.0, 1.0, 0.0, 0.0), 'move_x', InteractiveMarkerControl.MOVE_AXIS)
        add_control((1.0, 0.0, 1.0, 0.0), 'rotate_y', InteractiveMarkerControl.ROTATE_AXIS)
        add_control((1.0, 0.0, 1.0, 0.0), 'move_y', InteractiveMarkerControl.MOVE_AXIS)
        add_control((1.0, 0.0, 0.0, 1.0), 'rotate_z', InteractiveMarkerControl.ROTATE_AXIS)
        add_control((1.0, 0.0, 0.0, 1.0), 'move_z', InteractiveMarkerControl.MOVE_AXIS)

        # 插入并注册callback
        self.server.insert(marker)
        self.server.setCallback(marker.name, self.process_feedback)
        self.server.applyChanges()


def main():
    rclpy.init()
    node = DualMarkerPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
