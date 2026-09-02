import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import sys

class ThermalCameraNode(Node):
    def __init__(self):
        super().__init__('thermal_camera_node')

        self.declare_parameter('device_path', '/dev/video0')
        device_path = self.get_parameter('device_path').get_parameter_value().string_value

        self.get_logger().info(f'Attempting to open camera at: {device_path}')
        self.cap_ = cv2.VideoCapture(device_path)
        if not self.cap_.isOpened():
            self.get_logger().error(f"Failed to open camera at {device_path}. Node cannot start.")
            raise RuntimeError(f"Failed to open camera at {device_path}")
        self.get_logger().info(f"Successfully opened camera at {device_path}")

        self.publisher_ = self.create_publisher(Image, 'thermal_image_raw', 10)
        self.bridge_ = CvBridge()

        timer_period = 1.0 / 10.0
        self.timer_ = self.create_timer(timer_period, self.publish_frame)

        self.frame_read_attempts_ = 0
        self.frames_successfully_read_ = 0
        self.start_time_ = time.time()

        self.get_logger().info('Thermal camera node has started.')

    def publish_frame(self):
        self.frame_read_attempts_ += 1
        ret, frame = self.cap_.read()

        if ret:
            self.frames_successfully_read_ += 1

            if self.frames_successfully_read_ == 1:
                save_path = '/ros2_ws/first_thermal_frame.png'
                self.get_logger().info(f'Attempting to save first frame to {save_path}')
                try:
                    success = cv2.imwrite(save_path, frame)
                    if success:
                        self.get_logger().info(f'Successfully saved first frame to {save_path}')
                    else:
                        self.get_logger().error(f'cv2.imwrite failed for {save_path}.')
                except Exception as e:
                    self.get_logger().error(f'Failed to save frame due to exception: {e}')

            try:
                img_msg = self.bridge_.cv2_to_imgmsg(frame, encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Failed to convert frame: {e}')
                return

            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'thermal_camera_frame'
            self.publisher_.publish(img_msg)
            self.get_logger().debug('Published thermal image frame')

        else:
            self.get_logger().warn(f'Failed to capture frame (attempt {self.frame_read_attempts_})')

        if self.frames_successfully_read_ > 0 and self.frames_successfully_read_ % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time_
            if elapsed_time > 0:
                avg_fps = self.frames_successfully_read_ / elapsed_time
                self.get_logger().info(f'Avg FPS: {avg_fps:.2f}')

    def custom_destroy_node(self):
        self.get_logger().info('Shutting down thermal_camera_node...')
        if hasattr(self, 'timer_') and self.timer_ is not None:
            self.timer_.cancel()
        if hasattr(self, 'cap_') and self.cap_.isOpened():
            self.cap_.release()
            self.get_logger().info('Camera released.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    thermal_camera_node = None
    try:
        thermal_camera_node = ThermalCameraNode()
        rclpy.spin(thermal_camera_node)
    except RuntimeError as e:
        if thermal_camera_node:
            thermal_camera_node.get_logger().fatal(f'Node execution failed: {str(e)}')
        else:
            print(f'FATAL: Failed to initialize ThermalCameraNode: {str(e)}', file=sys.stderr)
    except KeyboardInterrupt:
        if thermal_camera_node:
            thermal_camera_node.get_logger().info('Node stopped cleanly by KeyboardInterrupt.')
    finally:
        if thermal_camera_node:
            thermal_camera_node.custom_destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
