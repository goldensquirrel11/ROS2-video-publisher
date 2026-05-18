import rclpy
from rclpy.node import Node
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pathlib import Path

class VideoPublisher(Node):

    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter('video_file', '/home/golde/Downloads/output.mp4')
        self.declare_parameter('publish_rate', 24.0)

        self.declare_parameter('camera.fx', 520.90862)
        self.declare_parameter('camera.fy', 521.007327)
        self.declare_parameter('camera.cx', 325.141442)
        self.declare_parameter('camera.cy', 249.701764)
        self.declare_parameter('camera.k1', 0.2312)
        self.declare_parameter('camera.k2', -0.7849)
        self.declare_parameter('camera.p1', -0.0033)
        self.declare_parameter('camera.p2', -0.0001)
        self.declare_parameter('camera.k3', 0.9172)
        self.declare_parameter('camera.width', 640)
        self.declare_parameter('camera.height', 480)
        self.declare_parameter('camera.distorted', True)

        self.video_file = self.get_parameter('video_file').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.fx = self.get_parameter('camera.fx').get_parameter_value().double_value
        self.fy = self.get_parameter('camera.fy').get_parameter_value().double_value
        self.cx = self.get_parameter('camera.cx').get_parameter_value().double_value
        self.cy = self.get_parameter('camera.cy').get_parameter_value().double_value
        self.k1 = self.get_parameter('camera.k1').get_parameter_value().double_value
        self.k2 = self.get_parameter('camera.k2').get_parameter_value().double_value
        self.p1 = self.get_parameter('camera.p1').get_parameter_value().double_value
        self.p2 = self.get_parameter('camera.p2').get_parameter_value().double_value
        self.k3 = self.get_parameter('camera.k3').get_parameter_value().double_value
        self.width = int(self.get_parameter('camera.width').get_parameter_value().integer_value)
        self.height = int(self.get_parameter('camera.height').get_parameter_value().integer_value)
        self.distorted = self.get_parameter('camera.distorted').get_parameter_value().bool_value

        self.get_logger().info(f'Publish rate: {self.publish_rate}')

        if not Path(self.video_file).exists():
            self.get_logger().error(f'Video file does not exist: {self.video_file}')
            return
        
        self.cap = cv.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open video file: {self.video_file}")
            return
        
        # Get video properties
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.get_logger().info(f"Video loaded: {self.video_file}, FPS: {self.fps}, Frames: {self.frame_count}")
        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Prepare camera info message
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.width = self.width
        self.camera_info_msg.height = self.height
        self.camera_info_msg.distortion_model = 'plumb_bob' if self.distorted else 'none'
        self.camera_info_msg.d = [self.k1, self.k2, self.p1, self.p2, self.k3]
        self.camera_info_msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info_msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_msg.header.frame_id = "camera_frame"
        
        # Create publishers
        self.image_publisher = self.create_publisher(Image, 'video_stream', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera_info', 10)
        
        # Create timer for publishing frames
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.frame_index = 0
        self.get_logger().info("Video publisher node initialized")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_frame"

                self.camera_info_msg.header.stamp = ros_image.header.stamp
                self.camera_info_msg.header.frame_id = ros_image.header.frame_id

                self.image_publisher.publish(ros_image)
                self.camera_info_publisher.publish(self.camera_info_msg)
                self.frame_index += 1
                self.get_logger().debug(f"Published frame {self.frame_index}")
            except Exception as e:
                self.get_logger().error(f"Failed to convert and publish frame: {str(e)}")
        else:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            self.get_logger().info("Reached end of video, looping back to start")

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)    # Prevents the node from quitting immediately and allow
                        # callbacks to be accessible
    node.cap.release()  # Release OpenCV video capture
    rclpy.shutdown()

if __name__ == '__main__':
    main()