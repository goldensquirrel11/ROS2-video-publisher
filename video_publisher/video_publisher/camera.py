import yaml
import rclpy
from rclpy.node import Node
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pathlib import Path

class VideoPublisher(Node):

    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter('video_file', '')
        self.declare_parameter('yaml_file', '')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('loop_video', False)

        video_file = self.get_parameter('video_file').get_parameter_value().string_value
        yaml_file = self.get_parameter('yaml_file').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.loop_video = self.get_parameter('loop_video').get_parameter_value().bool_value

        # Resolve paths
        video_path = Path(video_file)
        self.video_file = str(video_path)

        if not video_path.exists():
            self.get_logger().error(f'Video file does not exist: {self.video_file}')
            return

        # Determine yaml path
        yaml_path = None
        if yaml_file:
            yaml_path = Path(yaml_file)
        else:
            # Look for a yaml file in the same directory as the video file that has the same name
            candidate_yaml = video_path.with_suffix('.yaml')
            if candidate_yaml.exists():
                yaml_path = candidate_yaml
            else:
                candidate_yml = video_path.with_suffix('.yml')
                if candidate_yml.exists():
                    yaml_path = candidate_yml

        # Load camera parameters from yaml file if it exists
        if yaml_path and yaml_path.exists():
            self.get_logger().info(f"Loading camera parameters from YAML: {yaml_path}")
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                parsed_params = {}
                if yaml_data:
                    # Check Dataset -> Calibration
                    if 'Dataset' in yaml_data and isinstance(yaml_data['Dataset'], dict) and 'Calibration' in yaml_data['Dataset']:
                        parsed_params = yaml_data['Dataset']['Calibration']
                    # Check camera nested
                    elif 'camera' in yaml_data and isinstance(yaml_data['camera'], dict):
                        parsed_params = yaml_data['camera']
                    # Check standard ROS camera info
                    elif 'camera_matrix' in yaml_data or 'projection_matrix' in yaml_data:
                        if 'image_width' in yaml_data:
                            self.width = int(yaml_data['image_width'])
                        if 'image_height' in yaml_data:
                            self.height = int(yaml_data['image_height'])
                        if 'distortion_model' in yaml_data:
                            self.distorted = (yaml_data['distortion_model'] != 'none')
                        if 'camera_matrix' in yaml_data and isinstance(yaml_data['camera_matrix'], dict):
                            cm = yaml_data['camera_matrix'].get('data', [])
                            if len(cm) >= 9:
                                self.fx, self.cx = float(cm[0]), float(cm[2])
                                self.fy, self.cy = float(cm[4]), float(cm[5])
                        if 'distortion_coefficients' in yaml_data and isinstance(yaml_data['distortion_coefficients'], dict):
                            dc = yaml_data['distortion_coefficients'].get('data', [])
                            if len(dc) >= 4:
                                self.k1, self.k2, self.p1, self.p2 = float(dc[0]), float(dc[1]), float(dc[2]), float(dc[3])
                            if len(dc) >= 5:
                                self.k3 = float(dc[4])
                    # Check flat dict
                    elif isinstance(yaml_data, dict):
                        parsed_params = yaml_data

                    # Apply parsed_params if any
                    for k, v in parsed_params.items():
                        key_name = k.split('.')[-1]
                        if key_name == 'fx': self.fx = float(v)
                        elif key_name == 'fy': self.fy = float(v)
                        elif key_name == 'cx': self.cx = float(v)
                        elif key_name == 'cy': self.cy = float(v)
                        elif key_name == 'k1': self.k1 = float(v)
                        elif key_name == 'k2': self.k2 = float(v)
                        elif key_name == 'p1': self.p1 = float(v)
                        elif key_name == 'p2': self.p2 = float(v)
                        elif key_name == 'k3': self.k3 = float(v)
                        elif key_name == 'width': self.width = int(v)
                        elif key_name == 'height': self.height = int(v)
                        elif key_name == 'distorted': self.distorted = bool(v)
            except Exception as e:
                self.get_logger().error(f"Failed to parse camera yaml file: {str(e)}")
        else:
            if yaml_file:
                self.get_logger().warning(f"Specified YAML file does not exist: {yaml_file}")
            else:
                self.get_logger().info("No YAML file specified or found in video directory. Using default camera parameters.")

        self.get_logger().info(f'Publish rate: {self.publish_rate}')
        self.get_logger().info(
            f"Loaded camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, "
            f"k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3}, "
            f"width={self.width}, height={self.height}, distorted={self.distorted}"
        )
        
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
            if self.loop_video:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.get_logger().info("Reached end of video, looping back to start")
            else:
                self.get_logger().info("Reached end of video. Stopping publisher.")
                self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)    # Prevents the node from quitting immediately and allow
                        # callbacks to be accessible
    node.cap.release()  # Release OpenCV video capture
    rclpy.shutdown()

if __name__ == '__main__':
    main()