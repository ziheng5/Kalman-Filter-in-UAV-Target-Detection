# 相机参数相关的方法
import airsim
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import cv2


# 计算相机的内参矩阵
def get_IntrinsicMatrix(client, camera_name, vehicle_name):
    # 初始化内在参数矩阵
    intrinsic_matrix = np.zeros([3, 3])

    # 获取 FoV 信息
    fov = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name, external=False).fov
    # print(fov)

    # 获取摄像头视角
    request = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
    responses = client.simGetImages(request, vehicle_name=vehicle_name)

    # 获取分辨率
    img_width = responses[0].width
    img_height = responses[0].height
    # print(img_width, img_height)

    # 计算相机的内在参数矩阵
    intrinsic_matrix[0, 0] = img_width / 2 / math.tan(math.radians(fov / 2))
    intrinsic_matrix[1, 1] = img_width / 2 / math.tan(math.radians(fov / 2))
    intrinsic_matrix[0, 2] = img_width / 2
    intrinsic_matrix[1, 2] = img_height / 2
    intrinsic_matrix[2, 2] = 1

    return intrinsic_matrix


# def get_HomogeneousMatrix(client, camera_name, vehicle_name):
#     camera_info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
#
#     # 相机位置
#     position = camera_info.pose.position
#     tx, ty, tz = position.x_val, position.y_val, position.z_val
#
#     # 相机姿态（四元数形式）
#     orientation = camera_info.pose.orientation
#     w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
#
#     # 四元数 -> 旋转矩阵
#     R = np.array([
#         [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
#         [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
#         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
#     ])
#
#     # 构建齐次变换矩阵
#     T = np.eye(4)
#     T[:3, :3] = R # 旋转部分
#     T[:3, 3] = [tx, ty, tz]
#
#     return T, R


def get_HomogeneousMatrix(client, camera_name, vehicle_name):
    # Part 1
    camera_info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
    pose = camera_info.pose

    # 提取平移向量
    t = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    # 四元数转旋转矩阵
    rotation = R.from_quat([
        pose.orientation.x_val,
        pose.orientation.y_val,
        pose.orientation.z_val,
        pose.orientation.w_val
    ])
    R_matrix = rotation.as_matrix()

    # # 示例：相机朝下时的旋转修正
    # R_adjust = R.from_euler('x', 180, degrees=True).as_matrix()
    # R_matrix = R_matrix @ R_adjust  # 复合旋转

    # 构建齐次变换矩阵 T_body_camera (4x4)
    T_body_camera = np.eye(4)
    T_body_camera[:3, :3] = R_matrix  # 旋转部分
    T_body_camera[:3, 3] = t  # 平移部分

    # Part 2
    # 获取机体（无人机）在世界坐标系中的位姿
    body_pose = client.simGetVehiclePose()

    # 提取平移和旋转
    t_world = np.array([body_pose.position.x_val,
                        body_pose.position.y_val,
                        body_pose.position.z_val])
    rotation_world = R.from_quat([
        body_pose.orientation.x_val,
        body_pose.orientation.y_val,
        body_pose.orientation.z_val,
        body_pose.orientation.w_val
    ])
    R_world = rotation_world.as_matrix()

    # 构建 T_world_body (4x4)
    T_world_body = np.eye(4)
    T_world_body[:3, :3] = R_world
    T_world_body[:3, 3] = t_world

    T_world_camera = T_world_body @ T_body_camera

    return T_world_camera


def draw_camera_info(camera_info, frame_):
    """
    Draw camera information on the frame
    :param camera_info:
    :param frame_:
    :return: frame_:
    """

    color = (0, 0, 255)

    cv2.putText(frame_, 'X: ' + f'{camera_info.pose.position.x_val:.3f}', (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame_, 'Y: ' + f'{camera_info.pose.position.y_val:.3f}', (280, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame_, 'Z: ' + f'{camera_info.pose.position.z_val:.3f}', (530, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame_, 'qw: ' + f'{camera_info.pose.orientation.w_val:.3f}', (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame_, 'qx: ' + f'{camera_info.pose.orientation.x_val:.3f}', (280, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame_, 'qy: ' + f'{camera_info.pose.orientation.y_val:.3f}', (530, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame_, 'qz: ' + f'{camera_info.pose.orientation.z_val:.3f}', (780, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return frame_


def cal_target_state(x1, y1, x2, y2, depth_value, client):
    # TODO: 该函数用来实现基于视觉的目标状态估计
    pass

# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True, vehicle_name="Drone1")
# client.armDisarm(True, "Drone1")
#
# # 设置图像类型（可以是Scene, Depth, Segmentation等）
# image_type = airsim.ImageType.DepthPerspective
#
# # 设置摄像头名称
# camera_name = "front_center"
#
# # camera info
# camera_info = client.simGetCameraInfo(camera_name, vehicle_name="Drone1")
# print(camera_info)

# T = get_HomogeneousMatrix(client, camera_name, vehicle_name="Drone1")
# print(T)
#
# M = get_IntrinsicMatrix(client, camera_name, vehicle_name="Drone1")
# print(M)