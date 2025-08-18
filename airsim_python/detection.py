# 其实目前的方案有个小问题：当第一帧目标检测失误时，后面会一直错
from ultralytics import YOLO
from camera import *
from utils import *
import numpy as np
import airsim
import time
import cv2


def yolo_cv(model_name='yolov8n.pt'):

    """
    ×××
    获取前端摄像头图像并执行目标检测的函数，目前已为废案，可以用来调试
    ×××
    :param model_name: the model used to detect the objects
    :return: None
    """

    # 初始化 YOLOv8 模型
    model = YOLO(model_name)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 设置图像类型（可以是Scene, Depth, Segmentation等）
    image_type = airsim.ImageType.Scene

    # 设置摄像头名称
    camera_name = "front_center"

    # 设置显示窗口
    cv2.namedWindow("Drone FPV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone FPV View", 1000, 600)

    # FPS 计算变量
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # 获取前端摄像头图像
            responses = client.simGetImages([
                # 不返回浮点数，不压缩
                airsim.ImageRequest(camera_name=camera_name, image_type=image_type,
                                    pixels_as_float=False, compress=False),
            ])

            response = responses[0]

            # 处理场景图像
            ## 将图像数据转为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            ## 重塑数组为 3 通道图像
            frame = img1d.reshape(response.height, response.width, 3)

            ## 目标检测
            results = model.predict(frame, classes=[2])

            for result in results:
                annotated_frame = result.plot()

                # 计算并显示FPS
                frame_count += 1
                if frame_count >= 30:  # 每30帧计算一次FPS
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_count = 0
                    start_time = time.time()

                # 显示图像
                cv2.imshow("Drone FPV View", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


def dep_image(min_d, max_d):
    """
    获取深度图的函数，主要用于调试，目前已成废案（

    :param min_d: min depth
    :param max_d: max depth
    :return: None
    """

    AirSim_client = airsim.MultirotorClient()
    AirSim_client.confirmConnection()
    AirSim_client.enableApiControl(True)
    AirSim_client.armDisarm(True)

    # 设置图像类型（可以是Scene, Depth, Segmentation等）
    image_type = airsim.ImageType.DepthPerspective

    # 设置摄像头名称
    camera_name = "front_center"

    # 设置显示窗口
    cv2.namedWindow("Drone FPV DEPTH View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone FPV DEPTH View", 1000, 600)

    try:
        while True:
            responses = AirSim_client.simGetImages([
                airsim.ImageRequest(camera_name=camera_name, image_type=airsim.ImageType.DepthPerspective,
                                    pixels_as_float=True, compress=False)
            ])

            depth_response = responses[0]

            # 1. 处理深度图像
            depth_data = airsim.list_to_2d_float_array(depth_response.image_data_float,
                                                       depth_response.width, depth_response.height)
            # 截断有效范围（min_d ~ max_d）
            valid_depth = np.clip(depth_data, min_d, max_d)  # 小于 min_d 设为 min_d，大于 max_d 设为 max_d

            # 归一化
            depth_normalized = (255- (valid_depth - min_d) / (max_d - min_d) * 255).astype(np.uint8)

            # 2. **应用颜色映射（可选）**
            # 方案1：黑白灰度图（推荐）
            depth_visualized = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Drone FPV DEPTH View", depth_visualized)

            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    finally:
        AirSim_client.armDisarm(False)
        AirSim_client.enableApiControl(False)


def yolo_and_depth(disappear, shared_list, lock, model_name='yolov8n.pt', model_conf=0.25):

    """
    目标检测，同时将检测信息置入共享的目标状态存储列表中
    :param shared_list: The list that contains all the historical states of the target. [[x1, y1, x2, y2, depth_value], ...]
    :param lock: The process lock.
    :param min_d: Min depth
    :param max_d: Max depth
    :param model_name: The model used to detect the target
    :param model_conf: The confidence level of the model.
    :return:
    """
    # 初始化 YOLOv8 模型
    model = YOLO(model_name)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")

    # 设置图像类型（可以是Scene, Depth, Segmentation等）
    image_type = airsim.ImageType.Scene

    # 设置摄像头名称
    camera_name = "front_center"

    IOU_Threshold = 0.2  # 匹配时的阈值

    # 状态转移矩阵，上一时刻的状态转移到当前时刻
    A = np.array([[1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    # 状态观测矩阵
    H = np.eye(6)

    # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
    # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
    Q = np.eye(6) * 0.1

    # 观测噪声协方差矩阵R，p(v)~N(0,R)
    # 观测噪声来自于检测框丢失、重叠等
    R = np.eye(6) * 1

    # 控制输入矩阵B
    B = None
    # 状态估计协方差矩阵P初始化
    P = np.eye(6)

    # 初始化标记
    initialized = False

    # 检测失败标记
    times_of_disappearance = 0

    try:
        while True:
            if not initialized and len(shared_list) > 0:
                # 初始化
                initial_target_box = shared_list[0][0:4]
                initial_box_state = xyxy_to_xywh(initial_target_box)
                initial_state = np.array(
                    [[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3],
                      0, 0]]).T  # [中心x,中心y,宽w,高h,dx,dy]
                X_posterior = np.array(initial_state)
                P_posterior = np.array(P)
                Z = np.array(initial_state)
                times_of_disappearance = 0

                with lock:
                    disappear.value = False
                initialized = True

            # 获取前端摄像头图像
            responses = client.simGetImages([
                # 不返回浮点数，不压缩
                airsim.ImageRequest(camera_name=camera_name, image_type=image_type,
                                    pixels_as_float=False, compress=False),
                airsim.ImageRequest(camera_name=camera_name, image_type=airsim.ImageType.DepthPerspective,
                                    pixels_as_float=True, compress=False),
            ])

            response = responses[0]
            depth_response = responses[1]

            # 1. 处理深度图像
            depth_data = airsim.list_to_2d_float_array(depth_response.image_data_float,
                                                       depth_response.width, depth_response.height)

            # 2. 处理场景图像
            ## 将图像数据转为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            ## 重塑数组为 3 通道图像
            frame = img1d.reshape(response.height, response.width, 3)

            ## 目标检测
            results = model.predict(frame, classes=[3], conf=model_conf, verbose=False)

            tmp_lis = []

            if initialized:

                # 如果已经初始化过，且目标没有失踪，启动卡尔曼滤波
                for result in results:

                    max_iou = IOU_Threshold
                    max_iou_matched = False

                    # 失踪判定部分
                    if len(result.boxes) == 0:
                        # 如果没有检测到目标
                        times_of_disappearance += 1

                    else:
                        # 如果检测到目标了


                        # 将检测结果映射到深度图上
                        for box in result.boxes:
                            # x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            xyxy = np.array(box.xyxy[0].tolist(), dtype="float")
                            xyxy = xyxy//3
                            iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                            print(iou)
                            if iou > max_iou:
                                times_of_disappearance = 0
                                target_box = xyxy
                                max_iou = iou
                                max_iou_matched = True

                        if max_iou_matched:
                            # 如果找到了
                            xywh = xyxy_to_xywh(target_box)

                            # 计算dx,dy
                            dx = xywh[0] - X_posterior[0]
                            dy = xywh[1] - X_posterior[1]

                            Z[0:4] = np.array([xywh]).T
                            Z[4::] = np.array([dx, dy])

                            # 下面是绘图部分的信息
                            x1, y1, x2, y2 = map(int, target_box.tolist())
                            # x1, y1, x2, y2 = x1//3, y1//3, x2//3, y2//3

                            # 获取深度信息（边界框中心点的深度）
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            depth_value = depth_data[center_y, center_x] if (0 <= center_x < depth_data.shape[1] and
                                                                             0 <= center_y < depth_data.shape[0]) else 0

                            # 在确保线程安全的情况，将目标检测获取到的信息转移到共享的存储空间

                            color = (0, 0, 255)

                            with lock:
                                shared_list.append([x1, y1, x2, y2, depth_value, color])

                    if max_iou_matched:
                        # 如果 IOU 匹配成功，正常计算
                        # -----进行先验估计-----------------
                        X_prior = np.dot(A, X_posterior)
                        box_prior = xywh_to_xyxy(X_prior[0:4])
                        # -----计算状态估计协方差矩阵P--------
                        P_prior_1 = np.dot(A, P_posterior)
                        P_prior = np.dot(P_prior_1, A.T) + Q
                        # ------计算卡尔曼增益---------------------
                        k1 = np.dot(P_prior, H.T)
                        k2 = np.dot(np.dot(H, P_prior), H.T) + R
                        K = np.dot(k1, np.linalg.inv(k2))
                        # --------------后验估计------------
                        X_posterior_1 = Z - np.dot(H, X_prior)
                        X_posterior = X_prior + np.dot(K, X_posterior_1)
                        box_posterior = xywh_to_xyxy(X_posterior[0:4])
                        # ---------更新状态估计协方差矩阵P-----
                        P_posterior_1 = np.eye(6) - np.dot(K, H)
                        P_posterior = np.dot(P_posterior_1, P_prior)
                    else:
                        # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
                        # 此时直接迭代，不使用卡尔曼滤波
                        times_of_disappearance += 1

                        X_posterior = np.dot(A, X_posterior)
                        box_posterior = xywh_to_xyxy(X_posterior[0:4])
                        x1, y1, x2, y2 = map(int, box_posterior)
                        # x1, y1, x2, y2 = x1//3, y1//3, x2//3, y2//3

                        # 获取深度信息（边界框中心点的深度）
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        depth_value = depth_data[center_y, center_x] if (0 <= center_x < depth_data.shape[1] and
                                                                         0 <= center_y < depth_data.shape[0]) else 0

                        # 在确保线程安全的情况，将目标检测获取到的信息转移到共享的存储空间
                        color = (0, 255, 0)

                        with lock:
                            shared_list.append([x1, y1, x2, y2, depth_value, color])


            else:
                # 如果尚未初始化，保持最基本的检测，不启动卡尔曼滤波
                for result in results:
                    # 将检测结果映射到深度图上

                    # 失踪判定部分
                    if len(result.boxes) == 0:
                        times_of_disappearance += 1
                    else:
                        times_of_disappearance = 0

                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        x1, y1, x2, y2 = x1 // 3, y1 // 3, x2 // 3, y2 // 3

                        # 获取深度信息（边界框中心点的深度）
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        depth_value = depth_data[center_y, center_x] if (0 <= center_x < depth_data.shape[1] and
                                                                         0 <= center_y < depth_data.shape[0]) else 0

                        # 在确保线程安全的情况，将目标检测获取到的信息转移到共享的存储空间
                        color = (0, 0, 255)
                        with lock:
                            shared_list.append([x1, y1, x2, y2, depth_value, color])
                    # ===================================================================

            # 如果目标检测失败超过 20 次，判定为失踪，检测从头开始
            if times_of_disappearance == 20:
                initialized = False
                with lock:
                    del shared_list[:]
                    disappear.value = True

    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


def draw_vision(shared_list, show_depth, min_d, max_d, disappear, k):
    """
    专用于渲染无人机视角的函数，便于调试与查看检测效果
    :param shared_list: The list that contains all the historical states of the target. --> list[[x1, y1, x2, y2, depth_value], ...]
    :param show_depth: Either show depth vision or not. --> bool
    :param min_d: Min depth --> int
    :param max_d: Max depth --> int
    :param disappear: Either disappear or not. --> bool
    :param k: The camera's intrinsic parameter matrix. --> np.array
    :return:
    """

    # 调整参数
    k_inv = np.linalg.inv(k)

    # Create the client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")

    # 设置图像类型（可以是 Scene, Depth, Segmentation 等）
    image_type = airsim.ImageType.Scene

    # 设置摄像头名称
    camera_name = "front_center"

    # 设置显示窗口 1
    cv2.namedWindow("Drone FPV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone FPV View", 1000, 600)

    if show_depth:
        # 设置显示窗口 2
        cv2.namedWindow("Drone FPV DEPTH View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone FPV DEPTH View", 1000, 600)

    try:
        while True:
            # print(len(shared_list))
            # 获取摄像头状态信息
            camera_info = client.simGetCameraInfo(camera_name)

            if show_depth:
                # 如果要显示深度视角
                responses = client.simGetImages([
                    # 不返回浮点数，不压缩
                    airsim.ImageRequest(camera_name=camera_name, image_type=image_type,
                                        pixels_as_float=False, compress=False),
                    airsim.ImageRequest(camera_name=camera_name, image_type=airsim.ImageType.DepthPerspective,
                                        pixels_as_float=True, compress=False),
                ])

                response = responses[0]
                depth_response = responses[1]

                # 1. 处理深度图像
                depth_data = airsim.list_to_2d_float_array(depth_response.image_data_float,
                                                           depth_response.width, depth_response.height)
                # 截断有效范围（min_d ~ max_d）
                valid_depth = np.clip(depth_data, min_d, max_d)  # 小于 min_d 设为 min_d，大于 max_d 设为 max_d

                # 归一化
                depth_normalized = (255 - (valid_depth - min_d) / (max_d - min_d) * 255).astype(np.uint8)

                # 绘制黑白灰度图
                depth_visualized = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

                # 2. 处理场景图像
                ## 将图像数据转为 numpy 数组
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

                ## 重塑数组为 3 通道图像
                frame = img1d.reshape(response.height, response.width, 3)

                frame_ = frame.copy()

                if len(shared_list) and not disappear.value:
                    # 如果目标依然存在，并且被检测到
                    # 从目标信息池中获取最新的目标 bounding box 信息

                    x1, y1, x2, y2, depth_value, color = shared_list[-1]

                    # 获取边界框中心坐标（之前在 yolo_and_depth 函数中计算过了，这里再计算，是否需要优化？）
                    x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
                    center_vec = np.array([x_c, y_c, 1.])

                    # 从像素坐标 [u, v] 反投影到相机归一化平面（去除内参影响）
                    c = depth_value * k_inv @ center_vec
                    c_ = np.array([c[0], c[1], depth_value, 1.0])

                    # 获取机体的齐次变换矩阵
                    T = get_HomogeneousMatrix(client, camera_name, vehicle_name="Drone1")

                    # 获取 3D 坐标
                    tar_distance = T @ c_

                    cv2.rectangle(depth_visualized, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame_, (x1*3, y1*3), (x2*3, y2*3), color, 2)

                    cv2.putText(frame_, f'x_d: {tar_distance[0]:.3f} y_d: {tar_distance[1]:.3f} z_d: {tar_distance[2]:.3f}', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                    cv2.putText(depth_visualized, f'Target | {depth_value}m', (x1 - 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    cv2.putText(frame_, f'Target | {depth_value}m', (x1*3 - 10, y1*3 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)


                else:
                    # 在 RGB 视图中添加字幕
                    cv2.putText(frame_, 'Disappear!', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
                    # 在深度图中添加字幕
                    cv2.putText(depth_visualized, 'Disappear!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                frame_ = draw_camera_info(camera_info, frame_)

                # 显示图像
                cv2.imshow("Drone FPV View", frame_)

                # 显示深度图
                cv2.imshow("Drone FPV DEPTH View", depth_visualized)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                # 如果不显示深度视角
                responses = client.simGetImages([
                    # 不返回浮点数，不压缩
                    airsim.ImageRequest(camera_name=camera_name, image_type=image_type,
                                        pixels_as_float=False, compress=False),
                ])

                response = responses[0]

                # 2. 处理场景图像
                ## 将图像数据转为 numpy 数组
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

                ## 重塑数组为 3 通道图像
                frame = img1d.reshape(response.height, response.width, 3)

                color = (0, 0, 255)

                frame_ = frame.copy()
                if len(shared_list) and not disappear.value:
                    # 如果目标依然存在，并且被检测到
                    x1, y1, x2, y2, depth_value, color = shared_list[-1]

                    # 获取边界框中心坐标（之前在 yolo_and_depth 函数中计算过了，这里再计算，是否需要优化？）
                    x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
                    center_vec = np.array([x_c, y_c, 1.])

                    # 从像素坐标 [u, v] 反投影到相机归一化平面（去除内参影响）
                    c = depth_value * k_inv @ center_vec
                    c_ = np.array([c[0], c[1], depth_value, 1.0])

                    # 获取机体的齐次变换矩阵
                    T = get_HomogeneousMatrix(client, camera_name, vehicle_name="Drone1")

                    # 获取 3D 坐标
                    tar_distance = T @ c_

                    cv2.putText(frame_,
                                f'x_d: {tar_distance[0]:.3f} y_d: {tar_distance[1]:.3f} z_d: {tar_distance[2]:.3f}',
                                (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                    cv2.rectangle(frame_, (x1*3, y1*3), (x2*3, y2*3), color, 2)

                    if color[1] == 255:
                        cv2.putText(frame_, f'Predict | {depth_value}m', (x1*3 - 10, y1*3 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
                    else:
                        cv2.putText(frame_, f'Target | {depth_value}m', (x1 * 3 - 10, y1 * 3 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

                else:
                    # 如果目标失踪
                    cv2.putText(frame_, 'Disappear!', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

                frame_ = draw_camera_info(camera_info, frame_)

                # 显示图像
                cv2.imshow("Drone FPV View", frame_)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


# def check_disappearance(last_list_length, shared_list, disappear, lock):
#     """
#     用来判断检测目标是否已经丢失
#     :param last_list_length: The length of last shared list. --> int
#     :param shared_list: The list that contains all the historical states of the target. --> list[[x1, y1, x2, y2, depth_value], ...]
#     :param disappear: Either the target disappear or not. --> bool
#     :param lock: The process lock. --> Lock()
#     :return:
#     """
#
#     while True:
#         a = len(shared_list)
#         if a == last_list_length:
#             # 如果隔了 2 秒，list 长度和原来的长度还是相同的，则判定目标消失
#             with lock:
#                 disappear.value = True
#         else:
#             # 否则，判定目标未消失
#             with lock:
#                 disappear.value = False
#                 last_list_length = a
#
#         time.sleep(2)

