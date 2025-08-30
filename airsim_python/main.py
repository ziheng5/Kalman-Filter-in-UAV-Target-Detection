from keyboard_control import *
from detection import *
from pos_vis import *
from multiprocessing import Process, Manager, Lock
import time
import numpy as np


if __name__ == "__main__":
    # ========================================================================
    # TODO: Airsim 默认参数信息表
    ## 1. 相机内参矩阵（深度图和RGB图是一样的）
    ## [[961.61613003   0.              960.]   (   0.)
    ## [0.              1709.53978672    540.]   (   0.)
    ## [0.              0.              1.]]    (   0.)
    ## 2. 相机视场角 FoV：89.90362548828125（约等于 90 度）
    ## 3. 相机分辨率：1920 * 1080（w * h）
    # ========================================================================
    # TODO:
    ## 当前要处理的小问题：
    ## （已解决）1. 目标识别得到的目标状态信息无法存储在共享的外部空间，以被决策进程读取利用
    ## 2. 分层强化学习框架训练时间和计算成本太高，想一想有没有更多资源，或者有没有优化空间
    ## 3. 目标状态估计有一点问题，需要重新验算坐标轴旋转矩阵（目前计算上似乎没有什么问题了，直接作为状态向量输入神经网络计算出动作向量即可）
    # ========================================================================
    # TODO:
    ## 目前目标状态估计下的几个标注：
    ## x_d 目标与无人机前摄像头在摄像头坐标系 x 轴方向（左右）上的距离，向右变小，向左变大（即 x 轴正方向向左）
    ## y_d 目标与无人机前摄像头在摄像头坐标系 y 轴方向（上下）上的距离，向下变小，向上变大（即 y 轴正方向向下）
    ## z_d 目标与无人机前摄像头在摄像头坐标系 z 轴方向（前后）上的距离，向前变小，向后变大（即 z 轴正方向向前）
    # 调参
    depth_min_d = 0.5
    depth_max_d = 50
    model_name = 'runs/train2/weights/best.pt'   # 对于无人机来说，yolov8n 和 yolov8s 已经够大了
    model_conf = 0.2
    show_depth = False
    show_trajectory = False

    vehicle_name = 'Drone1'

    # 相机参数矩阵
    # k = np.array([
    #     [961.61613003, 0.0, 960.0],
    #     [0.0, 1709.53978672, 540.0],
    #     [0.0, 0.0, 1.0]
    # ])
    k = np.array([
        [961.61613003, 0.0, 960.0],
        [0.0, 961.61613003, 540.0],
        [0.0, 0.0, 1.0]
    ])
    # ========================================================================

    with Manager() as manager:
        shared_list = manager.list()
        shared_list2 = manager.list()
        disappear = manager.Value('b', True)
        last_list_length = manager.Value('i', 0)
        lock = Lock()
        lock2 = Lock()

        # 创建进程池
        process = [Process(target=keyboard_control,
                           args=()),
                   # Process(target=yolo_cv, args=()),
                   # Process(target=dep_image, args=(depth_min_d, depth_max_d)),
                   Process(target=yolo_and_depth,
                           args=(disappear, shared_list, lock,
                                 model_name, model_conf)),
                   Process(target=draw_vision,
                           args=(shared_list, show_depth, depth_min_d,
                                 depth_max_d, disappear, k)),]

        if show_trajectory:
            process.append(Process(target=show_position, args=()))

        # # 创建目标消失检测进程
        # check = Process(target=check_disappearance, args=(last_list_length, shared_list, disappear, lock2))

        # 启动进程池
        [p.start() for p in process]
        # check.start()

        # 等待进程池进程全部结束
        [p.join() for p in process]
        # check.terminate()