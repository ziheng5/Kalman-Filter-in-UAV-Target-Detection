import numpy as np
import airsim
import time
import cv2

def show_position():
    # 这个函数目前只能绘制二维轨迹
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    trajectory = []

    while True:
        position = client.getMultirotorState().kinematics_estimated.position
        trajectory.append((position.x_val, position.y_val))
        img = np.zeros((512, 512, 3))
        for x, y in trajectory:
            cv2.circle(img, (int(x)+250, int(y)+250), 2, (255, 0, 0), -1)
        cv2.imshow("Trajectory", img)
        cv2.waitKey(1)