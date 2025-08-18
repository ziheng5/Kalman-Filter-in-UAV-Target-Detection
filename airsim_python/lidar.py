import airsim
import numpy as np
import open3d as o3d

client = airsim.MultirotorClient()
client.confirmConnection()

data = client.getLidarData("LidarSensor1", "Drone1")

if data.point_cloud:
    points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)

    # 用 open3d 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
else:
    print("No lidar data received")
