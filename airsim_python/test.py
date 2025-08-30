import airsim
import numpy as np


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone1")
client.armDisarm(True, vehicle_name="Drone1")

# camera_info = client.simGetCameraInfo("front_center")
# print(camera_info)

responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
])
img = responses[0]
print(img.width, img.height)
