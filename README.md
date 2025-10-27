# Kalman-Filter-in-UAV-Target-Detection
利用 Kalman Filter 与 YOLOv8s 实现目标稳健实时检测与状态估计

本项目基于 Unreal Engine 4.27 的 AirSim 插件，并调用 AirSim 的 Python API 实现

> 实际问题：目标实时检测与状态估计过程中，较大的目标检测模型难以做到每一帧都及时检测，过小的目标检测模型又无法实现较高的精度，于是尝试采用 Kalman Filter 来增强 YOLOv8s 的目标检测与状态估计的实时性

（目前仅做了**汽车**的目标检测）

## 食用指南

本仓库为 `private` 仓库，仅供内部成员更新项目源码与文件

### 1. 安装依赖包

可以先直接按照 `requirements.txt` 文件一键安装所有需要用到的 `Python` 包：

```bash
pip install -r requirements.txt
```

> 如果你在项目中**使用了新的包**且需要上传到本仓库时，记得更新 `requirements.txt` 文件并上传到本仓库
>
> **Tips:** 这里推荐使用 `pipreqs` 包自动生成 `requirements.txt` 文件
> ```bash
> pip install pipreqs
> ```

当然，如果你担心 `requirements.txt` 并没有更新到最新版本的话，也可以根据项目依赖手动安装依赖包。


### 2. 地图文件
由于地图文件过大，这里就不上传了，可以自行配置地图文件。

### 3. 配置文件
即本仓库中的 `settings.json` 文件
