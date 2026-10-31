# CMT + AsyncBEV 与 ROS 2 接入

本次接入提供独立模型进程、无标注的传感器输入和 ROS 2 节点。模型配置固定为
官方 CMT + AsyncBEV LiDAR-delay 主模型。它同时需要六路相机和 LiDAR；文件名
中的 `lidar` 指延迟模态，不代表只使用 LiDAR。

## 代码与权重

- 官方源码：<https://github.com/tudelft-iv/AsyncBEV>
- 固定提交：`c51ced190a8095cc1723b9f7aee08c39f1bc482e`。
- 权重：`cmt_asyncbev_lidar_epoch12.pth`，440,052,953 字节。
- SHA-256：`324986141a3628c5c7c38209da186dd9dbed49847a29d2c043bf145e2d142bfd`。
- 来源与下载地址：`integrations/asyncbev/upstream.json`。
- 外部源码、权重、构建产物放在 Git 忽略的目录中。

`scripts/prepare_asyncbev.py` 下载并校验权重，把可审查的
`integrations/asyncbev/label_free_inference.patch` 应用到固定版本的外部源码。
重复执行不会重复应用补丁。已有不同版本的 checkout 会停止并提示，不会自动覆盖。

## 部署适配的含义

官方评估流水线会加载场景 map、真值流场和框内点。这个版本直接接收图像、
点云、相机内参、外参、时间戳与自车位姿，不加载 nuScenes 目标标注或流场标签。

补丁只让监督用的 GT-flow 读取和 FlowEmbedder 在训练时执行。推理仍保留学习到的
速度估计、乘以传感器时间差、位姿补偿和检测头；不改变权重键。推理中的 GT-flow
诊断张量设为零，该张量不用于检测输出。`scripts/check_asyncbev_label_free.py`
直接抽取并执行补丁后的对齐函数，检查无真值运行、真值独立性和训练监督分支。
这项检查不等价于完整模型的精度验证。

## 本机环境

本机已有 Ubuntu 20.04 的 BEVFusion 环境，以及 Ubuntu 22.04 的 ROS 2 Humble。
本次使用后者验证 ROS 2，未安装或声称验证 Jazzy。GPU 为 RTX 4070 Laptop 8GB。

模型放在独立的 `/home/yan/micromamba/envs/asyncbev`，与原 `bevfusion` 环境分开。
ROS 运行系统 Python 3.10；模型使用 Python 3.8、PyTorch 1.11.0+cu113、
CUDA 11.3、MMCV 1.6.0、MMDetection3D 1.0.0rc5 和 FlashAttention 0.2.2。
PyTorch/CUDA 选择与官方提供的 FlashAttention wheel 一致。旧 spconv 还需要
`cumm-cu113==0.2.9`；未锁定时 pip 会选择不兼容的新版 cumm。
完整安装清单保存在 `output/asyncbev/model_packages.txt`。这个部署环境使用
Numba 0.58.1、NetworkX 2.8.8、Trimesh 3.23.5，覆盖了旧 MMDetection3D 包元数据
中的部分限制；只验证了本文列出的部署路径，未验证训练和全部评估工具兼容性。

两个 WSL 发行版通过仅绑定 `127.0.0.1` 的模型服务通信，NPZ 输入包放在共同的
`/mnt/d/.../output/asyncbev/ipc` 下。这个服务依赖共享文件路径，只用于同一主机；
它不提供远程文件传输或公开网络服务。在同一 Linux 系统中也可以直接启动模型子进程。

## 准备与安装

项目根目录，Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_asyncbev.py
```

模型环境，WSL Ubuntu 20.04（首次安装；已有环境可跳过创建）：

```bash
cd /mnt/d/my_project/multi-camera-lidar-bev-perception
/home/yan/.local/bin/micromamba create -y -r /home/yan/micromamba -n asyncbev -c conda-forge python=3.8 pip
/home/yan/.local/bin/micromamba install -y -r /home/yan/micromamba -n asyncbev -c conda-forge cudatoolkit-dev=11.3
/home/yan/.local/bin/micromamba run -r /home/yan/micromamba -n asyncbev bash integrations/asyncbev/setup_model_env.sh
```

CUDA development tools 只装进模型环境，用来编译旧版 MMDetection3D 算子。编译脚本
使用 `8.6+PTX` 为本机 Ada GPU 提供前向兼容代码。更换硬件需重新检查此设置。

检查模型导入和真实 CUDA NMS 算子：

```bash
bash integrations/asyncbev/model_python.sh /home/yan/micromamba/envs/asyncbev integrations/asyncbev/worker.py --doctor
bash integrations/asyncbev/model_python.sh /home/yan/micromamba/envs/asyncbev scripts/check_asyncbev_label_free.py
```

## 离线真实模型推理

需要已有的 `data/external/nuscenes/v1.0-mini`、图像和点云。下列命令读取传感器和
标定位姿，不读取目标标注。`--delay-frames` 沿 LiDAR sample-data 历史向前取帧，
实际时间差由时间戳决定。

```bash
bash integrations/asyncbev/model_python.sh /home/yan/micromamba/envs/asyncbev scripts/run_asyncbev_nuscenes.py \
  --scene-index 0 --max-frames 3 --output output/asyncbev/scene.json
```

输出包括可供现有语义渲染脚本读取的 scene JSON，以及无标注的 NPZ 输入包。
新输出的框中心明确为 **gravity center**，尺寸顺序为 L/W/H。模型内部的 LiDAR
box bottom-center 会先用 `gravity_center` 转换，避免 ROS 显示框产生半个高度偏移。

## ROS 2 构建

Ubuntu 22.04 / Humble：

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends ros-humble-vision-msgs ros-humble-message-filters \
  ros-humble-cv-bridge python3-scipy python3-colcon-common-extensions
source /opt/ros/humble/setup.bash
cd /mnt/d/my_project/multi-camera-lidar-bev-perception/ros2
colcon build --symlink-install --packages-select parking_bev_ros
source install/setup.bash
export PYTHONPATH=/mnt/d/my_project/multi-camera-lidar-bev-perception/src:$PYTHONPATH
```

本仓库核心包通过 `PYTHONPATH` 供 ROS 导入，不向系统 Python 安装模型依赖。
`integrations/asyncbev/ros_python.sh` 可以自动加载本机 ROS、workspace 和核心代码。

## ROS 2 实时节点

先在 Ubuntu 20.04 启动驻留模型服务：

```bash
cd /mnt/d/my_project/multi-camera-lidar-bev-perception
bash integrations/asyncbev/model_python.sh /home/yan/micromamba/envs/asyncbev \
  integrations/asyncbev/worker.py --listen 127.0.0.1:18765
```

再在 Ubuntu 22.04 加载上述 ROS 环境并启动：

```bash
ros2 launch parking_bev_ros perception.launch.py \
  worker_endpoint:=127.0.0.1:18765 \
  packet_directory:=/mnt/d/my_project/multi-camera-lidar-bev-perception/output/asyncbev/ipc
```

同一 Linux 环境下，可以改为 `worker_command:='/absolute/model-env/bin/python /absolute/repo/integrations/asyncbev/worker.py'`。
若需要 CUDA 库环境，用 `worker_command:='bash /absolute/repo/integrations/asyncbev/model_python.sh /absolute/model-env /absolute/repo/integrations/asyncbev/worker.py'`。
命令与 endpoint 二选一。

传感器参数见 `ros2/parking_bev_ros/config/asyncbev.yaml`。接真实硬件前必须修改：

- 六路相机顺序：FRONT、FRONT_RIGHT、FRONT_LEFT、BACK、BACK_LEFT、BACK_RIGHT。
- `Image` 与对应 `CameraInfo` 的分辨率、光学坐标系必须一致。
- 默认输入已经矫正的 `image_rect`，使用 CameraInfo.P 的 3x3 部分。原始普通针孔
  相机可以设置 `images_rectified=false` 做去畸变；鱼眼需在上游矫正并匹配训练视场。
- TF 必须提供 `odom -> base_link -> lidar/camera_optical_frame` 的真实标定位姿。
  缺少位姿不使用单位矩阵替代。
- 点云需要 x/y/z/intensity 字段。通过 `intensity_scale` 匹配传感器强度量纲。
- 各相机时间差默认不超过 0.08s，LiDAR/参考时刻差不超过 0.6s；超限输入不推理。
- 默认积累最多十个历史扫描，以测量到的位姿补偿到源 LiDAR 坐标，时间通道单位为秒。
  不包含逐点扫描去畸变。单扫描运行与训练输入分布有差异。
- 回放 rosbag2 时设置 `use_sim_time:=true`，使用 `ros2 bag play BAG --clock`，并回放 TF
  与 CameraInfo。不能以历史时间戳配合当前墙钟。

发布内容：

| Topic | 类型 | 坐标/用途 |
| --- | --- | --- |
| `/parking_bev/detections` | `vision_msgs/Detection3DArray` | base_link，gravity centers |
| `/parking_bev/tracks` | `vision_msgs/Detection3DArray` | odom，id 为稳定跟踪编号 |
| `/parking_bev/track_states` | `std_msgs/String` | 含速度、最后观测时间与命中次数的停车节点输入快照 |
| `/parking_bev/markers` | `visualization_msgs/MarkerArray` | RViz 检测框 |
| `/parking_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 模型耗时、输入到输出延迟、丢帧、错误 |

跟踪复用项目已有 Kalman/Hungarian 实现，先转换到固定坐标再关联。输入只保留一个
最新待处理包，防止慢推理导致队列无限累积。模型错误和超时会停止该模型连接并发布
错误诊断，不会输出占位框或悄悄切回其他模型。

RViz 配置：`ros2/parking_bev_ros/rviz/perception.rviz`。

## 验证命令与边界

Windows 核心测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Ubuntu 22.04，保存结果到 ROS topic 的回放检查（不执行模型）：

```bash
bash integrations/asyncbev/ros_python.sh scripts/smoke_ros2.py \
  --predictions output/bevfusion_mini/scenes/00_scene-0061.json \
  --report output/asyncbev/ros_replay_smoke.json
```

真实模型服务启动后，使用记录的六路相机、点云与标定测试完整 ROS 链路：

```bash
bash integrations/asyncbev/ros_python.sh scripts/smoke_ros2.py \
  --packet output/asyncbev/packet_000.npz --endpoint 127.0.0.1:18765 \
  --report output/asyncbev/ros_sensor_smoke.json
```

这个传感器检查重复发布一个静态记录样本，不是实际驾驶序列，也不能证明实时性或
检测精度。`output/asyncbev/` 保存本机验证日志、预测和检查报告；最终状态见本文末尾。

## 本次验证状态

2026-10-31，本机完成以下验证：

| 检查 | 实测结果 |
| --- | --- |
| 核心 pytest | 57 项通过，包含 IPC、时间/坐标校验、点云格式、跟踪高度和后续停车用例 |
| 无真值对齐函数 | 无标签运行通过；改变真值流场不改变推理输出；训练监督仍被执行 |
| 权重 | 主模型下载完成，文件大小及 SHA-256 匹配官方清单 |
| GPU 环境 | 模型插件导入、CUDA NMS 执行通过 |
| 真实 nuScenes 推理 | scene-0061 连续 3 帧输出 84 / 89 / 88 个目标（score >= 0.2） |
| 模型计时 | 0.832 / 0.770 / 0.837 秒，平均 0.813 秒；不含完整输入预处理与 ROS 耗时 |
| 首次单帧显存 | 峰值已分配张量显存 4.103 GiB；不等于进程/设备总显存占用 |
| 非零 LiDAR 延迟 | 实测 0.249954 秒延迟的样本推理成功，输出 81 个目标 |
| ROS 构建 | Ubuntu 22.04 / Humble，colcon 构建通过 |
| 保存结果回放 | 实际 ROS 发布/订阅验证通过 |
| 原始传感器到真实模型 | 两次 Detection3DArray 与跟踪消息均收到，frame 为 base_link / odom，跟踪 ID 非空，诊断无 ERROR |
| ROS 输入到输出 | 静态记录样本回放实测约 1.41 / 1.93 秒，丢弃积压输入的计数正常增加 |
| 可视化兼容 | 新模型 scene JSON 成功生成 `output/asyncbev/semantic_preview.jpg` 与 MP4 |

旧 spconv 对本机 SM 8.9 GPU 会打印没有原生编译该架构的警告；上述样本在兼容
路径下执行成功。部署其他显卡、Jetson 或升级 CUDA 时需要重新验证，不应直接复制
这个环境并假定兼容。

这些结果证明模型和 ROS 数据链路已接通。它们不证明精度优于 BEVFusion、达到
实时帧率、已复现论文指标，或适用于实际驾驶。当前模型速度比项目已有的
BEVFusion 历史记录慢，下一步应先做相同数据、相同计时范围的精度与性能评测，
再决定是否替换默认后端。真实传感器尚未接入；需要设备参数、真实外参/位姿与
场景数据进行后续验证。测试结束会停止临时模型服务，按上面的命令可重新启动。
