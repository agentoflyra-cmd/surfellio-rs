# Oxford Spires 回环闭环（LiDAR-only 离线验证）

本文记录在 Oxford Spires 数据集上从数据准备到轨迹/地图修正的完整离线验证。
结论：**LiDAR-only 链路（Scan Context + 局部 ICP + GTSAM + 地图重融合）在 Spires 上可行**，
轨迹 ATE 0.51 → 0.12 m，回访区地图层残差 52 → 20 mm（mean）。

## 1. 数据准备

序列：`2024-03-18-christ-church-01`（Ouster OS1-64 + IMU，约 795 s，含真实回访）。

```bash
# 仅下载点云 / IMU / 参考轨迹（不下载图像，见 Hugging Face 仓库）
# ori-drs/oxford_spires_dataset
#   sequences/2024-03-18-christ-church-01/raw/lidar-clouds.zip
#   sequences/2024-03-18-christ-church-01/raw/imu.csv
#   sequences/.../processed/trajectory/*.txt
#   calibration/*.yaml
```

`datasets/oxford_spires/lidar-clouds.zip` 解压出 7959 个 PCD（首访到回访约 795 s）。
注意 PCD 的逐点 `timestamp` 字段为绝对时间（秒），转换时需要按 header 中 `SIZE/TYPE/COUNT`
正确保留填充字段（漏掉会得到 46 字节 stride 的错误解析）。

```bash
# PCD + IMU CSV -> ROS1 bag -> MCAP
/home/lyra/.local/share/uv/tools/rosbags/bin/python scripts/spires_convert.py
rosbags-convert --src datasets/oxford_spires/sequences/2024-03-18-christ-church-01/spires.bag \
  --dst datasets/oxford_spires/sequences/2024-03-18-christ-church-01/spires_mcap --dst-storage mcap
```

外参来自 `calibration/cam-lidar-imu.yaml` 的 `T_cam_lidar` 与 cam0 的 `C_q_CI/C_r_CI`，
推出 `T_IMU_LIDAR` 后写入 `config/oxford_spires.yaml`（R ≈ 90° yaw，t ≈ (-0.00065, -0.0155, 0.0846)）。

## 2. LIO 基线

```bash
./target/release/fastlio-replay \
  datasets/oxford_spires/sequences/2024-03-18-christ-church-01/spires_mcap/spires_mcap.mcap \
  config/oxford_spires.yaml 0.0 1024 output/spires/trajectory.csv output/spires/map.pcd \
  --keyframe-dir output/spires/keyframes --surfel-dump output/spires/surfels.csv --map-stats
```

| 指标 | 数值 |
|---|---:|
| 同步帧 | 7958 / 7958（failed_groups=0） |
| 时长 | 794.9 s |
| 关键帧（2 s 间隔） | 398 |
| surfel 数 | 1,276,645 |
| 帧延迟 | mean 86.9 ms / p99 158.3 ms |

## 3. 闭环链路

```
keyframes (398, world-frame point clouds)
  -> Scan Context 候选（20 rings x 60 sectors, r<40 m, 时间间隔>120 s）
  -> 局部 ICP（±3 帧子图, point-to-plane, 小初值偏移）
  -> 物理过滤（fitness >= 0.6, 修正量 <= 2 m）
  -> GTSAM Pose3 位姿图（odometry + loop, Cauchy kernel）
  -> 修正位姿 -> 全频率轨迹 warp -> 地图重融合
```

已整理为可复现脚本：

```bash
# 需要 gtsam/open3d（见下）
output/venv/bin/python scripts/spires_loop_pipeline.py
python3 scripts/spires_refusion.py
```

实验环境：`uv venv output/venv && uv pip install gtsam open3d`（uv 缓存需指向可写目录，
例如 `UV_CACHE_DIR=output/uv-cache`）。

### 参数

| 模块 | 参数 |
|---|---|
| Scan Context | rings=20, sectors=60, r_max=40 m, 每帧取 ring-key 最近 5 个候选, descriptor 距离阈值 8 |
| 候选去重 | 按 (i//20, j//20) 每块保留一对 |
| 局部 ICP | 子图 ±3 关键帧、voxel 0.25、法向半径 0.5/30nn、max_corr 1.2、初值沿漂移 {-1,0,1} × 横向 {-0.5,0,0.5} |
| 过滤 | fitness ≥ 0.6，相对 odometry 的修正量 ≤ 2 m |
| GTSAM | odom σ=(0.02 rad, 0.05 m)，loop σ=(0.05 rad, 0.15 m)，Cauchy k=0.08 |

## 4. 结果

### 4.1 轨迹（对三条参考轨迹，全轨迹 Umeyama 对齐）

| 参考 | ATE RMSE before | after | median before | after |
|---|---:|---:|---:|---:|
| hba | 0.511 m | **0.124 m** | 0.493 | 0.062 |
| vilens-slam | 0.504 m | 0.133 m | 0.493 | 0.075 |
| colmap | 0.518 m | 0.138 m | 0.499 | 0.084 |

- 候选 76 对 → 接受 28 条回环边（48 条被过滤）；
- 首尾闭合：2.384 m → 1.913 m（参考轨迹自身首尾差约 2.3 m）。

### 4.2 地图重融合（简单方案，当前主线）

重融合的做法：把每个关键帧的原始点云用修正变换 `dT_k = T'_k T_k^-1` 整体变换，
再以 0.25 m voxel 重新分桶、每个 voxel 用协方差特征分解重拟合平面（中心 + 法向），
得到新地图；不需要对旧 surfel 做任何变形。

```bash
python3 scripts/spires_refusion.py --render
```

输出：`output/spires/refusion/surfels.npz`（centers/normals/eigenvalues/counts）、
`map.pcd`（与 fastlio-replay 相同的 surfel PCD 布局）、`map.png`。

| 指标 | 数值 |
|---|---|
| 重融合 surfel 数（0.25 m） | 170,126 |
| 平均每 voxel 点数 | 24.2 |
| 漂移地图回访层残差 | mean 52.2 / median 33.3 / p90 139.0 mm |
| 重融合地图回访层残差 | **mean 20.3 / median 16.4 / p90 39.8 mm** |

### 4.3 消融结论

| 实验 | 结果 |
|---|---|
| FPFH+RANSAC vs 局部 ICP（同一批正确候选对） | 两者 ATE 完全相同（0.122–0.136 m）→ **FPFH+RANSAC 不必要** |
| 参考轨迹提候选 vs Scan Context 提候选 | ATE 0.122 vs 0.124 m → **SC 候选生成可用** |
| 局部 GICP（世界系 identity/多初值） | 在对称庭院中歧义（多个解 fit 几乎相同），不可用 |
| 参考轨迹作 oracle 约束（即“正确相对位姿”） | ATE 0.303 m，但比 LiDAR 测量差（参考本身有偏差） |
| 变形图 vs 重融合 | 变形实现数学正确（与重融合点位移 median 0.2 mm），但从漂移初值优化无法恢复 1.78 m 漂移（锚点收敛域问题）；**有原始点云时重融合更优** |

## 5. 评估口径注意事项

1. 三条 TUM 参考轨迹来自同一离线处理流程（VILENS/COLMAP/hba），**互相只差约 0.13 m，
   同源相关，不是 ground truth**；只能用于相对比较；
2. 对齐窗口影响很大：同一 LIO 轨迹“前 100 s 对齐”ATE 0.97 m，“全轨迹对齐”0.51 m，
   本文统一使用**全轨迹 Umeyama 对齐**；
3. 相对距离指标（随机关键帧对距离误差）显示 LIO 本身局部几何已经很准；
   闭环收益主要体现在全局形状/闭合与地图层一致性。

## 6. 已知限制与下一步

- SC 参数（rings/sectors/阈值）在 Spires 上手工标定，换数据集需要重新评估；
- 局部 ICP 依赖“候选对正确”，错误候选会给出 |t| 很大的解，靠物理过滤 + Cauchy 兜底；
- 计算量：SC 描述子 + 候选 ICP（每个 ~0.3 s）离线可接受，在线需要优化；
- 变形图若要在线使用，需要“正确初始对齐 + 地图锚点优化”，否则收敛域不足；
- 下一步：把该链路移植进 Rust（SC 检测 + 局部 ICP + 位姿图后端 + 重融合/变形）。
