# Surfel 生命周期阶段总结（2026-11）

本阶段的目标是提升 FAST-LIO 前端在长序列上的精度与地图质量，并评估回环/后端的
可行性。范围包含：growing surfel 关联、成熟共面合并、tombstone 压实与删除策略、
诊断工具链、初始化鲁棒性、Velodyne PointCloud2 支持，以及回环/变形图的可行性
调查（后者为负结果，详见 `docs/loop-closure-feasibility.md`）。

分支：`surfel-lifecycle` / `merge-baseline`。基线对照：`baseline` / `dev-ka`。

## 落地的功能

1. **Growing surfel 参与关联**（`growing_association_weight = 0.2`，默认开启）：
   - yuanqu ATE3d 0.95 -> 0.106，upstairs 0.104 -> 0.084，KA e2e 4.30 -> 3.49；
2. **成熟共面合并**（dirty-root 延迟队列 + union-find tombstone + 星型森林）：
   - 重复共面面对估计下降 12%~82%，p99 延迟普遍下降，四 bag 精度不退化；
   - `maintenance_max_tasks = 32`、`merge_queue_limit = 4096` 为精度最优（扫描数据
     记录在配置注释里）；
3. **删除策略（默认落地）**：tombstone + 每 256 帧压实；growing-age 删除关闭
   （负结果）；合并即删（eager）被否定（精度回退）；
4. **初始化鲁棒性**：静止判定阈值 0.02 -> 3e-3；滑动窗口运动检测（60 样本均值、
   0.35 m/s² / 0.04 rad/s、>=300 样本后启用）冻结静止前缀；日志输出
   `motion_detected` / `rel_var` / 样本数。四 bag 回归不退化；
5. **Velodyne / PointCloud2 支持**：数据集适配层新增 `sensor_msgs/PointCloud2`
   解码（x/y/z/intensity、ring/line、逐点 time），新增
   `config/corridor_velodyne.yaml`；
6. **诊断与导出**：生命周期统计、assoc 召回/残差/被选 surfel 年龄、surfel epoch
   dump（含协方差特征值）、world keyframe 导出、`--state-dump`、
   `--assoc-geometry-dump`、3DGS Gaussian PLY、maintenance 计时、`--init-groups`。

## 当前基线（默认配置）

| bag | 指标 | 值 |
|---|---|---|
| KA-Urban-East | e2e@10s / ATE | 3.54 / 3.83 |
| FR-IOSB-Short | ATE / e2e | 3.033 / 4.274 |
| upstairs | ATE3d | 0.0820 |
| yuanqu | ATE3d | 0.102（z_rmse 0.065） |
| corridor_04 | 闭环误差 / 帧 p99 | 1.13 m / 20 ms |

## 已验证的负结果（避免重复投入）

- eager 删除、growing-age GC：精度回退；
- 后台线程/时间预算 maintenance 对 p99 收益 <1 ms：不做；
- 回环/后端：KA 无紧回访（最小 3D 回访差 4.3 m），corridor 长直走廊配准多解；
  - 真 plane-to-plane GICP、submap-to-submap 最小验证均无改善；
  - RGB-D 验证给出真实相对位姿 0.49 m / 2.65°，但纯 LiDAR 无法稳定恢复；
- z 漂移：非初始化/重力/姿态/可观测性问题，本质是无绝对参考/回环下的随机游走
  （2.2 m / 1.25 km 环路）。

## 常用命令

```bash
# 单 bag 回放 + 诊断
./target/release/fastlio-replay <bag.mcap> <config.yaml> 0.0 1024 out.csv --map-stats
# 关键帧/surfel 导出（回环/后端分析）
./target/release/fastlio-replay <bag> <cfg> 0.0 1024 out.csv \
  --keyframe-dir <dir> --keyframe-stride 5 \
  --surfel-dump <csv> --surfel-dump-stride 2
# 状态/关联几何 dump
./target/release/fastlio-replay <bag> <cfg> 0.0 1024 out.csv \
  --state-dump <csv> --assoc-geometry-dump <csv>
```

评估脚本与离线分析脚本位于 `output/ka_diag/`（分析产物，不入库）。

## 未开展/遗留

- 回归验收脚本（4 bag 一键对比 baseline）；
- 额外 bag（aneng / corridor / undist）的配置泛化与 indoor/outdoor profile；
- 长时地图内存上界（空间保留/滑窗）；
- 纯 LiDAR 回环需要换场景（非对称几何、真实同点回访）再验证。
