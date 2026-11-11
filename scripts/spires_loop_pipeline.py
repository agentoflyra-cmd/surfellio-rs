#!/usr/bin/env python3
"""LiDAR-only loop-closure pipeline for Oxford Spires sequences.

Stages:
  1. Scan Context descriptors on the replay keyframes -> candidate pairs;
  2. local point-to-plane ICP measurement on sub-maps (+-3 keyframes);
  3. physical filters (ICP fitness, correction magnitude vs odometry);
  4. GTSAM Pose3 pose graph (odometry + loop factors, Cauchy kernel);
  5. evaluate the corrected trajectory against the reference TUM trajectories
     (full-trajectory Umeyama alignment; the references are SLAM outputs, not
     ground truth, so treat the numbers as relative improvements).

Requires numpy, gtsam and open3d (see docs/spires-loop-closure.md for the uv
environment used by the offline experiments).

Usage:
  output/venv/bin/python scripts/spires_loop_pipeline.py \
      --keyframes output/spires/keyframes \
      --trajectory output/spires/trajectory.csv \
      --reference-dir datasets/oxford_spires/sequences/2024-03-18-christ-church-01/processed/trajectory \
      --out output/spires/loop/pipeline_result.npz
"""
import argparse
import csv
import os
import struct

import numpy as np

try:
    import gtsam
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit('gtsam is required: uv pip install --python output/venv/bin/python gtsam') from exc
try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit('open3d is required: uv pip install --python output/venv/bin/python open3d') from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scan Context parameters (tuned on Oxford Spires; see docs).
SC_RINGS = 20
SC_SECTORS = 60
SC_MAX_RADIUS_M = 40.0
SC_MAX_DESCRIPTOR_DISTANCE = 8.0
SC_TOP_PER_KEYFRAME = 5
MIN_LOOP_GAP_SEC = 120.0
PAIR_BIN = 20

# Local ICP parameters.
SUBMAP_HALF = 3
SUBMAP_VOXEL = 0.25
ICP_MAX_CORRESPONDENCE_DIST = 1.2
ICP_MIN_FITNESS = 0.6
ICP_DRIFT_OFFSETS = (-1.0, 0.0, 1.0)
ICP_LATERAL_OFFSETS = (-0.5, 0.0, 0.5)
MAX_CORRECTION_M = 2.0

# GTSAM noise models.
ODOM_SIGMA = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
LOOP_SIGMA = np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15])
LOOP_CAUCHY_K = 0.08


def quat_to_rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_keyframes(directory):
    frames = []
    for name in sorted(f for f in os.listdir(directory) if f.endswith('.bin')):
        with open(os.path.join(directory, name), 'rb') as f:
            f.read(8)
            t = struct.unpack('<d', f.read(8))[0]
            pos = np.frombuffer(f.read(24), dtype='<f8').copy()
            quat = np.frombuffer(f.read(32), dtype='<f8').copy()
            count = struct.unpack('<I', f.read(4))[0]
            points = np.frombuffer(f.read(count * 12), dtype='<f4').reshape(count, 3).astype(np.float64)
        frames.append(dict(t=t, pos=pos, quat=quat, points=points))
    if not frames:
        raise SystemExit(f'no keyframes in {directory}')
    return frames


def scan_context(points, pos, rot):
    local = (rot.T @ (points - pos).T).T
    radius = np.linalg.norm(local[:, :2], axis=1)
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    ring = np.clip((radius / SC_MAX_RADIUS_M * SC_RINGS).astype(int), 0, SC_RINGS - 1)
    sector = np.clip(((azimuth + np.pi) / (2 * np.pi) * SC_SECTORS).astype(int), 0, SC_SECTORS - 1)
    desc = np.full((SC_RINGS, SC_SECTORS), -1e9)
    np.maximum.at(desc, (ring, sector), local[:, 2])
    return desc


def propose_candidates(frames, pos, rot, times):
    n = len(frames)
    descriptors = np.array([scan_context(f['points'], pos[k], rot[k]) for k, f in enumerate(frames)])
    ring_keys = descriptors.mean(axis=2)
    raw = []
    for k in range(n):
        order = np.argsort(np.linalg.norm(ring_keys - ring_keys[k], axis=1))
        kept = 0
        for j in order:
            if j <= k or times[j] - times[k] < MIN_LOOP_GAP_SEC:
                continue
            a, b = descriptors[k], descriptors[j]
            valid = (a > -1e8) & (b > -1e8)
            if valid.sum() < 50:
                continue
            raw.append((float(np.abs(a - b)[valid].mean()), k, int(j)))
            kept += 1
            if kept >= SC_TOP_PER_KEYFRAME:
                break
    raw.sort()
    seen, pairs = set(), []
    for sim, i, j in raw:
        if sim > SC_MAX_DESCRIPTOR_DISTANCE:
            continue
        key = (i // PAIR_BIN, j // PAIR_BIN)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j, sim))
    return pairs


def submap(frames, k, half=SUBMAP_HALF, voxel=SUBMAP_VOXEL):
    points = np.concatenate([frames[m]['points'] for m in range(max(0, k - half), min(len(frames), k + half + 1))])
    keys = np.floor(points / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[index]


def to_open3d(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return cloud


def measure_loop(frames, i, j, pos, rot):
    target = to_open3d(submap(frames, i))
    source = to_open3d(submap(frames, j))
    drift = pos[j] - pos[i]
    direction = drift / max(np.linalg.norm(drift), 1e-9)
    lateral = np.array([-direction[1], direction[0], 0.0])
    best = None
    for offset in ICP_DRIFT_OFFSETS:
        for side in ICP_LATERAL_OFFSETS:
            init = np.eye(4)
            init[:3, 3] = direction * offset + lateral * side
            result = o3d.pipelines.registration.registration_icp(
                source, target, ICP_MAX_CORRESPONDENCE_DIST, init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            if best is None or result.fitness > best.fitness:
                best = result
    if best is None or best.fitness < ICP_MIN_FITNESS:
        return None
    transform = best.transformation
    world_rot, world_t = transform[:3, :3], transform[:3, 3]
    # Loop measurement z_ij = T_i^-1 * (G * T_j), where G maps world points.
    z_rot = rot[i].T @ world_rot @ rot[j]
    z_t = rot[i].T @ (world_rot @ pos[j] + world_t - pos[i])
    odom_t = rot[i].T @ (pos[j] - pos[i])
    correction = float(np.linalg.norm(z_t - odom_t))
    if correction > MAX_CORRECTION_M:
        return None
    return z_rot, z_t, float(best.fitness), correction


def solve_pose_graph(frames, pos, rot, edges):
    graph = gtsam.NonlinearFactorGraph()
    prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(gtsam.Rot3(rot[0]), gtsam.Point3(pos[0])), prior))
    odom = gtsam.noiseModel.Diagonal.Sigmas(ODOM_SIGMA)
    for k in range(len(frames) - 1):
        rel_rot = rot[k].T @ rot[k + 1]
        rel_t = rot[k].T @ (pos[k + 1] - pos[k])
        graph.add(gtsam.BetweenFactorPose3(k, k + 1, gtsam.Pose3(gtsam.Rot3(rel_rot), gtsam.Point3(rel_t)), odom))
    loop = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(LOOP_CAUCHY_K),
        gtsam.noiseModel.Diagonal.Sigmas(LOOP_SIGMA))
    for (i, j, z_rot, z_t, _, _, _) in edges:
        graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(gtsam.Rot3(z_rot), gtsam.Point3(z_t)), loop))
    initial = gtsam.Values()
    for k in range(len(frames)):
        initial.insert(k, gtsam.Pose3(gtsam.Rot3(rot[k]), gtsam.Point3(pos[k])))
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    corrected_pos = np.array([result.atPose3(k).translation() for k in range(len(frames))])
    corrected_rot = np.array([result.atPose3(k).rotation().matrix() for k in range(len(frames))])
    return corrected_pos, corrected_rot


def warp_trajectory(traj_t, traj_pos, node_t, node_pos, node_rot, corrected_pos, corrected_rot):
    warped = np.zeros_like(traj_pos)
    for idx, t in enumerate(traj_t):
        a = int(np.clip(np.searchsorted(node_t, t), 1, len(node_t) - 1))
        alpha = float(np.clip((t - node_t[a - 1]) / max(node_t[a] - node_t[a - 1], 1e-9), 0.0, 1.0))
        d_rot_a = corrected_rot[a - 1] @ node_rot[a - 1].T
        d_t_a = corrected_pos[a - 1] - d_rot_a @ node_pos[a - 1]
        d_rot_b = corrected_rot[a] @ node_rot[a].T
        d_t_b = corrected_pos[a] - d_rot_b @ node_pos[a]
        d_rot = (1 - alpha) * d_rot_a + alpha * d_rot_b
        u, _, vt = np.linalg.svd(d_rot)
        d_rot = u @ vt
        warped[idx] = d_rot @ traj_pos[idx] + (1 - alpha) * d_t_a + alpha * d_t_b
    return warped


def umeyama(source, target):
    mu_s = source.mean(0)
    mu_t = target.mean(0)
    cov = (target - mu_t).T @ (source - mu_s) / len(source)
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot, mu_t - rot @ mu_s


def evaluate(traj_t, baseline, corrected, reference_dir):
    if not reference_dir or not os.path.isdir(reference_dir):
        return {}
    report = {}
    for name in ('hba-tum.txt', 'vilens-slam-tum.txt', 'colmap-tum.txt'):
        path = os.path.join(reference_dir, name)
        if not os.path.isfile(path):
            continue
        data = np.loadtxt(path, comments='#')
        ref_t, ref_p = data[:, 0], data[:, 1:4]
        mask = (traj_t >= ref_t[0]) & (traj_t <= ref_t[-1])
        times = traj_t[mask]
        ref = np.column_stack([np.interp(times, ref_t, ref_p[:, k]) for k in range(3)])
        errors = []
        for cloud in (baseline, corrected):
            rot, trans = umeyama(cloud[mask], ref)
            err = np.linalg.norm((rot @ cloud[mask].T).T + trans - ref, axis=1)
            errors.append((float(np.sqrt((err ** 2).mean())), float(np.median(err))))
        report[name] = {'before': errors[0], 'after': errors[1]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--keyframes', default=os.path.join(ROOT, 'output/spires/keyframes'))
    parser.add_argument('--trajectory', default=os.path.join(ROOT, 'output/spires/trajectory.csv'))
    parser.add_argument('--reference-dir', default=os.path.join(
        ROOT, 'datasets/oxford_spires/sequences/2024-03-18-christ-church-01/processed/trajectory'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'output/spires/loop/pipeline_result.npz'))
    args = parser.parse_args()

    frames = load_keyframes(args.keyframes)
    pos = np.array([f['pos'] for f in frames])
    rot = np.array([quat_to_rotation(f['quat']) for f in frames])
    times = np.array([f['t'] for f in frames])
    print(f'keyframes: {len(frames)}')

    pairs = propose_candidates(frames, pos, rot, times)
    print(f'scan-context candidate pairs: {len(pairs)}')
    edges, rejected = [], 0
    for (i, j, sim) in pairs:
        measured = measure_loop(frames, i, j, pos, rot)
        if measured is None:
            rejected += 1
            continue
        edges.append((i, j, measured[0], measured[1], measured[2], measured[3], sim))
        print(f'  loop ({i},{j}) sc_dist={sim:.2f} fitness={measured[2]:.2f} correction={measured[3]:.2f} m')
    print(f'accepted loop edges: {len(edges)} (rejected {rejected})')
    if not edges:
        raise SystemExit('no loop edges accepted')
    corrected_pos, corrected_rot = solve_pose_graph(frames, pos, rot, edges)

    traj_rows = list(csv.DictReader(open(args.trajectory)))
    traj_t = np.array([float(r['timestamp_sec']) for r in traj_rows])
    traj_pos = np.array([[float(r['px']), float(r['py']), float(r['pz'])] for r in traj_rows])
    warped = warp_trajectory(traj_t, traj_pos, times, pos, rot, corrected_pos, corrected_rot)

    closure_before = float(np.linalg.norm(pos[-1] - pos[0]))
    closure_after = float(np.linalg.norm(corrected_pos[-1] - corrected_pos[0]))
    report = evaluate(traj_t, traj_pos, warped, args.reference_dir)
    print(f'closure: {closure_before:.3f} -> {closure_after:.3f} m')
    for name, values in report.items():
        before, after = values['before'], values['after']
        print(f'  {name}: ATE {before[0]:.3f}/{before[1]:.3f} -> {after[0]:.3f}/{after[1]:.3f} m (rmse/median)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             corrected_pos=corrected_pos, corrected_rot=corrected_rot,
             edges=np.array([(e[0], e[1], e[4], e[5]) for e in edges]),
             closure=np.array([closure_before, closure_after]))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
