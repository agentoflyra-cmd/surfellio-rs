#!/usr/bin/env python3
"""Spires map re-fusion check: compare the drifted map with the map rebuilt from
LiDAR-loop-corrected keyframe poses.

Metric: for every 0.4 m voxel that contains points from both the first visit and
the loop (revisit) window, fit a plane on the first-visit points and measure the
loop-visit point-to-plane residual. Lower is better; this is the same metric used
in docs/spires-loop-closure.md.

Usage:
  python3 scripts/spires_refusion.py \
      --keyframes output/spires/keyframes \
      --corrected output/spires/loop/pipeline_result.npz
"""
import argparse
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    return frames


def collect(frames, rot, d_rot, d_t, visits, corrected, roi):
    points = []
    for k, frame in enumerate(frames):
        if not visits[k]:
            continue
        p = frame['points']
        if corrected:
            p = (d_rot[k] @ p.T).T + d_t[k]
        mask = (np.abs(p[:, 0]) < roi) & (np.abs(p[:, 1]) < roi)
        p = p[mask]
        if len(p):
            points.append(p)
    return np.concatenate(points) if points else np.zeros((0, 3))


def voxel_planes(points, voxel, min_points):
    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse, minlength=len(uniq))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    planes = {}
    for v in range(len(uniq)):
        if counts[v] < min_points:
            continue
        chunk = points[order[bounds[v]:bounds[v + 1]]]
        center = chunk.mean(0)
        cov = (chunk - center).T @ (chunk - center) / len(chunk)
        _, eigenvectors = np.linalg.eigh(cov)
        planes[tuple(uniq[v])] = (center, eigenvectors[:, 0])
    return planes


def layer_metric(early, late, voxel, min_points):
    early_planes = voxel_planes(early, voxel, min_points)
    late_planes = voxel_planes(late, voxel, min_points)
    residuals = []
    for key, (center, _) in late_planes.items():
        if key not in early_planes:
            continue
        early_center, early_normal = early_planes[key]
        residuals.append(abs(float(early_normal @ (center - early_center))))
    r = np.array(residuals)
    if len(r) == 0:
        return None
    return dict(mean=float(r.mean()), median=float(np.median(r)), p90=float(np.percentile(r, 90)), voxels=len(r))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--keyframes', default=os.path.join(ROOT, 'output/spires/keyframes'))
    parser.add_argument('--corrected', default=os.path.join(ROOT, 'output/spires/loop/pipeline_result.npz'))
    parser.add_argument('--early-sec', type=float, default=60.0)
    parser.add_argument('--late-sec', type=float, default=650.0)
    parser.add_argument('--voxel', type=float, default=0.4)
    parser.add_argument('--roi', type=float, default=20.0)
    parser.add_argument('--min-points', type=int, default=10)
    args = parser.parse_args()

    frames = load_keyframes(args.keyframes)
    pos = np.array([f['pos'] for f in frames])
    rot = np.array([quat_to_rotation(f['quat']) for f in frames])
    times = np.array([f['t'] for f in frames])
    t0 = times[0]

    data = np.load(args.corrected)
    corrected_pos, corrected_rot = data['corrected_pos'], data['corrected_rot']
    d_rot = np.array([corrected_rot[k] @ rot[k].T for k in range(len(frames))])
    d_t = corrected_pos - np.einsum('kij,kj->ki', d_rot, pos)

    early = times - t0 < args.early_sec
    late = times - t0 > args.late_sec
    for label, corrected in (('drifted', False), ('re-fused', True)):
        early_points = collect(frames, rot, d_rot, d_t, early, corrected, args.roi)
        late_points = collect(frames, rot, d_rot, d_t, late, corrected, args.roi)
        metric = layer_metric(early_points, late_points, args.voxel, args.min_points)
        if metric is None:
            print(f'{label}: no overlapping voxels')
            continue
        print(f'{label}: layer mean={metric["mean"]*1000:.1f} mm '
              f'median={metric["median"]*1000:.1f} mm p90={metric["p90"]*1000:.1f} mm '
              f'(voxels={metric["voxels"]})')


if __name__ == '__main__':
    main()
