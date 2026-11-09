#!/usr/bin/env python3
"""Corridor surfel-level re-fusion check.

Applies the pose-graph corrected keyframe poses to the raw keyframe clouds,
re-inserts them into a fresh voxel map, refits one plane per voxel and measures
the loop-region layer residual (loop-visit points against first-visit planes).

This is the "re-fusion / rebuild" counterpart of surfel deformation: it needs
the raw keyframe clouds, but achieves a much smaller layer residual.

Usage:
  python3 scripts/corridor_surfel_refusion.py
"""
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KF = os.path.join(ROOT, 'output', 'ka_diag', 'keyframes_corridor')
POSES = os.path.join(ROOT, 'output', 'ka_diag', 'corridor_system_poses_vview.npz')
FIRST_VISIT_SEC = 8.0
LOOP_VISIT_RANGE = (126.0, 136.0)
ROI = 6.0


def load_frames(directory):
    frames = []
    for name in sorted(f for f in os.listdir(directory) if f.endswith('.bin')):
        with open(os.path.join(directory, name), 'rb') as f:
            f.read(8)
            t = struct.unpack('<d', f.read(8))[0]
            pos = np.frombuffer(f.read(24), dtype='<f8').copy()
            quat = np.frombuffer(f.read(32), dtype='<f8').copy()
            n = struct.unpack('<I', f.read(4))[0]
            pts = np.frombuffer(f.read(n * 12), dtype='<f4').reshape(n, 3).astype(np.float64)
        frames.append(dict(t=t, pos=pos, quat=quat, pts=pts))
    return frames


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def voxel_groups(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return order, bounds, counts


def build_surfels(points, tags, voxel):
    order, bounds, counts = voxel_groups(points, voxel)
    surfels = []
    for v in range(len(counts)):
        idx = order[bounds[v]:bounds[v + 1]]
        P = points[idx]
        c = P.mean(0)
        C = (P - c).T @ (P - c) / len(P)
        ev, evec = np.linalg.eigh(C)
        surfels.append(dict(center=c, normal=evec[:, 0], points=P, tags=tags[idx],
                            first=np.any(tags[idx] == 0), loop=np.any(tags[idx] == 1)))
    return surfels


def layer_metric(surfels):
    values = []
    per_voxel = []
    for s in surfels:
        if not (s['first'] and s['loop']):
            continue
        first = s['points'][s['tags'] == 0]
        loop = s['points'][s['tags'] == 1]
        if len(first) < 6 or len(loop) == 0:
            continue
        c = first.mean(0)
        C = (first - c).T @ (first - c) / len(first)
        ev, evec = np.linalg.eigh(C)
        n = evec[:, 0]
        r = np.abs((loop - c) @ n)
        values.extend(r.tolist())
        per_voxel.append(float(np.mean(r)))
    if not values:
        return None
    vm = np.array(per_voxel)
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)),
                frac_below_5cm=float((vm < 0.05).mean()), voxels=len(vm))


def main():
    frames = load_frames(KF)
    n = len(frames)
    pos = np.array([f['pos'] for f in frames])
    R = np.array([quat_to_R(f['quat']) for f in frames])
    data = np.load(POSES)
    corr_pos, corr_R = data['pos'], data['rot']
    t = np.array([f['t'] for f in frames])
    t0 = t[0]
    for label, corrected in (('drifted', False), ('corrected', True)):
        first, loop = [], []
        for k in range(n):
            dt = t[k] - t0
            is_first = dt < FIRST_VISIT_SEC
            is_loop = LOOP_VISIT_RANGE[0] < dt < LOOP_VISIT_RANGE[1]
            if not (is_first or is_loop):
                continue
            p = frames[k]['pts']
            if corrected:
                dR = corr_R[k] @ R[k].T
                dtc = corr_pos[k] - dR @ pos[k]
                p = (dR @ p.T).T + dtc
            m = (np.abs(p[:, 0]) < ROI) & (np.abs(p[:, 1]) < ROI)
            (first if is_first else loop).append(p[m])
        first_pts = np.concatenate(first)
        loop_pts = np.concatenate(loop)
        print(f'{label}: first={len(first_pts)} loop={len(loop_pts)}')
        for voxel in (0.4, 0.25):
            surfels = build_surfels(np.concatenate([first_pts, loop_pts]),
                                    np.concatenate([np.zeros(len(first_pts)), np.ones(len(loop_pts))]),
                                    voxel)
            m = layer_metric(surfels)
            if m:
                print(f'  voxel={voxel:.2f}: mean={m["mean"]*1000:.1f} median={m["median"]*1000:.1f} '
                      f'p90={m["p90"]*1000:.1f} mm  voxels<5cm={m["frac_below_5cm"]*100:.0f}% '
                      f'(n={m["voxels"]})')


if __name__ == '__main__':
    main()
