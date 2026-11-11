#!/usr/bin/env python3
"""Spires map re-fusion (simple option): rebuild the surfel map with the
LiDAR-loop-corrected keyframe poses.

The re-fusion itself is deliberately simple:
  1. for every keyframe, transform its raw point cloud with the correction
     dT_k = T'_k T_k^-1 implied by the corrected poses;
  2. re-voxelise (0.25 m) and re-fit one plane per voxel (center + normal from
     the covariance eigen-decomposition);
  3. report map statistics and the loop-region layer residual, and export the
     map as NPZ + the same binary PCD layout used by fastlio-replay.

Only numpy is required; matplotlib is optional and only used for --render.

Usage:
  python3 scripts/spires_refusion.py --render
  python3 scripts/spires_refusion.py --keyframes output/spires/keyframes \
      --corrected output/spires/loop/pipeline_result.npz \
      --out-dir output/spires/refusion
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
    if not frames:
        raise SystemExit(f'no keyframes in {directory}')
    return frames


def corrections(frames, corrected):
    """Return per-keyframe dR, dt with p_corrected_world = dR @ p_world + dt."""
    rot = np.array([quat_to_rotation(f['quat']) for f in frames])
    pos = np.array([f['pos'] for f in frames])
    corrected_pos = corrected['corrected_pos']
    corrected_rot = corrected['corrected_rot']
    d_rot = np.array([corrected_rot[k] @ rot[k].T for k in range(len(frames))])
    d_t = corrected_pos - np.einsum('kij,kj->ki', d_rot, pos)
    return rot, pos, d_rot, d_t


def transform_points(points, d_rot, d_t):
    return (d_rot @ points.T).T + d_t


def voxel_planes(points, voxel, min_points):
    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse, minlength=len(uniq))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    centers, normals, eigenvalues, kept_counts = [], [], [], []
    for v in range(len(uniq)):
        if counts[v] < min_points:
            continue
        chunk = points[order[bounds[v]:bounds[v + 1]]]
        center = chunk.mean(0)
        cov = (chunk - center).T @ (chunk - center) / len(chunk)
        ev, evec = np.linalg.eigh(cov)
        centers.append(center)
        normals.append(evec[:, 0])
        eigenvalues.append(ev)
        kept_counts.append(counts[v])
    if not centers:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    return np.array(centers), np.array(normals), np.array(eigenvalues), np.array(kept_counts)


def collect_visit(frames, d_rot, d_t, visit_mask, corrected, roi):
    points = []
    for k, frame in enumerate(frames):
        if not visit_mask[k]:
            continue
        p = frame['points']
        if corrected:
            p = transform_points(p, d_rot[k], d_t[k])
        mask = (np.abs(p[:, 0]) < roi) & (np.abs(p[:, 1]) < roi)
        p = p[mask]
        if len(p):
            points.append(p)
    return np.concatenate(points) if points else np.zeros((0, 3))


def layer_metric(early, late, voxel, min_points):
    early_planes = voxel_planes(early, voxel, min_points)
    late_planes = voxel_planes(late, voxel, min_points)
    early_map = {}
    for v in range(len(early_planes[0])):
        key = tuple(np.floor(early_planes[0][v] / voxel).astype(np.int64))
        early_map[key] = (early_planes[0][v], early_planes[1][v])
    residuals = []
    for v in range(len(late_planes[0])):
        key = tuple(np.floor(late_planes[0][v] / voxel).astype(np.int64))
        if key not in early_map:
            continue
        center, normal = early_map[key]
        residuals.append(abs(float(normal @ (late_planes[0][v] - center))))
    if not residuals:
        return None
    r = np.array(residuals)
    return dict(mean=float(r.mean()), median=float(np.median(r)), p90=float(np.percentile(r, 90)),
                voxels=len(r))


def write_pcd(path, centers, normals):
    fields = 'x y z intensity normal_x normal_y normal_z class_id'
    header = (f'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
              f'FIELDS {fields}\nSIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\n'
              f'COUNT 1 1 1 1 1 1 1 1\nWIDTH {len(centers)}\nHEIGHT 1\n'
              f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(centers)}\nDATA binary\n')
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                      ('normal_x', '<f4'), ('normal_y', '<f4'), ('normal_z', '<f4'),
                      ('class_id', '<f4')])
    packed = np.zeros(len(centers), dtype=dtype)
    packed['x'], packed['y'], packed['z'] = centers[:, 0], centers[:, 1], centers[:, 2]
    packed['normal_x'], packed['normal_y'], packed['normal_z'] = normals[:, 0], normals[:, 1], normals[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(packed.tobytes())


def render(path, centers, early, late):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skip render')
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=140)
    ax = axes[0]
    sc = ax.scatter(centers[::2, 0], centers[::2, 1], s=0.2, c=centers[::2, 2], cmap='turbo', linewidths=0)
    ax.set_aspect('equal')
    ax.set_title(f're-fused surfel map ({len(centers)} surfaces, color=z)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='z (m)')
    ax = axes[1]
    ax.scatter(early[::5, 0], early[::5, 1], s=0.15, c='tab:red', alpha=0.4, linewidths=0, label='first visit')
    ax.scatter(late[::5, 0], late[::5, 1], s=0.15, c='tab:blue', alpha=0.4, linewidths=0, label='loop visit')
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=8)
    ax.set_title('loop region after re-fusion')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--keyframes', default=os.path.join(ROOT, 'output/spires/keyframes'))
    parser.add_argument('--corrected', default=os.path.join(ROOT, 'output/spires/loop/pipeline_result.npz'))
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'output/spires/refusion'))
    parser.add_argument('--voxel', type=float, default=0.25)
    parser.add_argument('--min-points', type=int, default=10)
    parser.add_argument('--early-sec', type=float, default=60.0)
    parser.add_argument('--late-sec', type=float, default=650.0)
    parser.add_argument('--roi', type=float, default=20.0)
    parser.add_argument('--render', action='store_true', help='write a top-down preview PNG')
    args = parser.parse_args()

    frames = load_keyframes(args.keyframes)
    corrected = np.load(args.corrected)
    rot, pos, d_rot, d_t = corrections(frames, corrected)
    times = np.array([f['t'] for f in frames])
    t0 = times[0]

    # 1) rebuild the full map with corrected poses
    corrected_points = [transform_points(f['points'], d_rot[k], d_t[k]) for k, f in enumerate(frames)]
    all_points = np.concatenate(corrected_points)
    centers, normals, eigenvalues, counts = voxel_planes(all_points, args.voxel, args.min_points)
    print(f're-fused surfaces: {len(centers)} (voxel {args.voxel} m, points {len(all_points)})')
    print(f'mean count/voxel: {counts.mean():.1f}; planar (l0/l2<0.02): '
          f'{np.mean(eigenvalues[:, 0] / np.maximum(eigenvalues[:, 2], 1e-12) < 0.02):.2f}')

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'surfels.npz'),
             centers=centers, normals=normals, eigenvalues=eigenvalues, counts=counts)
    write_pcd(os.path.join(args.out_dir, 'map.pcd'), centers, normals)
    print(f'saved {args.out_dir}/surfels.npz and {args.out_dir}/map.pcd')

    # 2) map consistency: visit-separated layer residual, drifted vs re-fused
    early = times - t0 < args.early_sec
    late = times - t0 > args.late_sec
    for label, is_corrected in (('drifted', False), ('re-fused', True)):
        early_points = collect_visit(frames, d_rot, d_t, early, is_corrected, args.roi)
        late_points = collect_visit(frames, d_rot, d_t, late, is_corrected, args.roi)
        metric = layer_metric(early_points, late_points, 0.4, args.min_points)
        if metric is None:
            print(f'{label}: no overlapping voxels')
            continue
        print(f'{label}: layer mean={metric["mean"]*1000:.1f} mm median={metric["median"]*1000:.1f} mm '
              f'p90={metric["p90"]*1000:.1f} mm (voxels={metric["voxels"]})')

    if args.render:
        early_points = collect_visit(frames, d_rot, d_t, early, True, args.roi)
        late_points = collect_visit(frames, d_rot, d_t, late, True, args.roi)
        png = os.path.join(args.out_dir, 'map.png')
        render(png, centers, early_points, late_points)
        print(f'saved {png}')


if __name__ == '__main__':
    main()
