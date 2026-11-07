#!/usr/bin/env python3
"""Generate a diagnostic package for each revisit sample.

Package contents (one directory per sample under output/revisit_packages/<dataset>/):
  topdown_xy.png        historical map + current scan at reference / GICP
  slices_xz.png         several x-z cross sections around the overlap
  virtual_view_ref.png  real range / virtual range / error / violation (reference)
  virtual_view_gicp.png same for the GICP alignment
  metrics.json          point-to-plane RMSE, range agreement, free-space
                        violations, overlap, local height differences
  summary.md            one-line conclusion: which alignment is more
                        self-consistent under the historical-map constraint

Usage:
  python3 scripts/revisit_packages.py --dataset ka
  python3 scripts/revisit_packages.py --dataset corridor
"""
import argparse
import csv
import json
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, 'output', 'revisit_packages')

DATASETS = {
    'ka': {
        'keyframes': 'output/ka_diag/keyframes_ka5',
        'surfels': 'output/ka_diag/ka_surfels.csv',
        'reference': {'kind': 'trajectory', 'path': 'output/trajectory-lili-ka-urban-east.csv'},
        'sample_mode': 'reference_proximity',
        'max_samples': 6,
        'sc_R': 0.0,  # not used by this generator
    },
    'corridor': {
        'keyframes': 'output/ka_diag/keyframes_corridor',
        'surfels': 'output/ka_diag/corridor_surfels_cov.csv',
        'reference': {'kind': 'corrected_poses', 'path': 'output/ka_diag/corridor_system_poses_vview.npz'},
        'sample_mode': 'corrected_proximity',
        'max_samples': 5,
    },
}

AZ_BINS = 1440
EL_MIN, EL_MAX = np.radians(-25.0), np.radians(15.0)
EL_BINS = 32


# ---------------------------------------------------------------- geometry ---
def quat_to_R(q):
    """q is (w, x, y, z) like the keyframe binary export."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def voxel_downsample(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def build_grid(points, cell):
    idx = np.floor(points / cell).astype(np.int64)
    grid = {}
    for i in range(len(points)):
        grid.setdefault((idx[i, 0], idx[i, 1], idx[i, 2]), []).append(i)
    return grid


def nearest(points, dst, grid, cell, max_dist):
    match = np.full(len(points), -1, dtype=np.int64)
    dist = np.full(len(points), 1e9)
    for k, p in enumerate(points):
        ci = np.floor(p / cell).astype(np.int64)
        best = max_dist
        bj = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((ci[0]+dx, ci[1]+dy, ci[2]+dz), ()):
                        d = np.linalg.norm(dst[j] - p)
                        if d < best:
                            best = d
                            bj = j
        match[k] = bj
        dist[k] = best
    return match, dist


# ------------------------------------------------------------------- gicp ---
def gicp(src, dst_pts, dst_nrm, init_R, init_t, scales=(0.6, 0.3), max_dist=1.0, iters=8):
    R = init_R.copy()
    t = init_t.copy()
    cur = (R @ src.T).T + t
    for voxel in scales:
        src_s = voxel_downsample(cur, voxel)
        grid = build_grid(dst_pts, voxel)
        for _ in range(iters):
            match, dist = nearest(src_s, dst_pts, grid, voxel, max_dist)
            valid = match >= 0
            if valid.sum() < 50:
                return None
            p = src_s[valid]
            q = dst_pts[match[valid]]
            n = dst_nrm[match[valid]]
            r = (n * (p - q)).sum(axis=1)
            huber = np.where(np.abs(r) < 0.05, 1.0, 0.05 / np.maximum(np.abs(r), 1e-9))
            J = np.hstack([np.cross(p, n), n]) * huber[:, None]
            A = J.T @ J + 1e-6 * np.eye(6)
            b = -J.T @ (r * huber)
            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            dR = so3_exp(delta[:3])
            R = dR @ R
            t = dR @ t + delta[3:]
            cur = (R @ src.T).T + t
        max_dist *= 0.5
    return R, t


def best_gicp_alignment(src, dst_pts, dst_nrm, map_world, pose_j, pdR, pos_i, pos_j):
    drift = pos_j - pos_i
    drift_dir = drift / max(np.linalg.norm(drift), 1e-6)
    best = None
    for deg in (-10.0, 0.0, 10.0):
        for off in (-0.5, 0.0, 0.5):
            init_R = rotz(deg) @ pdR
            init_t = (pos_i - init_R @ pos_j) + drift_dir * off
            out = gicp(src, dst_pts, dst_nrm, init_R, init_t)
            if out is None:
                continue
            Rg, tg = out
            _, _, _, agree, viol = range_metrics(map_world, src, Rg, tg, pose_j)
            score = -(agree - 0.5 * viol)
            if best is None or score < best[0]:
                best = (score, Rg, tg)
    return best


# --------------------------------------------------------- virtual camera ---
def render_range(points_sensor):
    r = np.linalg.norm(points_sensor, axis=1)
    az = np.arctan2(points_sensor[:, 1], points_sensor[:, 0])
    el = np.arctan2(points_sensor[:, 2], np.hypot(points_sensor[:, 0], points_sensor[:, 1]))
    ai = np.clip(((az + np.pi) / (2 * np.pi) * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
    ei = np.clip(((el - EL_MIN) / (EL_MAX - EL_MIN) * EL_BINS).astype(np.int64), 0, EL_BINS - 1)
    flat = ei * AZ_BINS + ai
    depth = np.full(EL_BINS * AZ_BINS, np.inf)
    np.minimum.at(depth, flat, r)
    return depth.reshape(EL_BINS, AZ_BINS)


def range_metrics(map_world, scan_world, R_align, t_align, pose_j):
    Rj, tj = pose_j
    Tj = np.eye(4); Tj[:3, :3] = Rj; Tj[:3, 3] = tj
    Ta = np.eye(4); Ta[:3, :3] = R_align; Ta[:3, 3] = t_align
    Tp = Ta @ Tj
    Rpi = Tp[:3, :3].T
    tpi = -Rpi @ Tp[:3, 3]
    map_sensor = (Rpi @ map_world.T).T + tpi
    scan_sensor = (Rj.T @ (scan_world - tj).T).T
    virtual = render_range(map_sensor)
    actual = render_range(scan_sensor)
    both = np.isfinite(virtual) & np.isfinite(actual)
    if not both.any():
        return virtual, actual, np.zeros_like(virtual), 0.0, 1.0
    diff = virtual - actual
    agree = float((np.abs(diff[both]) < 0.3).mean())
    viol = float((virtual[both] < actual[both] - 0.5).mean())
    return virtual, actual, diff, agree, viol


# ------------------------------------------------------------------ metrics ---
def align_scan(scan_world, R_align, t_align):
    return (R_align @ scan_world.T).T + t_align


def pt_plane_rmse(aligned, map_pts, map_nrm, grid):
    match, dist = nearest(aligned, map_pts, grid, 0.5, 1.0)
    valid = (match >= 0) & (dist < 0.6)
    if valid.sum() < 20:
        return 9.9, 0.0
    r = np.abs((map_nrm[match[valid]] * (aligned[valid] - map_pts[match[valid]])).sum(axis=1))
    return float(np.sqrt((r ** 2).mean())), float(valid.sum() / len(aligned))


def overlap_ratio(aligned, map_pts, grid):
    match, dist = nearest(aligned, map_pts, grid, 0.5, 1.0)
    return float(((match >= 0) & (dist < 0.5)).mean())


def local_height_diff(aligned, map_pts, grid):
    match, dist = nearest(aligned, map_pts, grid, 0.5, 1.0)
    valid = (match >= 0) & (dist < 0.5)
    if valid.sum() < 20:
        return 0.0, 0.0
    dz = np.abs(aligned[valid, 2] - map_pts[match[valid], 2])
    return float(np.median(dz)), float(np.percentile(dz, 90))


# -------------------------------------------------------------------- plots ---
def plot_topdown(path, map_pts, scan, ref_aligned, gicp_aligned, title):
    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111)
    ax.scatter(map_pts[::3, 0], map_pts[::3, 1], s=0.5, c='0.6', linewidths=0, label='historical map')
    ax.scatter(ref_aligned[::3, 0], ref_aligned[::3, 1], s=0.6, c='tab:green', alpha=0.5, linewidths=0, label='scan @ reference')
    ax.scatter(gicp_aligned[::3, 0], gicp_aligned[::3, 1], s=0.6, c='tab:red', alpha=0.5, linewidths=0, label='scan @ GICP')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8, markerscale=6)
    ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def plot_slices(path, map_pts, ref_aligned, gicp_aligned, title):
    ys = np.percentile(map_pts[:, 1], [25, 50, 75])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=140)
    for ax, y0 in zip(axes, ys):
        band = lambda p: p[np.abs(p[:, 1] - y0) < 0.6]
        mp, rp, gp = band(map_pts), band(ref_aligned), band(gicp_aligned)
        if len(mp): ax.scatter(mp[:, 0], mp[:, 2], s=1.0, c='0.6', linewidths=0, label='map')
        if len(rp): ax.scatter(rp[:, 0], rp[:, 2], s=1.5, c='tab:green', linewidths=0, label='ref')
        if len(gp): ax.scatter(gp[:, 0], gp[:, 2], s=1.5, c='tab:red', linewidths=0, label='GICP')
        ax.set_title(f'y={y0:.1f} m'); ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def plot_virtual(path, virtual, actual, diff, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=140)
    both = np.isfinite(virtual) & np.isfinite(actual)
    im0 = axes[0, 0].imshow(np.where(np.isfinite(actual), actual, np.nan), aspect='auto', cmap='viridis', vmin=0, vmax=40)
    axes[0, 0].set_title('real range'); plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    im1 = axes[0, 1].imshow(np.where(np.isfinite(virtual), virtual, np.nan), aspect='auto', cmap='viridis', vmin=0, vmax=40)
    axes[0, 1].set_title('virtual range'); plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    im2 = axes[1, 0].imshow(np.where(both, diff, np.nan), aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('range error (virtual - real)'); plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    viol = both & (virtual < actual - 0.5)
    im3 = axes[1, 1].imshow(viol.astype(float), aspect='auto', cmap='Reds')
    axes[1, 1].set_title('free-space violations'); plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------ inputs ---
def load_keyframes(directory):
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


def load_reference(spec):
    kind = spec['kind']
    path = os.path.join(ROOT, spec['path'])
    if kind == 'trajectory':
        rows = list(csv.DictReader(open(path)))
        t = np.array([float(r['timestamp_sec']) for r in rows])
        pos = np.array([[float(r['px']), float(r['py']), float(r['pz'])] for r in rows])
        quat = np.array([[float(r['qx']), float(r['qy']), float(r['qz']), float(r['qw'])] for r in rows])
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        return dict(kind=kind, t=t, pos=pos, quat=quat)
    data = np.load(path)
    return dict(kind=kind, pos=data['pos'], rot=data['rot'])


def reference_pose(ref, query_t, index=None):
    if ref['kind'] == 'corrected_poses':
        return ref['pos'][index], ref['rot'][index]
    t, pos, quat = ref['t'], ref['pos'], ref['quat']
    i = int(np.clip(np.searchsorted(t, query_t), 1, len(t) - 1))
    a = (query_t - t[i - 1]) / max(t[i] - t[i - 1], 1e-9)
    p = (1 - a) * pos[i - 1] + a * pos[i]
    q0, q1 = quat[i - 1], quat[i]
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    ang = np.arccos(dot)
    qq = q0 if ang < 1e-6 else (np.sin((1 - a) * ang) * q0 + np.sin(a * ang) * q1) / np.sin(ang)
    qq /= np.linalg.norm(qq)
    return p, quat_to_R(np.array([qq[3], qq[0], qq[1], qq[2]]))


def select_samples(dataset, frames, ref):
    pos = np.array([f['pos'] for f in frames])
    n = len(frames)
    rng = np.random.default_rng(0)
    if ref['kind'] == 'trajectory':
        rp = np.array([reference_pose(ref, fr['t'])[0] for fr in frames])
        candidates = []
        for i in range(0, n, 7):
            d = np.linalg.norm(rp - rp[i], axis=1)
            for j in np.flatnonzero((d < 5.0) & (d > 0.8)):
                if int(j) <= i or frames[j]['t'] - frames[i]['t'] < 120.0:
                    continue
                candidates.append((i, int(j), float(d[j])))
        rng.shuffle(candidates)
        samples = []
        for lo, hi in ((0.8, 2.0), (2.0, 3.5), (3.5, 5.0)):
            added = 0
            for i, j, dist in candidates:
                if lo <= dist < hi:
                    samples.append((i, j, dist))
                    added += 1
                    if added >= 2:
                        break
        return samples
    # corridor: pairs of early and late keyframes that are close in the
    # corrected (reference) trajectory
    early = list(range(0, 18))
    late = [252, 258, 264, 270, 274]
    samples = []
    for j in late:
        i = min(early, key=lambda k: np.linalg.norm(ref['pos'][k] - ref['pos'][j]))
        samples.append((i, j, float(np.linalg.norm(ref['pos'][i] - ref['pos'][j]))))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    out_dir = os.path.join(OUT_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    frames = load_keyframes(os.path.join(ROOT, spec['keyframes']))
    n = len(frames)
    pos = np.array([f['pos'] for f in frames])
    R = np.array([quat_to_R(f['quat']) for f in frames])
    ref = load_reference(spec['reference'])
    rows = list(csv.reader(open(os.path.join(ROOT, spec['surfels']))))
    data = np.loadtxt(os.path.join(ROOT, spec['surfels']), delimiter=',', skiprows=1)
    sx, sn, sc = data[:, :3], data[:, 3:6], data[:, 7]
    sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-6)
    samples = select_samples(args.dataset, frames, ref)
    if args.samples:
        samples = samples[:args.samples]
    print(f'dataset={args.dataset} keyframes={n} samples={len(samples)}')
    summaries = []
    for idx, (i, j, ref_dist) in enumerate(samples):
        name = f'sample_{idx:02d}_i{i}_j{j}'
        sample_dir = os.path.join(out_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        ids = np.flatnonzero((sc < frames[j]['t'] - 30.0) & (np.linalg.norm(sx - pos[i], axis=1) < 40.0))
        if len(ids) < 300:
            print(name, 'too few map surfels'); continue
        map_pts, map_nrm = sx[ids], sn[ids]
        grid = build_grid(map_pts, 0.5)
        map_world = np.concatenate([frames[m]['pts'] for m in range(max(0, i - 10), min(n, i + 10))])[::4]
        src = frames[j]['pts']
        # reference alignment: relative pose from the reference trajectory
        pi, Ri_ref = reference_pose(ref, frames[i]['t'], index=i)
        pj, Rj_ref = reference_pose(ref, frames[j]['t'], index=j)
        R_rel = Ri_ref.T @ Rj_ref
        t_rel = Ri_ref.T @ (pj - pi)
        R_ref = R[i] @ R_rel @ R[j].T
        t_ref = pos[i] + R[i] @ t_rel - R_ref @ pos[j]
        # GICP alignment
        best = best_gicp_alignment(src, map_pts, map_nrm, map_world, (R[j], pos[j]),
                                   R[i] @ R[j].T, pos[i], pos[j])
        if best is None:
            print(name, 'no GICP alignment'); continue
        _, R_gicp, t_gicp = best
        ref_aligned = align_scan(src, R_ref, t_ref)
        gicp_aligned = align_scan(src, R_gicp, t_gicp)
        metrics = {}
        for tag, Ra, ta in (('ref', R_ref, t_ref), ('gicp', R_gicp, t_gicp)):
            aligned = align_scan(src, Ra, ta)
            rmse, valid = pt_plane_rmse(aligned, map_pts, map_nrm, grid)
            overlap = overlap_ratio(aligned, map_pts, grid)
            virtual, actual, diff, agree, viol = range_metrics(map_world, src, Ra, ta, (R[j], pos[j]))
            hm, hp = local_height_diff(aligned, map_pts, grid)
            metrics[tag] = dict(pt_plane_rmse_m=rmse, pt_plane_valid_frac=valid, overlap=overlap,
                                range_agree=agree, free_space_violation=viol,
                                local_height_diff_mean_m=hm, local_height_diff_p90_m=hp)
            suffix = 'ref' if tag == 'ref' else 'gicp'
            plot_virtual(os.path.join(sample_dir, f'virtual_view_{suffix}.png'),
                         virtual, actual, diff, f'virtual view @ {"reference" if tag == "ref" else "GICP"}')
        metrics['pair'] = dict(i=i, j=j, ref_distance_m=float(ref_dist))
        json.dump(metrics, open(os.path.join(sample_dir, 'metrics.json'), 'w'), indent=2, sort_keys=True)
        plot_topdown(os.path.join(sample_dir, 'topdown_xy.png'), map_pts, src, ref_aligned, gicp_aligned,
                     f'{args.dataset} sample {idx}: i={i} j={j}')
        plot_slices(os.path.join(sample_dir, 'slices_xz.png'), map_pts, ref_aligned, gicp_aligned,
                    f'{args.dataset} sample {idx}: x-z slices')

        def score(tag):
            m = metrics[tag]
            return m['range_agree'] - m['free_space_violation'] - 0.1 * m['pt_plane_rmse_m']

        s_ref, s_gicp = score('ref'), score('gicp')
        if s_ref > s_gicp + 0.02:
            verdict = '参考位姿更自洽'
        elif s_gicp > s_ref + 0.02:
            verdict = 'GICP 位姿更自洽'
        else:
            verdict = '两者自洽程度接近'
        line = (
            f'# {args.dataset} sample {idx} (i={i}, j={j})\n\n'
            f'- reference: agree={metrics["ref"]["range_agree"]:.2f}, '
            f'violation={metrics["ref"]["free_space_violation"]:.2f}, '
            f'pt-plane RMSE={metrics["ref"]["pt_plane_rmse_m"]*1000:.1f} mm, '
            f'overlap={metrics["ref"]["overlap"]:.2f}\n'
            f'- GICP: agree={metrics["gicp"]["range_agree"]:.2f}, '
            f'violation={metrics["gicp"]["free_space_violation"]:.2f}, '
            f'pt-plane RMSE={metrics["gicp"]["pt_plane_rmse_m"]*1000:.1f} mm, '
            f'overlap={metrics["gicp"]["overlap"]:.2f}\n\n'
            f'**结论：在历史地图约束下，{verdict}。**\n'
        )
        open(os.path.join(sample_dir, 'summary.md'), 'w').write(line)
        summaries.append(line)
        print(f'{name}: ref_score={s_ref:.3f} gicp_score={s_gicp:.3f} -> {verdict}')
    open(os.path.join(out_dir, 'ALL_SUMMARIES.md'), 'w').write('\n'.join(summaries))
    print(f'packages written to {out_dir}')


if __name__ == '__main__':
    sys.exit(main())
