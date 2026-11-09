#!/usr/bin/env python3
"""Regression check for the fastlio-rs replay baseline.

Runs the four development bags with their pinned configs, computes trajectory
accuracy and map/latency statistics, and compares them against a committed
baseline JSON. Exits non-zero if any metric regresses beyond its tolerance.

Usage:
    python3 scripts/regression_check.py --write-baseline   # refresh baseline
    python3 scripts/regression_check.py                    # check
    python3 scripts/regression_check.py --bags KA,FR        # subset

The reference trajectories and raw bags are local artifacts (output/ and
rosbags/ are both gitignored); this script only pins the *checks* that matter
so the numbers cannot silently drift.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output', 'regression')
BIN = os.path.join(ROOT, 'target', 'release', 'fastlio-replay')
BASELINE = os.path.join(ROOT, 'scripts', 'regression_baseline.json')

BAGS = {
    'KA': {
        'bag': 'rosbags/KA_Urban_East/KA_Urban_East.mcap',
        'config': 'config/mid360_FR_IOSB_Short.yaml',
        'eval': 'ka',
    },
    'FR': {
        'bag': 'rosbags/FR_IOSB_Short/FR_IOSB_Short.mcap',
        'config': 'config/mid360_FR_IOSB_Short.yaml',
        'eval': 'fr',
    },
    'upstairs': {
        'bag': 'rosbags/rosbag2_upstairs/rosbag2_2026_06_23-15_13_52_0.mcap',
        'config': 'config/mid360.yaml',
        'eval': 'upstairs',
    },
    'yuanqu': {
        'bag': 'rosbags/yuanqu_with_odom/yuanqu_with_odom_0.mcap',
        'config': 'config/mid360.yaml',
        'eval': 'yuanqu',
    },
}

# name -> (relative tolerance on lower-is-better metrics, absolute tolerance)
TOLERANCE = {
    'ate': 0.05,
    'e2e': 0.05,
    'z_rmse': 0.05,
    'duplicates': 0.15,
    'latency_p99': 0.20,
    'latency_p50': 0.20,
    'latency_mean': 0.20,
    'surfels': 0.05,
    'merged': 0.15,
    'compacted': 0.15,
}


def read_csv(path):
    import numpy as np
    rows = list(csv.DictReader(open(path)))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            pass
    return out


def umeyama_2d(src, dst):
    import numpy as np
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    a = src - mu_s
    b = dst - mu_d
    cov = b.T @ a / len(src)
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, mu_d - r @ mu_s


def umeyama_3d(src, dst):
    import numpy as np
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    a = src - mu_s
    b = dst - mu_d
    cov = b.T @ a / len(src)
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, mu_d - r @ mu_s


def lla_to_ecef(lat, lon, alt):
    import numpy as np
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    la = np.radians(lat)
    lo = np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.column_stack([
        (n + alt) * np.cos(la) * np.cos(lo),
        (n + alt) * np.cos(la) * np.sin(lo),
        (n * (1 - e2) + alt) * np.sin(la),
    ])


def ecef_to_enu(ecef, lat0, lon0, alt0):
    import numpy as np
    x0 = lla_to_ecef(np.array([lat0]), np.array([lon0]), np.array([alt0]))[0]
    la, lo = np.radians(lat0), np.radians(lon0)
    sl, cl = np.sin(la), np.cos(la)
    so, co = np.sin(lo), np.cos(lo)
    e = ecef - x0
    return np.column_stack([
        -so * e[:, 0] + co * e[:, 1],
        -sl * co * e[:, 0] - sl * so * e[:, 1] + cl * e[:, 2],
        cl * co * e[:, 0] + cl * so * e[:, 1] + sl * e[:, 2],
    ])


def eval_ka(traj_path):
    import numpy as np
    gnss = read_csv(os.path.join(ROOT, 'output', 'ka_diag', 'gnss.csv'))
    lla = read_csv(os.path.join(ROOT, 'output', 'ka_diag', 'positionlla.csv'))
    lat0, lon0, alt0 = gnss['lat'][0], gnss['lon'][0], gnss['alt'][0]
    ref = ecef_to_enu(lla_to_ecef(lla['x'], lla['y'], lla['z']), lat0, lon0, alt0)
    ours = read_csv(traj_path)
    t = ours['timestamp_sec']
    m = (t >= lla['t'][0]) & (t <= lla['t'][-1])
    to = t[m]
    ro = np.column_stack([np.interp(to, lla['t'], ref[:, i]) for i in range(2)])
    oo = np.column_stack([ours['px'][m], ours['py'][m]])
    r, tr = umeyama_2d(oo, ro)
    err = np.linalg.norm(oo @ r.T + tr - ro, axis=1)
    start = (to - to[0]) <= 10.0
    rs, ts = umeyama_2d(oo[start], ro[start])
    errs = np.linalg.norm(oo @ rs.T + ts - ro, axis=1)
    return {'ate': float(np.sqrt((err ** 2).mean())), 'e2e': float(errs[-1])}


def eval_fr(traj_path):
    import numpy as np
    gnss = read_csv(os.path.join(ROOT, 'output', 'ka_diag', 'fr_gnss.csv'))
    ref = ecef_to_enu(lla_to_ecef(gnss['lat'], gnss['lon'], gnss['alt']), gnss['lat'][0], gnss['lon'][0], gnss['alt'][0])
    ours = read_csv(traj_path)
    t = ours['timestamp_sec']
    m = (gnss['t'] >= t[0]) & (gnss['t'] <= t[-1])
    tg = gnss['t'][m]
    gg = ref[m, :2]
    op = np.column_stack([np.interp(tg, t, ours['px']), np.interp(tg, t, ours['py'])])
    r, tr = umeyama_2d(op, gg)
    err = np.linalg.norm(op @ r.T + tr - gg, axis=1)
    start = (tg - tg[0]) <= 10.0
    rs, ts = umeyama_2d(op[start], gg[start])
    errs = np.linalg.norm(op @ rs.T + ts - gg, axis=1)
    return {'ate': float(np.sqrt((err ** 2).mean())), 'e2e': float(errs[-1])}


def eval_ref3d(traj_path, ref_path):
    import numpy as np
    ref = read_csv(ref_path)
    ours = read_csv(traj_path)
    t = ours['timestamp_sec']
    tr = ref['timestamp_sec']
    m = (tr >= t[0]) & (tr <= t[-1])
    trm = tr[m]
    oo = np.column_stack([
        np.interp(trm, t, ours['px']),
        np.interp(trm, t, ours['py']),
        np.interp(trm, t, ours['pz']),
    ])
    rr = np.column_stack([ref['px'][m], ref['py'][m], ref['pz'][m]])
    align = (trm - trm[0]) <= 100.0
    r, tt = umeyama_3d(oo[align], rr[align])
    al = oo @ r.T + tt
    err = np.linalg.norm(al - rr, axis=1)
    z_err = al[:, 2] - rr[:, 2]
    return {
        'ate': float(np.sqrt((err ** 2).mean())),
        'z_rmse': float(np.sqrt((z_err ** 2).mean())),
    }


def parse_stats(stdout):
    stats = {}
    m = re.search(r'coplanar duplicate pairs \(estimated\): (\d+)', stdout)
    if m:
        stats['duplicates'] = int(m.group(1))
    m = re.search(r'total surfels: (\d+)', stdout)
    if m:
        stats['surfels'] = int(m.group(1))
    m = re.search(r'merged_total=(\d+)', stdout)
    if m:
        stats['merged'] = int(m.group(1))
    m = re.search(r'compacted_total=(\d+)', stdout)
    if m:
        stats['compacted'] = int(m.group(1))
    m = re.search(r'frame latency: mean=([\d.]+) ms p50=([\d.]+) p99=([\d.]+)', stdout)
    if m:
        stats['latency_mean'] = float(m.group(1))
        stats['latency_p50'] = float(m.group(2))
        stats['latency_p99'] = float(m.group(3))
    return stats


def run_bag(name, spec):
    bag = os.path.join(ROOT, spec['bag'])
    config = os.path.join(ROOT, spec['config'])
    os.makedirs(OUT, exist_ok=True)
    traj = os.path.join(OUT, f'{name}.csv')
    log_path = os.path.join(OUT, f'{name}.log')
    cmd = [BIN, bag, config, '0.0', '1024', traj, '--map-stats']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f'{name}: replay failed (exit {result.returncode})\n{result.stderr[-2000:]}')
    with open(log_path, 'w') as f:
        f.write(result.stdout)
        f.write('\n[stderr]\n')
        f.write(result.stderr)
    if spec['eval'] == 'ka':
        metrics = eval_ka(traj)
    elif spec['eval'] == 'fr':
        metrics = eval_fr(traj)
    elif spec['eval'] == 'upstairs':
        metrics = eval_ref3d(traj, os.path.join(ROOT, 'output', 'trajectory-fastlio-sam-output-upstairs.csv'))
    elif spec['eval'] == 'yuanqu':
        metrics = eval_ref3d(traj, os.path.join(ROOT, 'output', 'trajectory-fastlio-sam-output-yuanqu.csv'))
    else:
        raise ValueError(spec['eval'])
    metrics.update(parse_stats(result.stdout))
    return metrics


def compare(name, metrics, baseline):
    failures = []
    base = baseline['bags'][name]['metrics']
    for key, expected in base.items():
        if key not in metrics:
            failures.append(f'{name}.{key}: missing from current run')
            continue
        actual = metrics[key]
        if key in ('duplicates', 'latency_p99', 'latency_p50'):
            limit = expected * (1.0 + TOLERANCE[key])
        else:
            limit = expected * (1.0 + TOLERANCE[key]) + 1e-6
        ok = actual <= limit
        if not ok:
            failures.append(f'{name}.{key}: {actual:.4f} > {limit:.4f} (baseline {expected:.4f})')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-baseline', action='store_true')
    parser.add_argument('--bags', default=','.join(BAGS.keys()))
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    if not args.skip_build:
        subprocess.run(['cargo', 'build', '--release', '-p', 'fastlio-replay'], cwd=ROOT, check=True)
    selected = [name.strip() for name in args.bags.split(',') if name.strip()]
    results = {}
    for name in selected:
        print(f'== {name}', flush=True)
        metrics = run_bag(name, BAGS[name])
        results[name] = {'metrics': metrics}
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.write_baseline:
        with open(BASELINE, 'w') as f:
            json.dump({'bags': results}, f, indent=2, sort_keys=True)
        print(f'baseline written to {BASELINE}')
        return 0
    baseline = json.load(open(BASELINE))
    failures = []
    for name in selected:
        failures.extend(compare(name, results[name]['metrics'], baseline))
    print()
    if failures:
        print('REGRESSION DETECTED:')
        for failure in failures:
            print(f'  - {failure}')
        return 1
    print('regression check passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
