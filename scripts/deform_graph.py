"""Full deformation-graph prototype on corridor_04.

Nodes: keyframe poses (278). Variables: per-node SE(3) correction DeltaT_n.
Constraints:
  * loop anchors: corrected relative pose between verified loop keyframes
    matches the virtual-view verified loop measurement;
  * smoothness: consecutive node corrections are close to each other;
  * map anchors: loop-visit surfels, transformed by the (time-blended) node
    corrections, must lie on the old-visit surfel planes.
Solved with LM (dense numeric Jacobian). Finally the surfel map is deformed and
the loop-region layer residual is measured.
"""
import os, struct
import numpy as np

KF = 'output/ka_diag/keyframes_corridor'
DUMP = 'output/ka_diag/corridor_surfels.csv'
EDGES = 'output/ka_diag/corridor_system_edges.npz'

def load_frames():
    out = []
    for name in sorted(f for f in os.listdir(KF) if f.endswith('.bin')):
        with open(os.path.join(KF, name), 'rb') as f:
            f.read(8)
            t = struct.unpack('<d', f.read(8))[0]
            pos = np.frombuffer(f.read(24), dtype='<f8').copy()
            quat = np.frombuffer(f.read(32), dtype='<f8').copy()
            f.read(struct.unpack('<I', f.read(4))[0] * 12)
        out.append(dict(t=t, pos=pos, quat=quat))
    return out

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def so3_log(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1)
    a = np.arccos(c)
    if a < 1e-8:
        return np.zeros(3)
    return a * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(a))

def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w/a
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)

def build_grid(points, cell):
    idx = np.floor(points/cell).astype(np.int64)
    grid = {}
    for i in range(len(points)):
        grid.setdefault((idx[i,0],idx[i,1],idx[i,2]), []).append(i)
    return grid

def nearest(points, dst, grid, cell, maxd):
    out = np.full(len(points), -1, dtype=np.int64)
    dist = np.full(len(points), 1e9)
    for k, p in enumerate(points):
        ci = np.floor(p/cell).astype(np.int64)
        best = maxd; bj = -1
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    for j in grid.get((ci[0]+dx,ci[1]+dy,ci[2]+dz), ()):
                        d = np.linalg.norm(dst[j]-p)
                        if d < best: best = d; bj = j
        out[k] = bj; dist[k] = best
    return out, dist

def main():
    frames = load_frames(); n = len(frames)
    pos = np.array([f['pos'] for f in frames])
    R = np.array([quat_to_R(f['quat']) for f in frames])
    t = np.array([f['t'] for f in frames])
    data = np.load(EDGES, allow_pickle=True)
    loop_edges = [(int(i), int(j), np.array(Rm), np.array(tm)) for i, j, Rm, tm in data['vview']]
    print(f'nodes={n} loop edges={len(loop_edges)}')
    # surfels
    with open(DUMP) as f:
        f.readline()
        d = np.loadtxt(f, delimiter=',')
    sx, sn, sc = d[:,:3], d[:,3:6], d[:,7]
    sn /= np.maximum(np.linalg.norm(sn,axis=1,keepdims=True),1e-6)
    t0 = t[0]
    old_mask = (sc < t0+8.0) & (np.abs(sx[:,0])<6) & (np.abs(sx[:,1])<6)
    loop_mask = (sc > t0+126.0) & (sc < t0+136.0) & (np.abs(sx[:,0])<6) & (np.abs(sx[:,1])<6)
    old_pts, old_nrm = sx[old_mask], sn[old_mask]
    loop_pts, loop_nrm = sx[loop_mask], sn[loop_mask]
    keep = np.arange(len(loop_pts))[::2]
    loop_pts, loop_nrm = loop_pts[keep], loop_nrm[keep]
    loop_created = sc[loop_mask][keep]
    print(f'old surfels={len(old_pts)} loop surfels(anchor)={len(loop_pts)}')
    grid = build_grid(old_pts, 0.4)
    match, dist = nearest(loop_pts, old_pts, grid, 0.4, 1.5)
    valid = (match >= 0) & (dist < 1.2)
    loop_pts, loop_nrm, loop_created = loop_pts[valid], loop_nrm[valid], loop_created[valid]
    target_pts, target_nrm = old_pts[match[valid]], old_nrm[match[valid]]
    print(f'anchored loop surfels={len(loop_pts)}')
    # node transforms: R_n, t_n  (correction, init identity)
    # initialise the node corrections from the pose-graph solution (this is
    # the field the previous "warp" experiment used); the map anchors then
    # refine it locally instead of starting from identity.
    pg = np.load('output/ka_diag/corridor_system_poses_vview.npz')
    cpos, cR = pg['pos'], pg['rot']
    Rn = [cR[k] @ R[k].T for k in range(n)]
    tn = [cpos[k] - Rn[k] @ pos[k] for k in range(n)]
    print('initialised from pose-graph corrections')
    # time bracketing for each anchored surfel
    br = np.clip(np.searchsorted(t, loop_created, side='right')-1, 0, n-2)
    br_next = br+1
    alpha = np.clip((loop_created - t[br]) / np.maximum(t[br_next]-t[br],1e-9), 0, 1)
    w_loop = (1.0/0.05, 1.0/0.10)
    w_smooth = (1.0/0.30, 1.0/0.30)
    w_map = 1.0/0.02

    def corrected(k):
        return Rn[k], tn[k]

    def residual():
        r = []
        # loop anchors
        for (i, j, Rm, tm) in loop_edges:
            Ri, ti = Rn[i], tn[i]
            Rj, tj = Rn[j], tn[j]
            Rp = (Ri @ R[i]).T @ (Rj @ R[j])
            tp = (Ri @ R[i]).T @ ((Rj @ pos[j] + tj) - (Ri @ pos[i] + ti))
            Rerr = Rm.T @ Rp
            terr = Rm.T @ (tp - tm)
            r.extend(so3_log(Rerr)*w_loop[0]); r.extend(terr*w_loop[1])
        # smoothness
        for k in range(n-1):
            Ra, ta = Rn[k], tn[k]
            Rb, tb = Rn[k+1], tn[k+1]
            Rrel = Ra.T @ Rb
            trel = Ra.T @ (tb-ta)
            r.extend(so3_log(Rrel)*w_smooth[0]); r.extend(trel*w_smooth[1])
        # map anchors
        for idx in range(len(loop_pts)):
            a = alpha[idx]
            ab = 1.0-a
            R1, t1 = Rn[br[idx]], tn[br[idx]]
            R2, t2 = Rn[br_next[idx]], tn[br_next[idx]]
            p = loop_pts[idx]
            p1 = R1 @ p + t1
            p2 = R2 @ p + t2
            pd = ab*p1 + a*p2
            r.append(w_map * float(target_nrm[idx] @ (pd - target_pts[idx])))
        return np.array(r)

    dim = n*6
    lam = 1e-3
    for iteration in range(10):
        r0 = residual()
        J = np.zeros((len(r0), dim))
        eps = 1e-6
        for k in range(n):
            for comp in range(6):
                wv = np.zeros(6); wv[comp] = 1.0
                dR = so3_exp(wv[:3]*eps); dt = wv[3:]*eps
                oldR, oldt = Rn[k], tn[k]
                Rn[k] = dR @ oldR; tn[k] = dR @ oldt + dt
                r1 = residual()
                Rn[k], tn[k] = oldR, oldt
                J[:, k*6+comp] = (r1-r0)/eps
        A = J.T @ J; b = -J.T @ r0
        accepted = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(A + lam*np.eye(dim), b)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            backup = [(Rn[k].copy(), tn[k].copy()) for k in range(n)]
            for k in range(n):
                dR = so3_exp(delta[k*6:k*6+3]); dt = delta[k*6+3:k*6+6]
                Rn[k] = dR @ Rn[k]; tn[k] = dR @ tn[k] + dt
            r1 = residual()
            if r1 @ r1 < r0 @ r0:
                lam = max(lam/3.0, 1e-6); accepted = True; break
            for k in range(n): Rn[k], tn[k] = backup[k]
            lam *= 10
        print(f'  iter {iteration}: rms={np.sqrt((r0**2).mean()):.5f} accepted={accepted}')
        if not accepted:
            break
    # warp surfels and measure layer residual
    def warp(points, created):
        br2 = np.clip(np.searchsorted(t, created, side='right')-1, 0, n-2)
        b2 = br2+1
        al = np.clip((created - t[br2]) / np.maximum(t[b2]-t[br2],1e-9), 0, 1)
        out = np.zeros_like(points)
        outn = np.zeros_like(points)
        for idx in range(len(points)):
            a = al[idx]
            R1, tt1 = Rn[br2[idx]], tn[br2[idx]]
            R2, tt2 = Rn[b2[idx]], tn[b2[idx]]
            out[idx] = (1-a)*(R1@points[idx]+tt1) + a*(R2@points[idx]+tt2)
        return out
    old_created = sc[old_mask]
    wold = warp(old_pts, old_created)
    wloop = warp(loop_pts, loop_created)
    wold_n = warp(old_pts, old_created)  # normals omitted; use unit normals below
    # normal rotation for old planes
    br2 = np.clip(np.searchsorted(t, old_created, side='right')-1, 0, n-2)
    b2 = br2+1
    al = np.clip((old_created - t[br2]) / np.maximum(t[b2]-t[br2],1e-9), 0, 1)
    wold_nrm = np.zeros_like(old_nrm)
    for idx in range(len(old_pts)):
        a = al[idx]
        R1, R2 = Rn[br2[idx]], Rn[b2[idx]]
        nrm = (1-a)*(R1@old_nrm[idx]) + a*(R2@old_nrm[idx])
        wold_nrm[idx] = nrm/np.linalg.norm(nrm)
    grid2 = build_grid(wold, 0.4)
    m, dd = nearest(wloop, wold, grid2, 0.4, 1.0)
    valid = (m >= 0) & (dd < 0.8)
    res = np.abs((wold_nrm[m[valid]] * (wloop[valid]-wold[m[valid]])).sum(axis=1))
    print(f'deformed map layer residual: mean={res.mean()*1000:.1f} mm '
          f'median={np.median(res)*1000:.1f} p90={np.percentile(res,90)*1000:.1f} mm (n={valid.sum()})')
    np.savez('output/ka_diag/deform_graph_full_nodes.npz', R=np.array(Rn), t=np.array(tn))

if __name__ == '__main__':
    main()
