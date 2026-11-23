#!/usr/bin/env python3
"""Pilot: register corridor revisit keyframes to a first-visit-only map.

This is an offline point-to-plane registration probe, not an online localizer.
Requires the existing output/venv environment with Open3D, NumPy and PyYAML.
"""

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parent.parent
FRAME_DIR = ROOT / "output/ka_diag/keyframes_corridor"
REFERENCE = ROOT / "output/ka_diag/corridor_system_poses_vview.npz"
CONFIG = ROOT / "config/corridor_velodyne.yaml"


def quat_to_R(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def load_keyframes(directory):
    frames = []
    for path in sorted(directory.glob("*.bin")):
        with path.open("rb") as source:
            source.read(8)
            timestamp = struct.unpack("<d", source.read(8))[0]
            position = np.frombuffer(source.read(24), dtype="<f8").copy()
            quat = np.frombuffer(source.read(32), dtype="<f8").copy()
            count = struct.unpack("<I", source.read(4))[0]
            points = np.frombuffer(source.read(count * 12), dtype="<f4").reshape(count, 3).astype(float)
        frames.append({"t": timestamp, "pos": position, "quat": quat, "pts": points})
    return frames


def render_range(points_sensor):
    radius = np.linalg.norm(points_sensor, axis=1)
    azimuth = np.arctan2(points_sensor[:, 1], points_sensor[:, 0])
    elevation = np.arctan2(points_sensor[:, 2], np.hypot(points_sensor[:, 0], points_sensor[:, 1]))
    az_bin = np.clip(((azimuth + np.pi) / (2 * np.pi) * 1440).astype(int), 0, 1439)
    el_bin = np.clip(((elevation - np.radians(-25)) / np.radians(40) * 32).astype(int), 0, 31)
    image = np.full(1440 * 32, np.inf)
    np.minimum.at(image, el_bin * 1440 + az_bin, radius)
    return image.reshape(32, 1440)


def range_metrics(map_world, scan_world, rotation_align, translation_align, raw_pose,
                  rotation_il, translation_il):
    rotation_raw, position_raw = raw_pose
    aligned_pose = pose(rotation_align, translation_align) @ pose(rotation_raw, position_raw)
    map_imu = (aligned_pose[:3, :3].T @ (map_world - aligned_pose[:3, 3]).T).T
    scan_imu = (rotation_raw.T @ (scan_world - position_raw).T).T
    map_sensor = (rotation_il.T @ (map_imu - translation_il).T).T
    scan_sensor = (rotation_il.T @ (scan_imu - translation_il).T).T
    virtual = render_range(map_sensor)
    actual = render_range(scan_sensor)
    both = np.isfinite(virtual) & np.isfinite(actual)
    if not both.any():
        return 0.0, 1.0
    difference = virtual[both] - actual[both]
    return float((np.abs(difference) < 0.3).mean()), float((difference < -0.5).mean())


def pose(rotation, position):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def point_cloud(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def changed_scan(points_w, rotation_wi, position_wi, rotation_il, translation_il,
                 frame_number, seed, scenario):
    if scenario == "clean":
        return points_w.copy(), 0
    rng = np.random.default_rng(seed + frame_number * 1009)
    points_i = (rotation_wi.T @ (points_w - position_wi).T).T
    points_l = (rotation_il.T @ (points_i - translation_il).T).T
    ranges = np.linalg.norm(points_l, axis=1)
    rays = points_l / np.maximum(ranges[:, None], 1e-9)
    if scenario == "range_dropout":
        keep = rng.random(len(points_l)) >= 0.20
        new_ranges = np.maximum(0.1, ranges[keep] + rng.normal(0.0, 0.03, keep.sum()))
        changed_l = rays[keep] * new_ranges[:, None]
        modified = int(len(points_l))  # retained ranges changed; missing rays are counted too
    elif scenario == "moving_occluder":
        azimuth = np.arctan2(points_l[:, 1], points_l[:, 0])
        elevation = np.arcsin(np.clip(rays[:, 2], -1.0, 1.0))
        center = float(np.median(azimuth)) + (frame_number - 270) * 0.06
        wrapped = np.arctan2(np.sin(azimuth - center), np.cos(azimuth - center))
        mask = (np.abs(wrapped) < 0.16) & (np.abs(elevation) < 0.30) & (ranges > 3.0)
        changed_l = points_l.copy()
        # The foreground replaces the background returns on the same rays.
        changed_l[mask] = rays[mask] * (2.5 + rng.normal(0.0, 0.03, mask.sum()))[:, None]
        modified = int(mask.sum())
    else:
        raise ValueError(scenario)
    changed_i = (rotation_il @ changed_l.T).T + translation_il
    changed_w = (rotation_wi @ changed_i.T).T + position_wi
    return changed_w, modified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/frozen_map_probe.json")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--start-frame", type=int, default=264)
    parser.add_argument("--end-frame", type=int, default=275, help="exclusive")
    parser.add_argument("--init-x-offset-m", type=float, default=0.0)
    parser.add_argument("--init-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    frames = load_keyframes(FRAME_DIR)
    ref = np.load(REFERENCE)
    mapping = yaml.safe_load(CONFIG.read_text())["mapping"]
    rotation_il = np.asarray(mapping["extrinsic_R"], dtype=float).reshape(3, 3)
    translation_il = np.asarray(mapping["extrinsic_T"], dtype=float)

    first_visit = frames[:18]
    first_visit_end = first_visit[-1]["t"]
    if not (18 <= args.start_frame < args.end_frame <= len(frames)):
        parser.error("frame range must be nonempty and after the first-visit map")
    if not first_visit_end < frames[args.start_frame]["t"]:
        parser.error("query frames must occur after the first-visit map")
    target_raw = np.concatenate([frame["pts"] for frame in first_visit])
    target = point_cloud(target_raw).voxel_down_sample(0.3)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    map_world = np.asarray(target.points)
    map_digest = hashlib.sha256(
        map_world.tobytes() + np.asarray(target.normals).tobytes()
    ).hexdigest()

    # The corrected trajectory supplies a fixed near-pose initial guess only.
    # It is not independent ground truth for this experiment.
    anchor = 0
    raw_anchor = pose(quat_to_R(frames[anchor]["quat"]), frames[anchor]["pos"])
    corrected_anchor = pose(ref["rot"][anchor], ref["pos"][anchor])
    rows = []
    for frame_number in range(args.start_frame, args.end_frame):
        frame = frames[frame_number]
        rotation_wi = quat_to_R(frame["quat"])
        raw_pose = pose(rotation_wi, frame["pos"])
        corrected_pose = pose(ref["rot"][frame_number], ref["pos"][frame_number])
        initial = raw_anchor @ np.linalg.inv(corrected_anchor) @ corrected_pose @ np.linalg.inv(raw_pose)
        yaw = np.radians(args.init_yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        initial = pose(yaw_rotation, np.array([args.init_x_offset_m, 0.0, 0.0])) @ initial
        for scenario in ("clean", "moving_occluder", "range_dropout"):
            sample_start = time.perf_counter()
            changed, modified = changed_scan(
                frame["pts"], rotation_wi, frame["pos"], rotation_il, translation_il,
                frame_number, args.seed, scenario,
            )
            source = point_cloud(changed)
            start = time.perf_counter()
            result = o3d.pipelines.registration.registration_icp(
                source, target, 1.2, initial,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.HuberLoss(0.05)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            solve_ms = (time.perf_counter() - start) * 1000.0
            transform = result.transformation
            agreement, violation = range_metrics(
                map_world, changed, transform[:3, :3], transform[:3, 3],
                (rotation_wi, frame["pos"]), rotation_il, translation_il,
            )
            total_ms = (time.perf_counter() - sample_start) * 1000.0
            rows.append({
                "frame": frame_number, "timestamp_sec": frame["t"],
                "scenario": scenario, "input_points": len(frame["pts"]),
                "points": len(changed), "modified_points": modified,
                "modified_fraction": modified / len(frame["pts"]),
                "dropped_points": len(frame["pts"]) - len(changed),
                "fitness": result.fitness, "inlier_rmse_m": result.inlier_rmse,
                "range_agreement": agreement, "free_space_violation": violation,
                "solve_ms": solve_ms, "sample_total_ms": total_ms,
                "transform": transform.tolist(),
            })
            print(f"{frame_number} {scenario}: fitness={result.fitness:.3f} "
                  f"agree={agreement:.3f} violation={violation:.3f} "
                  f"solve={solve_ms:.1f} ms modified={modified}", flush=True)

    summary = {}
    for scenario in ("clean", "moving_occluder", "range_dropout"):
        selected = [row for row in rows if row["scenario"] == scenario]
        durations = np.asarray([row["solve_ms"] for row in selected])
        totals = np.asarray([row["sample_total_ms"] for row in selected])
        differences = []
        if scenario != "clean":
            clean_by_frame = {row["frame"]: row for row in rows if row["scenario"] == "clean"}
            for row in selected:
                clean = np.asarray(clean_by_frame[row["frame"]]["transform"])
                changed = np.asarray(row["transform"])
                delta = np.linalg.inv(clean) @ changed
                angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
                differences.append((float(np.linalg.norm(delta[:3, 3])), float(np.degrees(angle))))
        summary[scenario] = {
            "frames": len(selected),
            "solve_p95_ms": float(np.percentile(durations, 95)),
            "sample_total_p95_ms": float(np.percentile(totals, 95)),
            "fitness_min": min(row["fitness"] for row in selected),
            "range_agreement_min": min(row["range_agreement"] for row in selected),
            "free_space_violation_max": max(row["free_space_violation"] for row in selected),
            "paired_translation_p95_m": float(np.percentile([v[0] for v in differences], 95)) if differences else None,
            "paired_rotation_p95_deg": float(np.percentile([v[1] for v in differences], 95)) if differences else None,
        }

    output = {
        "method": "Open3D point-to-plane ICP with Huber loss; virtual-view check",
        "scope": "offline per-keyframe local registration, not continuous localization",
        "seed": args.seed, "start_frame": args.start_frame, "end_frame": args.end_frame,
        "init_x_offset_m": args.init_x_offset_m, "init_yaw_deg": args.init_yaw_deg,
        "map_first_visit_end_sec": first_visit_end,
        "map_points": len(map_world), "map_sha256": map_digest,
        "reference_role": "near-pose initialization only, not independent truth",
        "summary": summary, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
