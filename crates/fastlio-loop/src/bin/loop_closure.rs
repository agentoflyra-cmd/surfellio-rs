//! End-to-end LiDAR-only loop closure for fastlio-rs keyframe dumps.
//!
//! Pipeline: keyframe dumps -> Scan Context candidates -> local ICP loop
//! measurements -> robust SE(3) pose graph -> full-rate trajectory warp ->
//! surfel map re-fusion.
//!
//! Usage:
//!   loop_closure --keyframes <dir> --trajectory <trajectory.csv> --out-dir <dir>
//!                [--voxel 0.25] [--min-points 10] [--no-refusion]
//!
//! Outputs (under `--out-dir`): `poses.txt` (corrected keyframes),
//! `measurements.txt`, `warped_trajectory.csv`, `refusion/map.pcd` and
//! `refusion/surfels.csv`.

use anyhow::{Context, Result, bail};
use fastlio_loop::candidates::{CandidateParams, propose_candidates};
use fastlio_loop::icp::{MeasureParams, measure_loop};
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::pose_graph::{
    Edge, EdgeKind, EdgeNoise, PoseGraphParams, optimize_graduated, pose_graph_cost,
};
use fastlio_loop::refusion::{refusion_planes, write_pcd, write_surfels_text};
use fastlio_loop::scan_context::{ScanContextParams, descriptors};
use fastlio_loop::warp::{corrections, warp_trajectory};
use fastlio_types::Pose3;
use nalgebra::Vector3;
use std::io::Write;
use std::path::PathBuf;

struct Args {
    keyframes: PathBuf,
    trajectory: Option<PathBuf>,
    out_dir: PathBuf,
    voxel: f64,
    min_points: usize,
    refusion: bool,
}

fn parse_args() -> Result<Args> {
    let mut keyframes = None;
    let mut trajectory = None;
    let mut out_dir = None;
    let mut voxel = 0.25;
    let mut min_points = 10;
    let mut refusion = true;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--keyframes" => {
                keyframes = Some(PathBuf::from(
                    arguments.next().context("--keyframes needs a path")?,
                ))
            }
            "--trajectory" => {
                trajectory = Some(PathBuf::from(
                    arguments.next().context("--trajectory needs a path")?,
                ))
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    arguments.next().context("--out-dir needs a path")?,
                ))
            }
            "--voxel" => voxel = arguments.next().context("--voxel needs a value")?.parse()?,
            "--min-points" => {
                min_points = arguments
                    .next()
                    .context("--min-points needs a value")?
                    .parse()?
            }
            "--no-refusion" => refusion = false,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(Args {
        keyframes: keyframes.context("missing --keyframes")?,
        trajectory,
        out_dir: out_dir.context("missing --out-dir")?,
        voxel,
        min_points,
        refusion,
    })
}

fn write_poses(path: &std::path::Path, poses: &[Pose3]) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for (index, pose) in poses.iter().enumerate() {
        let rotation = pose.rotation.to_rotation_matrix().matrix().clone_owned();
        writeln!(
            file,
            "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            index,
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
        )?;
    }
    Ok(())
}

fn write_measurements(
    path: &std::path::Path,
    measurements: &[fastlio_loop::icp::LoopMeasurement],
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for measurement in measurements {
        let rotation = measurement.rotation;
        writeln!(
            file,
            "{} {} {:.6} {:.6} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            measurement.i,
            measurement.j,
            measurement.fitness,
            measurement.correction_m,
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            measurement.translation.x,
            measurement.translation.y,
            measurement.translation.z,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out_dir)?;

    let keyframes = load_keyframes(&args.keyframes)?;
    let n = keyframes.len();
    let timestamps: Vec<f64> = keyframes
        .iter()
        .map(|keyframe| keyframe.timestamp_sec)
        .collect();
    let initial: Vec<Pose3> = keyframes
        .iter()
        .map(|keyframe| Pose3::new(keyframe.rotation, keyframe.position))
        .collect();

    let scan_contexts = descriptors(&keyframes, &ScanContextParams::default());
    let candidates = propose_candidates(&scan_contexts, &timestamps, &CandidateParams::default());
    println!("keyframes {n} scan-context candidates {}", candidates.len());

    let measure_params = MeasureParams::default();
    let mut measurements = Vec::new();
    for candidate in &candidates {
        if let Some(measurement) =
            measure_loop(&keyframes, candidate.i, candidate.j, &measure_params)
        {
            measurements.push(measurement);
        }
    }
    if measurements.is_empty() {
        bail!("no loop measurements accepted; nothing to optimise");
    }
    println!("accepted loop measurements {}", measurements.len());
    write_measurements(&args.out_dir.join("measurements.txt"), &measurements)?;

    let mut edges = Vec::new();
    edges.push(Edge::prior(0, &initial[0], 1e-6, 1e-6));
    for index in 0..n.saturating_sub(1) {
        let rotation_a = initial[index]
            .rotation
            .to_rotation_matrix()
            .matrix()
            .clone_owned();
        let rotation_b = initial[index + 1]
            .rotation
            .to_rotation_matrix()
            .matrix()
            .clone_owned();
        let z_rotation = rotation_a.transpose() * rotation_b;
        let z_translation =
            rotation_a.transpose() * (initial[index + 1].translation - initial[index].translation);
        edges.push(Edge::between(
            EdgeKind::Odometry,
            index,
            index + 1,
            z_rotation,
            z_translation,
            EdgeNoise::new(0.02, 0.05, 0.0),
        ));
    }
    for measurement in &measurements {
        edges.push(Edge::between(
            EdgeKind::Loop,
            measurement.i,
            measurement.j,
            measurement.rotation,
            measurement.translation,
            EdgeNoise::new(0.05, 0.15, 0.08),
        ));
    }
    let schedule = [1.0, 0.4, 0.15, 0.08];
    let started = std::time::Instant::now();
    let cost_before = pose_graph_cost(&initial, &edges);
    let corrected = optimize_graduated(&initial, &edges, &PoseGraphParams::default(), &schedule)?;
    println!(
        "pose graph: cost {:.3} -> {:.3} in {:.2} s",
        cost_before,
        pose_graph_cost(&corrected, &edges),
        started.elapsed().as_secs_f64()
    );
    let closure_before = (initial[n - 1].translation - initial[0].translation).norm();
    let closure_after = (corrected[n - 1].translation - corrected[0].translation).norm();
    println!("closure {closure_before:.3} -> {closure_after:.3} m");
    write_poses(&args.out_dir.join("poses.txt"), &corrected)?;

    let corrections = corrections(&keyframes, &corrected)?;
    if let Some(trajectory_file) = &args.trajectory {
        let text = std::fs::read_to_string(trajectory_file)
            .with_context(|| format!("failed to read {}", trajectory_file.display()))?;
        let mut samples = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            if line_index == 0 {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 4 {
                continue;
            }
            samples.push((
                fields[0].parse::<f64>().context("trajectory timestamp")?,
                Vector3::new(
                    fields[1].parse().context("trajectory px")?,
                    fields[2].parse().context("trajectory py")?,
                    fields[3].parse().context("trajectory pz")?,
                ),
            ));
        }
        let warped = warp_trajectory(&samples, &timestamps, &corrections);
        let mut file = std::fs::File::create(args.out_dir.join("warped_trajectory.csv"))?;
        writeln!(file, "timestamp_sec,x,y,z")?;
        for ((timestamp, _), position) in samples.iter().zip(warped.iter()) {
            writeln!(
                file,
                "{timestamp:.9},{:.9},{:.9},{:.9}",
                position.x, position.y, position.z
            )?;
        }
        println!("warped {} trajectory samples", samples.len());
    }

    if args.refusion {
        let planes = refusion_planes(&keyframes, &corrections, args.voxel, args.min_points);
        let refusion_dir = args.out_dir.join("refusion");
        std::fs::create_dir_all(&refusion_dir)?;
        write_pcd(refusion_dir.join("map.pcd"), &planes)?;
        write_surfels_text(refusion_dir.join("surfels.csv"), &planes)?;
        println!(
            "re-fused {} surfaces -> {}",
            planes.len(),
            refusion_dir.display()
        );
    }
    Ok(())
}
