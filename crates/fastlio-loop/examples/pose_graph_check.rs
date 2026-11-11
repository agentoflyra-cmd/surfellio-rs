//! Run the pose graph on Spires keyframes and dumped loop measurements.
//!
//! Output format (one line per keyframe):
//!   index tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22
//!
//! Usage:
//!   cargo run --release -p fastlio-loop --example pose_graph_check -- \
//!     <keyframes-dir> <measurements.txt> <out-poses.txt>

use anyhow::{Context, Result, bail};
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::pose_graph::{
    Edge, EdgeKind, EdgeNoise, PoseGraphParams, optimize_graduated, pose_graph_cost,
};
use fastlio_types::Pose3;
use nalgebra::{Matrix3, Vector3};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let keyframe_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/spires/keyframes".to_string());
    let measurements = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "output/spires/loop/rust_measurements.txt".to_string());
    let output = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "output/spires/loop/rust_poses.txt".to_string());
    let init_file = args.get(4).cloned();

    let keyframes = load_keyframes(&keyframe_dir)?;
    let n = keyframes.len();
    let mut initial: Vec<Pose3> = keyframes
        .iter()
        .map(|keyframe| Pose3::new(keyframe.rotation, keyframe.position))
        .collect();
    if let Some(path) = &init_file {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("failed to read init {path}"))?;
        let mut overwritten = 0usize;
        for line in text.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 13 {
                continue;
            }
            let index: usize = fields[0].parse().context("init index")?;
            let translation = Vector3::new(
                fields[1].parse().context("init tx")?,
                fields[2].parse().context("init ty")?,
                fields[3].parse().context("init tz")?,
            );
            let mut rotation = Matrix3::<f64>::zeros();
            for row in 0..3 {
                for column in 0..3 {
                    rotation[(row, column)] = fields[4 + row * 3 + column]
                        .parse()
                        .context("init rotation")?;
                }
            }
            initial[index] = Pose3::new(
                nalgebra::UnitQuaternion::from_rotation_matrix(
                    &nalgebra::Rotation3::from_matrix_unchecked(rotation),
                ),
                translation,
            );
            overwritten += 1;
        }
        println!("init poses from {path} ({overwritten})");
    }

    let mut edges = Vec::new();
    edges.push(Edge::prior(0, &initial[0], 1e-6, 1e-6));
    for index in 0..n.saturating_sub(1) {
        let a = &initial[index];
        let b = &initial[index + 1];
        let rotation_a = a.rotation.to_rotation_matrix().matrix().clone_owned();
        let rotation_b = b.rotation.to_rotation_matrix().matrix().clone_owned();
        let z_rotation = rotation_a.transpose() * rotation_b;
        let z_translation = rotation_a.transpose() * (b.translation - a.translation);
        edges.push(Edge::between(
            EdgeKind::Odometry,
            index,
            index + 1,
            z_rotation,
            z_translation,
            EdgeNoise::new(0.02, 0.05, 0.0),
        ));
    }

    let text = std::fs::read_to_string(&measurements)
        .with_context(|| format!("failed to read {measurements}"))?;
    let mut loop_count = 0;
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 16 {
            continue;
        }
        let i: usize = fields[0].parse().context("loop i")?;
        let j: usize = fields[1].parse().context("loop j")?;
        let mut rotation = Matrix3::<f64>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                rotation[(row, column)] = fields[4 + row * 3 + column]
                    .parse()
                    .context("loop rotation")?;
            }
        }
        let translation = Vector3::new(
            fields[13].parse().context("loop tx")?,
            fields[14].parse().context("loop ty")?,
            fields[15].parse().context("loop tz")?,
        );
        edges.push(Edge::between(
            EdgeKind::Loop,
            i,
            j,
            rotation,
            translation,
            EdgeNoise::new(0.05, 0.15, 0.08),
        ));
        loop_count += 1;
    }
    if loop_count == 0 {
        bail!("no loop measurements parsed from {measurements}");
    }
    println!("keyframes {n} loop edges {loop_count}");

    let started = std::time::Instant::now();
    let cost_before = pose_graph_cost(&initial, &edges);
    let schedule = [1.0, 0.4, 0.15, 0.08];
    let optimized = optimize_graduated(&initial, &edges, &PoseGraphParams::default(), &schedule)?;
    let cost_after = pose_graph_cost(&optimized, &edges);
    println!(
        "cost {cost_before:.3} -> {cost_after:.3} (optimised in {:.2} s)",
        started.elapsed().as_secs_f64()
    );

    let closure_before = (initial[n - 1].translation - initial[0].translation).norm();
    let closure_after = (optimized[n - 1].translation - optimized[0].translation).norm();
    println!("closure {closure_before:.3} -> {closure_after:.3} m");

    let mut file = std::fs::File::create(&output)?;
    for (index, pose) in optimized.iter().enumerate() {
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
    println!("wrote {output}");
    Ok(())
}
