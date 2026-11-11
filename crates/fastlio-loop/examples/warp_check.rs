//! Warp the full-rate trajectory with the corrected keyframe poses.
//!
//! Input trajectory CSV header: `timestamp_sec,px,py,pz,...` (as written by
//! `fastlio-replay`). Output: `timestamp_sec,x,y,z` CSV.
//!
//! Usage:
//!   cargo run --release -p fastlio-loop --example warp_check -- \
//!     <keyframes-dir> <poses.txt> <trajectory.csv> <out.csv>

use anyhow::{Context, Result};
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::warp::{corrections, load_corrected_poses, warp_trajectory};
use nalgebra::Vector3;
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let keyframe_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/spires/keyframes".to_string());
    let poses_file = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "output/spires/loop/rust_poses.txt".to_string());
    let trajectory_file = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "output/spires/trajectory.csv".to_string());
    let output = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "output/spires/loop/rust_warped.csv".to_string());

    let keyframes = load_keyframes(&keyframe_dir)?;
    let corrected = load_corrected_poses(&poses_file)?;
    let corrections = corrections(&keyframes, &corrected)?;
    let keyframe_times: Vec<f64> = keyframes
        .iter()
        .map(|keyframe| keyframe.timestamp_sec)
        .collect();

    let text = std::fs::read_to_string(&trajectory_file)
        .with_context(|| format!("failed to read {trajectory_file}"))?;
    let mut samples = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let timestamp: f64 = fields[0].parse().context("trajectory timestamp")?;
        let position = Vector3::new(
            fields[1].parse().context("trajectory px")?,
            fields[2].parse().context("trajectory py")?,
            fields[3].parse().context("trajectory pz")?,
        );
        samples.push((timestamp, position));
    }
    let warped = warp_trajectory(&samples, &keyframe_times, &corrections);

    let mut file = std::fs::File::create(&output)?;
    writeln!(file, "timestamp_sec,x,y,z")?;
    for ((timestamp, _), position) in samples.iter().zip(warped.iter()) {
        writeln!(
            file,
            "{timestamp:.9},{:.9},{:.9},{:.9}",
            position.x, position.y, position.z
        )?;
    }
    println!("warped {} samples -> {output}", samples.len());
    Ok(())
}
