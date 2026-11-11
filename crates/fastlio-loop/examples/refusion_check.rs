//! Re-fuse the surfel map from the corrected keyframe poses.
//!
//! Writes `map.pcd` (same binary layout as fastlio-replay) and `surfels.csv`
//! (center, normal, eigenvalues, count) and prints map statistics plus the
//! visit-separated layer residual before/after re-fusion.
//!
//! Usage:
//!   cargo run --release -p fastlio-loop --example refusion_check -- \
//!     <keyframes-dir> <poses.txt> <out-dir>

use anyhow::Result;
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::refusion::{
    layer_metric, refusion_planes, visit_points, write_pcd, write_surfels_text,
};
use fastlio_loop::warp::{corrections, load_corrected_poses};

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
    let out_dir = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "output/spires/refusion_rust".to_string());

    let keyframes = load_keyframes(&keyframe_dir)?;
    let corrected = load_corrected_poses(&poses_file)?;
    let corrections = corrections(&keyframes, &corrected)?;
    std::fs::create_dir_all(&out_dir)?;

    let started = std::time::Instant::now();
    let planes = refusion_planes(&keyframes, &corrections, 0.25, 10);
    println!(
        "re-fused surfaces: {} (0.25 m voxel) in {:.2} s",
        planes.len(),
        started.elapsed().as_secs_f64()
    );
    let mean_count =
        planes.iter().map(|plane| plane.count as f64).sum::<f64>() / planes.len().max(1) as f64;
    let planar = planes
        .iter()
        .filter(|plane| plane.eigenvalues.x / plane.eigenvalues.z.max(1e-12) < 0.02)
        .count() as f64
        / planes.len().max(1) as f64;
    println!("mean count/voxel: {mean_count:.1} planar (l0/l2<0.02): {planar:.2}");
    write_pcd(format!("{out_dir}/map.pcd"), &planes)?;
    write_surfels_text(format!("{out_dir}/surfels.csv"), &planes)?;
    println!("wrote {out_dir}/map.pcd and {out_dir}/surfels.csv");

    let times: Vec<f64> = keyframes
        .iter()
        .map(|keyframe| keyframe.timestamp_sec)
        .collect();
    let t0 = times[0];
    let early = |index: usize| times[index] - t0 < 60.0;
    let late = |index: usize| times[index] - t0 > 650.0;
    for (label, apply_correction) in [("drifted", false), ("re-fused", true)] {
        let early_points = visit_points(&keyframes, &corrections, early, 20.0, apply_correction);
        let late_points = visit_points(&keyframes, &corrections, late, 20.0, apply_correction);
        match layer_metric(&early_points, &late_points, 0.4, 10) {
            Some(metric) => println!(
                "{label}: layer mean={:.1} mm median={:.1} mm p90={:.1} mm (voxels={})",
                metric.mean_m * 1000.0,
                metric.median_m * 1000.0,
                metric.p90_m * 1000.0,
                metric.voxels
            ),
            None => println!("{label}: no overlapping voxels"),
        }
    }
    Ok(())
}
