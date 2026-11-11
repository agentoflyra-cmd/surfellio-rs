//! Measure loops for Scan Context candidates and dump the measurements.
//!
//! Output format (one line per accepted measurement):
//!   i j fitness correction r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
//!
//! Usage:
//!   cargo run --release -p fastlio-loop --example loop_measure_check -- <keyframes-dir> <out.txt>

use anyhow::Result;
use fastlio_loop::candidates::{CandidateParams, propose_candidates};
use fastlio_loop::icp::{MeasureParams, measure_loop};
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::scan_context::{ScanContextParams, descriptors};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let directory = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/spires/keyframes".to_string());
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "output/spires/loop/rust_measurements.txt".to_string());

    let keyframes = load_keyframes(&directory)?;
    let timestamps: Vec<f64> = keyframes
        .iter()
        .map(|keyframe| keyframe.timestamp_sec)
        .collect();
    let scan_contexts = descriptors(&keyframes, &ScanContextParams::default());
    let candidates = propose_candidates(&scan_contexts, &timestamps, &CandidateParams::default());
    println!(
        "keyframes {} candidates {}",
        keyframes.len(),
        candidates.len()
    );

    let params = MeasureParams::default();
    let mut accepted = 0usize;
    let mut file = std::fs::File::create(&output)?;
    for candidate in &candidates {
        let Some(measurement) = measure_loop(&keyframes, candidate.i, candidate.j, &params) else {
            continue;
        };
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
        accepted += 1;
        println!(
            "loop ({},{}) fitness={:.3} correction={:.3}",
            measurement.i, measurement.j, measurement.fitness, measurement.correction_m
        );
    }
    println!("accepted {}", accepted);
    Ok(())
}
