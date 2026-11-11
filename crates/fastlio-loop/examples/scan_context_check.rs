//! Dump Scan Context loop candidates for a keyframe directory.
//!
//! Output format (sorted by descriptor distance):
//!   i j distance
//!
//! Usage:
//!   cargo run -p fastlio-loop --example scan_context_check -- <keyframes-dir>

use anyhow::Result;
use fastlio_loop::candidates::{CandidateParams, propose_candidates};
use fastlio_loop::keyframe::load_keyframes;
use fastlio_loop::scan_context::{ScanContextParams, descriptors};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let directory = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/spires/keyframes".to_string());
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
    for candidate in &candidates {
        println!("{} {} {:.6}", candidate.i, candidate.j, candidate.distance);
    }
    Ok(())
}
