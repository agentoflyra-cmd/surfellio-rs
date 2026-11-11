//! Loop candidate proposal from Scan Context descriptors.
//!
//! For every keyframe the ring keys are compared against all later keyframes;
//! the best `top_per_keyframe` descriptor distances are kept, filtered by a
//! maximum descriptor distance and de-duplicated per (i-block, j-block) so that
//! nearby keyframes do not all produce the same candidate pair.

use crate::scan_context::ScanContext;

/// Candidate selection parameters (matching the offline Python pipeline).
#[derive(Debug, Clone, Copy)]
pub struct CandidateParams {
    /// Minimum time gap between the two keyframes, seconds.
    pub min_gap_sec: f64,
    /// How many candidates to keep per keyframe after the ring-key pre-filter.
    pub top_per_keyframe: usize,
    /// Maximum descriptor distance accepted.
    pub max_descriptor_distance: f64,
    /// Keyframe index block size used for de-duplication.
    pub pair_bin: usize,
}

impl Default for CandidateParams {
    fn default() -> Self {
        Self {
            min_gap_sec: 120.0,
            top_per_keyframe: 5,
            max_descriptor_distance: 8.0,
            pair_bin: 20,
        }
    }
}

/// One proposed loop candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopCandidate {
    /// Earlier keyframe index.
    pub i: usize,
    /// Later keyframe index.
    pub j: usize,
    /// Scan Context descriptor distance (lower is better).
    pub distance: f64,
}

/// Propose loop candidates from precomputed Scan Context descriptors.
///
/// `timestamps_sec` must have the same length as `descriptors`.
pub fn propose_candidates(
    descriptors: &[ScanContext],
    timestamps_sec: &[f64],
    params: &CandidateParams,
) -> Vec<LoopCandidate> {
    assert_eq!(descriptors.len(), timestamps_sec.len());
    let mut raw = Vec::new();
    for (i, i_descriptor) in descriptors.iter().enumerate() {
        let mut order: Vec<usize> = (0..descriptors.len()).collect();
        order.sort_by(|&a, &b| {
            let da = i_descriptor.ring_key_distance(&descriptors[a]);
            let db = i_descriptor.ring_key_distance(&descriptors[b]);
            da.total_cmp(&db).then(a.cmp(&b))
        });
        let mut accepted = 0usize;
        for &j in &order {
            if j <= i {
                continue;
            }
            if timestamps_sec[j] - timestamps_sec[i] < params.min_gap_sec {
                continue;
            }
            let Some(distance) = i_descriptor.distance(&descriptors[j]) else {
                continue;
            };
            raw.push(LoopCandidate { i, j, distance });
            accepted += 1;
            if accepted >= params.top_per_keyframe {
                break;
            }
        }
    }
    raw.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then(a.i.cmp(&b.i))
            .then(a.j.cmp(&b.j))
    });

    let mut seen = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    for candidate in raw {
        if candidate.distance > params.max_descriptor_distance {
            continue;
        }
        let key = (candidate.i / params.pair_bin, candidate.j / params.pair_bin);
        if !seen.insert(key) {
            continue;
        }
        candidates.push(candidate);
    }
    candidates
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::scan_context::keyframe_with_points;
    use crate::scan_context::{ScanContextParams, descriptors};

    fn ring_cloud(height: f32) -> Vec<(f32, f32, f32)> {
        let mut points = Vec::new();
        for ring in 0..10 {
            for sector in 0..36 {
                let azimuth = (sector as f32 + 0.5) / 36.0 * std::f32::consts::TAU;
                let radius = (ring as f32 + 0.5) * 3.0;
                points.push((radius * azimuth.cos(), radius * azimuth.sin(), height));
            }
        }
        points
    }

    #[test]
    fn proposes_pairs_for_similar_places_and_skips_close_times() {
        let params = ScanContextParams::default();
        // Scene A (height 1) at t=0, scene B (height 4) at t=10,
        // scene A again at t=200, then alternating A/B.
        let a = ring_cloud(1.0);
        let b = ring_cloud(4.0);
        let keyframes = vec![
            keyframe_with_points(&a),
            keyframe_with_points(&b),
            keyframe_with_points(&a),
            keyframe_with_points(&b),
            keyframe_with_points(&a),
            keyframe_with_points(&b),
        ];
        let timestamps = [0.0, 10.0, 200.0, 210.0, 400.0, 410.0];
        let sc = descriptors(&keyframes, &params);
        let candidates = propose_candidates(&sc, &timestamps, &CandidateParams::default());
        assert!(!candidates.is_empty(), "no candidates");
        assert!(
            candidates.iter().any(|c| c.i == 0 && c.j == 2),
            "candidates: {candidates:?}"
        );
        assert!(!candidates.iter().any(|c| c.i == 0 && c.j == 1));
    }
}
