//! Offline LiDAR-only loop closure for fastlio-rs.
//!
//! This crate ports the verified Python offline pipeline to Rust in stages:
//!
//! 1. keyframe loading + Scan Context place recognition (this stage);
//! 2. local point-to-plane ICP on sub-maps;
//! 3. SE(3) pose graph with a robust kernel;
//! 4. trajectory correction and map re-fusion.
//!
//! The input keyframes are the binary dumps produced by
//! `fastlio-replay --keyframe-dir`: world-frame points with the estimated
//! world pose at the keyframe timestamp.

pub mod candidates;
pub mod keyframe;
pub mod scan_context;

pub use candidates::{CandidateParams, LoopCandidate, propose_candidates};
pub use keyframe::{Keyframe, load_keyframes};
pub use scan_context::{ScanContext, ScanContextParams};
