//! Scan Context descriptor used for LiDAR place recognition.
//!
//! The descriptor is a polar grid (rings x sectors): each cell stores the
//! maximum local-frame height of the points falling into it, and empty cells
//! keep a large negative sentinel. Two descriptors are compared by the mean
//! absolute difference over cells that are valid in both. This mirrors the
//! verified Python offline pipeline exactly.

use fastlio_types::PointXYZI;

use crate::keyframe::Keyframe;

/// Sentinel for empty Scan Context cells (kept identical to the Python
/// implementation so that ring keys and distances match).
pub const EMPTY_CELL: f64 = -1.0e9;
/// Cells with a value below this threshold are treated as invalid.
pub const INVALID_CELL_THRESHOLD: f64 = -1.0e8;
/// Minimum number of jointly valid cells for a descriptor distance.
pub const MIN_VALID_CELLS: usize = 50;

/// Scan Context parameters (tuned on Oxford Spires in the offline pipeline).
#[derive(Debug, Clone, Copy)]
pub struct ScanContextParams {
    pub rings: usize,
    pub sectors: usize,
    pub max_radius_m: f64,
}

impl Default for ScanContextParams {
    fn default() -> Self {
        Self {
            rings: 20,
            sectors: 60,
            max_radius_m: 40.0,
        }
    }
}

/// A single Scan Context descriptor.
#[derive(Debug, Clone)]
pub struct ScanContext {
    pub params: ScanContextParams,
    /// `rings * sectors` cells, row-major by ring then sector.
    pub values: Vec<f64>,
}

impl ScanContext {
    /// Build the descriptor of a keyframe by transforming its world points into
    /// the keyframe body frame.
    pub fn from_keyframe(keyframe: &Keyframe, params: &ScanContextParams) -> Self {
        let mut values = vec![EMPTY_CELL; params.rings * params.sectors];
        for point in &keyframe.points {
            let world = keyframe.point_world(point);
            let local = keyframe.transform_point_world_to_body(&world);
            let radius = (local.x * local.x + local.y * local.y).sqrt();
            let azimuth = local.y.atan2(local.x);
            let ring = ((radius / params.max_radius_m * params.rings as f64) as usize)
                .min(params.rings - 1);
            let sector = (((azimuth + std::f64::consts::PI) / (2.0 * std::f64::consts::PI))
                * params.sectors as f64) as usize;
            let sector = sector.min(params.sectors - 1);
            let index = ring * params.sectors + sector;
            if local.z > values[index] {
                values[index] = local.z;
            }
        }
        Self {
            params: *params,
            values,
        }
    }

    pub fn get(&self, ring: usize, sector: usize) -> f64 {
        self.values[ring * self.params.sectors + sector]
    }

    /// Mean value per ring (the "ring key" used for fast pre-filtering).
    pub fn ring_key(&self) -> Vec<f64> {
        (0..self.params.rings)
            .map(|ring| {
                let sum: f64 = (0..self.params.sectors)
                    .map(|sector| self.get(ring, sector))
                    .sum();
                sum / self.params.sectors as f64
            })
            .collect()
    }

    /// Mean absolute cell difference over cells valid in both descriptors.
    pub fn distance(&self, other: &Self) -> Option<f64> {
        debug_assert_eq!(self.values.len(), other.values.len());
        if self.params.rings != other.params.rings || self.params.sectors != other.params.sectors {
            return None;
        }
        let mut valid = 0usize;
        let mut sum = 0.0;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            if *a > INVALID_CELL_THRESHOLD && *b > INVALID_CELL_THRESHOLD {
                sum += (a - b).abs();
                valid += 1;
            }
        }
        if valid < MIN_VALID_CELLS {
            None
        } else {
            Some(sum / valid as f64)
        }
    }

    /// L2 distance between ring keys (used for candidate pre-selection).
    pub fn ring_key_distance(&self, other: &Self) -> f64 {
        self.ring_key()
            .iter()
            .zip(other.ring_key().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// Convenience helper: build descriptors for a whole keyframe sequence.
pub fn descriptors(keyframes: &[Keyframe], params: &ScanContextParams) -> Vec<ScanContext> {
    keyframes
        .iter()
        .map(|keyframe| ScanContext::from_keyframe(keyframe, params))
        .collect()
}

/// Build a synthetic keyframe from world-frame points for tests and examples.
#[allow(dead_code)]
pub(crate) fn keyframe_with_points(points: &[(f32, f32, f32)]) -> Keyframe {
    use nalgebra::{UnitQuaternion, Vector3};
    Keyframe {
        index: 0,
        timestamp_sec: 0.0,
        position: Vector3::zeros(),
        rotation: UnitQuaternion::identity(),
        points: points
            .iter()
            .map(|(x, y, z)| PointXYZI {
                x: *x,
                y: *y,
                z: *z,
                intensity: 0.0,
            })
            .collect(),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Dense synthetic cloud covering many rings/sectors so that descriptors
    /// share at least `MIN_VALID_CELLS` cells.
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

    fn keyframe(points: &[(f32, f32, f32)]) -> Keyframe {
        keyframe_with_points(points)
    }

    #[test]
    fn identical_clouds_have_zero_distance() {
        let params = ScanContextParams::default();
        let points = ring_cloud(1.0);
        let a = ScanContext::from_keyframe(&keyframe(&points), &params);
        let b = ScanContext::from_keyframe(&keyframe(&points), &params);
        assert_eq!(a.distance(&b), Some(0.0));
    }

    #[test]
    fn shifted_cloud_increases_distance() {
        let params = ScanContextParams::default();
        let a = ScanContext::from_keyframe(&keyframe(&ring_cloud(1.0)), &params);
        let b = ScanContext::from_keyframe(&keyframe(&ring_cloud(4.0)), &params);
        let distance = a.distance(&b).unwrap();
        assert!((distance - 3.0).abs() < 1e-9, "distance {distance}");
    }

    #[test]
    fn descriptor_maps_polar_cells() {
        let params = ScanContextParams::default();
        // +x direction, radius ~5 m, height 2 m -> ring = 5/40*20 = 2,
        // sector = (0+pi)/(2pi)*60 = 30.
        let sc = ScanContext::from_keyframe(&keyframe(&[(5.0, 0.0, 2.0)]), &params);
        assert!((sc.get(2, 30) - 2.0).abs() < 1e-12);
        // +y direction -> sector 45.
        let sc = ScanContext::from_keyframe(&keyframe(&[(0.0, 5.0, 1.0)]), &params);
        assert!((sc.get(2, 45) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ring_key_matches_descriptor_mean() {
        let params = ScanContextParams::default();
        let sc = ScanContext::from_keyframe(&keyframe(&ring_cloud(2.0)), &params);
        let keys = sc.ring_key();
        let expected = sc.values.iter().sum::<f64>() / (params.rings * params.sectors) as f64;
        assert!((keys.iter().sum::<f64>() / params.rings as f64 - expected).abs() < 1e-9);
    }
}
