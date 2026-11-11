//! Sub-map construction and voxel plane fitting for loop measurement.
//!
//! A loop measurement compares the local map around keyframe `i` (target
//! planes) with the sub-map around keyframe `j` (source points). The helpers
//! here mirror the verified Python pipeline: deterministic voxel down-sampling
//! and per-voxel plane fitting with a covariance eigen-decomposition.

use std::collections::HashMap;

use fastlio_types::PointXYZI;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::keyframe::Keyframe;

/// Parameters for sub-map construction.
#[derive(Debug, Clone, Copy)]
pub struct SubmapParams {
    /// Number of keyframes on each side of the centre keyframe.
    pub half_frames: usize,
    /// Voxel size used to down-sample the sub-map, meters.
    pub voxel_size_m: f64,
}

impl Default for SubmapParams {
    fn default() -> Self {
        Self {
            half_frames: 3,
            voxel_size_m: 0.25,
        }
    }
}

/// One fitted surfel-like plane.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub center: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub count: usize,
}

/// Deterministic voxel down-sampling: keep the first point of each voxel in the
/// original point order.
pub fn voxel_downsample(points: &[PointXYZI], voxel_size_m: f64) -> Vec<PointXYZI> {
    if points.is_empty() || voxel_size_m <= 0.0 {
        return points.to_vec();
    }
    let mut seen: HashMap<(i64, i64, i64), ()> = HashMap::new();
    let mut kept = Vec::new();
    for point in points {
        let key = (
            (point.x as f64 / voxel_size_m).floor() as i64,
            (point.y as f64 / voxel_size_m).floor() as i64,
            (point.z as f64 / voxel_size_m).floor() as i64,
        );
        if seen.insert(key, ()).is_none() {
            kept.push(point.clone());
        }
    }
    kept
}

/// Build the sub-map around `center` keyframe.
pub fn build_submap(
    keyframes: &[Keyframe],
    center: usize,
    params: &SubmapParams,
) -> Vec<PointXYZI> {
    let start = center.saturating_sub(params.half_frames);
    let end = (center + params.half_frames + 1).min(keyframes.len());
    let total: usize = keyframes[start..end].iter().map(|kf| kf.points.len()).sum();
    let mut points = Vec::with_capacity(total);
    for keyframe in &keyframes[start..end] {
        points.extend_from_slice(&keyframe.points);
    }
    voxel_downsample(&points, params.voxel_size_m)
}

/// Fit one plane per occupied voxel with at least `min_points` points.
pub fn fit_planes(points: &[PointXYZI], voxel_size_m: f64, min_points: usize) -> Vec<Plane> {
    let mut voxels: HashMap<(i64, i64, i64), Vec<Vector3<f64>>> = HashMap::new();
    for point in points {
        let key = (
            (point.x as f64 / voxel_size_m).floor() as i64,
            (point.y as f64 / voxel_size_m).floor() as i64,
            (point.z as f64 / voxel_size_m).floor() as i64,
        );
        voxels.entry(key).or_default().push(Vector3::new(
            point.x as f64,
            point.y as f64,
            point.z as f64,
        ));
    }
    let mut planes = Vec::new();
    for chunk in voxels.values() {
        if chunk.len() < min_points {
            continue;
        }
        let mut center = Vector3::zeros();
        for point in chunk {
            center += point;
        }
        center /= chunk.len() as f64;
        let mut covariance = Matrix3::zeros();
        for point in chunk {
            let delta = point - center;
            covariance += delta * delta.transpose();
        }
        covariance /= chunk.len() as f64;
        let eigen = SymmetricEigen::new(covariance);
        // SymmetricEigen does not sort eigenvalues; pick the smallest.
        let mut smallest = 0usize;
        for index in 1..3 {
            if eigen.eigenvalues[index] < eigen.eigenvalues[smallest] {
                smallest = index;
            }
        }
        planes.push(Plane {
            center,
            normal: Vector3::new(
                eigen.eigenvectors[(0, smallest)],
                eigen.eigenvectors[(1, smallest)],
                eigen.eigenvectors[(2, smallest)],
            ),
            count: chunk.len(),
        });
    }
    planes
}

#[cfg(test)]
mod test {
    use super::*;

    fn point(x: f32, y: f32, z: f32) -> PointXYZI {
        PointXYZI {
            x,
            y,
            z,
            intensity: 0.0,
        }
    }

    #[test]
    fn voxel_downsample_keeps_first_point_per_voxel() {
        let points = vec![
            point(0.1, 0.1, 0.1),
            point(0.2, 0.2, 0.2),
            point(1.5, 0.0, 0.0),
            point(0.05, 0.05, 0.05),
        ];
        let kept = voxel_downsample(&points, 1.0);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].x - 0.1).abs() < 1e-6);
        assert!((kept[1].x - 1.5).abs() < 1e-6);
    }

    #[test]
    fn fit_planes_recovers_horizontal_plane_normal() {
        let mut points = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                points.push(point(x as f32 * 0.2, y as f32 * 0.2, 1.0));
            }
        }
        let planes = fit_planes(&points, 1.0, 8);
        assert_eq!(planes.len(), 1);
        assert!(
            planes[0].normal.z.abs() > 0.99,
            "normal {:?}",
            planes[0].normal
        );
        assert!((planes[0].center.z - 1.0).abs() < 1e-6);
    }
}
