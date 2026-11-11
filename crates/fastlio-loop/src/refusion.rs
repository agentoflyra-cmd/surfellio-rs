//! Map re-fusion: rebuild the surfel plane map from the loop-corrected
//! keyframe poses.
//!
//! For every keyframe the raw points are transformed with the correction
//! `dT_k = T'_k T_k^-1`, accumulated per 0.25 m voxel (running count, sum and
//! sum of outer products, so the raw points never all have to be retained) and
//! one plane is fitted per voxel from the covariance eigen-decomposition.
//! The module also reproduces the visit-separated layer residual used to verify
//! the map update: first-visit planes vs. loop-visit planes in the overlap ROE.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use fastlio_types::PointXYZI;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::keyframe::Keyframe;
use crate::warp::Correction;

/// One re-fused surfel plane.
#[derive(Debug, Clone)]
pub struct SurfelPlane {
    pub center: Vector3<f64>,
    pub normal: Vector3<f64>,
    /// Covariance eigenvalues in ascending order.
    pub eigenvalues: Vector3<f64>,
    pub count: u64,
}

#[derive(Default, Clone)]
struct VoxelStats {
    count: u64,
    sum: Vector3<f64>,
    outer: Matrix3<f64>,
}

impl VoxelStats {
    fn add(&mut self, point: &Vector3<f64>) {
        self.count += 1;
        self.sum += point;
        self.outer += point * point.transpose();
    }

    fn plane(&self) -> Option<SurfelPlane> {
        if self.count == 0 {
            return None;
        }
        let count = self.count as f64;
        let center = self.sum / count;
        let covariance = self.outer / count - center * center.transpose();
        let eigen = SymmetricEigen::new(covariance);
        let mut values = [0.0_f64; 3];
        let mut vectors = [Vector3::zeros(); 3];
        for index in 0..3 {
            values[index] = eigen.eigenvalues[index];
            vectors[index] = eigen.eigenvectors.column(index).into_owned();
        }
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        Some(SurfelPlane {
            center,
            normal: vectors[order[0]].normalize(),
            eigenvalues: Vector3::new(values[order[0]], values[order[1]], values[order[2]]),
            count: self.count,
        })
    }
}

type PlaneMap = HashMap<(i64, i64, i64), (Vector3<f64>, Vector3<f64>)>;

fn voxel_key(point: &Vector3<f64>, voxel_size_m: f64) -> (i64, i64, i64) {
    (
        (point.x / voxel_size_m).floor() as i64,
        (point.y / voxel_size_m).floor() as i64,
        (point.z / voxel_size_m).floor() as i64,
    )
}

fn corrected_point(point: &PointXYZI, correction: &Correction) -> Vector3<f64> {
    correction.d_rotation * Vector3::new(point.x as f64, point.y as f64, point.z as f64)
        + correction.d_translation
}

/// Voxel plane fitting over arbitrary points.
pub fn voxel_planes(
    points: impl Iterator<Item = Vector3<f64>>,
    voxel_size_m: f64,
    min_points: usize,
) -> Vec<SurfelPlane> {
    let mut voxels: HashMap<(i64, i64, i64), VoxelStats> = HashMap::new();
    for point in points {
        voxels
            .entry(voxel_key(&point, voxel_size_m))
            .or_default()
            .add(&point);
    }
    voxels
        .values()
        .filter(|stats| stats.count as usize >= min_points)
        .filter_map(VoxelStats::plane)
        .collect()
}

/// Rebuild the full map from all corrected keyframe points.
pub fn refusion_planes(
    keyframes: &[Keyframe],
    corrections: &[Correction],
    voxel_size_m: f64,
    min_points: usize,
) -> Vec<SurfelPlane> {
    let points = keyframes
        .iter()
        .zip(corrections)
        .flat_map(|(keyframe, correction)| {
            keyframe
                .points
                .iter()
                .map(move |point| corrected_point(point, correction))
        });
    voxel_planes(points, voxel_size_m, min_points)
}

/// Collect keyframe points inside the XY region of interest.
///
/// `visit` selects the keyframes to include; `apply_correction` chooses between
/// the drifted (raw) and re-fused (corrected) map.
pub fn visit_points(
    keyframes: &[Keyframe],
    corrections: &[Correction],
    visit: impl Fn(usize) -> bool,
    roi_m: f64,
    apply_correction: bool,
) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    for (index, (keyframe, correction)) in keyframes.iter().zip(corrections).enumerate() {
        if !visit(index) {
            continue;
        }
        for point in &keyframe.points {
            let world = if apply_correction {
                corrected_point(point, correction)
            } else {
                Vector3::new(point.x as f64, point.y as f64, point.z as f64)
            };
            if world.x.abs() < roi_m && world.y.abs() < roi_m {
                points.push(world);
            }
        }
    }
    points
}

/// Visit-separated layer residual statistics.
#[derive(Debug, Clone, Copy)]
pub struct LayerMetric {
    pub mean_m: f64,
    pub median_m: f64,
    pub p90_m: f64,
    pub voxels: usize,
}

/// Layer residual: for loop-visit planes whose voxel also has a first-visit
/// plane, the point-to-plane distance to the first-visit plane.
pub fn layer_metric(
    early: &[Vector3<f64>],
    late: &[Vector3<f64>],
    voxel_size_m: f64,
    min_points: usize,
) -> Option<LayerMetric> {
    let early_planes = voxel_planes(early.iter().copied(), voxel_size_m, min_points);
    let late_planes = voxel_planes(late.iter().copied(), voxel_size_m, min_points);
    let mut early_map: PlaneMap = HashMap::new();
    for plane in &early_planes {
        early_map.insert(
            voxel_key(&plane.center, voxel_size_m),
            (plane.center, plane.normal),
        );
    }
    let mut residuals = Vec::new();
    for plane in &late_planes {
        if let Some((center, normal)) = early_map.get(&voxel_key(&plane.center, voxel_size_m)) {
            residuals.push(normal.dot(&(plane.center - center)).abs());
        }
    }
    if residuals.is_empty() {
        return None;
    }
    residuals.sort_by(f64::total_cmp);
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let median = residuals[residuals.len() / 2];
    let p90 = residuals[(residuals.len() * 90 / 100).min(residuals.len() - 1)];
    Some(LayerMetric {
        mean_m: mean,
        median_m: median,
        p90_m: p90,
        voxels: residuals.len(),
    })
}

/// Write the surfel map in the same binary PCD layout as `fastlio-replay`
/// (`x y z intensity normal_x normal_y normal_z class_id`, all f32).
pub fn write_pcd(path: impl AsRef<Path>, planes: &[SurfelPlane]) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let header = format!(
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity normal_x normal_y normal_z class_id\nSIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\nWIDTH {0}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {0}\nDATA binary\n",
        planes.len()
    );
    file.write_all(header.as_bytes())?;
    for plane in planes {
        for value in [
            plane.center.x,
            plane.center.y,
            plane.center.z,
            0.0,
            plane.normal.x,
            plane.normal.y,
            plane.normal.z,
            0.0,
        ] {
            file.write_all(&(value as f32).to_le_bytes())?;
        }
    }
    Ok(())
}

/// Write the surfel map as a text table (`cx cy cz nx ny nz l0 l1 l2 count`).
pub fn write_surfels_text(path: impl AsRef<Path>, planes: &[SurfelPlane]) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writeln!(file, "cx,cy,cz,nx,ny,nz,l0,l1,l2,count")?;
    for plane in planes {
        writeln!(
            file,
            "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
            plane.center.x,
            plane.center.y,
            plane.center.z,
            plane.normal.x,
            plane.normal.y,
            plane.normal.z,
            plane.eigenvalues.x,
            plane.eigenvalues.y,
            plane.eigenvalues.z,
            plane.count
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::keyframe::Keyframe;
    use fastlio_types::Pose3;
    use nalgebra::UnitQuaternion;

    fn keyframe(t: f64, points: Vec<(f32, f32, f32)>) -> Keyframe {
        Keyframe {
            index: 0,
            timestamp_sec: t,
            position: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            points: points
                .into_iter()
                .map(|(x, y, z)| PointXYZI {
                    x,
                    y,
                    z,
                    intensity: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn refusion_planes_fit_a_floor() {
        let mut points = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                points.push((x as f32 * 0.2, y as f32 * 0.2, 1.0));
            }
        }
        let keyframes = vec![keyframe(0.0, points)];
        let corrected = vec![Pose3::new(UnitQuaternion::identity(), Vector3::zeros())];
        let corrections = crate::warp::corrections(&keyframes, &corrected).unwrap();
        let planes = refusion_planes(&keyframes, &corrections, 1.0, 8);
        assert_eq!(planes.len(), 1);
        assert!(planes[0].normal.z.abs() > 0.99);
        assert!((planes[0].center.z - 1.0).abs() < 1e-9);
    }

    #[test]
    fn layer_metric_is_zero_for_consistent_layers() {
        let mut points = Vec::new();
        for x in 0..20 {
            for y in 0..20 {
                points.push(Vector3::new(x as f64 * 0.1, y as f64 * 0.1, 1.0));
            }
        }
        let metric = layer_metric(&points, &points, 0.4, 8).expect("metric");
        assert!(metric.mean_m < 1e-9);
        assert!(metric.voxels > 0);
    }
}
