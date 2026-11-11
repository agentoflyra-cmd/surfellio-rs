//! Local point-to-plane ICP used to measure a loop between two sub-maps.
//!
//! This mirrors the verified Python pipeline: the source sub-map (world frame)
//! is aligned to the target plane centres, several small initial offsets along
//! the drift direction are tried, and the best result (by inlier ratio) is
//! kept. The point-to-plane Jacobian is `[p x n, n]` with a Huber weight on the
//! residual; the update is `R <- exp(dtheta) R`, `t <- exp(dtheta) t + dt`.

use std::collections::HashMap;

use fastlio_types::PointXYZI;
use nalgebra::{Matrix3, Matrix6, SymmetricEigen, Vector3, Vector6};

use crate::keyframe::Keyframe;
use crate::submap::{SubmapParams, build_submap};

/// ICP parameters matching the offline pipeline.
#[derive(Debug, Clone, Copy)]
pub struct IcpParams {
    /// Multi-scale voxel sizes.
    pub scales: [f64; 2],
    /// Maximum correspondence distance, meters.
    pub max_correspondence_distance: f64,
    /// Number of iterations per scale.
    pub iterations: usize,
    /// Huber threshold on the point-to-plane residual.
    pub huber_threshold: f64,
    /// Maximum number of source points kept per scale (deterministic stride).
    pub max_source_points: usize,
    /// Minimum number of correspondences required.
    pub min_correspondences: usize,
    /// Final inlier ratio distance used for the fitness score.
    pub fitness_distance: f64,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            scales: [0.6, 0.3],
            max_correspondence_distance: 1.2,
            iterations: 20,
            huber_threshold: 0.05,
            max_source_points: 1500,
            min_correspondences: 60,
            fitness_distance: 1.2,
        }
    }
}

/// Result of one ICP alignment.
#[derive(Debug, Clone)]
pub struct IcpResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// Fraction of source points with a target correspondence within
    /// `fitness_distance` at the final alignment.
    pub fitness: f64,
    /// RMS point-to-plane residual at the final alignment.
    pub rms: f64,
}

struct Grid {
    cell: f64,
    buckets: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(points: &[Vector3<f64>], cell: f64) -> Self {
        let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (index, point) in points.iter().enumerate() {
            let key = Self::key(point, cell);
            buckets.entry(key).or_default().push(index);
        }
        Self { cell, buckets }
    }

    fn key(point: &Vector3<f64>, cell: f64) -> (i64, i64, i64) {
        (
            (point.x / cell).floor() as i64,
            (point.y / cell).floor() as i64,
            (point.z / cell).floor() as i64,
        )
    }

    fn nearest(
        &self,
        point: &Vector3<f64>,
        target: &[Vector3<f64>],
        max_distance: f64,
    ) -> Option<usize> {
        let base = Self::key(point, self.cell);
        let range = (max_distance / self.cell).ceil() as i64;
        let mut best = max_distance;
        let mut best_index = None;
        for dx in -range..=range {
            for dy in -range..=range {
                for dz in -range..=range {
                    if let Some(indices) =
                        self.buckets.get(&(base.0 + dx, base.1 + dy, base.2 + dz))
                    {
                        for &index in indices {
                            let distance = (target[index] - point).norm();
                            if distance < best {
                                best = distance;
                                best_index = Some(index);
                            }
                        }
                    }
                }
            }
        }
        best_index
    }
}

/// One target point with its local surface normal.
#[derive(Debug, Clone, Copy)]
pub struct TargetPoint {
    pub position: Vector3<f64>,
    pub normal: Vector3<f64>,
}

/// Build the target point cloud with per-point normals estimated by local PCA
/// (the analogue of Open3D's `estimate_normals` with a search radius).
pub fn build_target_points(
    target: &[PointXYZI],
    radius_m: f64,
    min_neighbors: usize,
) -> Vec<TargetPoint> {
    if target.is_empty() {
        return Vec::new();
    }
    let positions: Vec<Vector3<f64>> = target
        .iter()
        .map(|point| Vector3::new(point.x as f64, point.y as f64, point.z as f64))
        .collect();
    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (index, position) in positions.iter().enumerate() {
        let key = (
            (position.x / radius_m).floor() as i64,
            (position.y / radius_m).floor() as i64,
            (position.z / radius_m).floor() as i64,
        );
        buckets.entry(key).or_default().push(index);
    }
    let mut out = Vec::with_capacity(positions.len());
    for position in positions.iter() {
        let key = (
            (position.x / radius_m).floor() as i64,
            (position.y / radius_m).floor() as i64,
            (position.z / radius_m).floor() as i64,
        );
        let mut neighbors = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = buckets.get(&(key.0 + dx, key.1 + dy, key.2 + dz)) {
                        for &candidate in indices {
                            if (positions[candidate] - *position).norm() <= radius_m {
                                neighbors.push(positions[candidate]);
                            }
                        }
                    }
                }
            }
        }
        let normal = if neighbors.len() < min_neighbors {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            let mut center = Vector3::zeros();
            for point in &neighbors {
                center += point;
            }
            center /= neighbors.len() as f64;
            let mut covariance = Matrix3::zeros();
            for point in &neighbors {
                let delta = point - center;
                covariance += delta * delta.transpose();
            }
            covariance /= neighbors.len() as f64;
            let eigen = SymmetricEigen::new(covariance);
            let mut smallest = 0usize;
            for k in 1..3 {
                if eigen.eigenvalues[k] < eigen.eigenvalues[smallest] {
                    smallest = k;
                }
            }
            Vector3::new(
                eigen.eigenvectors[(0, smallest)],
                eigen.eigenvectors[(1, smallest)],
                eigen.eigenvectors[(2, smallest)],
            )
        };
        out.push(TargetPoint {
            position: *position,
            normal,
        });
    }
    out
}

fn downsample_cap(points: &[Vector3<f64>], voxel: f64, max_points: usize) -> Vec<Vector3<f64>> {
    let mut seen: HashMap<(i64, i64, i64), ()> = HashMap::new();
    let mut kept = Vec::new();
    for point in points {
        let key = (
            (point.x / voxel).floor() as i64,
            (point.y / voxel).floor() as i64,
            (point.z / voxel).floor() as i64,
        );
        if seen.insert(key, ()).is_none() {
            kept.push(*point);
        }
    }
    if kept.len() > max_points {
        let stride = kept.len().div_ceil(max_points);
        kept = kept.into_iter().step_by(stride).collect();
    }
    kept
}

fn se3_exp(omega: Vector3<f64>) -> Matrix3<f64> {
    let angle = omega.norm();
    if angle < 1e-12 {
        return Matrix3::identity();
    }
    let axis = omega / angle;
    let k = Matrix3::new(
        0.0, -axis.z, axis.y, axis.z, 0.0, -axis.x, -axis.y, axis.x, 0.0,
    );
    Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * (k * k)
}

fn source_world(
    points: &[PointXYZI],
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Vec<Vector3<f64>> {
    points
        .iter()
        .map(|point| {
            rotation * Vector3::new(point.x as f64, point.y as f64, point.z as f64) + translation
        })
        .collect()
}

fn measure_fitness(source: &[Vector3<f64>], target: &[Vector3<f64>], params: &IcpParams) -> f64 {
    if source.is_empty() {
        return 0.0;
    }
    let grid = Grid::new(target, params.fitness_distance);
    let mut inliers = 0usize;
    for point in source {
        if grid
            .nearest(point, target, params.fitness_distance)
            .is_some()
        {
            inliers += 1;
        }
    }
    inliers as f64 / source.len() as f64
}

/// Align `source` (world-frame points of keyframe `j`) to the target planes of
/// keyframe `i` with the given initial world-frame transform.
pub fn align_point_to_plane(
    source: &[PointXYZI],
    target: &[TargetPoint],
    init_rotation: &Matrix3<f64>,
    init_translation: &Vector3<f64>,
    params: &IcpParams,
) -> Option<IcpResult> {
    let debug = std::env::var_os("FASTLIO_LOOP_DEBUG").is_some();
    if target.is_empty() || source.is_empty() {
        if debug {
            eprintln!("icp: empty target={} source={}", target.len(), source.len());
        }
        return None;
    }
    let target_positions: Vec<Vector3<f64>> = target.iter().map(|point| point.position).collect();
    let target_normals: Vec<Vector3<f64>> = target.iter().map(|point| point.normal).collect();

    let mut rotation = *init_rotation;
    let mut translation = *init_translation;
    let mut max_distance = params.max_correspondence_distance;
    let mut rms = f64::INFINITY;

    for scale in params.scales {
        let cell = scale.max(0.5);
        let grid = Grid::new(&target_positions, cell);
        for _ in 0..params.iterations {
            let world = source_world(source, &rotation, &translation);
            let downsampled = downsample_cap(&world, scale, params.max_source_points);
            let mut hessian = Matrix6::<f64>::zeros();
            let mut gradient = Vector6::<f64>::zeros();
            let mut residual_sq = 0.0;
            let mut used = 0usize;
            for point in &downsampled {
                let Some(index) = grid.nearest(point, &target_positions, max_distance) else {
                    continue;
                };
                let normal = &target_normals[index];
                let residual = normal.dot(&(point - target_positions[index]));
                let abs = residual.abs();
                let huber = if abs < params.huber_threshold {
                    1.0
                } else {
                    params.huber_threshold / abs.max(1e-9)
                };
                let cross = point.cross(normal);
                let jacobian =
                    Vector6::new(cross.x, cross.y, cross.z, normal.x, normal.y, normal.z);
                hessian += jacobian * jacobian.transpose() * huber;
                gradient += jacobian * (residual * huber * huber);
                residual_sq += residual * residual * huber;
                used += 1;
            }
            if used < params.min_correspondences {
                if debug {
                    eprintln!("icp: used {used} < {}", params.min_correspondences);
                }
                return None;
            }
            let damping = Matrix6::<f64>::identity() * 1e-6;
            let Some(delta) = (hessian + damping).lu().solve(&(-gradient)) else {
                if debug {
                    eprintln!("icp: linear solve failed");
                }
                return None;
            };
            let omega = Vector3::new(delta[0], delta[1], delta[2]);
            let dt = Vector3::new(delta[3], delta[4], delta[5]);
            let increment = se3_exp(omega);
            rotation = increment * rotation;
            translation = increment * translation + dt;
            rms = (residual_sq / used as f64).sqrt();
        }
        max_distance = (max_distance * 0.5).max(0.3);
    }

    let aligned = source_world(source, &rotation, &translation);
    let fitness = measure_fitness(&aligned, &target_positions, params);
    Some(IcpResult {
        rotation,
        translation,
        fitness,
        rms,
    })
}

/// A loop measurement between keyframes `i` and `j`.
#[derive(Debug, Clone)]
pub struct LoopMeasurement {
    pub i: usize,
    pub j: usize,
    /// Measurement `z_ij = T_i^-1 * (G * T_j)`, rotation part.
    pub rotation: Matrix3<f64>,
    /// Measurement translation part.
    pub translation: Vector3<f64>,
    /// ICP inlier ratio.
    pub fitness: f64,
    /// Distance between the measured and odometry relative translations, m.
    pub correction_m: f64,
}

/// Parameters for loop measurement.
#[derive(Debug, Clone, Copy)]
pub struct MeasureParams {
    pub submap: SubmapParams,
    /// Voxel size used to build the target sub-map before plane fitting. It is
    /// denser than the source sub-map so that each 0.4 m plane voxel has
    /// enough points.
    pub target_voxel_m: f64,
    pub icp: IcpParams,
    pub fit_min_points: usize,
    pub plane_voxel_m: f64,
    pub min_fitness: f64,
    pub max_correction_m: f64,
    pub drift_offsets: [f64; 3],
    pub lateral_offsets: [f64; 3],
}

impl Default for MeasureParams {
    fn default() -> Self {
        Self {
            submap: SubmapParams::default(),
            target_voxel_m: 0.15,
            icp: IcpParams::default(),
            fit_min_points: 10,
            plane_voxel_m: 0.4,
            min_fitness: 0.6,
            max_correction_m: 2.0,
            drift_offsets: [-1.0, 0.0, 1.0],
            lateral_offsets: [-0.5, 0.0, 0.5],
        }
    }
}

/// Measure a loop between two keyframes; returns `None` when the result is
/// rejected by the fitness or correction-magnitude filters.
pub fn measure_loop(
    keyframes: &[Keyframe],
    i: usize,
    j: usize,
    params: &MeasureParams,
) -> Option<LoopMeasurement> {
    let target_points = build_submap(keyframes, i, &params.submap);
    let source_points = build_submap(keyframes, j, &params.submap);
    if target_points.is_empty() || source_points.is_empty() {
        return None;
    }
    let target = build_target_points(&target_points, 0.5, 5);
    if target.is_empty() {
        return None;
    }
    if std::env::var_os("FASTLIO_LOOP_DEBUG").is_some() {
        eprintln!(
            "loop ({i},{j}) pos_i={:?} pos_j={:?}",
            keyframes[i].position, keyframes[j].position
        );
    }
    let drift = keyframes[j].position - keyframes[i].position;
    let direction = if drift.norm() > 1e-9 {
        drift.normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let lateral = Vector3::new(-direction.y, direction.x, 0.0);
    let mut best: Option<IcpResult> = None;
    for drift_offset in params.drift_offsets {
        for lateral_offset in params.lateral_offsets {
            let translation = direction * drift_offset + lateral * lateral_offset;
            if let Some(result) = align_point_to_plane(
                &source_points,
                &target,
                &Matrix3::identity(),
                &translation,
                &params.icp,
            ) && best
                .as_ref()
                .is_none_or(|current| result.fitness > current.fitness)
            {
                best = Some(result);
            }
        }
    }
    let debug = std::env::var_os("FASTLIO_LOOP_DEBUG").is_some();
    let Some(best) = best else {
        if debug {
            eprintln!("loop ({i},{j}): no icp result");
        }
        return None;
    };
    if best.fitness < params.min_fitness {
        if debug {
            eprintln!(
                "loop ({i},{j}): fitness {:.3} < {:.3}",
                best.fitness, params.min_fitness
            );
        }
        return None;
    }
    let rotation_i = keyframes[i].rotation.to_rotation_matrix();
    let rotation_j = keyframes[j].rotation.to_rotation_matrix();
    let world_rotation = rotation_i.matrix().transpose() * best.rotation * rotation_j.matrix();
    let world_translation = rotation_i.matrix().transpose()
        * (best.rotation * keyframes[j].position + best.translation - keyframes[i].position);
    let odometry_translation =
        rotation_i.matrix().transpose() * (keyframes[j].position - keyframes[i].position);
    let correction_m = (world_translation - odometry_translation).norm();
    if correction_m > params.max_correction_m {
        return None;
    }
    Some(LoopMeasurement {
        i,
        j,
        rotation: world_rotation,
        translation: world_translation,
        fitness: best.fitness,
        correction_m,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::keyframe::Keyframe;
    use nalgebra::UnitQuaternion;

    fn keyframe(
        index: u64,
        timestamp_sec: f64,
        position: Vector3<f64>,
        points: Vec<(f32, f32, f32)>,
    ) -> Keyframe {
        Keyframe {
            index,
            timestamp_sec,
            position,
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

    fn room_cloud(shift: Vector3<f64>) -> Vec<(f32, f32, f32)> {
        let mut points = Vec::new();
        for x in -10..=10 {
            for y in -10..=10 {
                // Floor and ceiling at z = 0 and z = 3.
                points.push((
                    (x as f64 * 0.5 + shift.x) as f32,
                    (y as f64 * 0.5 + shift.y) as f32,
                    shift.z as f32,
                ));
                points.push((
                    (x as f64 * 0.5 + shift.x) as f32,
                    (y as f64 * 0.5 + shift.y) as f32,
                    (3.0 + shift.z) as f32,
                ));
            }
        }
        for y in 0..=20 {
            for z in 0..=30 {
                points.push((
                    (-5.0 + shift.x) as f32,
                    (y as f64 * 0.5 - 5.0 + shift.y) as f32,
                    (z as f64 * 0.1 + shift.z) as f32,
                ));
                points.push((
                    (5.0 + shift.x) as f32,
                    (y as f64 * 0.5 - 5.0 + shift.y) as f32,
                    (z as f64 * 0.1 + shift.z) as f32,
                ));
            }
        }
        points
    }

    #[test]
    fn icp_recovers_small_known_offset() {
        let cloud = room_cloud(Vector3::zeros());
        let shifted = room_cloud(Vector3::new(0.3, -0.2, 0.1));
        let target_points: Vec<PointXYZI> = cloud
            .iter()
            .map(|(x, y, z)| PointXYZI {
                x: *x,
                y: *y,
                z: *z,
                intensity: 0.0,
            })
            .collect();
        let source_points: Vec<PointXYZI> = shifted
            .iter()
            .map(|(x, y, z)| PointXYZI {
                x: *x,
                y: *y,
                z: *z,
                intensity: 0.0,
            })
            .collect();
        let target = build_target_points(&target_points, 1.0, 4);
        assert!(!target.is_empty());
        let result = align_point_to_plane(
            &source_points,
            &target,
            &Matrix3::identity(),
            &Vector3::zeros(),
            &IcpParams {
                max_correspondence_distance: 1.0,
                ..IcpParams::default()
            },
        )
        .expect("icp result");
        // The correction should be approximately -(shift).
        let recovered = result.translation;
        assert!((recovered.x - -0.3).abs() < 0.1, "x {}", recovered.x);
        assert!((recovered.y - 0.2).abs() < 0.1, "y {}", recovered.y);
    }

    #[test]
    fn measure_loop_rejects_large_displacement() {
        let a = keyframe(0, 0.0, Vector3::zeros(), room_cloud(Vector3::zeros()));
        let b = keyframe(
            1,
            200.0,
            Vector3::new(20.0, 0.0, 0.0),
            room_cloud(Vector3::new(20.0, 0.0, 0.0)),
        );
        let keyframes = vec![a, b];
        let params = MeasureParams {
            submap: SubmapParams {
                half_frames: 0,
                voxel_size_m: 0.25,
            },
            ..MeasureParams::default()
        };
        assert!(measure_loop(&keyframes, 0, 1, &params).is_none());
    }
}
