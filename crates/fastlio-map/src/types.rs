use anyhow::Result;
use fastlio_types::{Mat3, PointXYZI, SurfelConfig, Vec3};
use smallvec::SmallVec;

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct VoxelKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelKey {
    const AXIS_BITS: u32 = 21;
    const AXIS_MASK: u64 = (1 << Self::AXIS_BITS) - 1;
    const AXIS_BIAS: i64 = 1 << 20;

    #[inline]
    pub fn new(point: &PointXYZI, voxel_size: f32) -> Result<Self> {
        if !voxel_size.is_normal() || voxel_size < 0.0 {
            anyhow::bail!("voxel_size should be a valid number(f32).")
        }
        let x = (point.x / voxel_size).floor() as i32;
        let y = (point.y / voxel_size).floor() as i32;
        let z = (point.z / voxel_size).floor() as i32;
        Ok(Self { x, y, z })
    }

    #[inline]
    pub fn from_vec3(point: Vec3<f32>, voxel_size: f32) -> Result<Self> {
        if !voxel_size.is_normal() || voxel_size < 0.0 {
            anyhow::bail!("voxel_size should be a valid number(f32).")
        }

        let x = (point[0] / voxel_size).floor() as i32;
        let y = (point[1] / voxel_size).floor() as i32;
        let z = (point[2] / voxel_size).floor() as i32;
        Ok(Self { x, y, z })
    }

    #[inline]
    pub(crate) fn encode_axis(value: i32) -> u64 {
        let encoded = value as i64 + Self::AXIS_BIAS;

        debug_assert!(
            (0..=Self::AXIS_MASK as i64).contains(&encoded),
            "map axis exceed max: 21 bit"
        );
        encoded as u64
    }

    #[inline]
    pub(crate) fn decode_axis(value: u64) -> i32 {
        (value as i64 - Self::AXIS_BIAS) as i32
    }

    #[inline]
    pub fn pack(self) -> u64 {
        let x = Self::encode_axis(self.x);
        let y = Self::encode_axis(self.y);
        let z = Self::encode_axis(self.z);

        (x << 42) | (y << 21) | z
    }

    #[inline]
    pub fn unpack(key: u64) -> Self {
        Self {
            x: Self::decode_axis((key >> 42) & Self::AXIS_MASK),
            y: Self::decode_axis((key >> 21) & Self::AXIS_MASK),
            z: Self::decode_axis(key & Self::AXIS_MASK),
        }
    }
}

pub enum GeometryClass {
    Plane,
    Line,
    Scatter,
    Degenerate,
    Growing,
}

#[derive(Default)]
pub struct Surfel {
    pub mean_w: Vec3<f64>,
    pub eigenvectors: Mat3<f64>,
    pub eigenvalues: Vec3<f64>,
    pub m2: Mat3<f64>,

    pub count: usize,
    pub last_refit: usize,
    /// Timestamp (s) of the last plane refit / creation. Association drift grows
    /// with the time since this anchor, because the pose that anchored the surfel
    /// may have drifted since then.
    pub last_refit_time_sec: f64,
    pub indexed_voxels: SmallVec<[u64; 8]>,
}

impl Surfel {
    pub fn from_first_point(point: &PointXYZI) -> Self {
        let mean_w = point.to_vec3_f64();
        let m2 = Mat3::<f64>::zeros();
        let init_vec3 = Vec3::<f64>::zeros();

        Surfel {
            mean_w,
            eigenvectors: m2,
            eigenvalues: init_vec3,
            m2,
            count: 1,
            last_refit: 0,
            last_refit_time_sec: 0.0,
            indexed_voxels: SmallVec::new(),
        }
    }

    pub fn plane_distance(&self, point: &PointXYZI) -> f32 {
        let norm_w = self.eigenvectors.column(0);
        ((point.to_vec3_f64() - self.mean_w).dot(&norm_w).abs()) as f32
    }

    pub fn within_plane_distance(&self, point: &PointXYZI, config: &SurfelConfig) -> bool {
        let max_distance = config.max_plane_distance;
        self.plane_distance(point) <= max_distance
    }

    pub fn support_aabb_extent(&self, config: &SurfelConfig) -> Vec3<f64> {
        let support_sigma = config.max_mahalanobis_distance as f64;
        let covariance_floor = config.covariance_eigenvalue_floor as f64;
        let lambda_0 = support_sigma * support_sigma * self.eigenvalues[0].max(covariance_floor);
        let lambda_1 = support_sigma * support_sigma * self.eigenvalues[1].max(covariance_floor);
        let lambda_2 = support_sigma * support_sigma * self.eigenvalues[2].max(covariance_floor);

        let t0 = self.eigenvectors.column(0);
        let t1 = self.eigenvectors.column(1);
        let t2 = self.eigenvectors.column(2);

        Vec3::new(
            (lambda_0 * t0.x * t0.x + lambda_1 * t1.x * t1.x + lambda_2 * t2.x * t2.x).sqrt(),
            (lambda_0 * t0.y * t0.y + lambda_1 * t1.y * t1.y + lambda_2 * t2.y * t2.y).sqrt(),
            (lambda_0 * t0.z * t0.z + lambda_1 * t1.z * t1.z + lambda_2 * t2.z * t2.z).sqrt(),
        )
    }

    pub fn mahalanobis_squared(&self, point: &PointXYZI, config: &SurfelConfig) -> f64 {
        let delta = point.to_vec3_f64() - self.mean_w;
        let d0 = self.eigenvectors.column(0).dot(&delta);
        let d1 = self.eigenvectors.column(1).dot(&delta);
        let d2 = self.eigenvectors.column(2).dot(&delta);

        let covariance_floor = config.covariance_eigenvalue_floor as f64;
        let l0 = self.eigenvalues[0].max(covariance_floor);
        let l1 = self.eigenvalues[1].max(covariance_floor);
        let l2 = self.eigenvalues[2].max(covariance_floor);

        d0 * d0 / l0 + d1 * d1 / l1 + d2 * d2 / l2
    }

    pub fn within_support(&self, point: &PointXYZI, config: &SurfelConfig) -> bool {
        let support_sigma = config.max_mahalanobis_distance;
        self.mahalanobis_squared(point, config) as f32 <= support_sigma * support_sigma
    }

    /// Score a point against this surfel for association and return the
    /// covariance pieces used by the update.
    ///
    /// The returned tuple is `(geometric_covariance_w, plane_covariance_w,
    /// score)`:
    ///
    /// - geometric covariance = floored surfel scatter (`m2/(n-1)` shape);
    /// - plane covariance = uncertainty of the fitted plane parameters
    ///   propagated to this point: offset variance `l0/n` along the normal,
    ///   plus normal angular variance `l0/(n*l1)` times the tangential lever
    ///   arm `|t|^2`; used by the IEKF whitening;
    /// - score = Mahalanobis distance using `geo + plane + point covariance`
    ///   plus the isotropic map-drift prior `association_map_drift_sigma^2`.
    ///   The drift term widens only the gate; it is not part of the returned
    ///   covariances.
    ///
    /// The support-domain test rejects points whose tangential projection
    /// falls outside the surfel's bounded patch, so an infinite-plane
    /// extrapolation is never accepted.
    /// Association drift prior: the gate widens with the time since the surfel
    /// was last anchored by a refit, because the anchoring pose may have
    /// drifted. `base` is the floor, `rate` the growth in m/s and `max` the cap.
    pub(crate) fn association_drift_sigma(age_sec: f64, config: &SurfelConfig) -> f64 {
        let base = config.association_map_drift_sigma as f64;
        let rate = config.association_drift_rate as f64;
        let max = config.association_drift_max as f64;
        (base + rate * age_sec.max(0.0)).clamp(0.0, max.max(0.0))
    }

    /// Linear fade of the drift prior beyond `association_drift_max_range`:
    /// near returns keep the full prior, far returns fall back to the tight
    /// fit-only gate. `0` disables the range gating (always 1).
    pub(crate) fn drift_range_scale(
        point_w: &PointXYZI,
        sensor_w: &Vec3<f64>,
        config: &SurfelConfig,
    ) -> f64 {
        let max_range = config.association_drift_max_range as f64;
        if max_range <= 0.0 {
            return 1.0;
        }
        let range = (point_w.to_vec3_f64() - *sensor_w).norm();
        ((max_range - range) / max_range).clamp(0.0, 1.0)
    }

    /// Local map density scale for the drift prior: a dense neighbourhood (many
    /// surfels within `association_drift_density_radius`, i.e. close-range
    /// indoor scenes) keeps the full prior; a sparse long-range neighbourhood
    /// gets none, because there a loose gate picks wrong distant surfels.
    /// `high <= low` disables the density gating (always 1).
    pub(crate) fn drift_density_scale(local_surfels: usize, config: &SurfelConfig) -> f64 {
        let low = config.association_drift_density_low as f64;
        let high = config.association_drift_density_high as f64;
        if high <= low {
            return 1.0;
        }
        ((local_surfels as f64 - low) / (high - low)).clamp(0.0, 1.0)
    }

    pub fn surfel_score(
        &self,
        point_w: &PointXYZI,
        point_w_variance: &Mat3<f64>,
        config: &SurfelConfig,
        age_sec: f64,
        drift_scale: f64,
    ) -> Option<(Mat3<f64>, Mat3<f64>, f64)> {
        let delta = point_w.to_vec3_f64() - self.mean_w;
        let e0 = self.eigenvectors.column(0);
        let e1 = self.eigenvectors.column(1);
        let e2 = self.eigenvectors.column(2);

        let support_floor = config.covariance_eigenvalue_floor as f64;
        let support_sigma = config.max_mahalanobis_distance as f64;

        // Support domain: bounded plane patch (tangential ellipse).
        let s1 = self.eigenvalues[1].max(support_floor);
        let s2 = self.eigenvalues[2].max(support_floor);
        let t1 = e1.dot(&delta);
        let t2 = e2.dot(&delta);
        let tangent_quad = t1 * t1 / s1 + t2 * t2 / s2;
        if tangent_quad > support_sigma * support_sigma {
            return None;
        }

        // Geometric scatter with the association floor (numerical stability).
        let floor = config.association_covariance_eigenvalue_floor as f64;
        let l0 = self.eigenvalues[0].max(floor);
        let l1 = self.eigenvalues[1].max(floor);
        let l2 = self.eigenvalues[2].max(floor);
        let geometric = e0 * e0.transpose() * l0
            + e1 * e1.transpose() * l1
            + e2 * e2.transpose() * l2;

        // Plane-parameter covariance propagated to this point. This is the
        // *fit* uncertainty and is what the IEKF whitens with; the map drift
        // term below only widens the association gate, because treating drift
        // as measurement noise would weaken the actual plane constraints and
        // can let the filter diverge.
        let count = self.count.max(2) as f64;
        let normal_spread = self.eigenvalues[0].max(0.0);
        let angular_variance = normal_spread / (count * s1);
        let lever_arm = t1 * t1 + t2 * t2;
        let plane = e0 * e0.transpose() * (normal_spread / count)
            + (e1 * e1.transpose() + e2 * e2.transpose()) * (lever_arm * angular_variance);

        // Association gate covariance: fit covariance + drift prior. The drift
        // prior grows with the age since the last refit, so stale/indoor
        // surfaces get a wider gate while freshly anchored ones stay tight.
        let drift_sigma = Self::association_drift_sigma(age_sec, config) * drift_scale;
        let gate_covariance = geometric
            + plane
            + point_w_variance
            + Mat3::identity() * (drift_sigma * drift_sigma);
        let chol = gate_covariance.cholesky()?;
        let l = chol.l();
        let y = l.solve_lower_triangular(&delta)?;
        let score = y.dot(&y);
        Some((geometric, plane, score))
    }

    pub fn within_tangent_support(&self, point: &PointXYZI, config: &SurfelConfig) -> bool {
        let delta = point.to_vec3_f64() - self.mean_w;
        let d1 = self.eigenvectors.column(1).dot(&delta);
        let d2 = self.eigenvectors.column(2).dot(&delta);

        let covariance_floor = config.covariance_eigenvalue_floor as f64;
        let l1 = self.eigenvalues[1].max(covariance_floor);
        let l2 = self.eigenvalues[2].max(covariance_floor);
        let support_sigma = config.max_mahalanobis_distance as f64;

        d1 * d1 / l1 + d2 * d2 / l2 <= support_sigma * support_sigma
    }

    pub(crate) fn planarity(&self) -> f64 {
        self.eigenvalues[0] / self.eigenvalues[1]
    }

    pub(crate) fn linearity(&self) -> f64 {
        (self.eigenvalues[2] - self.eigenvalues[1]) / self.eigenvalues[2]
    }

    fn scattering(&self) -> f64 {
        self.eigenvalues[0] / self.eigenvalues[2]
    }

    pub(crate) fn is_growing(&self, min_mature_surfel_count: usize) -> bool {
        self.count < min_mature_surfel_count
    }

    pub(crate) fn is_planar(&self, config: &SurfelConfig) -> bool {
        if self.eigenvalues[1] < config.min_plane_spread_eigenvalue as f64 {
            return false;
        }
        self.planarity() <= config.max_planarity_ratio as f64
    }

    pub fn geometry_class(&self, config: &SurfelConfig) -> GeometryClass {
        let planarity = self.planarity();
        let linearity = self.linearity();
        let scattering = self.scattering();
        if self.count < config.min_mature_surfel_count {
            GeometryClass::Growing
        } else if planarity < config.max_planarity_ratio as f64
            && self.eigenvalues[1] > config.min_plane_spread_eigenvalue as f64
        {
            GeometryClass::Plane
        } else if linearity > config.min_linearity as f64
            && self.eigenvalues[2] > config.min_line_spread_eigenvalue as f64
        {
            GeometryClass::Line
        } else if scattering > config.min_scattering as f64 {
            GeometryClass::Scatter
        } else {
            GeometryClass::Degenerate
        }
    }

    pub(crate) fn within_growing_radius(&self, point: &PointXYZI, config: &SurfelConfig) -> bool {
        let dist2 = (point.to_vec3_f64() - self.mean_w).norm_squared();
        let radius = config.growing_radius as f64;
        dist2 <= radius * radius
    }

    pub(crate) fn line_distance(&self, point: &PointXYZI) -> f64 {
        let n0 = self.eigenvectors.column(0);
        let n1 = self.eigenvectors.column(1);
        let delta = point.to_vec3_f64() - self.mean_w;
        let r0 = n0.dot(&delta);
        let r1 = n1.dot(&delta);
        (r0 * r0 + r1 * r1).sqrt()
    }

    // merge an similar surfel
    // pub(crate) fn merge_surfel(&mut self, rhs: &Surfel) {
    //     let na = self.count as f32;
    //     let nb = rhs.count as f32;
    //     let n = na + nb;

    //     let delta = rhs.mean_w - self.mean_w;

    //     self.mean_w += delta * (nb / n) as f64;
    //     self.m2 += rhs.m2 + delta * delta.transpose() * (na * nb / n) as f64;
    //     self.count += rhs.count;
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drift_config() -> SurfelConfig {
        SurfelConfig {
            association_map_drift_sigma: 0.05,
            association_drift_rate: 0.01,
            association_drift_max: 1.0,
            ..SurfelConfig::default()
        }
    }

    #[test]
    fn drift_sigma_is_base_plus_rate_times_age() {
        let config = drift_config();
        assert!((Surfel::association_drift_sigma(0.0, &config) - 0.05).abs() < 1e-6);
        assert!((Surfel::association_drift_sigma(10.0, &config) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn drift_sigma_is_capped_and_ignores_negative_age() {
        let config = SurfelConfig {
            association_map_drift_sigma: 0.05,
            association_drift_rate: 0.01,
            association_drift_max: 0.20,
            ..SurfelConfig::default()
        };
        assert!((Surfel::association_drift_sigma(-5.0, &config) - 0.05).abs() < 1e-6);
        assert!((Surfel::association_drift_sigma(1000.0, &config) - 0.20).abs() < 1e-6);
    }

    #[test]
    fn density_scale_enables_drift_only_in_dense_maps() {
        let config = SurfelConfig {
            association_drift_density_low: 5.0,
            association_drift_density_high: 30.0,
            ..SurfelConfig::default()
        };
        // Sparse long-range neighbourhood: drift disabled.
        assert_eq!(Surfel::drift_density_scale(1, &config), 0.0);
        assert_eq!(Surfel::drift_density_scale(5, &config), 0.0);
        // Mid-density: linear ramp.
        assert!((Surfel::drift_density_scale(17, &config) - 0.48).abs() < 1e-6);
        // Dense close-range neighbourhood: full drift prior.
        assert_eq!(Surfel::drift_density_scale(30, &config), 1.0);
        assert_eq!(Surfel::drift_density_scale(500, &config), 1.0);
        // Disabled when high <= low.
        let disabled = SurfelConfig::default();
        assert_eq!(Surfel::drift_density_scale(0, &disabled), 1.0);
    }

    #[test]
    fn drift_range_scale_fades_long_range() {
        let config = SurfelConfig {
            association_drift_max_range: 20.0,
            ..SurfelConfig::default()
        };
        let sensor = Vec3::new(0.0, 0.0, 0.0);
        let near = PointXYZI {
            x: 5.0,
            y: 0.0,
            z: 0.0,
            intensity: 0.0,
        };
        let far = PointXYZI {
            x: 25.0,
            y: 0.0,
            z: 0.0,
            intensity: 0.0,
        };
        assert!((Surfel::drift_range_scale(&near, &sensor, &config) - 0.75).abs() < 1e-6);
        assert_eq!(Surfel::drift_range_scale(&far, &sensor, &config), 0.0);

        let unlimited = SurfelConfig {
            association_drift_max_range: 0.0,
            ..SurfelConfig::default()
        };
        assert_eq!(Surfel::drift_range_scale(&far, &sensor, &unlimited), 1.0);
    }

    #[test]
    fn surfel_score_widens_gate_with_age() {
        let mut surfel = Surfel::from_first_point(&PointXYZI {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            intensity: 0.0,
        });
        // 1 cm thick, wide patch: the query point is inside the support domain.
        surfel.eigenvalues = Vec3::new(1e-4, 1.0, 1.0);
        surfel.eigenvectors = Mat3::identity();
        surfel.count = 32;

        let config = SurfelConfig {
            association_map_drift_sigma: 0.0,
            association_drift_rate: 0.01,
            association_drift_max: 1.0,
            ..SurfelConfig::default()
        };
        let point = PointXYZI {
            x: 0.0,
            y: 0.0,
            z: 0.2,
            intensity: 0.0,
        };
        let point_covariance = Mat3::identity() * 1e-4;

        let fresh = surfel
            .surfel_score(&point, &point_covariance, &config, 0.0, 1.0)
            .unwrap()
            .2;
        let stale = surfel
            .surfel_score(&point, &point_covariance, &config, 60.0, 1.0)
            .unwrap()
            .2;
        assert!(
            stale < fresh,
            "stale surfels must have a wider gate: fresh={fresh} stale={stale}"
        );
    }
}
