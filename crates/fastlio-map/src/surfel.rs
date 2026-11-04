use crate::VoxelKey;
use crate::types::{GeometryClass, Surfel};
use anyhow::Result;
use fastlio_types::{Mat3, PointXYZI, SurfelConfig, SurfelMapConfig, Vec3};
use hashbrown::{HashMap, HashSet};
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::VecDeque;

pub use crate::types::SurfelID;

const LINE_SECOND_BEST_MIN_DISTANCE_MARGIN: f64 = 0.05;
const LINE_SECOND_BEST_MAX_DISTANCE_RATIO: f64 = 0.70;

#[derive(Default)]
pub struct SurfelMap {
    pub(super) buckets: HashMap<u64, SmallVec<[SurfelID; 4]>>,
    pub(super) surfels: SlotMap<SurfelID, Surfel>,
    surfel_map_config: SurfelMapConfig,
    surfel_config: SurfelConfig,
    /// Current frame time (s), used for surfel lifetime preference.
    time_sec: f64,
    /// Deferred maintenance queue of dirty roots whose coplanar neighbours are
    /// scanned at maintenance time (never on the hot path).
    merge_queue: VecDeque<SurfelID>,
    merged_total: usize,
    max_chain_depth: usize,
}

/// Result of one deferred map-maintenance step (diagnostic).
#[derive(Debug, Default, Clone, Copy)]
pub struct MaintenanceStats {
    pub merged: usize,
    pub queue_len: usize,
    pub max_chain_depth: usize,
    pub merged_total: usize,
}

#[derive(Debug, Default, Clone)]
pub struct MapLifecycleStats {
    pub total: usize,
    pub mature: usize,
    pub growing: usize,
    pub mean_count: f64,
    pub mean_lambda0: f64,
    /// Count buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+.
    pub count_hist: [u32; 8],
    /// Surfel-per-voxel buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+.
    pub occupancy_hist: [u32; 8],
    /// Estimated number of adjacent surfels describing the same plane.
    pub coplanar_duplicate_pairs: usize,
}

fn bucket_index(value: usize) -> usize {
    match value {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        _ => 7,
    }
}

#[derive(Debug, Clone)]
pub struct SurfelObservation {
    pub surfel_id: SurfelID,
    pub mean_w: Vec3<f64>,
    pub covariance_w: Mat3<f64>,
    pub eigenvectors: Mat3<f64>,
    pub eigenvalues: Vec3<f64>,
    pub best_score: f64,
    pub second_best_score: Option<f64>,
    /// Association confidence in (0, 1]: 1 for mature surfels, less for growing
    /// surfels taking part with `growing_association_weight`.
    pub confidence: f64,
}

/// A planar surface observation returned by [`SurfelMap::query`].
///
/// It describes the *plane* attached to the best-matching planar surfel
/// near a query point, expressed entirely in the world / map frame `W`:
///
/// - `mean_w`: a point on the plane (the surfel centroid).
/// - `norm_w`: a unit plane normal, expressed in `W`.
/// - `eigenvalues`: the surfel covariance eigenvalues sorted ascending; for
///   a plane the smallest entry is near zero while `eigenvalues[1]` measures
///   spread along the plane.
/// - `plane_distance`: the perpendicular distance from the queried point to
///   the plane: `|(query_w - mean_w) · norm_w|`.
/// - `planarity`: the ratio `eigenvalues[0] / eigenvalues[1]`; smaller means
///   more planar.
///
/// All coordinates and vectors are in meters and expressed in the world
/// frame. The sign of `norm_w` is arbitrary (it follows the principal
/// eigenvector) and must not be depended upon; use its absolute value or
/// align it against a known reference.
#[derive(Debug, Clone)]
pub struct SurfelPlaneObservation {
    pub surfel_id: SurfelID,
    pub mean_w: Vec3<f64>,
    pub norm_w: Vec3<f64>,
    pub eigenvalues: Vec3<f64>,
    pub plane_distance: f64,
    pub planarity: f64,
    pub signed_residual: f64,
}

impl SurfelPlaneObservation {
    pub fn new(
        surfel_id: SurfelID,
        mean_w: Vec3<f64>,
        norm_w: Vec3<f64>,
        eigenvalues: Vec3<f64>,
        plane_distance: f64,
        planarity: f64,
        signed_residual: f64,
    ) -> Self {
        Self {
            surfel_id,
            mean_w,
            norm_w,
            eigenvalues,
            plane_distance,
            planarity,
            signed_residual,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfelLineObservation {
    pub surfel_id: SurfelID,
    pub mean_w: Vec3<f64>,
    pub eigenvalues: Vec3<f64>,
    pub direction_w: Vec3<f64>,
    pub normal0_w: Vec3<f64>,
    pub normal1_w: Vec3<f64>,
    pub residual0: f64,
    pub residual1: f64,
    pub distance: f64,
    pub linearity: f64,
    pub second_best_distance: Option<f64>,
    pub ambiguity_ratio: Option<f64>,
}

fn helper_interval(count: usize) -> usize {
    match count {
        0..=16 => 4,
        17..=64 => 8,
        _ => 16,
    }
}

impl SurfelMap {
    pub fn new(map_config: SurfelMapConfig, surfel_config: SurfelConfig) -> Self {
        let buckets = HashMap::new();
        let surfels = SlotMap::with_key();
        Self {
            buckets,
            surfels,
            surfel_map_config: map_config,
            surfel_config,
            time_sec: 0.0,
            merge_queue: VecDeque::new(),
            merged_total: 0,
            max_chain_depth: 0,
        }
    }

    /// Set the timestamp (s) of the frame currently being processed.
    pub fn set_time(&mut self, time_sec: f64) {
        self.time_sec = time_sec;
    }

    /// Resolve the union-find root of a surfel and report the chain length.
    pub fn resolve_root(&self, surfel_id: SurfelID) -> (SurfelID, usize) {
        let mut id = surfel_id;
        let mut depth = 0;
        loop {
            let Some(surfel) = self.surfels.get(id) else {
                return (id, depth);
            };
            match surfel.parent {
                Some(parent) if parent != id => {
                    id = parent;
                    depth += 1;
                    if depth > 64 {
                        return (id, depth);
                    }
                }
                _ => return (id, depth),
            }
        }
    }

    pub fn query_plane(&self, point: &PointXYZI) -> Result<Option<SurfelPlaneObservation>> {
        let radius = self.surfel_map_config.search_radius;
        if !point.is_valid() {
            return Ok(None);
        }
        let mut voxel_key = VoxelKey::new(point, self.surfel_map_config.voxel_size)?;
        let x = voxel_key.x;
        let y = voxel_key.y;
        let z = voxel_key.z;
        let candidate_ids = (x - radius..=x + radius)
            .flat_map(move |x| {
                (y - radius..=y + radius).flat_map(move |y| {
                    (z - radius..=z + radius).map(move |z| {
                        voxel_key.x = x;
                        voxel_key.y = y;
                        voxel_key.z = z;
                        voxel_key.pack()
                    })
                })
            })
            .filter_map(|v| self.buckets.get(&v))
            .flat_map(|bucket| bucket.iter().copied())
            .filter_map(|id| self.surfels.get(id).map(|s| (id, s)))
            .filter(|(_, surfel)| {
                surfel.is_planar(&self.surfel_config)
                    && surfel.within_tangent_support(point, &self.surfel_config)
                    && surfel.within_plane_distance(point, &self.surfel_config)
            });
        let mut best: Option<(SurfelID, f32)> = None;
        let mut seen = HashSet::new();
        for (id, surfel) in candidate_ids {
            if seen.contains(&id) {
                continue;
            } else {
                seen.insert(id);
            }
            match surfel.geometry_class(&self.surfel_config) {
                GeometryClass::Plane => {
                    let score = surfel.plane_distance(point);
                    if best.is_none_or(|(_, best)| score < best) {
                        best = Some((id, score));
                    }
                }
                _ => continue,
            }
        }

        if let Some((best_id, score)) = best {
            // best_id is selected from candidate_ids, which only yields ids that were
            let surfel = self.surfels.get(best_id).expect(
                "UnexpectedError: on query: Logically, there should be no missing value here.",
            );
            let mean_w = surfel.mean_w.cast();
            let norm_w = surfel.eigenvectors.column(0).into_owned().cast();
            let eigenvalues = surfel.eigenvalues.cast::<f64>();
            let plane_distance = score as f64;
            let planarity = surfel.planarity();
            let signed_residual = norm_w.dot(&(point.to_vec3_f64() - mean_w));
            Ok(Some(SurfelPlaneObservation::new(
                best_id,
                mean_w,
                norm_w,
                eigenvalues,
                plane_distance,
                planarity,
                signed_residual,
            )))
        } else {
            Ok(None)
        }
    }

    pub fn query_line(&self, point: &PointXYZI) -> Result<Option<SurfelLineObservation>> {
        let radius = self.surfel_map_config.search_radius;
        if !point.is_valid() {
            return Ok(None);
        }
        let mut voxel_key = VoxelKey::new(point, self.surfel_map_config.voxel_size)?;
        let x = voxel_key.x;
        let y = voxel_key.y;
        let z = voxel_key.z;
        let candidate_ids = (x - radius..=x + radius)
            .flat_map(move |x| {
                (y - radius..=y + radius).flat_map(move |y| {
                    (z - radius..=z + radius).map(move |z| {
                        voxel_key.x = x;
                        voxel_key.y = y;
                        voxel_key.z = z;
                        voxel_key.pack()
                    })
                })
            })
            .filter_map(|v| self.buckets.get(&v))
            .flat_map(|buckets| buckets.iter().copied())
            .filter_map(|id| self.surfels.get(id).map(|surfel| (id, surfel)));

        let mut best: Option<(SurfelID, f64)> = None;
        let mut second_best: Option<(SurfelID, f64)> = None;
        let mut seen = HashSet::new();
        for (id, surfel) in candidate_ids {
            if seen.contains(&id) {
                continue;
            } else {
                seen.insert(id);
            }
            match surfel.geometry_class(&self.surfel_config) {
                GeometryClass::Line => {
                    let score = surfel.line_distance(point);
                    if score > self.surfel_config.max_plane_distance as f64
                        || !surfel.within_support(point, &self.surfel_config)
                    {
                        continue;
                    }
                    if best.is_none_or(|(_, best_score)| score < best_score) {
                        second_best = best;
                        best = Some((id, score));
                    } else if second_best.is_none_or(|(_, second_score)| score < second_score) {
                        second_best = Some((id, score));
                    }
                }
                _ => continue,
            }
        }

        if let Some((best_id, best_distance)) = best {
            if let Some((_, second_best_distance)) = second_best
                && line_association_is_ambiguous(best_distance, second_best_distance)
            {
                return Ok(None);
            }

            // best_id is selected from candidate_ids, which only yields ids that were
            let surfel = self.surfels.get(best_id).expect(
                "UnexpectedError: on query: Logically, there should be no missing value here.",
            );
            let delta = point.to_vec3_f64() - surfel.mean_w;
            let normal0_w = surfel.eigenvectors.column(0).into_owned().normalize();
            let normal1_w = surfel.eigenvectors.column(1).into_owned().normalize();
            let direction_w = surfel.eigenvectors.column(2).into_owned().normalize();
            let residual0 = normal0_w.dot(&delta);
            let residual1 = normal1_w.dot(&delta);
            let distance = (residual0 * residual0 + residual1 * residual1).sqrt();
            let linearity = surfel.linearity();
            let second_best_distance = second_best.map(|(_, distance)| distance);
            let ambiguity_ratio = second_best_distance.map(|second| {
                if second <= f64::EPSILON {
                    1.0
                } else {
                    (distance / second).clamp(0.0, 1.0)
                }
            });

            Ok(Some(SurfelLineObservation {
                surfel_id: best_id,
                mean_w: surfel.mean_w,
                eigenvalues: surfel.eigenvalues,
                direction_w,
                normal0_w,
                normal1_w,
                residual0,
                residual1,
                distance,
                linearity,
                second_best_distance,
                ambiguity_ratio,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn query_surfel(
        &self,
        point: &PointXYZI,
        point_w_covariance: Mat3<f64>,
    ) -> Result<Option<SurfelObservation>> {
        let radius = self.surfel_map_config.search_radius;
        if !point.is_valid() {
            return Ok(None);
        }
        let mut voxel_key = VoxelKey::new(point, self.surfel_map_config.voxel_size)?;
        let x = voxel_key.x;
        let y = voxel_key.y;
        let z = voxel_key.z;
        let candidate_ids = (x - radius..=x + radius)
            .flat_map(move |x| {
                (y - radius..=y + radius).flat_map(move |y| {
                    (z - radius..=z + radius).map(move |z| {
                        voxel_key.x = x;
                        voxel_key.y = y;
                        voxel_key.z = z;
                        voxel_key.pack()
                    })
                })
            })
            .filter_map(|v| self.buckets.get(&v))
            .flat_map(|buckets| buckets.iter().copied());

        let min_mature = self.surfel_config.min_mature_surfel_count;
        let grossing_weight = self.surfel_config.growing_association_weight;
        let mut best: Option<(SurfelID, f64)> = None;
        let mut second_best: Option<(SurfelID, f64)> = None;
        let mut seen_roots = HashSet::new();
        for candidate_id in candidate_ids {
            let (id, _depth) = self.resolve_root(candidate_id);
            // The same root may be reachable through several indexed voxels /
            // stale children; score it only once per query.
            if !seen_roots.insert(id) {
                continue;
            }
            let Some(surfel) = self.surfels.get(id) else {
                continue;
            };
            if surfel.count < 2 {
                continue;
            }
            let growing = surfel.is_growing(min_mature);
            if growing && grossing_weight <= 0.0 {
                continue;
            }
            let Some((_, _, score)) = surfel.surfel_score(point, &point_w_covariance, &self.surfel_config) else {
                continue;
            };
            if score > self.surfel_config.max_mahalanobis_distance.powi(2) as f64 {
                continue;
            }

            if best.is_none_or(|(_, best_score)| score < best_score) {
                second_best = best;
                best = Some((id, score));
            } else if second_best.is_none_or(|(_, second_score)| score < second_score) {
                second_best = Some((id, score));
            }
        }
        if let Some((best_id, best_score)) = best {
            let surfel = self.surfels.get(best_id).expect(
                "UnexpectedError: on query: Logically, there should be no missing value here.",
            );
            let confidence = if surfel.is_growing(min_mature) {
                let maturity = (surfel.count as f32 / min_mature.max(1) as f32).clamp(0.0, 1.0);
                (grossing_weight * maturity) as f64
            } else {
                1.0
            };
            Ok(Some(SurfelObservation {
                surfel_id: best_id,
                mean_w: surfel.mean_w,
                covariance_w: surfel.m2 / (surfel.count - 1) as f64,
                eigenvalues: surfel.eigenvalues,
                eigenvectors: surfel.eigenvectors,
                best_score,
                second_best_score: second_best.map(|(_, score)| score),
                confidence,
            }))
        } else {
            Ok(None)
        }
    }

    /// Lifecycle statistics of the surfel map (diagnostic for map-quality work):
    /// maturity split, count/eigenvalue distributions, voxel occupancy and an
    /// estimate of adjacent surfels describing the same plane.
    pub fn lifecycle_stats(&self) -> MapLifecycleStats {
        let mut stats = MapLifecycleStats::default();
        let min_mature = self.surfel_config.min_mature_surfel_count;
        let mut total_count = 0u64;
        let mut total_lambda0 = 0.0;
        for surfel in self.surfels.values() {
            stats.total += 1;
            total_count += surfel.count as u64;
            total_lambda0 += surfel.eigenvalues[0].max(0.0);
            if surfel.count >= min_mature {
                stats.mature += 1;
            } else {
                stats.growing += 1;
            }
            stats.count_hist[bucket_index(surfel.count)] += 1;
        }
        if stats.total > 0 {
            stats.mean_count = total_count as f64 / stats.total as f64;
            stats.mean_lambda0 = total_lambda0 / stats.total as f64;
        }
        for bucket in self.buckets.values() {
            stats.occupancy_hist[bucket_index(bucket.len())] += 1;
        }

        // Coplanar duplicate estimate: sample surfels and count adjacent ones
        // whose normal and offset match, then scale back up.
        const SAMPLE_STRIDE: usize = 8;
        let mut sampled = 0usize;
        let mut duplicates = 0usize;
        for (index, (id, surfel)) in self.surfels.iter().enumerate() {
            if index % SAMPLE_STRIDE != 0 {
                continue;
            }
            sampled += 1;
            duplicates += self.count_coplanar_neighbours(id, surfel);
        }
        if sampled > 0 {
            stats.coplanar_duplicate_pairs = duplicates * stats.total / sampled;
        }
        stats
    }

    fn count_coplanar_neighbours(&self, id: SurfelID, surfel: &Surfel) -> usize {
        let voxel = Vec3::new(
            surfel.mean_w.x as f32,
            surfel.mean_w.y as f32,
            surfel.mean_w.z as f32,
        );
        let Ok(mut voxel_key) = VoxelKey::from_vec3(voxel, self.surfel_map_config.voxel_size)
        else {
            return 0;
        };
        let radius = self.surfel_map_config.search_radius;
        let x = voxel_key.x;
        let y = voxel_key.y;
        let z = voxel_key.z;
        let normal = surfel.eigenvectors.column(0);
        let mut count = 0;
        let mut seen = HashSet::new();
        for vx in x - radius..=x + radius {
            for vy in y - radius..=y + radius {
                for vz in z - radius..=z + radius {
                    voxel_key.x = vx;
                    voxel_key.y = vy;
                    voxel_key.z = vz;
                    let Some(bucket) = self.buckets.get(&voxel_key.pack()) else {
                        continue;
                    };
                    for other_id in bucket.iter().copied() {
                        if other_id == id || !seen.insert(other_id) {
                            continue;
                        }
                        let Some(other) = self.surfels.get(other_id) else {
                            continue;
                        };
                        let alignment = normal.dot(&other.eigenvectors.column(0)).abs();
                        if alignment < 0.999 {
                            continue;
                        }
                        let delta = other.mean_w - surfel.mean_w;
                        if normal.dot(&delta).abs() > 0.05 {
                            continue;
                        }
                        if delta.norm() > 0.6 {
                            continue;
                        }
                        count += 1;
                    }
                }
            }
        }
        count
    }

    pub fn insert(&mut self, points: impl Iterator<Item = PointXYZI>) -> Result<()> {
        let radius = self.surfel_map_config.search_radius;
        for point in points {
            if !point.is_valid() {
                continue;
            }
            let mut voxel_key = VoxelKey::new(&point, self.surfel_map_config.voxel_size)?;
            let x = voxel_key.x;
            let y = voxel_key.y;
            let z = voxel_key.z;
            let candidate_ids = (x - radius..=x + radius)
                .flat_map(move |x| {
                    (y - radius..=y + radius).flat_map(move |y| {
                        (z - radius..=z + radius).map(move |z| {
                            voxel_key.x = x;
                            voxel_key.y = y;
                            voxel_key.z = z;
                            voxel_key.pack()
                        })
                    })
                })
                .filter_map(|v| self.buckets.get(&v))
                .flat_map(|bucket| bucket.iter().copied());
            let mut best_mature: Option<(SurfelID, f64)> = None;
            let mut best_growing: Option<(SurfelID, f64)> = None;

            let mut seen_roots = HashSet::new();

            for candidate_id in candidate_ids {
                // Stale children resolve to their root; each root is tested once.
                let (id, _depth) = self.resolve_root(candidate_id);
                if !seen_roots.insert(id) {
                    continue;
                }
                let Some(s) = self.surfels.get(id) else {
                    continue;
                };
                if s.is_growing(self.surfel_config.min_mature_surfel_count) {
                    if !s.within_growing_radius(&point, &self.surfel_config) {
                        continue;
                    }
                    let score = (point.to_vec3().cast::<f64>() - s.mean_w).norm_squared();
                    if best_growing.is_none_or(|(_, best)| score < best) {
                        best_growing = Some((id, score));
                    }
                } else {
                    if !s.within_support(&point, &self.surfel_config) {
                        continue;
                    }
                    let score = s.plane_distance(&point) as f64;
                    if best_mature.is_none_or(|(_, best)| score < best) {
                        best_mature = Some((id, score));
                    }
                }
            }

            if let Some((best_id, _)) = best_mature {
                self.update_surfel(best_id, &point)?;
            } else if let Some((best_id, _)) = best_growing {
                self.update_surfel(best_id, &point)?;
            } else {
                let _id = self.create_surfel(&point)?;
            }
        }
        Ok(())
    }

    fn create_surfel(&mut self, point: &PointXYZI) -> Result<SurfelID> {
        let mut surfel = Surfel::from_first_point(point);
        surfel.created_time_sec = self.time_sec;
        let id = self.surfels.insert(surfel);
        self.reindex_surfel(id)?;
        self.enqueue_merge_candidate(id);
        Ok(id)
    }

    fn update_surfel(&mut self, id: SurfelID, point: &PointXYZI) -> Result<()> {
        // best_id is selected from candidate_ids, which only yields ids that were
        let surfel = self.surfels.get_mut(id).expect(
            "UnexpectedError: on update: Logically, there should be no missing value here.",
        );
        let mut needs_reindex = { surfel.is_growing(self.surfel_config.min_mature_surfel_count) };

        surfel.count += 1;
        let delta = point.to_vec3().cast::<f64>() - surfel.mean_w;
        let n = surfel.count as f64;
        surfel.mean_w += delta / n;
        surfel.m2 += delta * delta.transpose() * ((n - 1.0) / n);
        if surfel.count >= self.surfel_config.min_mature_surfel_count
            && (surfel.count - surfel.last_refit) > helper_interval(surfel.count)
        {
            surfel.refit_from_m2();
            needs_reindex = true;
        }

        if needs_reindex {
            self.reindex_surfel(id)?;
            // Only refit/creation makes a surfel a new merge candidate; queuing
            // on every point update would flood the maintenance queue.
            self.enqueue_merge_candidate(id);
        }

        Ok(())
    }

    /// Two mature surfels describe the same plane when their normals align,
    /// their offsets agree and they are close enough to belong to one patch.
    fn coplanar_pair(a: &Surfel, b: &Surfel, config: &SurfelConfig) -> bool {
        let normal_a = a.eigenvectors.column(0);
        let normal_b = b.eigenvectors.column(0);
        if normal_a.dot(&normal_b).abs() < config.merge_max_normal_alignment as f64 {
            return false;
        }
        let delta = b.mean_w - a.mean_w;
        if normal_a.dot(&delta).abs() > config.merge_max_offset as f64 {
            return false;
        }
        delta.norm() <= config.merge_max_distance as f64
    }

    /// Enqueue a surfel root for deferred coplanar maintenance. The neighbour
    /// scan happens later in [`Self::process_maintenance`], so the insert path
    /// only pays an O(1) queue push.
    fn enqueue_merge_candidate(&mut self, id: SurfelID) {
        if !self.surfel_config.merge_enabled {
            return;
        }
        if self.merge_queue.len() >= self.surfel_config.merge_queue_limit {
            return;
        }
        let (root, _) = self.resolve_root(id);
        let Some(surfel) = self.surfels.get_mut(root) else {
            return;
        };
        if surfel.is_growing(self.surfel_config.min_mature_surfel_count) || surfel.merge_queued {
            return;
        }
        surfel.merge_queued = true;
        self.merge_queue.push_back(root);
    }

    /// Coplanar root neighbours of a surfel, scanned at maintenance time.
    fn coplanar_neighbours(&self, root: SurfelID, surfel: &Surfel) -> Vec<SurfelID> {
        const MAX_TARGETS: usize = 8;
        let Ok(voxel_key) = VoxelKey::from_vec3(
            surfel.mean_w.cast(),
            self.surfel_map_config.voxel_size,
        ) else {
            return Vec::new();
        };
        let scan_radius = ((self.surfel_config.merge_max_distance
            / self.surfel_map_config.voxel_size)
            .ceil() as i32)
            .max(1);
        let mut targets = Vec::new();
        let mut seen = HashSet::new();
        for vx in voxel_key.x - scan_radius..=voxel_key.x + scan_radius {
            for vy in voxel_key.y - scan_radius..=voxel_key.y + scan_radius {
                for vz in voxel_key.z - scan_radius..=voxel_key.z + scan_radius {
                    let Some(bucket) = self.buckets.get(&VoxelKey { x: vx, y: vy, z: vz }.pack())
                    else {
                        continue;
                    };
                    for other_id in bucket.iter().copied() {
                        let (other, _) = self.resolve_root(other_id);
                        if other == root || !seen.insert(other) {
                            continue;
                        }
                        let Some(other_surfel) = self.surfels.get(other) else {
                            continue;
                        };
                        if other_surfel.is_growing(self.surfel_config.min_mature_surfel_count) {
                            continue;
                        }
                        if Self::coplanar_pair(surfel, other_surfel, &self.surfel_config) {
                            targets.push(other);
                            if targets.len() >= MAX_TARGETS {
                                return targets;
                            }
                        }
                    }
                }
            }
        }
        targets
    }

    /// Remove a surfel id from all indexed voxel buckets.
    fn detach_from_index(&mut self, id: SurfelID) {
        let Some(old_keys) = self.surfels.get(id).map(|s| s.indexed_voxels.clone()) else {
            return;
        };
        for key in old_keys {
            let empty = if let Some(bucket) = self.buckets.get_mut(&key) {
                bucket.retain(|surfel_id| *surfel_id != id);
                bucket.is_empty()
            } else {
                false
            };
            if empty {
                self.buckets.remove(&key);
            }
        }
        if let Some(surfel) = self.surfels.get_mut(id) {
            surfel.indexed_voxels.clear();
        }
    }

    /// Run deferred maintenance: consume up to `maintenance_max_tasks` dirty
    /// roots per frame, scanning their neighbourhood and merging coplanar
    /// surfels. All neighbour scans and reindexing happen here, off the hot path.
    pub fn process_maintenance(&mut self) -> MaintenanceStats {
        let mut merged = 0usize;
        if self.surfel_config.merge_enabled {
            let budget = self.surfel_config.maintenance_max_tasks;
            let max_depth = self.surfel_config.maintenance_max_depth;
            let min_mature = self.surfel_config.min_mature_surfel_count;
            for _ in 0..budget {
                let Some(dirty) = self.merge_queue.pop_front() else {
                    break;
                };
                let (root, depth) = self.resolve_root(dirty);
                self.max_chain_depth = self.max_chain_depth.max(depth);
                if let Some(dirty_root) = self.surfels.get_mut(root) {
                    dirty_root.merge_queued = false;
                }
                if depth >= max_depth {
                    continue;
                }
                let Some(surfel) = self.surfels.get(root).cloned() else {
                    continue;
                };
                if surfel.is_growing(min_mature) {
                    continue;
                }
                let targets = self.coplanar_neighbours(root, &surfel);
                if targets.is_empty() {
                    continue;
                }
                let mut current_root = root;
                for target in targets {
                    let (target_root, target_depth) = self.resolve_root(target);
                    self.max_chain_depth = self.max_chain_depth.max(target_depth);
                    if target_root == current_root || target_depth >= max_depth {
                        continue;
                    }
                    let Some(current_surfel) = self.surfels.get(current_root) else {
                        break;
                    };
                    let Some(target_surfel) = self.surfels.get(target_root) else {
                        continue;
                    };
                    if target_surfel.is_growing(min_mature)
                        || !Self::coplanar_pair(current_surfel, target_surfel, &self.surfel_config)
                    {
                        continue;
                    }
                    // Heuristic root choice: prefer the root covering more
                    // voxels, then the one that has been alive longer.
                    let (new_root, child) = if current_surfel.indexed_voxels.len()
                        > target_surfel.indexed_voxels.len()
                        || (current_surfel.indexed_voxels.len() == target_surfel.indexed_voxels.len()
                            && current_surfel.created_time_sec <= target_surfel.created_time_sec)
                    {
                        (current_root, target_root)
                    } else {
                        (target_root, current_root)
                    };
                    let Some(child_surfel) = self.surfels.get(child).cloned() else {
                        continue;
                    };
                    {
                        let Some(root_surfel) = self.surfels.get_mut(new_root) else {
                            continue;
                        };
                        root_surfel.merge_from(&child_surfel);
                    }
                    self.detach_from_index(child);
                    if let Some(child_surfel) = self.surfels.get_mut(child) {
                        child_surfel.parent = Some(new_root);
                    }
                    if self.reindex_surfel(new_root).is_err() {
                        continue;
                    }
                    if let Some(merged_root) = self.surfels.get_mut(new_root) {
                        merged_root.merge_queued = false;
                    }
                    current_root = new_root;
                    merged += 1;
                    self.merged_total += 1;
                }
            }
        }
        MaintenanceStats {
            merged,
            queue_len: self.merge_queue.len(),
            max_chain_depth: self.max_chain_depth,
            merged_total: self.merged_total,
        }
    }

    pub fn maintenance_stats(&self) -> MaintenanceStats {
        MaintenanceStats {
            merged: 0,
            queue_len: self.merge_queue.len(),
            max_chain_depth: self.max_chain_depth,
            merged_total: self.merged_total,
        }
    }

    fn reindex_surfel(&mut self, id: SurfelID) -> Result<()> {
        let (old_keys, new_keys) = {
            let surfel = self.surfels.get(id).expect("surfel must exist");

            let extent = if surfel.is_growing(self.surfel_config.min_mature_surfel_count) {
                Vec3::repeat(self.surfel_config.growing_radius as f64)
            } else {
                surfel.support_aabb_extent(&self.surfel_config)
            };

            let min_w = surfel.mean_w - extent;
            let max_w = surfel.mean_w + extent;

            let min_key = VoxelKey::from_vec3(min_w.cast(), self.surfel_map_config.voxel_size)?;
            let max_key = VoxelKey::from_vec3(max_w.cast(), self.surfel_map_config.voxel_size)?;

            let mut new_keys = SmallVec::<[u64; 8]>::new();

            for x in min_key.x..=max_key.x {
                for y in min_key.y..=max_key.y {
                    for z in min_key.z..=max_key.z {
                        new_keys.push(VoxelKey { x, y, z }.pack());
                    }
                }
            }

            (surfel.indexed_voxels.clone(), new_keys)
        };

        if old_keys == new_keys {
            return Ok(());
        }

        for &key in &old_keys {
            let should_remove = if let Some(bucket) = self.buckets.get_mut(&key) {
                bucket.retain(|sid| *sid != id);
                bucket.is_empty()
            } else {
                false
            };
            if should_remove {
                self.buckets.remove(&key);
            }
        }

        for &key in &new_keys {
            self.buckets.entry(key).or_default().push(id);
        }

        self.surfels
            .get_mut(id)
            .expect("surfel must exist")
            .indexed_voxels = new_keys;

        Ok(())
    }

    // pub fn merge_compatible_mature_surfels(&mut self) -> Result<()> {

    // }

    pub fn surfel_map_config(&self) -> &SurfelMapConfig {
        &self.surfel_map_config
    }

    pub fn surfel_config(&self) -> &SurfelConfig {
        &self.surfel_config
    }

    pub fn surfels(&self) -> impl Iterator<Item = &Surfel> {
        self.surfels.values()
    }

    pub fn is_empty(&self) -> bool {
        self.surfels.is_empty()
    }
}

fn line_association_is_ambiguous(best_distance: f64, second_best_distance: f64) -> bool {
    if !best_distance.is_finite() || !second_best_distance.is_finite() {
        return true;
    }
    if second_best_distance <= best_distance {
        return true;
    }
    let margin = second_best_distance - best_distance;
    let ratio = best_distance / second_best_distance;
    margin < LINE_SECOND_BEST_MIN_DISTANCE_MARGIN || ratio > LINE_SECOND_BEST_MAX_DISTANCE_RATIO
}
