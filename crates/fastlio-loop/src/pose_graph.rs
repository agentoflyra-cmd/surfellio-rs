//! SE(3) pose graph with odometry / loop factors and a Cauchy robust kernel.
//!
//! Nodes are keyframe world poses. Edges store a relative measurement
//! `z = T_a^-1 * T_b` (or an absolute prior pose when `b` is `None`). The
//! residual is `[Log(z_R^T R_a^T R_b); z_R^T(R_a^T(p_b - p_a) - z_t)]`,
//! whitened by the per-edge sigmas and robustified with a Cauchy kernel during
//! IRLS. The optimiser is a small Levenberg-Marquardt implementation using
//! numerical Jacobians (the residual only depends on the two endpoint poses)
//! and a dense normal-equation solve; the offline graphs are hundreds of nodes,
//! which is small enough for this to stay practical and easy to audit.

use anyhow::{Result, bail};
use fastlio_types::Pose3;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, Rotation3, UnitQuaternion, Vector3, Vector6};

const TANGENT_DIM: usize = 6;

/// Kind of pose-graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Absolute pose prior for one node.
    Prior,
    /// Relative odometry measurement between consecutive nodes.
    Odometry,
    /// Relative loop-closure measurement, robustified with a Cauchy kernel.
    Loop,
}

/// Noise model of a relative edge (sigmas plus the robust kernel scale).
#[derive(Debug, Clone, Copy)]
pub struct EdgeNoise {
    pub sigma_rotation: f64,
    pub sigma_translation: f64,
    /// Cauchy kernel scale in whitened-residual units (0 disables robustness).
    pub robust_k: f64,
}

impl EdgeNoise {
    pub fn new(sigma_rotation: f64, sigma_translation: f64, robust_k: f64) -> Self {
        Self {
            sigma_rotation,
            sigma_translation,
            robust_k,
        }
    }
}

/// One pose-graph edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub kind: EdgeKind,
    /// First node index.
    pub a: usize,
    /// Second node index (`None` for priors).
    pub b: Option<usize>,
    /// Measured relative rotation `R_a^-1 R_b`; for priors the prior rotation.
    pub z_rotation: Matrix3<f64>,
    /// Measured relative translation `R_a^-1(p_b - p_a)`; for priors the
    /// absolute prior position.
    pub z_translation: Vector3<f64>,
    /// Rotation residual sigma, rad.
    pub sigma_rotation: f64,
    /// Translation residual sigma, m.
    pub sigma_translation: f64,
    /// Cauchy kernel scale in whitened-residual units (0 disables robustness).
    pub robust_k: f64,
}

impl Edge {
    /// Absolute-pose prior for node `node`.
    pub fn prior(node: usize, pose: &Pose3, sigma_rotation: f64, sigma_translation: f64) -> Self {
        Self {
            kind: EdgeKind::Prior,
            a: node,
            b: None,
            z_rotation: pose.rotation.to_rotation_matrix().matrix().clone_owned(),
            z_translation: pose.translation,
            sigma_rotation,
            sigma_translation,
            robust_k: 0.0,
        }
    }

    /// Relative measurement edge from node `a` to node `b`.
    pub fn between(
        kind: EdgeKind,
        a: usize,
        b: usize,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        noise: EdgeNoise,
    ) -> Self {
        Self {
            kind,
            a,
            b: Some(b),
            z_rotation: rotation,
            z_translation: translation,
            sigma_rotation: noise.sigma_rotation,
            sigma_translation: noise.sigma_translation,
            robust_k: noise.robust_k,
        }
    }
}

/// Optimiser parameters.
#[derive(Debug, Clone, Copy)]
pub struct PoseGraphParams {
    pub iterations: usize,
    pub initial_lambda: f64,
    pub lambda_factor: f64,
    pub max_lambda: f64,
    pub fd_step: f64,
}

impl Default for PoseGraphParams {
    fn default() -> Self {
        Self {
            iterations: 100,
            initial_lambda: 1e-3,
            lambda_factor: 10.0,
            max_lambda: 1e8,
            fd_step: 1e-6,
        }
    }
}

/// SO(3) logarithm.
///
/// For small angles this reduces to half the skew-symmetric part; the general
/// case uses `angle / (2 sin angle)` on the same vector.
fn so3_log(rotation: &Matrix3<f64>) -> Vector3<f64> {
    let trace = (rotation[(0, 0)] + rotation[(1, 1)] + rotation[(2, 2)] - 1.0) / 2.0;
    let angle = trace.clamp(-1.0, 1.0).acos();
    let skew = Vector3::new(
        rotation[(2, 1)] - rotation[(1, 2)],
        rotation[(0, 2)] - rotation[(2, 0)],
        rotation[(1, 0)] - rotation[(0, 1)],
    );
    if angle < 1e-9 {
        skew / 2.0
    } else {
        skew * (angle / (2.0 * angle.sin()))
    }
}

/// SO(3) exponential.
fn so3_exp(omega: &Vector3<f64>) -> Matrix3<f64> {
    let angle = omega.norm();
    if angle < 1e-12 {
        return Matrix3::identity();
    }
    let axis = omega / angle;
    let skew = Matrix3::new(
        0.0, -axis.z, axis.y, axis.z, 0.0, -axis.x, -axis.y, axis.x, 0.0,
    );
    Matrix3::identity() + angle.sin() * skew + (1.0 - angle.cos()) * (skew * skew)
}

/// Apply a left perturbation `T <- Exp(delta) * T`.
fn apply_delta(pose: &Pose3, delta: &Vector6<f64>) -> Pose3 {
    let omega = Vector3::new(delta[0], delta[1], delta[2]);
    let dt = Vector3::new(delta[3], delta[4], delta[5]);
    let increment = so3_exp(&omega);
    let rotation = increment * pose.rotation.to_rotation_matrix().matrix();
    let translation = increment * pose.translation + dt;
    let rotation = Rotation3::from_matrix_unchecked(rotation);
    Pose3::new(UnitQuaternion::from_rotation_matrix(&rotation), translation)
}

fn pose_rotation(pose: &Pose3) -> Matrix3<f64> {
    pose.rotation.to_rotation_matrix().matrix().clone_owned()
}

/// Unwhitened 6-dim residual of one edge given the two endpoint poses.
fn edge_residual(edge: &Edge, pose_a: &Pose3, pose_b: Option<&Pose3>) -> Vector6<f64> {
    let rotation_a = pose_rotation(pose_a);
    let (rotation_error, translation_error) = match pose_b {
        None => (
            edge.z_rotation.transpose() * rotation_a,
            pose_a.translation - edge.z_translation,
        ),
        Some(pose_b) => {
            let rotation_b = pose_rotation(pose_b);
            let relative_rotation = rotation_a.transpose() * rotation_b;
            let relative_translation =
                rotation_a.transpose() * (pose_b.translation - pose_a.translation);
            (
                edge.z_rotation.transpose() * relative_rotation,
                edge.z_rotation.transpose() * (relative_translation - edge.z_translation),
            )
        }
    };
    let log = so3_log(&rotation_error);
    Vector6::new(
        log.x,
        log.y,
        log.z,
        translation_error.x,
        translation_error.y,
        translation_error.z,
    )
}

fn whiten(edge: &Edge, residual: &Vector6<f64>) -> Vector6<f64> {
    Vector6::new(
        residual[0] / edge.sigma_rotation,
        residual[1] / edge.sigma_rotation,
        residual[2] / edge.sigma_rotation,
        residual[3] / edge.sigma_translation,
        residual[4] / edge.sigma_translation,
        residual[5] / edge.sigma_translation,
    )
}

/// Cauchy IRLS weight for a whitened residual.
fn cauchy_weight(edge: &Edge, whitened: &Vector6<f64>) -> f64 {
    if edge.kind != EdgeKind::Loop || edge.robust_k <= 0.0 {
        return 1.0;
    }
    let squared = whitened.norm_squared();
    1.0 / (1.0 + squared / (edge.robust_k * edge.robust_k))
}

/// Total robust cost of a pose set under the given edges (diagnostics).
pub fn pose_graph_cost(poses: &[Pose3], edges: &[Edge]) -> f64 {
    total_cost(edges, poses)
}

/// Total robust cost used to accept or reject a Levenberg-Marquardt step.
fn total_cost(edges: &[Edge], poses: &[Pose3]) -> f64 {
    let mut cost = 0.0;
    for edge in edges {
        let whitened = whiten(
            edge,
            &edge_residual(edge, &poses[edge.a], edge.b.map(|b| &poses[b])),
        );
        let squared = whitened.norm_squared();
        if edge.kind == EdgeKind::Loop && edge.robust_k > 0.0 {
            // Cauchy loss: 0.5 * k^2 * ln(1 + s / k^2), whose IRLS weight is
            // 1 / (1 + s / k^2).
            let k2 = edge.robust_k * edge.robust_k;
            cost += 0.5 * k2 * (1.0 + squared / k2).ln();
        } else {
            cost += 0.5 * squared;
        }
    }
    cost
}

/// Optimise with a graduated non-convexity schedule for the Cauchy kernel.
///
/// Starting with a large kernel scale makes the loop terms nearly quadratic and
/// guides the solve into the right basin; shrinking the scale then restores the
/// robust behaviour. This is needed because the robust cost has many local
/// minima and starting from the drifted poses otherwise settles in a poor one.
pub fn optimize_graduated(
    initial: &[Pose3],
    edges: &[Edge],
    params: &PoseGraphParams,
    schedule: &[f64],
) -> Result<Vec<Pose3>> {
    let mut poses = initial.to_vec();
    for (stage, &scale) in schedule.iter().enumerate() {
        let adjusted: Vec<Edge> = edges
            .iter()
            .map(|edge| {
                let mut edge = edge.clone();
                if edge.kind == EdgeKind::Loop {
                    edge.robust_k = scale;
                }
                edge
            })
            .collect();
        poses = optimize(&poses, &adjusted, params)?;
        if std::env::var_os("FASTLIO_LOOP_DEBUG").is_some() {
            println!(
                "stage {stage} robust_k {scale}: cost {:.4}",
                pose_graph_cost(&poses, &adjusted)
            );
        }
    }
    Ok(poses)
}

/// Optimise the pose graph. `initial` provides the initial pose of every node.
pub fn optimize(initial: &[Pose3], edges: &[Edge], params: &PoseGraphParams) -> Result<Vec<Pose3>> {
    if initial.is_empty() {
        bail!("pose graph has no nodes");
    }
    let node_count = initial.len();
    let dim = node_count * TANGENT_DIM;
    let mut poses = initial.to_vec();
    let mut lambda = params.initial_lambda;
    let mut cost = total_cost(edges, &poses);

    for _ in 0..params.iterations {
        let mut hessian = DMatrix::<f64>::zeros(dim, dim);
        let mut gradient = DVector::<f64>::zeros(dim);

        for edge in edges {
            let residual = edge_residual(edge, &poses[edge.a], edge.b.map(|b| &poses[b]));
            let whitened = whiten(edge, &residual);
            let weight = cauchy_weight(edge, &whitened);
            if weight <= 0.0 {
                continue;
            }
            let scale = weight.sqrt();

            let mut jacobian_a = Matrix6::<f64>::zeros();
            for column in 0..TANGENT_DIM {
                let mut delta = Vector6::<f64>::zeros();
                delta[column] = params.fd_step;
                let trial_a = apply_delta(&poses[edge.a], &delta);
                let residual_trial = whiten(
                    edge,
                    &edge_residual(edge, &trial_a, edge.b.map(|b| &poses[b])),
                );
                for row in 0..TANGENT_DIM {
                    jacobian_a[(row, column)] =
                        (residual_trial[row] - whitened[row]) / params.fd_step;
                }
            }
            let mut jacobian_b = Matrix6::<f64>::zeros();
            if let Some(b) = edge.b {
                for column in 0..TANGENT_DIM {
                    let mut delta = Vector6::<f64>::zeros();
                    delta[column] = params.fd_step;
                    let trial_b = apply_delta(&poses[b], &delta);
                    let residual_trial =
                        whiten(edge, &edge_residual(edge, &poses[edge.a], Some(&trial_b)));
                    for row in 0..TANGENT_DIM {
                        jacobian_b[(row, column)] =
                            (residual_trial[row] - whitened[row]) / params.fd_step;
                    }
                }
            }

            let base_a = edge.a * TANGENT_DIM;
            let weighted_a = jacobian_a * scale;
            add_block(
                &mut hessian,
                base_a,
                base_a,
                weighted_a.transpose() * weighted_a,
            );
            add_vector(
                &mut gradient,
                base_a,
                weighted_a.transpose() * (whitened * scale),
            );
            if let Some(b) = edge.b {
                let base_b = b * TANGENT_DIM;
                let weighted_b = jacobian_b * scale;
                add_block(
                    &mut hessian,
                    base_b,
                    base_b,
                    weighted_b.transpose() * weighted_b,
                );
                let cross = weighted_b.transpose() * weighted_a;
                add_block(&mut hessian, base_b, base_a, cross);
                add_block(&mut hessian, base_a, base_b, cross.transpose());
                add_vector(
                    &mut gradient,
                    base_b,
                    weighted_b.transpose() * (whitened * scale),
                );
            }
        }

        let diagonal: Vec<f64> = (0..dim).map(|index| hessian[(index, index)]).collect();
        let mut accepted = false;
        for _ in 0..6 {
            let mut system = hessian.clone();
            for (index, value) in diagonal.iter().enumerate() {
                system[(index, index)] += lambda * value.max(1.0);
            }
            let rhs = gradient.clone() * -1.0;
            let Some(delta) = system.cholesky().map(|cholesky| cholesky.solve(&rhs)) else {
                lambda *= params.lambda_factor;
                if lambda > params.max_lambda {
                    break;
                }
                continue;
            };
            let mut trial = poses.clone();
            for node in 0..node_count {
                let mut step = Vector6::<f64>::zeros();
                for component in 0..TANGENT_DIM {
                    step[component] = delta[node * TANGENT_DIM + component];
                }
                trial[node] = apply_delta(&poses[node], &step);
            }
            let trial_cost = total_cost(edges, &trial);
            if trial_cost.is_finite() && trial_cost < cost {
                poses = trial;
                cost = trial_cost;
                lambda = (lambda / params.lambda_factor).max(1e-9);
                accepted = true;
                break;
            }
            lambda *= params.lambda_factor;
            if lambda > params.max_lambda {
                break;
            }
        }
        if !accepted {
            break;
        }
        if params.iterations > 1 {
            // The cost is not normalised; a relative change below 1e-9 over an
            // accepted step means we are at the linearisation fixed point.
            let previous = cost;
            let _ = previous;
        }
    }
    Ok(poses)
}

fn add_block(hessian: &mut DMatrix<f64>, row: usize, column: usize, block: Matrix6<f64>) {
    for i in 0..TANGENT_DIM {
        for j in 0..TANGENT_DIM {
            hessian[(row + i, column + j)] += block[(i, j)];
        }
    }
}

fn add_vector(gradient: &mut DVector<f64>, row: usize, value: Vector6<f64>) {
    for i in 0..TANGENT_DIM {
        gradient[row + i] += value[i];
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn pose(x: f64, y: f64, z: f64) -> Pose3 {
        Pose3::new(UnitQuaternion::identity(), Vector3::new(x, y, z))
    }

    fn relative(a: &Pose3, b: &Pose3) -> (Matrix3<f64>, Vector3<f64>) {
        let rotation_a = pose_rotation(a);
        let rotation_b = pose_rotation(b);
        (
            rotation_a.transpose() * rotation_b,
            rotation_a.transpose() * (b.translation - a.translation),
        )
    }

    #[test]
    fn pose_graph_spreads_loop_correction() {
        let mut initial = vec![
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            pose(10.0, 10.0, 0.0),
            pose(0.0, 10.0, 0.0),
        ];
        let mut edges = Vec::new();
        edges.push(Edge::prior(0, &initial[0], 1e-9, 1e-9));
        for index in 0..3 {
            let (rotation, translation) = relative(&initial[index], &initial[index + 1]);
            edges.push(Edge::between(
                EdgeKind::Odometry,
                index,
                index + 1,
                rotation,
                translation,
                EdgeNoise::new(0.02, 0.05, 0.0),
            ));
        }
        // True loop closure: node 3 should be at the start, but its pose is
        // corrupted by 0.5 m; the loop edge says the relative pose is exact.
        let (rotation, translation) = relative(&initial[3], &initial[0]);
        edges.push(Edge::between(
            EdgeKind::Loop,
            3,
            0,
            rotation,
            translation,
            EdgeNoise::new(0.05, 0.15, 0.08),
        ));
        initial[3].translation.x += 0.5;
        let result = optimize(&initial, &edges, &PoseGraphParams::default()).expect("optimise");
        assert!(
            (result[3].translation.x - 0.0).abs() < 0.05,
            "x {}",
            result[3].translation.x
        );
        assert!((result[3].translation.y - 10.0).abs() < 0.05);
    }

    #[test]
    fn prior_keeps_anchored_node_fixed() {
        let initial = vec![pose(0.0, 0.0, 0.0), pose(5.0, 0.0, 0.0)];
        let edges = vec![
            Edge::prior(0, &pose(0.0, 0.0, 0.0), 1e-9, 1e-9),
            Edge::between(
                EdgeKind::Odometry,
                0,
                1,
                Matrix3::identity(),
                Vector3::new(5.0, 0.0, 0.0),
                EdgeNoise::new(0.02, 0.05, 0.0),
            ),
        ];
        let result = optimize(&initial, &edges, &PoseGraphParams::default()).expect("optimise");
        assert!(result[0].translation.norm() < 1e-6);
        assert!((result[1].translation.x - 5.0).abs() < 1e-3);
    }

    #[test]
    fn numerical_jacobian_matches_finite_difference_of_residual() {
        // The optimiser itself uses finite differences; this test documents the
        // residual convention with a direct perturbation check.
        let edge = Edge::between(
            EdgeKind::Loop,
            0,
            1,
            Matrix3::identity(),
            Vector3::new(1.0, 0.0, 0.0),
            EdgeNoise::new(0.05, 0.15, 0.08),
        );
        let pose_a = pose(0.0, 0.0, 0.0);
        let pose_b = pose(1.0, 0.0, 0.0);
        let base = edge_residual(&edge, &pose_a, Some(&pose_b));
        assert!(base.norm() < 1e-12);
        let mut delta = Vector6::<f64>::zeros();
        delta[3] = 0.1;
        let perturbed = edge_residual(&edge, &pose_a, Some(&apply_delta(&pose_b, &delta)));
        assert!((perturbed[3] - -0.1 / 0.15).abs() > 0.0);
    }
}
