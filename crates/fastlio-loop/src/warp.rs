//! Full-rate trajectory warping with the keyframe loop correction.
//!
//! The pose graph produces corrected keyframe poses `T'_k`; the implied
//! correction field is `dT_k = T'_k T_k^-1`, i.e.
//! `p_corrected = dR_k p + dt_k`. To warp a full-rate trajectory sample we
//! interpolate `dR` and `dt` between the bracketing keyframes (the rotation is
//! re-orthonormalised with an SVD) and apply the result to the sample position.

use anyhow::{Context, Result, bail};
use fastlio_types::Pose3;
use nalgebra::{Matrix3, Vector3};

use crate::keyframe::Keyframe;

/// Per-keyframe correction `p_corrected = d_rotation * p + d_translation`.
#[derive(Debug, Clone, Copy)]
pub struct Correction {
    pub d_rotation: Matrix3<f64>,
    pub d_translation: Vector3<f64>,
}

/// Build the per-keyframe corrections from the drifted and corrected poses.
pub fn corrections(keyframes: &[Keyframe], corrected: &[Pose3]) -> Result<Vec<Correction>> {
    if keyframes.len() != corrected.len() {
        bail!(
            "corrections: {} keyframes but {} corrected poses",
            keyframes.len(),
            corrected.len()
        );
    }
    Ok(keyframes
        .iter()
        .zip(corrected)
        .map(|(keyframe, corrected_pose)| {
            let drifted_rotation = keyframe
                .rotation
                .to_rotation_matrix()
                .matrix()
                .clone_owned();
            let corrected_rotation = corrected_pose
                .rotation
                .to_rotation_matrix()
                .matrix()
                .clone_owned();
            let d_rotation = corrected_rotation * drifted_rotation.transpose();
            let d_translation = corrected_pose.translation - d_rotation * keyframe.position;
            Correction {
                d_rotation,
                d_translation,
            }
        })
        .collect())
}

/// Re-orthonormalise a rotation matrix whose determinant may be negative after
/// interpolation (`U V^T`, flipping the last singular direction if needed).
fn orthonormalise(matrix: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = matrix.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return *matrix;
    };
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        let mut u = u;
        u.column_mut(2).neg_mut();
        rotation = u * v_t;
    }
    rotation
}

/// Load corrected keyframe poses from a text file with one line per keyframe:
/// `index tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22`.
pub fn load_corrected_poses(path: impl AsRef<std::path::Path>) -> Result<Vec<Pose3>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut poses = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 13 {
            continue;
        }
        let translation = Vector3::new(
            fields[1].parse().context("pose tx")?,
            fields[2].parse().context("pose ty")?,
            fields[3].parse().context("pose tz")?,
        );
        let mut rotation = Matrix3::<f64>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                rotation[(row, column)] = fields[4 + row * 3 + column]
                    .parse()
                    .context("pose rotation")?;
            }
        }
        let rotation = nalgebra::Rotation3::from_matrix_unchecked(rotation);
        poses.push(Pose3::new(
            nalgebra::UnitQuaternion::from_rotation_matrix(&rotation),
            translation,
        ));
    }
    Ok(poses)
}

/// Warp one trajectory position at time `timestamp_sec`.
pub fn warp_position(
    timestamp_sec: f64,
    position: &Vector3<f64>,
    keyframe_times: &[f64],
    corrections: &[Correction],
) -> Vector3<f64> {
    if keyframe_times.len() < 2 || corrections.len() != keyframe_times.len() {
        return *position;
    }
    let upper = keyframe_times.partition_point(|&t| t < timestamp_sec);
    let upper = upper.clamp(1, keyframe_times.len() - 1);
    let lower = upper - 1;
    let span = keyframe_times[upper] - keyframe_times[lower];
    let alpha = if span > 0.0 {
        ((timestamp_sec - keyframe_times[lower]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let d_rotation = orthonormalise(
        &((1.0 - alpha) * corrections[lower].d_rotation + alpha * corrections[upper].d_rotation),
    );
    let d_translation =
        (1.0 - alpha) * corrections[lower].d_translation + alpha * corrections[upper].d_translation;
    d_rotation * position + d_translation
}

/// Warp a whole trajectory (`(timestamp, position)` samples).
pub fn warp_trajectory(
    samples: &[(f64, Vector3<f64>)],
    keyframe_times: &[f64],
    corrections: &[Correction],
) -> Vec<Vector3<f64>> {
    samples
        .iter()
        .map(|(timestamp, position)| {
            warp_position(*timestamp, position, keyframe_times, corrections)
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::keyframe::Keyframe;
    use fastlio_types::PointXYZI;
    use nalgebra::UnitQuaternion;

    fn keyframe(t: f64, position: Vector3<f64>) -> Keyframe {
        Keyframe {
            index: 0,
            timestamp_sec: t,
            position,
            rotation: UnitQuaternion::identity(),
            points: Vec::<PointXYZI>::new(),
        }
    }

    #[test]
    fn constant_correction_is_applied_to_all_samples() {
        let keyframes = vec![
            keyframe(0.0, Vector3::zeros()),
            keyframe(10.0, Vector3::new(10.0, 0.0, 0.0)),
        ];
        let corrected = vec![
            Pose3::new(UnitQuaternion::identity(), Vector3::new(0.0, 1.0, 0.0)),
            Pose3::new(UnitQuaternion::identity(), Vector3::new(10.0, 1.0, 0.0)),
        ];
        let corrections = corrections(&keyframes, &corrected).unwrap();
        let samples = vec![
            (0.0, Vector3::new(0.0, 0.0, 0.0)),
            (5.0, Vector3::new(5.0, 0.0, 0.0)),
            (10.0, Vector3::new(10.0, 0.0, 0.0)),
        ];
        let warped = warp_trajectory(&samples, &[0.0, 10.0], &corrections);
        assert!((warped[0] - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
        assert!((warped[1] - Vector3::new(5.0, 1.0, 0.0)).norm() < 1e-9);
        assert!((warped[2] - Vector3::new(10.0, 1.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn interpolation_between_keyframes_is_linear() {
        let keyframes = vec![
            keyframe(0.0, Vector3::zeros()),
            keyframe(10.0, Vector3::new(10.0, 0.0, 0.0)),
        ];
        let corrected = vec![
            Pose3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
            Pose3::new(UnitQuaternion::identity(), Vector3::new(10.0, 2.0, 0.0)),
        ];
        let corrections = corrections(&keyframes, &corrected).unwrap();
        let warped = warp_position(5.0, &Vector3::zeros(), &[0.0, 10.0], &corrections);
        assert!((warped - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
    }
}
