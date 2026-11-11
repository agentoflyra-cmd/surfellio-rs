//! Keyframe loading for the offline loop-closure pipeline.
//!
//! Binary layout written by `fastlio-pipeline` (`--keyframe-dir`):
//!
//! ```text
//! u64 index (LE)
//! f64 timestamp seconds (LE)
//! 3 x f64 position (LE)
//! 4 x f64 quaternion w, i, j, k (LE)
//! u32 point count (LE)
//! count x 3 x f32 world-frame points (LE)
//! ```
//!
//! Points are stored in the world frame (as produced during replay), which is
//! what the Scan Context descriptor and the ICP stage expect.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use fastlio_types::PointXYZI;
use nalgebra::{UnitQuaternion, Vector3};

/// One exported keyframe.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Sequential index of the keyframe (the `--keyframe-stride` counter).
    pub index: u64,
    /// Timestamp of the frame in seconds.
    pub timestamp_sec: f64,
    /// World position of the IMU/body frame.
    pub position: Vector3<f64>,
    /// World orientation (rotation world <- body).
    pub rotation: UnitQuaternion<f64>,
    /// World-frame points observed at this keyframe.
    pub points: Vec<PointXYZI>,
}

impl Keyframe {
    /// Transform a world-frame point into the body frame of this keyframe.
    pub fn transform_point_world_to_body(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.inverse() * (point - self.position)
    }

    /// Transform a body-frame point into the world frame of this keyframe.
    pub fn transform_point_body_to_world(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * point + self.position
    }

    /// World-frame point of a stored cloud point.
    pub fn point_world(&self, point: &PointXYZI) -> Vector3<f64> {
        Vector3::new(point.x as f64, point.y as f64, point.z as f64)
    }
}

/// Load every `kf_*.bin` file from `directory`, sorted by file name.
pub fn load_keyframes(directory: impl AsRef<Path>) -> Result<Vec<Keyframe>> {
    let directory = directory.as_ref();
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read keyframe directory {}", directory.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", directory.display()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        bail!("no keyframe .bin files found in {}", directory.display());
    }
    let mut keyframes = Vec::with_capacity(paths.len());
    for path in paths {
        keyframes.push(
            load_keyframe(&path).with_context(|| format!("failed to load {}", path.display()))?,
        );
    }
    Ok(keyframes)
}

fn load_keyframe(path: &Path) -> Result<Keyframe> {
    let bytes = fs::read(path)?;
    let mut reader = Reader::new(&bytes);
    let index = reader.read_u64()?;
    let timestamp_sec = reader.read_f64()?;
    let position = Vector3::new(reader.read_f64()?, reader.read_f64()?, reader.read_f64()?);
    let w = reader.read_f64()?;
    let i = reader.read_f64()?;
    let j = reader.read_f64()?;
    let k = reader.read_f64()?;
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, i, j, k));
    let count = reader.read_u32()? as usize;
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        let x = reader.read_f32()?;
        let y = reader.read_f32()?;
        let z = reader.read_f32()?;
        points.push(PointXYZI {
            x,
            y,
            z,
            intensity: 0.0,
        });
    }
    if !reader.is_empty() {
        bail!(
            "trailing {} bytes in {}",
            reader.remaining(),
            path.display()
        );
    }
    Ok(Keyframe {
        index,
        timestamp_sec,
        position,
        rotation,
        points,
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.offset + N > self.bytes.len() {
            bail!("unexpected end of keyframe file");
        }
        let mut buffer = [0u8; N];
        buffer.copy_from_slice(&self.bytes[self.offset..self.offset + N]);
        self.offset += N;
        Ok(buffer)
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take()?))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn is_empty(&self) -> bool {
        self.offset >= self.bytes.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn write_keyframe(
        path: &Path,
        index: u64,
        t: f64,
        pos: [f64; 3],
        quat: [f64; 4],
        points: &[(f32, f32, f32)],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&index.to_le_bytes());
        bytes.extend_from_slice(&t.to_le_bytes());
        for value in pos {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in quat {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&(points.len() as u32).to_le_bytes());
        for (x, y, z) in points {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&z.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn loads_keyframe_and_transforms_points_between_frames() {
        let dir = std::env::temp_dir().join(format!("fastlio-loop-kf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kf_000000.bin");
        // Body frame rotated by 90 deg around z, positioned at (10, 0, 0).
        let half = std::f64::consts::FRAC_1_SQRT_2;
        write_keyframe(
            &path,
            0,
            12.5,
            [10.0, 0.0, 0.0],
            [half, 0.0, 0.0, half],
            &[(1.0, 2.0, 3.0)],
        );
        let keyframe = load_keyframe(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(keyframe.index, 0);
        assert!((keyframe.timestamp_sec - 12.5).abs() < 1e-12);
        assert!((keyframe.position - Vector3::new(10.0, 0.0, 0.0)).norm() < 1e-12);

        let body = Vector3::new(1.0, 0.0, 0.0);
        let world = keyframe.transform_point_body_to_world(&body);
        let recovered = keyframe.transform_point_world_to_body(&world);
        assert!((recovered - body).norm() < 1e-12);
    }
}
