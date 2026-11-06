use std::{cmp::max, fs};

use anyhow::{Context, Result};
use camino::Utf8Path;
use fastlio_types::{ImuSample, LidarFrame, PointXYZI, TimedPoint, Vec3, transfer_from_header};
use memmap2::Mmap;
use ros2_dispatch::borrow_decode_message_by_schema;

#[derive(Debug)]
pub enum SensorEvent {
    Imu(ImuSample),
    Lidar(LidarFrame),
}

impl SensorEvent {
    pub fn timestamp_sec(&self) -> f64 {
        match self {
            SensorEvent::Imu(imu_sample) => imu_sample.time_stamp_sec,
            SensorEvent::Lidar(lidar_frame) => lidar_frame.base_timestamp_sec,
        }
    }

    pub fn dispatch(data: &[u8], schema_name: &str) -> Result<Self> {
        let result = borrow_decode_message_by_schema(schema_name, data)?;
        match result {
            ros2_dispatch::DecodedMessageBorrowed::LivoxRosDriver2CustomMsg(custom_msg) => {
                let mut max_offset = 0;
                let mut points: Vec<_> = custom_msg
                    .points
                    .iter()
                    .map(|e| {
                        max_offset = max(max_offset, e.offset_time);
                        TimedPoint {
                            offset_time_sec: e.offset_time as f64 / 1e9,
                            point: PointXYZI {
                                x: e.x,
                                y: e.y,
                                z: e.z,
                                intensity: f32::from(e.reflectivity),
                            },
                            tag: e.tag,
                            line: e.line,
                        }
                    })
                    .collect();
                sort_timepoint_by_offset(&mut points);
                let msg = LidarFrame::new(
                    custom_msg.timebase as f64 / 1e9,
                    custom_msg.timebase as f64 / 1e9 + max_offset as f64 / 1e9,
                    points,
                );
                Ok(Self::Lidar(msg))
            }
            ros2_dispatch::DecodedMessageBorrowed::SensorMsgsImu(imu) => {
                let msg = ImuSample {
                    time_stamp_sec: transfer_from_header(
                        imu.header.stamp.sec,
                        imu.header.stamp.nanosec,
                    ),
                    gyro: Vec3::<f64>::new(
                        imu.angular_velocity.x,
                        imu.angular_velocity.y,
                        imu.angular_velocity.z,
                    ),
                    accel: Vec3::<f64>::new(
                        imu.linear_acceleration.x,
                        imu.linear_acceleration.y,
                        imu.linear_acceleration.z,
                    ),
                };
                Ok(Self::Imu(msg))
            }
            ros2_dispatch::DecodedMessageBorrowed::SensorMsgsPointCloud2(cloud) => {
                Ok(Self::Lidar(decode_pointcloud2_frame(&cloud)?))
            }
        }
    }
}

/// Point layout of one `sensor_msgs/PointCloud2` message: byte offset and
/// datatype for the fields we consume.
struct PointCloudFields {
    x: (usize, u8),
    y: (usize, u8),
    z: (usize, u8),
    intensity: Option<(usize, u8)>,
    time: Option<(usize, u8)>,
    line: Option<(usize, u8)>,
}

fn find_field(
    fields: &[sensor_msgs::borrowed::PointField<'_>],
    names: &[&str],
) -> Option<(usize, u8)> {
    fields
        .iter()
        .find(|field| names.iter().any(|name| field.name == *name))
        .map(|field| (field.offset as usize, field.datatype))
}

fn read_scalar(data: &[u8], offset: usize, datatype: u8, big_endian: bool) -> Option<f64> {
    let read = |n: usize| -> Option<&[u8]> { data.get(offset..offset.checked_add(n)?) };
    macro_rules! from_bytes {
        ($ty:ty, $n:expr) => {{
            let bytes = read($n)?;
            let mut raw = [0u8; $n];
            raw.copy_from_slice(bytes);
            Some(if big_endian {
                <$ty>::from_be_bytes(raw) as f64
            } else {
                <$ty>::from_le_bytes(raw) as f64
            })
        }};
    }
    use sensor_msgs::borrowed::PointField;
    match datatype {
        PointField::INT8 => from_bytes!(i8, 1),
        PointField::UINT8 => from_bytes!(u8, 1),
        PointField::INT16 => from_bytes!(i16, 2),
        PointField::UINT16 => from_bytes!(u16, 2),
        PointField::INT32 => from_bytes!(i32, 4),
        PointField::UINT32 => from_bytes!(u32, 4),
        PointField::FLOAT32 => from_bytes!(f32, 4),
        PointField::FLOAT64 => from_bytes!(f64, 8),
        _ => None,
    }
}

/// Convert a `PointCloud2` sweep into a `LidarFrame`.
///
/// The base timestamp is the header stamp (ROS time). Per-point offsets come
/// from a `time`/`t` field when present: floating point values are seconds
/// (or absolute epoch seconds, in which case the base stamp is subtracted),
/// integer values are nanoseconds since the sweep start. Without a time field
/// every point gets offset zero, which is acceptable for 10 Hz spinning
/// LiDARs but disables motion compensation.
fn decode_pointcloud2_frame(cloud: &sensor_msgs::borrowed::PointCloud2<'_>) -> Result<LidarFrame> {
    let base_timestamp_sec =
        transfer_from_header(cloud.header.stamp.sec, cloud.header.stamp.nanosec);
    let width = cloud.width as usize;
    let height = cloud.height as usize;
    let point_step = cloud.point_step as usize;
    let row_step = cloud.row_step as usize;
    if point_step == 0 {
        anyhow::bail!("PointCloud2 point_step is zero");
    }
    let layout = PointCloudFields {
        x: find_field(&cloud.fields, &["x"]).context("PointCloud2 has no x field")?,
        y: find_field(&cloud.fields, &["y"]).context("PointCloud2 has no y field")?,
        z: find_field(&cloud.fields, &["z"]).context("PointCloud2 has no z field")?,
        intensity: find_field(&cloud.fields, &["intensity", "i"]),
        time: find_field(
            &cloud.fields,
            &["time", "t", "timestamp", "time_stamp", "offset_time"],
        ),
        line: find_field(&cloud.fields, &["ring", "line"]),
    };
    let count = width.saturating_mul(height);
    let mut points = Vec::with_capacity(count);
    let mut max_offset_sec = 0.0_f64;
    let big_endian = cloud.is_bigendian;
    for row in 0..height {
        let row_base = row.saturating_mul(row_step);
        for col in 0..width {
            let point_base = row_base.saturating_add(col.saturating_mul(point_step));
            let coord = |(offset, datatype): (usize, u8)| -> Option<f32> {
                read_scalar(&cloud.data, point_base + offset, datatype, big_endian)
                    .map(|value| value as f32)
            };
            let (Some(x), Some(y), Some(z)) = (coord(layout.x), coord(layout.y), coord(layout.z))
            else {
                continue;
            };
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                continue;
            }
            let intensity = layout
                .intensity
                .and_then(|field| coord(field))
                .unwrap_or(0.0);
            let offset_time_sec = layout
                .time
                .and_then(|(offset, datatype)| {
                    let raw = read_scalar(&cloud.data, point_base + offset, datatype, big_endian)?;
                    Some(match datatype {
                        sensor_msgs::borrowed::PointField::FLOAT32
                        | sensor_msgs::borrowed::PointField::FLOAT64 => {
                            if raw > 1.0e6 {
                                raw - base_timestamp_sec
                            } else {
                                raw
                            }
                        }
                        _ => {
                            if raw > 1.0e6 {
                                raw / 1.0e9
                            } else if raw > 1.0e3 {
                                raw / 1.0e6
                            } else {
                                raw
                            }
                        }
                    })
                })
                .unwrap_or(0.0)
                .max(0.0);
            max_offset_sec = max_offset_sec.max(offset_time_sec);
            let line = layout
                .line
                .and_then(|(offset, datatype)| {
                    read_scalar(&cloud.data, point_base + offset, datatype, big_endian)
                })
                .unwrap_or(0.0) as u8;
            points.push(TimedPoint {
                offset_time_sec,
                point: PointXYZI { x, y, z, intensity },
                tag: 0,
                line,
            });
        }
    }
    sort_timepoint_by_offset(&mut points);
    Ok(LidarFrame::new(
        base_timestamp_sec,
        base_timestamp_sec + max_offset_sec,
        points,
    ))
}

fn sort_timepoint_by_offset(points: &mut [TimedPoint]) {
    points.sort_by(|a, b| a.offset_time_sec.total_cmp(&b.offset_time_sec));
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReadStats {
    pub total_messages: usize,
    pub lidar_messages: usize,
    pub emitted_events: usize,
    pub emitted_lidar_events: usize,
    pub reordered_imu_events: usize,
    pub max_imu_arrival_regression_sec: f64,
    pub skipped_missing_schema: usize,
    pub skipped_unsupported_schema: usize,
    pub skipped_unsupported_topic: usize,
    pub last_timestamp_sec: Option<u64>,
}

// Xsens samples in KA-Urban-East arrive up to 8 ms out of header-time order.
// Delay only IMU emission; LiDAR remains in bag order and the synchronizer
// still rejects any timestamp regression that exceeds this bounded window.
const IMU_REORDER_WINDOW_SEC: f64 = 0.02;

#[derive(Default)]
struct ImuReorderBuffer {
    pending: Vec<ImuSample>,
    max_seen_time_sec: Option<f64>,
    last_emitted_time_sec: Option<f64>,
}

impl ImuReorderBuffer {
    fn push<F>(&mut self, imu: ImuSample, stats: &mut ReadStats, on_event: &mut F) -> Result<()>
    where
        F: FnMut(SensorEvent) -> Result<()>,
    {
        let time = imu.time_stamp_sec;
        if !time.is_finite() {
            anyhow::bail!("non-finite IMU header timestamp: {time}");
        }
        if let Some(last_emitted) = self.last_emitted_time_sec
            && time < last_emitted
        {
            anyhow::bail!(
                "IMU header timestamp {time:.9} arrived after {last_emitted:.9} was emitted; disorder exceeds {IMU_REORDER_WINDOW_SEC:.3} s window"
            );
        }
        if let Some(max_seen) = self.max_seen_time_sec
            && time < max_seen
        {
            stats.reordered_imu_events += 1;
            stats.max_imu_arrival_regression_sec =
                stats.max_imu_arrival_regression_sec.max(max_seen - time);
        }
        let max_seen = self.max_seen_time_sec.map_or(time, |seen| seen.max(time));
        self.max_seen_time_sec = Some(max_seen);
        let index = self
            .pending
            .partition_point(|sample| sample.time_stamp_sec <= time);
        self.pending.insert(index, imu);
        self.flush_until(max_seen - IMU_REORDER_WINDOW_SEC, stats, on_event)
    }

    fn flush_until<F>(
        &mut self,
        cutoff_sec: f64,
        stats: &mut ReadStats,
        on_event: &mut F,
    ) -> Result<()>
    where
        F: FnMut(SensorEvent) -> Result<()>,
    {
        while self
            .pending
            .first()
            .is_some_and(|sample| sample.time_stamp_sec <= cutoff_sec)
        {
            let imu = self.pending.remove(0);
            self.last_emitted_time_sec = Some(imu.time_stamp_sec);
            on_event(SensorEvent::Imu(imu))?;
            stats.emitted_events += 1;
        }
        Ok(())
    }
}

fn map_mcap<P: AsRef<Utf8Path>>(p: P) -> Result<Mmap> {
    let fd = fs::File::open(p.as_ref()).context("Couldn't open MCAP file")?;
    unsafe { Mmap::map(&fd) }.context("Couldn't map MCAP file")
}

fn is_supported_sensor_schema(schema_name: &str) -> bool {
    matches!(
        schema_name,
        "livox_ros_driver2/msg/CustomMsg"
            | "livox_ros_driver/msg/CustomMsg"
            | "sensor_msgs/msg/Imu"
            | "sensor_msgs/msg/PointCloud2"
    )
}

pub fn read_mcap_events<P, F>(p: P, mut on_event: F) -> Result<ReadStats>
where
    P: AsRef<Utf8Path>,
    F: FnMut(SensorEvent) -> Result<()>,
{
    read_mcap_events_on_topics(p, "/livox/lidar", "/livox/imu", &mut on_event)
}

/// Decode only the configured LiDAR and IMU topics from an MCAP bag.
/// Sensor timestamps come from each message; MCAP log time is used only to
/// check that the input stream does not regress.
pub fn read_mcap_events_on_topics<P, F>(
    p: P,
    lidar_topic: &str,
    imu_topic: &str,
    mut on_event: F,
) -> Result<ReadStats>
where
    P: AsRef<Utf8Path>,
    F: FnMut(SensorEvent) -> Result<()>,
{
    if lidar_topic.is_empty() || imu_topic.is_empty() || lidar_topic == imu_topic {
        anyhow::bail!("LiDAR and IMU topics must be nonempty and distinct");
    }
    let mapped = map_mcap(p)?;
    let mut stats = ReadStats::default();
    let mut imu_reorder = ImuReorderBuffer::default();

    for message in mcap::MessageStream::new(&mapped)? {
        let message = message?;
        let current = message.log_time;
        stats.total_messages += 1;
        if message.channel.topic == lidar_topic {
            stats.lidar_messages += 1;
        }

        if message.channel.topic != lidar_topic && message.channel.topic != imu_topic {
            stats.skipped_unsupported_topic += 1;
            continue;
        }

        let Some(schema) = message.channel.schema.as_ref() else {
            stats.skipped_missing_schema += 1;
            continue;
        };

        if !is_supported_sensor_schema(&schema.name) {
            stats.skipped_unsupported_schema += 1;
            continue;
        }

        let sensor_source =
            SensorEvent::dispatch(&message.data, &schema.name).with_context(|| {
                format!(
                    "failed to decode MCAP message on topic `{}` with schema `{}`",
                    message.channel.topic, schema.name
                )
            })?;
        if let Some(last_publish_timestamp) = stats.last_timestamp_sec
            && last_publish_timestamp > current
        {
            anyhow::bail!(
                "event timestamp regressed: current={current}, last={last_publish_timestamp}"
            )
        }
        stats.last_timestamp_sec = Some(current);
        match sensor_source {
            SensorEvent::Imu(imu) => imu_reorder.push(imu, &mut stats, &mut on_event)?,
            SensorEvent::Lidar(lidar) => {
                on_event(SensorEvent::Lidar(lidar))?;
                stats.emitted_lidar_events += 1;
                stats.emitted_events += 1;
            }
        }
    }

    imu_reorder.flush_until(f64::INFINITY, &mut stats, &mut on_event)?;

    if stats.emitted_lidar_events == 0 || stats.emitted_events == stats.emitted_lidar_events {
        anyhow::bail!(
            "no decodable LiDAR/IMU pair for topics lidar=`{lidar_topic}`, imu=`{imu_topic}` (emitted lidar={}, emitted imu={})",
            stats.emitted_lidar_events,
            stats.emitted_events - stats.emitted_lidar_events
        );
    }

    Ok(stats)
}

#[cfg(test)]
mod test {
    use crate::{
        ImuReorderBuffer, ReadStats, SensorEvent, read_mcap_events, read_mcap_events_on_topics,
    };
    use fastlio_types::Vec3;

    fn imu(time_stamp_sec: f64) -> fastlio_types::ImuSample {
        fastlio_types::ImuSample {
            time_stamp_sec,
            gyro: Vec3::zeros(),
            accel: Vec3::zeros(),
        }
    }

    #[test]
    fn reorders_bounded_imu_arrival_regression_without_losing_samples() {
        let mut buffer = ImuReorderBuffer::default();
        let mut stats = ReadStats::default();
        let mut emitted = Vec::new();
        let mut collect = |event| {
            if let SensorEvent::Imu(sample) = event {
                emitted.push(sample.time_stamp_sec);
            }
            Ok(())
        };
        for time in [1.0, 1.01, 1.03, 1.022, 1.04, 1.06] {
            buffer.push(imu(time), &mut stats, &mut collect).unwrap();
        }
        buffer
            .flush_until(f64::INFINITY, &mut stats, &mut collect)
            .unwrap();
        assert_eq!(emitted, [1.0, 1.01, 1.022, 1.03, 1.04, 1.06]);
        assert_eq!(stats.emitted_events, emitted.len());
        assert_eq!(stats.reordered_imu_events, 1);
        assert!((stats.max_imu_arrival_regression_sec - 0.008).abs() < 1e-12);
    }

    #[test]
    fn rejects_imu_arriving_after_reorder_window() {
        let mut buffer = ImuReorderBuffer::default();
        let mut stats = ReadStats::default();
        let mut collect = |_| Ok(());
        buffer.push(imu(1.0), &mut stats, &mut collect).unwrap();
        buffer.push(imu(1.03), &mut stats, &mut collect).unwrap();
        let error = buffer
            .push(imu(0.99), &mut stats, &mut collect)
            .unwrap_err();
        assert!(error.to_string().contains("disorder exceeds"));
    }

    #[test]
    fn rejects_ambiguous_topic_selection() {
        let result =
            read_mcap_events_on_topics("missing.mcap", "/imu/data", "/imu/data", |_| Ok(()));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("distinct"));
    }

    #[test]
    #[ignore]
    fn smoke_read_it() {
        let stats = read_mcap_events(
            "/home/lyra/Projects/fastlio-rs/rosbags/rosbag2_upstairs/rosbag2_2026_06_23-15_13_52_0.mcap",
            |_e| {
                // println!("{:?}", e);
                Ok(())
            },
        )
        .unwrap();
        assert!(stats.emitted_events > 0);
        assert!(stats.skipped_missing_schema == 0);
        assert!(stats.skipped_unsupported_topic == 0);
        // assert!(stats.skipped_unmonotonic_timestamp == 0);
    }

    fn pointcloud2_fixture() -> sensor_msgs::msg::PointCloud2 {
        use sensor_msgs::msg::{PointCloud2, PointField};
        let fields = vec![
            PointField {
                name: "x".into(),
                offset: 0,
                datatype: PointField::FLOAT32,
                count: 1,
            },
            PointField {
                name: "y".into(),
                offset: 4,
                datatype: PointField::FLOAT32,
                count: 1,
            },
            PointField {
                name: "z".into(),
                offset: 8,
                datatype: PointField::FLOAT32,
                count: 1,
            },
            PointField {
                name: "intensity".into(),
                offset: 12,
                datatype: PointField::FLOAT32,
                count: 1,
            },
            PointField {
                name: "time".into(),
                offset: 16,
                datatype: PointField::FLOAT32,
                count: 1,
            },
        ];
        let mut data = Vec::new();
        for (x, y, z, i, t) in [(1.0f32, 2.0, 3.0, 7.0, 0.01f32), (4.0, 5.0, 6.0, 9.0, 0.02)] {
            for value in [x, y, z, i, t] {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        PointCloud2 {
            header: std_msgs::msg::Header {
                stamp: builtin_interfaces::msg::Time {
                    sec: 100,
                    nanosec: 500_000_000,
                },
                frame_id: "velodyne".into(),
            },
            height: 1,
            width: 2,
            fields,
            is_bigendian: false,
            point_step: 20,
            row_step: 40,
            data,
            is_dense: true,
        }
    }

    #[test]
    fn pointcloud2_decodes_points_and_offsets() {
        let cloud = pointcloud2_fixture();
        let bytes = ros2_dispatch::DecodedMessage::SensorMsgsPointCloud2(cloud)
            .encode_to_vec()
            .expect("encode");
        let event = SensorEvent::dispatch(&bytes, "sensor_msgs/msg/PointCloud2").expect("decode");
        let SensorEvent::Lidar(frame) = event else {
            panic!("expected lidar frame");
        };
        assert!((frame.base_timestamp_sec - 100.5).abs() < 1e-9);
        assert_eq!(frame.points.len(), 2);
        assert!((frame.end_timestamp_sec() - 100.52).abs() < 1e-9);
        assert!((frame.points[0].offset_time_sec - 0.01).abs() < 1e-6);
        assert!((frame.points[1].offset_time_sec - 0.02).abs() < 1e-6);
        assert!((frame.points[0].point.x - 1.0).abs() < 1e-6);
        assert!((frame.points[1].point.z - 6.0).abs() < 1e-6);
        assert!((frame.points[0].point.intensity - 7.0).abs() < 1e-6);
    }

    #[test]
    fn pointcloud2_without_x_field_is_rejected() {
        let mut cloud = pointcloud2_fixture();
        cloud.fields.retain(|field| field.name != "x");
        let bytes = ros2_dispatch::DecodedMessage::SensorMsgsPointCloud2(cloud)
            .encode_to_vec()
            .expect("encode");
        let error = SensorEvent::dispatch(&bytes, "sensor_msgs/msg/PointCloud2")
            .expect_err("must reject cloud without x");
        assert!(error.to_string().contains("x field"), "error: {error}");
    }
}
