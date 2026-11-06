use anyhow::{Context, Result, anyhow, bail};
use camino::Utf8PathBuf;
use fastlio_dataset::{ReadStats, SensorEvent, read_mcap_events_on_topics};
use fastlio_map::{
    surfel::SurfelMap,
    types::{GeometryClass, Surfel},
};
use fastlio_pipeline::{MainPipeline, PipelineFrameSummary, synchronizer::MeasurementSynchronizer};
use fastlio_types::{Mat3, NavState, read_from_config_path};
use pcd_rs::{DataKind, PcdSerialize, WriterInit};
use ringbuffer_spsc::{RingBufferReader, RingBufferWriter, ringbuffer};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

enum ReplayMessage {
    Event(SensorEvent),
    Finished(Result<ReadStats, String>),
}

#[derive(Debug, Clone)]
struct ReplayConfig {
    playback_rate: f64,
    channel_capacity: usize,
    time_offset_lidar_to_imu_sec: f64,
    storm_mode: bool,
    collect_state: bool,
}

#[derive(Debug)]
struct ReplayArgs {
    bag_path: Utf8PathBuf,
    config_path: Utf8PathBuf,
    trajectory_path: Option<Utf8PathBuf>,
    surfel_map_path: Option<Utf8PathBuf>,
    gaussian_ply_path: Option<Utf8PathBuf>,
    surfel_dump_path: Option<Utf8PathBuf>,
    surfel_dump_stride: usize,
    keyframe_dir: Option<Utf8PathBuf>,
    keyframe_stride: usize,
    state_dump_path: Option<Utf8PathBuf>,
    revisit_mask_path: Option<Utf8PathBuf>,
    assoc_diag: bool,
    playback_rate: f64,
    channel_capacity: usize,
    storm_mode: bool,
    map_stats: bool,
}

/// Timestamp labels of revisit-region frames, used only to split diagnostics.
#[derive(Default)]
struct RevisitLabels {
    timestamps: Vec<f64>,
    cursor: usize,
}

impl RevisitLabels {
    fn load(path: &Utf8PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read revisit mask `{path}`"))?;
        let mut timestamps = Vec::new();
        for (line_number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let first = line.split(',').next().unwrap_or("").trim();
            let Ok(timestamp) = first.parse::<f64>() else {
                if line_number == 0 {
                    continue; // header
                }
                bail!("revisit mask line {}: invalid timestamp", line_number + 1);
            };
            timestamps.push(timestamp);
        }
        timestamps.sort_by(|a, b| a.total_cmp(b));
        Ok(Self {
            timestamps,
            cursor: 0,
        })
    }

    fn is_revisit(&mut self, timestamp_sec: f64) -> bool {
        while self.cursor < self.timestamps.len()
            && self.timestamps[self.cursor] < timestamp_sec - 1e-6
        {
            self.cursor += 1;
        }
        if let Some(&t) = self.timestamps.get(self.cursor) {
            if (t - timestamp_sec).abs() <= 1e-6 {
                self.cursor += 1;
                return true;
            }
        }
        false
    }
}

impl ReplayArgs {
    fn parse(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                "usage: {} <bag.mcap> <config.yaml> [playback_rate] [channel_capacity] [trajectory.csv] [surfel-map.pcd] [--storm] [--map-stats] [--gaussian-ply <out.ply>] [--surfel-dump <out.csv>] [--keyframe-dir <dir>] [--state-dump <csv>] [--assoc-diag] [--revisit-mask <mask.csv>]",
                args.first().map(String::as_str).unwrap_or("fastlio-replay")
            );
        }
        let mut storm_mode = false;
        let mut map_stats = false;
        let mut gaussian_ply_path = None;
        let mut surfel_dump_path = None;
        let mut surfel_dump_stride = 4usize;
        let mut keyframe_dir = None;
        let mut keyframe_stride = 20usize;
        let mut state_dump_path = None;
        let mut revisit_mask_path = None;
        let mut assoc_diag = false;
        let mut positional = Vec::new();
        let mut iter = args[3..].iter();
        while let Some(arg) = iter.next() {
            if arg == "--storm" {
                storm_mode = true;
            } else if arg == "--map-stats" {
                map_stats = true;
            } else if arg == "--assoc-diag" {
                assoc_diag = true;
            } else if arg == "--surfel-dump" {
                let Some(path) = iter.next() else {
                    bail!("--surfel-dump requires a CSV path");
                };
                surfel_dump_path = Some(Utf8PathBuf::from(path.as_str()));
            } else if arg == "--state-dump" {
                let Some(path) = iter.next() else {
                    bail!("--state-dump requires a CSV path");
                };
                state_dump_path = Some(Utf8PathBuf::from(path.as_str()));
            } else if arg == "--keyframe-dir" {
                let Some(path) = iter.next() else {
                    bail!("--keyframe-dir requires a directory path");
                };
                keyframe_dir = Some(Utf8PathBuf::from(path.as_str()));
            } else if arg == "--keyframe-stride" {
                let Some(value) = iter.next() else {
                    bail!("--keyframe-stride requires a value");
                };
                keyframe_stride = value.parse::<usize>().unwrap_or(20).max(1);
            } else if arg == "--surfel-dump-stride" {
                let Some(value) = iter.next() else {
                    bail!("--surfel-dump-stride requires a value");
                };
                surfel_dump_stride = value.parse::<usize>().unwrap_or(4).max(1);
            } else if arg == "--revisit-mask" {
                let Some(path) = iter.next() else {
                    bail!("--revisit-mask requires a CSV path");
                };
                revisit_mask_path = Some(Utf8PathBuf::from(path.as_str()));
            } else if arg == "--gaussian-ply" {
                let Some(path) = iter.next() else {
                    bail!("--gaussian-ply requires an output path");
                };
                gaussian_ply_path = Some(Utf8PathBuf::from(path.as_str()));
            } else if arg.starts_with("--") {
                bail!("unknown replay option `{arg}`");
            } else {
                positional.push(arg);
            }
        }
        if positional.len() > 4 {
            bail!("too many replay arguments");
        }
        let playback_rate = positional
            .first()
            .map(|v| v.parse::<f64>())
            .transpose()
            .context("invalid playback_rate")?
            .unwrap_or(0.0);
        if !playback_rate.is_finite() || playback_rate < 0.0 {
            bail!("playback_rate must be finite and non-negative");
        }
        let channel_capacity = positional
            .get(1)
            .map(|v| v.parse::<usize>())
            .transpose()
            .context("invalid channel_capacity")?
            .unwrap_or(1024);
        if channel_capacity == 0 {
            bail!("channel_capacity must be positive");
        }
        Ok(Self {
            bag_path: Utf8PathBuf::from(&args[1]),
            config_path: Utf8PathBuf::from(&args[2]),
            trajectory_path: positional
                .get(2)
                .map(|path| Utf8PathBuf::from(path.as_str())),
            surfel_map_path: positional
                .get(3)
                .map(|path| Utf8PathBuf::from(path.as_str())),
            gaussian_ply_path,
            surfel_dump_path,
            surfel_dump_stride,
            keyframe_dir,
            keyframe_stride,
            state_dump_path,
            revisit_mask_path,
            assoc_diag,
            playback_rate,
            channel_capacity,
            storm_mode,
            map_stats,
        })
    }
}

#[derive(PcdSerialize)]
struct SurfelPcdPoint {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    class_id: f32,
}

#[derive(Clone)]
struct TrajectoryRow {
    timestamp_sec: f64,
    state: NavState,
    frame_summary: PipelineFrameSummary,
}

#[derive(Default)]
struct ReplayStats {
    read: ReadStats,
    synchronized_groups: usize,
    processed_frames: usize,
    failed_groups: usize,
    max_pending_lidar: usize,
    dropped_lidar_before_first_imu: usize,
    pending_lidar_at_eof: usize,
    first_imu_raw_time_sec: Option<f64>,
    last_imu_raw_time_sec: Option<f64>,
    first_imu_time_sec: Option<f64>,
    last_imu_time_sec: Option<f64>,
    first_lidar_time_sec: Option<f64>,
    last_lidar_time_sec: Option<f64>,
    first_pipeline_error: Option<String>,
    trajectory: Vec<TrajectoryRow>,
    /// Wall-clock duration of each `process_measure_group` call (ms).
    frame_durations_ms: Vec<f64>,
    /// Per-frame filter states for offline drift analysis.
    state_rows: Vec<StateRow>,
}

#[derive(Clone)]
struct StateRow {
    timestamp_sec: f64,
    state: NavState,
    covariance_diag: [f64; 23],
}

struct ProducerAlive(Arc<AtomicBool>);

impl Drop for ProducerAlive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn push_blocking<T>(
    producer: &mut RingBufferWriter<T>,
    mut item: T,
    consumer_alive: &AtomicBool,
) -> Result<()> {
    loop {
        match producer.push(item) {
            None => return Ok(()),
            Some(returned) if consumer_alive.load(Ordering::Acquire) => {
                item = returned;
                thread::yield_now();
            }
            Some(_) => bail!("replay consumer stopped"),
        }
    }
}

fn produce_events(
    path: Utf8PathBuf,
    lidar_topic: String,
    imu_topic: String,
    producer: &mut RingBufferWriter<ReplayMessage>,
    consumer_alive: Arc<AtomicBool>,
) -> Result<()> {
    let result = read_mcap_events_on_topics(path, &lidar_topic, &imu_topic, |event| {
        push_blocking(producer, ReplayMessage::Event(event), &consumer_alive)
    })
    .map_err(|error| format!("{error:#}"));
    push_blocking(producer, ReplayMessage::Finished(result), &consumer_alive)
}

struct PlaybackClock {
    rate: f64,
    first_sensor_time_sec: Option<f64>,
    wall_start: Instant,
}

impl PlaybackClock {
    fn new(rate: f64) -> Self {
        Self {
            rate,
            first_sensor_time_sec: None,
            wall_start: Instant::now(),
        }
    }

    fn wait_for(&mut self, sensor_time_sec: f64) {
        if self.rate == 0.0 || !sensor_time_sec.is_finite() {
            return;
        }
        let first = *self.first_sensor_time_sec.get_or_insert_with(|| {
            self.wall_start = Instant::now();
            sensor_time_sec
        });
        let target = Duration::from_secs_f64((sensor_time_sec - first).max(0.0) / self.rate);
        if let Some(remaining) = target.checked_sub(self.wall_start.elapsed()) {
            thread::sleep(remaining);
        }
    }
}

fn apply_time_offset(event: SensorEvent, offset_sec: f64, stats: &mut ReplayStats) -> SensorEvent {
    match event {
        SensorEvent::Imu(mut imu) => {
            let raw = imu.time_stamp_sec;
            stats.first_imu_raw_time_sec.get_or_insert(raw);
            stats.last_imu_raw_time_sec = Some(raw);
            imu.time_stamp_sec = raw - offset_sec;
            stats.first_imu_time_sec.get_or_insert(imu.time_stamp_sec);
            stats.last_imu_time_sec = Some(imu.time_stamp_sec);
            SensorEvent::Imu(imu)
        }
        SensorEvent::Lidar(lidar) => {
            stats
                .first_lidar_time_sec
                .get_or_insert(lidar.base_timestamp_sec);
            stats.last_lidar_time_sec = Some(lidar.base_timestamp_sec);
            SensorEvent::Lidar(lidar)
        }
    }
}

fn process_ready_groups(
    synchronizer: &mut MeasurementSynchronizer,
    pipeline: &mut MainPipeline,
    stats: &mut ReplayStats,
    revisit_labels: &mut RevisitLabels,
    config: &ReplayConfig,
) {
    for group in synchronizer.drain_ready() {
        stats.synchronized_groups += 1;
        let scan_begin_sec = group.lidar.base_timestamp_sec;
        let timestamp_sec = group.lidar.end_timestamp_sec();
        let first_imu_sec = group.imu.first().map(|imu| imu.time_stamp_sec);
        let last_imu_sec = group.imu.last().map(|imu| imu.time_stamp_sec);
        let min_offset_sec = group
            .lidar
            .points
            .iter()
            .map(|point| point.offset_time_sec)
            .reduce(f64::min);
        let max_offset_sec = group
            .lidar
            .points
            .iter()
            .map(|point| point.offset_time_sec)
            .reduce(f64::max);
        let offset_regressions = group
            .lidar
            .points
            .windows(2)
            .filter(|pair| pair[1].offset_time_sec < pair[0].offset_time_sec)
            .count();
        let frame_start = Instant::now();
        pipeline
            .map
            .set_assoc_diag_revisit(revisit_labels.is_revisit(timestamp_sec));
        let result = pipeline.process_measure_group(group);
        stats
            .frame_durations_ms
            .push(frame_start.elapsed().as_secs_f64() * 1000.0);
        match result {
            Ok(frame_summary) => {
                stats.processed_frames += 1;
                if config.collect_state {
                    let mut covariance_diag = [0.0_f64; 23];
                    for (index, value) in covariance_diag.iter_mut().enumerate() {
                        *value = pipeline.filter.covariance[(index, index)];
                    }
                    stats.state_rows.push(StateRow {
                        timestamp_sec,
                        state: pipeline.filter.state.clone(),
                        covariance_diag,
                    });
                }
                stats.trajectory.push(TrajectoryRow {
                    timestamp_sec,
                    state: pipeline.filter.state.clone(),
                    frame_summary,
                });
            }
            Err(error) => {
                stats.failed_groups += 1;
                stats
                    .first_pipeline_error
                    .get_or_insert_with(|| {
                        format!(
                            "scan=[{scan_begin_sec:.9},{timestamp_sec:.9}], imu=[{first_imu_sec:?},{last_imu_sec:?}], offset=[{min_offset_sec:?},{max_offset_sec:?}], offset_regressions={offset_regressions}: {error:#}"
                        )
                    });
            }
        }
    }
}

fn receive_events(
    consumer: &mut RingBufferReader<ReplayMessage>,
    producer_alive: &AtomicBool,
    synchronizer: &mut MeasurementSynchronizer,
    pipeline: &mut MainPipeline,
    config: &ReplayConfig,
    revisit_labels: &mut RevisitLabels,
) -> Result<ReplayStats> {
    let mut stats = ReplayStats::default();
    let mut clock = PlaybackClock::new(if config.storm_mode {
        0.0
    } else {
        config.playback_rate
    });
    loop {
        match consumer.pull() {
            Some(ReplayMessage::Event(event)) => {
                clock.wait_for(event.timestamp_sec());
                match apply_time_offset(event, config.time_offset_lidar_to_imu_sec, &mut stats) {
                    SensorEvent::Imu(imu) => synchronizer.pend_imu(imu)?,
                    SensorEvent::Lidar(lidar) => synchronizer.pend_lidar(lidar)?,
                }
                stats.max_pending_lidar =
                    stats.max_pending_lidar.max(synchronizer.lidar_buffer.len());
                process_ready_groups(synchronizer, pipeline, &mut stats, revisit_labels, config);
            }
            Some(ReplayMessage::Finished(result)) => {
                process_ready_groups(synchronizer, pipeline, &mut stats, revisit_labels, config);
                stats.read = result.map_err(anyhow::Error::msg)?;
                stats.dropped_lidar_before_first_imu = synchronizer.dropped_lidar_before_first_imu;
                stats.pending_lidar_at_eof = synchronizer.lidar_buffer.len();
                return Ok(stats);
            }
            None if !producer_alive.load(Ordering::Acquire) => {
                bail!("replay producer stopped without a completion message")
            }
            None => thread::yield_now(),
        }
    }
}

fn run_spsc(
    path: Utf8PathBuf,
    pipeline: &mut MainPipeline,
    config: ReplayConfig,
    revisit_labels: &mut RevisitLabels,
) -> Result<ReplayStats> {
    let lidar_topic = pipeline.config.common.lid_topic.clone();
    let imu_topic = pipeline.config.common.imu_topic.clone();
    let (mut producer, mut consumer) = ringbuffer(config.channel_capacity);
    let producer_alive = Arc::new(AtomicBool::new(true));
    let consumer_alive = Arc::new(AtomicBool::new(true));
    let producer_guard = ProducerAlive(producer_alive.clone());
    let producer_consumer_alive = consumer_alive.clone();
    let handle = thread::spawn(move || {
        let _guard = producer_guard;
        produce_events(
            path,
            lidar_topic,
            imu_topic,
            &mut producer,
            producer_consumer_alive,
        )
    });
    let mut synchronizer = MeasurementSynchronizer::new();
    let result = receive_events(
        &mut consumer,
        &producer_alive,
        &mut synchronizer,
        pipeline,
        &config,
        revisit_labels,
    );
    consumer_alive.store(false, Ordering::Release);
    let producer_result = handle
        .join()
        .map_err(|_| anyhow!("replay producer thread panicked"))?;
    match result {
        Ok(stats) => {
            producer_result?;
            Ok(stats)
        }
        Err(error) => Err(error),
    }
}

fn write_trajectory(path: &Utf8PathBuf, rows: &[TrajectoryRow]) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create trajectory `{path}`"))?,
    );
    writeln!(
        writer,
        "timestamp_sec,px,py,pz,qx,qy,qz,qw,vx,vy,vz,tracking"
    )?;
    for row in rows {
        let p = row.state.position;
        let v = row.state.velocity;
        let q = row.state.orientation.quaternion();
        writeln!(
            writer,
            "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
            row.timestamp_sec,
            p.x,
            p.y,
            p.z,
            q.i,
            q.j,
            q.k,
            q.w,
            v.x,
            v.y,
            v.z,
            row.frame_summary.tracking,
        )?;
    }
    Ok(())
}

fn write_surfel_map(path: &Utf8PathBuf, map: &SurfelMap) -> Result<()> {
    // Only active roots describe the map; surfels merged away via the union-find
    // parent chain are tombstones and must not be exported.
    let active = || map.surfels().filter(|surfel| surfel.parent.is_none());
    let width = active().count();
    let config = map.surfel_config();
    let mut writer = WriterInit {
        width: width as u64,
        height: 1,
        viewpoint: Default::default(),
        data_kind: DataKind::Binary,
        schema: None,
        version: None,
    }
    .create::<SurfelPcdPoint, _>(path)
    .with_context(|| format!("failed to create surfel map PCD `{path}`"))?;

    for surfel in active() {
        let class_id = match surfel.geometry_class(config) {
            GeometryClass::Plane => 0.0,
            GeometryClass::Line => 1.0,
            GeometryClass::Scatter => 2.0,
            GeometryClass::Degenerate => 3.0,
            GeometryClass::Growing => 4.0,
        };
        writer.push(&SurfelPcdPoint {
            x: surfel.mean_w.x as f32,
            y: surfel.mean_w.y as f32,
            z: surfel.mean_w.z as f32,
            intensity: surfel.count as f32,
            normal_x: surfel.eigenvectors[(0, 0)] as f32,
            normal_y: surfel.eigenvectors[(1, 0)] as f32,
            normal_z: surfel.eigenvectors[(2, 0)] as f32,
            class_id,
        })?;
    }
    writer.finish()?;
    Ok(())
}

/// Number of little-endian `f32` fields written per Gaussian vertex.
const GAUSSIAN_PLY_FIELDS: usize = 17;
const SH_C0: f64 = 0.282_094_791_796_743_77;

/// Export the active surfel roots as a 3D Gaussian Splatting PLY (INRIA 3DGS
/// layout, degree-0 SH) that viewers such as SuperSplat can open directly.
///
/// Each surfel becomes one anisotropic Gaussian: center = `mean_w`, rotation =
/// the plane eigenbasis, scales = `sqrt(eigenvalues)` with a small floor on the
/// normal so thin disks stay visible, colour from the geometry class, opacity
/// from maturity.
fn write_gaussian_ply(path: &Utf8PathBuf, map: &SurfelMap) -> Result<()> {
    let config = map.surfel_config();
    let active: Vec<_> = map
        .surfels()
        .filter(|surfel| surfel.parent.is_none())
        .collect();
    let file = File::create(path)
        .with_context(|| format!("failed to create gaussian splat PLY `{path}`"))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"ply\n")?;
    writer.write_all(b"format binary_little_endian 1.0\n")?;
    writer.write_all(b"comment fastlio-rs surfel map exported as 3DGS degree-0 SH\n")?;
    writeln!(writer, "element vertex {}", active.len())?;
    for field in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(writer, "property float {field}")?;
    }
    writer.write_all(b"end_header\n")?;

    let mut batch: Vec<u8> = Vec::with_capacity(GAUSSIAN_PLY_FIELDS * 4 * 4096);
    for surfel in &active {
        let normal = surfel.eigenvectors.column(0);
        let quaternion = surfel_quaternion(surfel);
        let color = gaussian_color(surfel.geometry_class(config));
        let maturity =
            (surfel.count as f64 / config.min_mature_surfel_count.max(1) as f64).min(1.0);
        let opacity = (0.45 + 0.5 * maturity).clamp(0.3, 0.95);
        let scales = [
            surfel.eigenvalues[0].max(0.0).sqrt().clamp(0.02, 1.0),
            surfel.eigenvalues[1].max(0.0).sqrt().clamp(0.02, 3.0),
            surfel.eigenvalues[2].max(0.0).sqrt().clamp(0.02, 3.0),
        ];
        let values = [
            surfel.mean_w.x,
            surfel.mean_w.y,
            surfel.mean_w.z,
            normal.x,
            normal.y,
            normal.z,
            (color[0] - 0.5) / SH_C0,
            (color[1] - 0.5) / SH_C0,
            (color[2] - 0.5) / SH_C0,
            logit(opacity),
            scales[0].ln(),
            scales[1].ln(),
            scales[2].ln(),
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        ];
        for value in values {
            batch.extend_from_slice(&(value as f32).to_le_bytes());
        }
        if batch.len() >= GAUSSIAN_PLY_FIELDS * 4 * 4096 {
            writer.write_all(&batch)?;
            batch.clear();
        }
    }
    writer.write_all(&batch)?;
    writer.flush()?;
    Ok(())
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

/// Per-class colour (linear RGB) for the splat export.
fn gaussian_color(class: GeometryClass) -> [f64; 3] {
    match class {
        GeometryClass::Plane => [0.72, 0.76, 0.82],
        GeometryClass::Line => [0.95, 0.78, 0.28],
        GeometryClass::Scatter => [0.90, 0.55, 0.32],
        GeometryClass::Growing => [0.35, 0.85, 0.50],
        GeometryClass::Degenerate => [0.90, 0.35, 0.35],
    }
}

/// Rotation (w, x, y, z) of the surfel's plane eigenbasis, with the third axis
/// flipped when needed so the basis stays right-handed.
fn surfel_quaternion(surfel: &Surfel) -> [f64; 4] {
    let mut rotation = surfel.eigenvectors;
    if rotation.determinant() < 0.0 {
        let flipped = -rotation.column(2).into_owned();
        rotation.set_column(2, &flipped);
    }
    rotation_matrix_to_quaternion(&rotation)
}

/// Convert a right-handed rotation matrix to a normalized (w, x, y, z) quaternion.
fn rotation_matrix_to_quaternion(rotation: &Mat3<f64>) -> [f64; 4] {
    let trace = rotation[(0, 0)] + rotation[(1, 1)] + rotation[(2, 2)];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (rotation[(2, 1)] - rotation[(1, 2)]) / s,
            (rotation[(0, 2)] - rotation[(2, 0)]) / s,
            (rotation[(1, 0)] - rotation[(0, 1)]) / s,
        )
    } else if rotation[(0, 0)] > rotation[(1, 1)] && rotation[(0, 0)] > rotation[(2, 2)] {
        let s = (1.0 + rotation[(0, 0)] - rotation[(1, 1)] - rotation[(2, 2)]).sqrt() * 2.0;
        (
            (rotation[(2, 1)] - rotation[(1, 2)]) / s,
            0.25 * s,
            (rotation[(0, 1)] + rotation[(1, 0)]) / s,
            (rotation[(0, 2)] + rotation[(2, 0)]) / s,
        )
    } else if rotation[(1, 1)] > rotation[(2, 2)] {
        let s = (1.0 + rotation[(1, 1)] - rotation[(0, 0)] - rotation[(2, 2)]).sqrt() * 2.0;
        (
            (rotation[(0, 2)] - rotation[(2, 0)]) / s,
            (rotation[(0, 1)] + rotation[(1, 0)]) / s,
            0.25 * s,
            (rotation[(1, 2)] + rotation[(2, 1)]) / s,
        )
    } else {
        let s = (1.0 + rotation[(2, 2)] - rotation[(0, 0)] - rotation[(1, 1)]).sqrt() * 2.0;
        (
            (rotation[(1, 0)] - rotation[(0, 1)]) / s,
            (rotation[(0, 2)] + rotation[(2, 0)]) / s,
            (rotation[(1, 2)] + rotation[(2, 1)]) / s,
            0.25 * s,
        )
    };
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    [w / norm, x / norm, y / norm, z / norm]
}

/// Write per-frame filter states for offline z-drift analysis.
fn write_state_dump(path: &Utf8PathBuf, rows: &[StateRow]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create state dump `{path}`"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "timestamp_sec,px,py,pz,qw,qx,qy,qz,vx,vy,vz,gx,gy,gz,bgx,bgy,bgz,bax,bay,baz,cov_px,cov_py,cov_pz,cov_vx,cov_vy,cov_vz,cov_gx,cov_gy,cov_bgx,cov_bgy,cov_bgz,cov_bax,cov_bay,cov_baz"
    )?;
    for row in rows {
        let s = &row.state;
        writeln!(
            writer,
            "{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}",
            row.timestamp_sec,
            s.position.x,
            s.position.y,
            s.position.z,
            s.orientation.w,
            s.orientation.i,
            s.orientation.j,
            s.orientation.k,
            s.velocity.x,
            s.velocity.y,
            s.velocity.z,
            s.gravity.x,
            s.gravity.y,
            s.gravity.z,
            s.gyro_bias.x,
            s.gyro_bias.y,
            s.gyro_bias.z,
            s.accel_bias.x,
            s.accel_bias.y,
            s.accel_bias.z,
            row.covariance_diag[0],
            row.covariance_diag[1],
            row.covariance_diag[2],
            row.covariance_diag[3],
            row.covariance_diag[4],
            row.covariance_diag[5],
            row.covariance_diag[6],
            row.covariance_diag[7],
            row.covariance_diag[15],
            row.covariance_diag[16],
            row.covariance_diag[17],
            row.covariance_diag[18],
            row.covariance_diag[19],
            row.covariance_diag[20]
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Dump active surfels with their lifetime metadata for offline map-deformation
/// experiments: position, plane normal, count and creation/last-observation
/// times (epochs), sampled every `stride` surfels.
fn write_surfel_dump(path: &Utf8PathBuf, map: &SurfelMap, stride: usize) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create surfel dump `{path}`"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "x,y,z,nx,ny,nz,count,created_time_sec,last_observed_time_sec"
    )?;
    let stride = stride.max(1);
    let mut written = 0usize;
    for (index, surfel) in map.surfels().enumerate() {
        if surfel.parent.is_some() || index % stride != 0 {
            continue;
        }
        let normal = surfel.eigenvectors.column(0);
        writeln!(
            writer,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6}",
            surfel.mean_w.x,
            surfel.mean_w.y,
            surfel.mean_w.z,
            normal.x,
            normal.y,
            normal.z,
            surfel.count,
            surfel.created_time_sec,
            surfel.last_observed_time_sec
        )?;
        written += 1;
    }
    writer.flush()?;
    println!("surfel dump: {written} surfels -> {path}");
    Ok(())
}

fn print_summary(stats: &ReplayStats, config: &ReplayConfig) {
    println!("replay complete");
    println!("  total messages: {}", stats.read.total_messages);
    println!("  emitted events: {}", stats.read.emitted_events);
    println!(
        "  reordered imu events: {} (max arrival regression: {:.6} s)",
        stats.read.reordered_imu_events, stats.read.max_imu_arrival_regression_sec
    );
    println!("  synchronized groups: {}", stats.synchronized_groups);
    println!("  processed frames: {}", stats.processed_frames);
    println!("  failed groups: {}", stats.failed_groups);
    println!("  max pending lidar: {}", stats.max_pending_lidar);
    println!(
        "  dropped lidar before first imu: {}",
        stats.dropped_lidar_before_first_imu
    );
    println!("  pending lidar at eof: {}", stats.pending_lidar_at_eof);
    println!("  storm mode: {}", config.storm_mode);
    println!(
        "  time offset lidar-to-imu: {:.9} s (t_imu_sync = t_imu_raw - offset)",
        config.time_offset_lidar_to_imu_sec
    );
    println!(
        "  imu time raw/shifted first: {:?} / {:?}",
        stats.first_imu_raw_time_sec, stats.first_imu_time_sec
    );
    println!(
        "  imu time raw/shifted last: {:?} / {:?}",
        stats.last_imu_raw_time_sec, stats.last_imu_time_sec
    );
    println!(
        "  lidar time first/last: {:?} / {:?}",
        stats.first_lidar_time_sec, stats.last_lidar_time_sec
    );
    if let Some(error) = &stats.first_pipeline_error {
        println!("  first pipeline error: {error}");
    }
}

fn main() -> Result<()> {
    let args = ReplayArgs::parse(&env::args().collect::<Vec<_>>())?;
    let pipeline_config = read_from_config_path(&args.config_path)
        .with_context(|| format!("failed to read config `{}`", args.config_path))?;
    let offset = pipeline_config
        .common
        .time_offset_lidar_to_imu
        .unwrap_or(0.0);
    if !offset.is_finite() {
        bail!("time_offset_lidar_to_imu must be finite");
    }
    let replay_config = ReplayConfig {
        playback_rate: args.playback_rate,
        channel_capacity: args.channel_capacity,
        time_offset_lidar_to_imu_sec: offset,
        storm_mode: args.storm_mode,
        collect_state: args.state_dump_path.is_some(),
    };
    let mut pipeline = MainPipeline::new(pipeline_config);
    if let Some(dir) = &args.keyframe_dir {
        pipeline.set_keyframe_export(dir.as_std_path().to_path_buf(), args.keyframe_stride);
    }
    pipeline.map.set_assoc_diag_enabled(args.assoc_diag);
    let mut revisit_labels = match &args.revisit_mask_path {
        Some(path) => RevisitLabels::load(path)?,
        None => RevisitLabels::default(),
    };
    let stats = run_spsc(
        args.bag_path,
        &mut pipeline,
        replay_config.clone(),
        &mut revisit_labels,
    )?;
    if let Some(path) = &args.trajectory_path {
        write_trajectory(path, &stats.trajectory)?;
    }
    if let Some(path) = &args.surfel_map_path {
        write_surfel_map(path, &pipeline.map)?;
    }
    if let Some(path) = &args.gaussian_ply_path {
        write_gaussian_ply(path, &pipeline.map)?;
    }
    if let Some(path) = &args.surfel_dump_path {
        write_surfel_dump(path, &pipeline.map, args.surfel_dump_stride)?;
    }
    if let Some(path) = &args.state_dump_path {
        write_state_dump(path, &stats.state_rows)?;
    }
    print_summary(&stats, &replay_config);
    if args.map_stats {
        print_map_stats(&pipeline.map, &stats);
    }
    Ok(())
}

/// Print surfel-map lifecycle statistics (Phase 2 diagnostics).
fn print_map_stats(map: &SurfelMap, stats: &ReplayStats) {
    let lifecycle = map.lifecycle_stats();
    let active = map
        .surfels()
        .filter(|surfel| surfel.parent.is_none())
        .count();
    println!("map lifecycle stats");
    println!(
        "  total surfels: {} (active roots {}, merged tombstones {})",
        lifecycle.total,
        active,
        lifecycle.total - active
    );
    println!(
        "  active maturity: mature {}, growing {}",
        lifecycle.mature, lifecycle.growing
    );
    println!(
        "  mean count: {:.1}, mean lambda0: {:.6}",
        lifecycle.mean_count, lifecycle.mean_lambda0
    );
    println!(
        "  count buckets   [0,1,2-3,4-7,8-15,16-31,32-63,64+]: {:?}",
        lifecycle.count_hist
    );
    println!(
        "  voxel occupancy [0,1,2-3,4-7,8-15,16-31,32-63,64+]: {:?}",
        lifecycle.occupancy_hist
    );
    println!(
        "  coplanar duplicate pairs (estimated): {}",
        lifecycle.coplanar_duplicate_pairs
    );

    let maintenance = map.maintenance_stats();
    println!(
        "  maintenance: merged_total={} queue_len={} max_chain_depth={} compacted_total={} deleted_total={}",
        maintenance.merged_total,
        maintenance.queue_len,
        maintenance.max_chain_depth,
        maintenance.compacted_total,
        maintenance.deleted_total
    );

    let diag = map.assoc_diag_snapshot();
    let print_diag = |label: &str, c: &fastlio_map::surfel::AssocDiagCounters| {
        if c.queries == 0 {
            return;
        }
        let accepted = c.accepted as f64 / c.queries as f64 * 100.0;
        let no_candidate = c.no_candidate as f64 / c.queries as f64 * 100.0;
        let gate = c.gate_rejected as f64 / c.queries as f64 * 100.0;
        println!(
            "  assoc diag [{label}]: queries={} accepted={:.2}% no_candidate={:.2}% gate_rejected={:.2}%",
            c.queries, accepted, no_candidate, gate
        );
        if c.probes > 0 {
            println!(
                "    probes={} -> wide_accepted={} wide_rejected={} wide_none={}",
                c.probes, c.probe_accepted, c.probe_rejected, c.probe_none
            );
        }
        if c.accepted > 0 {
            let accepted = c.accepted as f64;
            let mean_residual_mm = c.plane_residual_sum / accepted * 1000.0;
            let residual_pct: Vec<String> = c
                .residual_hist
                .iter()
                .map(|&n| format!("{:.1}", n as f64 / accepted * 100.0))
                .collect();
            let score_pct: Vec<String> = c
                .score_hist
                .iter()
                .map(|&n| format!("{:.1}", n as f64 / accepted * 100.0))
                .collect();
            println!(
                "    accepted residual mean={mean_residual_mm:.2}mm hist[0-5,5-10,10-20,20-50,50-100,100-200,200-500,500+mm]%=[{}]",
                residual_pct.join(",")
            );
            println!(
                "    accepted score hist[0-.25,.25-.5,.5-1,1-2,2-4,4-8,8-16,16+]%=[{}]",
                score_pct.join(",")
            );
            let age_pct: Vec<String> = c
                .surfel_age_hist
                .iter()
                .map(|&n| format!("{:.1}", n as f64 / accepted * 100.0))
                .collect();
            println!(
                "    accepted surfel age hist[0-1,1-5,5-30,30-120,120-600,600+s]%=[{}]",
                age_pct.join(",")
            );
        }
    };
    print_diag("normal", &diag.normal);
    print_diag("revisit", &diag.revisit);

    if !stats.frame_durations_ms.is_empty() {
        let mut sorted = stats.frame_durations_ms.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let p50 = sorted[sorted.len() * 50 / 100];
        let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
        let max = *sorted.last().unwrap();
        println!(
            "  frame latency: mean={mean:.3} ms p50={p50:.3} p99={p99:.3} max={max:.3} ({} frames)",
            sorted.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastlio_types::{ImuSample, LidarFrame, PointXYZI, SurfelConfig, SurfelMapConfig, Vec3};
    use std::fs;

    fn imu_event(t: f64) -> SensorEvent {
        SensorEvent::Imu(ImuSample {
            time_stamp_sec: t,
            gyro: Vec3::zeros(),
            accel: Vec3::zeros(),
        })
    }

    #[test]
    fn parse_args_defaults_to_unlimited_playback() {
        let args = vec!["replay".into(), "bag.mcap".into(), "config.yaml".into()];
        let parsed = ReplayArgs::parse(&args).unwrap();
        assert_eq!(parsed.playback_rate, 0.0);
        assert_eq!(parsed.channel_capacity, 1024);
        assert!(!parsed.storm_mode);
        assert!(parsed.trajectory_path.is_none());
        assert!(parsed.surfel_map_path.is_none());
    }

    #[test]
    fn parse_args_accepts_storm_mode_with_positional_arguments() {
        let args = vec![
            "replay".into(),
            "bag.mcap".into(),
            "config.yaml".into(),
            "2.0".into(),
            "64".into(),
            "trajectory.csv".into(),
            "surfel-map.pcd".into(),
            "--storm".into(),
        ];
        let parsed = ReplayArgs::parse(&args).unwrap();
        assert!(parsed.storm_mode);
        assert_eq!(parsed.playback_rate, 2.0);
        assert_eq!(parsed.channel_capacity, 64);
        assert_eq!(
            parsed.surfel_map_path.as_deref(),
            Some("surfel-map.pcd".into())
        );
    }

    #[test]
    fn parse_args_rejects_invalid_rate_and_capacity() {
        let bad_rate = vec!["replay".into(), "bag".into(), "config".into(), "NaN".into()];
        assert!(ReplayArgs::parse(&bad_rate).is_err());
        let bad_capacity = vec![
            "replay".into(),
            "bag".into(),
            "config".into(),
            "0".into(),
            "0".into(),
        ];
        assert!(ReplayArgs::parse(&bad_capacity).is_err());
    }

    #[test]
    fn imu_offset_shifts_only_imu_timestamp() {
        let mut stats = ReplayStats::default();
        match apply_time_offset(imu_event(10.0), 0.25, &mut stats) {
            SensorEvent::Imu(imu) => assert_eq!(imu.time_stamp_sec, 9.75),
            SensorEvent::Lidar(_) => panic!("expected IMU event"),
        }
        assert_eq!(stats.first_imu_raw_time_sec, Some(10.0));
        assert_eq!(stats.first_imu_time_sec, Some(9.75));
        match apply_time_offset(
            SensorEvent::Lidar(LidarFrame::new(20.0, 20.1, vec![])),
            0.25,
            &mut stats,
        ) {
            SensorEvent::Lidar(frame) => assert_eq!(frame.base_timestamp_sec, 20.0),
            SensorEvent::Imu(_) => panic!("expected LiDAR event"),
        }
    }

    #[test]
    fn trajectory_csv_has_compact_tracking_fields() {
        let path = Utf8PathBuf::from("/tmp/fastlio-rs-trajectory-diagnostics-test.csv");
        let rows = vec![TrajectoryRow {
            timestamp_sec: 1.0,
            state: NavState::default(),
            frame_summary: PipelineFrameSummary::default(),
        }];

        write_trajectory(&path, &rows).unwrap();

        let csv = fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        let header = lines.next().unwrap();
        let row = lines.next().unwrap();
        assert_eq!(header.split(',').count(), row.split(',').count());
        assert_eq!(
            header,
            "timestamp_sec,px,py,pz,qx,qy,qz,qw,vx,vy,vz,tracking"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn blocking_push_preserves_items() {
        let (mut producer, mut consumer) = ringbuffer(1);
        let alive = AtomicBool::new(true);
        push_blocking(&mut producer, 1, &alive).unwrap();
        assert_eq!(consumer.pull(), Some(1));
        push_blocking(&mut producer, 2, &alive).unwrap();
        assert_eq!(consumer.pull(), Some(2));
    }

    #[test]
    fn blocking_push_stops_after_consumer_shutdown() {
        let (mut producer, _consumer) = ringbuffer(1);
        let dead = AtomicBool::new(false);
        assert!(producer.push(1).is_none());
        assert!(push_blocking(&mut producer, 2, &dead).is_err());
    }

    /// Rotate a vector by the (w, x, y, z) quaternion using Rodrigues' formula.
    fn rotate_by_quaternion(vector: Vec3<f64>, quaternion: [f64; 4]) -> Vec3<f64> {
        let [w, x, y, z] = quaternion;
        let q = Vec3::new(x, y, z);
        vector + 2.0 * w * q.cross(&vector) + 2.0 * q.cross(&q.cross(&vector))
    }

    #[test]
    fn rotation_matrix_converts_to_equivalent_quaternion() {
        // 90 degree rotation about z: maps (1, 0, 0) to (0, 1, 0).
        let rotation = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let quaternion = rotation_matrix_to_quaternion(&rotation);
        let image = rotate_by_quaternion(Vec3::new(1.0, 0.0, 0.0), quaternion);
        assert!((image - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-12);

        let identity = rotation_matrix_to_quaternion(&Mat3::identity());
        assert!((identity[0].abs() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_ply_writes_inria_header_and_vertex_payload() {
        let mut map = SurfelMap::new(
            SurfelMapConfig {
                voxel_size: 0.4,
                search_radius: 0,
            },
            SurfelConfig::default(),
        );
        let points: Vec<PointXYZI> = (0..16)
            .map(|i| PointXYZI {
                x: i as f32 * 0.01,
                y: (i % 3) as f32 * 0.01,
                z: 0.0,
                intensity: 0.0,
            })
            .collect();
        map.insert(points.into_iter()).unwrap();

        let path = std::env::temp_dir().join("fastlio_gaussian_ply_test.ply");
        let utf8_path = Utf8PathBuf::from_path_buf(path.clone()).expect("utf8 temp path");
        write_gaussian_ply(&utf8_path, &map).unwrap();

        let bytes = fs::read(&path).unwrap();
        let header_end = bytes
            .windows(b"end_header\n".len())
            .position(|window| window == b"end_header\n")
            .expect("header terminator")
            + b"end_header\n".len();
        let header = String::from_utf8_lossy(&bytes[..header_end]);
        assert!(header.starts_with("ply\n"));
        assert!(header.contains("format binary_little_endian 1.0"));
        for field in ["f_dc_0", "opacity", "scale_0", "rot_3"] {
            assert!(header.contains(field), "missing 3DGS field {field}");
        }
        let vertex_count: usize = header
            .lines()
            .find(|line| line.starts_with("element vertex"))
            .and_then(|line| line.split_whitespace().nth(2))
            .and_then(|value| value.parse().ok())
            .expect("vertex count");
        assert_eq!(
            vertex_count,
            map.surfels()
                .filter(|surfel| surfel.parent.is_none())
                .count()
        );
        assert_eq!(
            bytes.len() - header_end,
            vertex_count * GAUSSIAN_PLY_FIELDS * 4
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn revisit_labels_match_timestamps_only() {
        let mut labels = RevisitLabels {
            timestamps: vec![1.0, 2.0],
            cursor: 0,
        };
        assert!(!labels.is_revisit(0.5));
        assert!(labels.is_revisit(1.0));
        assert!(!labels.is_revisit(1.5));
        assert!(labels.is_revisit(2.0));
        assert!(!labels.is_revisit(3.0));
    }
}
