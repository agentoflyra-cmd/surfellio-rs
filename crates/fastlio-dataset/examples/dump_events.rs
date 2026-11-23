use fastlio_dataset::{SensorEvent, read_mcap_events_on_topics};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let lidar_topic = &args[2];
    let imu_topic = &args[3];
    let mut lidar = 0;
    let mut imu = 0;
    read_mcap_events_on_topics(path, lidar_topic, imu_topic, |ev| {
        match ev {
            SensorEvent::Lidar(f) => {
                if lidar < 5 {
                    println!(
                        "lidar base={:.6} end={:.6} npts={} off0={:?} off1={:?}",
                        f.base_timestamp_sec,
                        f.end_timestamp_sec(),
                        f.points.len(),
                        f.points.get(0).map(|p| p.offset_time_sec),
                        f.points.get(1).map(|p| p.offset_time_sec)
                    );
                }
                lidar += 1;
            }
            SensorEvent::Imu(i) => {
                if imu < 5 {
                    println!("imu t={:.6}", i.time_stamp_sec);
                }
                imu += 1;
            }
        }
        if lidar + imu > 15 {
            std::process::exit(0);
        }
        Ok(())
    })?;
    Ok(())
}
