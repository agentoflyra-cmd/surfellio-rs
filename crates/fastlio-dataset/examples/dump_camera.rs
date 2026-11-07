//! Dump a few camera colour/depth frames from an MCAP bag for the
//! LiDAR-inertial-visual / 3D Gaussian splatting data audit.
//!
//! Usage: cargo run --release -p fastlio-dataset --example dump_camera -- \
//!            <bag.mcap> <out_dir> [stride]
//!
//! Colour frames are written as binary PPM, 8-bit depth frames as binary PGM
//! (16-bit, big-endian), 32-bit float depth frames as raw `.f32` plus a `.txt`
//! with width/height/encoding. A `frames.csv` lists topic, timestamps and
//! encoding/size metadata for every dumped frame.

use anyhow::{Context, Result, bail};
use mcap::MessageStream;
use memmap2::Mmap;
use sensor_msgs::decode::decode_from_bytes;
use std::fs;
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: dump_camera <bag.mcap> <out_dir> [stride]");
    }
    let bag = &args[1];
    let out_dir = &args[2];
    let stride: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    fs::create_dir_all(out_dir)?;

    let fd = fs::File::open(bag).context("open MCAP file")?;
    let mapped = unsafe { Mmap::map(&fd) }.context("map MCAP file")?;
    let mut csv = fs::File::create(format!("{out_dir}/frames.csv"))?;
    writeln!(
        csv,
        "topic,log_time_sec,stamp_sec,encoding,width,height,step,bytes,file"
    )?;
    let mut per_topic = std::collections::HashMap::<String, usize>::new();
    for message in MessageStream::new(&mapped)? {
        let message = message?;
        let topic = message.channel.topic.as_str();
        if topic != "/camera/color" && topic != "/camera/depth" {
            continue;
        }
        let counter = per_topic.entry(topic.to_string()).or_insert(0);
        *counter += 1;
        let dumped = *counter;
        if dumped != 1 && dumped % stride != 0 {
            continue;
        }
        let Some(schema) = message.channel.schema.as_ref() else {
            continue;
        };
        if schema.name != "sensor_msgs/msg/Image" {
            continue;
        }
        let image = decode_from_bytes::<sensor_msgs::msg::Image>(&message.data)
            .with_context(|| format!("decode image on {topic}"))?;
        let stamp = image.header.stamp.sec as f64 + image.header.stamp.nanosec as f64 * 1e-9;
        println!(
            "{topic} t={:.3} enc={} {}x{} step={} bytes={}",
            stamp,
            image.encoding,
            image.width,
            image.height,
            image.step,
            image.data.len()
        );
        let tag = if topic.ends_with("color") {
            "color"
        } else {
            "depth"
        };
        let base = format!(
            "{out_dir}/{tag}_{:05}_{:.3}",
            dumped,
            message.log_time as f64 * 1e-9
        );
        let file = match image.encoding.as_str() {
            "rgb8" => {
                let path = format!("{base}.ppm");
                let mut f = fs::File::create(&path)?;
                write!(f, "P6\n{} {}\n255\n", image.width, image.height)?;
                f.write_all(&image.data)?;
                path
            }
            "mono16" | "16UC1" => {
                let path = format!("{base}.pgm");
                let mut f = fs::File::create(&path)?;
                write!(f, "P5\n{} {}\n65535\n", image.width, image.height)?;
                f.write_all(&image.data)?;
                path
            }
            "32FC1" => {
                let path = format!("{base}.f32");
                fs::write(&path, &image.data)?;
                fs::write(
                    format!("{base}.txt"),
                    format!("{} {} {}\n", image.width, image.height, image.encoding),
                )?;
                path
            }
            other => format!("unsupported-{other}"),
        };
        writeln!(
            csv,
            "{topic},{:.6},{:.6},{},{},{},{},{},{}",
            message.log_time as f64 * 1e-9,
            stamp,
            image.encoding,
            image.width,
            image.height,
            image.step,
            image.data.len(),
            file
        )?;
    }
    let total: usize = per_topic.values().sum();
    println!("dumped {total} camera messages (stride {stride})");
    Ok(())
}
