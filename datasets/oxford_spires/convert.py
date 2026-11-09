"""Convert Oxford Spires PCD lidar clouds + IMU CSV into a ROS1 bag, then
rosbags-convert can turn it into MCAP for fastlio-replay."""
import csv, os, re, sys
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

BASE = 'datasets/oxford_spires/sequences/2024-03-18-christ-church-01'
LIDAR = BASE + '/lidar-clouds'
IMU = BASE + '/raw/imu.csv'
OUT = BASE + '/spires.bag'

ts = get_typestore(Stores.ROS1_NOETIC)
Header = ts.types['std_msgs/msg/Header']
Time = ts.types['builtin_interfaces/msg/Time']
PointCloud2 = ts.types['sensor_msgs/msg/PointCloud2']
PointField = ts.types['sensor_msgs/msg/PointField']
Imu = ts.types['sensor_msgs/msg/Imu']
Quaternion = ts.types['geometry_msgs/msg/Quaternion']
Vector3 = ts.types['geometry_msgs/msg/Vector3']


def parse_pcd(path):
    raw = open(path, 'rb').read()
    header_end = raw.index(b'DATA binary\n') + len(b'DATA binary\n')
    text = raw[:header_end].decode('ascii', 'ignore')
    fields = re.search(r'FIELDS (.*)', text).group(1).split()
    sizes = [int(x) for x in re.search(r'SIZE (.*)', text).group(1).split()]
    types = re.search(r'TYPE (.*)', text).group(1).split()
    counts = [int(x) for x in re.search(r'COUNT (.*)', text).group(1).split()]
    n = int(re.search(r'POINTS (\d+)', text).group(1))
    types = re.search(r'TYPE (.*)', text).group(1).split()
    np_types = {'F': 'f', 'U': 'u', 'I': 'i'}
    dtype_fields = []
    offset = 0
    pad_idx = 0
    for name, size, typ, count in zip(fields, sizes, types, counts):
        base = np_types[typ] + str(size)
        if name == '_':
            # padding must stay in the dtype, otherwise the stride is wrong
            dtype_fields.append((f'pad{pad_idx}', 'u1', size * count))
            pad_idx += 1
            offset += size * count
            continue
        dtype_fields.append((name, base, count) if count > 1 else (name, base))
        offset += size * count
    assert offset % 4 == 0
    dt = np.dtype(dtype_fields)
    arr = np.frombuffer(raw[header_end:header_end + n * dt.itemsize], dtype=dt)
    return arr, n


def build_cloud(arr, stamp_ns):
    x = arr['x'].astype(np.float32)
    y = arr['y'].astype(np.float32)
    z = arr['z'].astype(np.float32)
    inten = arr['intensity'].astype(np.float32)
    t = arr['timestamp'].astype(np.float64) if 'timestamp' in arr.dtype.names else np.zeros(len(x))
    packed = np.zeros(len(x), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('time', '<f8')])
    packed['x'] = x; packed['y'] = y; packed['z'] = z
    packed['intensity'] = inten; packed['time'] = t
    sec = stamp_ns // 1_000_000_000; nsec = stamp_ns % 1_000_000_000
    stamp = Time(sec=sec, nanosec=nsec)
    header = Header(seq=0, stamp=stamp, frame_id='os1_sensor')
    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7, count=1),
        PointField(name='time', offset=16, datatype=8, count=1),
    ]
    data = np.frombuffer(packed.tobytes(), dtype=np.uint8)
    pcs = PointCloud2(header=header, height=1, width=len(x), fields=fields,
                      is_bigendian=False, point_step=24, row_step=24 * len(x),
                      data=data, is_dense=True)
    return stamp_ns, pcs


def imu_messages():
    out = []
    zeros9 = np.zeros(9, dtype=np.float64)
    with open(IMU) as f:
        reader = csv.DictReader(f)
        for row in reader:
            secs = int(row['secs']); nsecs = int(row['nsecs'])
            ns = secs * 1_000_000_000 + nsecs
            stamp = Time(sec=secs, nanosec=nsecs)
            header = Header(seq=0, stamp=stamp, frame_id='os1_imu')
            msg = Imu(header=header,
                      orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                      orientation_covariance=zeros9,
                      angular_velocity=Vector3(x=float(row['ang_vel_x']), y=float(row['ang_vel_y']), z=float(row['ang_vel_z'])),
                      angular_velocity_covariance=zeros9,
                      linear_acceleration=Vector3(x=float(row['acc_x']), y=float(row['acc_y']), z=float(row['acc_z'])),
                      linear_acceleration_covariance=zeros9)
            out.append((ns, msg))
    return out


def main():
    events = []
    names = sorted(f for f in os.listdir(LIDAR) if f.endswith('.pcd'))
    print(f'pcd files={len(names)}', flush=True)
    for i, name in enumerate(names):
        stamp_sec = float(name.replace('.pcd', ''))
        stamp_ns = int(round(stamp_sec * 1e9))
        arr, n = parse_pcd(os.path.join(LIDAR, name))
        events.append(('lidar', *build_cloud(arr, stamp_ns)))
        if i % 500 == 0:
            print(f'  lidar {i}/{len(names)}', flush=True)
    for item in imu_messages():
        events.append(('imu', *item))
    events.sort(key=lambda e: e[1])
    print(f'events={len(events)}', flush=True)
    with Writer(OUT) as writer:
        lidar_conn = writer.add_connection('/os1_cloud_node/points', 'sensor_msgs/msg/PointCloud2', typestore=ts)
        imu_conn = writer.add_connection('/os1_cloud_node/imu', 'sensor_msgs/msg/Imu', typestore=ts)
        for kind, stamp_ns, msg in events:
            conn = lidar_conn if kind == 'lidar' else imu_conn
            raw = ts.serialize_ros1(msg, msg.__msgtype__)
            writer.write(conn, stamp_ns, raw)
    print('bag written', OUT, os.path.getsize(OUT) / 1e9, 'GB', flush=True)


if __name__ == '__main__':
    main()
