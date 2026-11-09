# File Format

cam<n>.yaml: Cameras intrinsics obtained from Kalibr (https://github.com/ethz-asl/kalibr)
- camera_matrix (3x3) 
- distortion_coefficients ([k1, k2, k3, k4])
- distortion_model (equidistant, see https://github.com/ethz-asl/kalibr/wiki/supported-models)
- projection_matrix (3x4, same as camera matrix)
- rectification_matrix
- image_width, image_height

imu.yaml: IMU intrinsics obtained from Allen Variance ROS (https://github.com/ori-drs/allan_variance_ros)
- accelerometer_noise_density
- accelerometer_random_walk
- gyroscope_noise_density
- gyroscope_random_walk
- rostopic
- update_rate

cam-lidar-imu.yaml: Camera-LiDAR-IMU extrinsics obtained from DiffCal (https://github.com/ori-drs/diffcal)
- T_cam_lidar: 4x4 homogeneous matrix that converts points in lidar frame to camera<n> frame
- C_r_CL: translation (x,y,z) of T_cam<n>_lidar
- C_q_CL: rotation quaternion (x,y,z,w) of T_cam<n>_lidar
- C_r_CI: translation (x,y,z) of T_cam<n>_imu
- C_q_CI: rotation quaternion (x,y,z,w) of T_cam<n>_imu
- rostopic

Intra-camera calibration and IMU-LiDAR can be inferred from the transformations above.

# Base Frame
An artificial frame that is located at the bottom of the device with the x-axis forward (same direction as the front camera), y-axis to the left, and z-axis to the top. It is designed to make the visualisation of the sensor pose more intuitive.

The base frame is manually set with respect to the lidar as
T_base_lidar_t_xyz_q_xyzw = [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0]

The processed (undistorted) point clouds from VILENS are provided in the base frame, not the lidar frame. 

# Calibration sequences
There are three sequences recorded for calibration
- 2024-03-19
- 2024-06-29 (although imu calibration is done on 2024-06-27)
- 2024-10-29
 

The default calibration files provided in this folder are obtained from these sequences.
- cam<n>.yaml: 2024-03-19 sequence
- imu.yaml: 2024-06-29 sequence
- cam-lidar-imu.yaml
  - cam-lidar: 2024-03-19 sequence
  - cam-imu: 2024-06-29 sequence 
  
The rationale is that 
- most sequences we have are collected in March (2024-03-xx)
- for these sequences, the closest IMU calibration we did is in June (2024-06-29)


We recommend using the calibration that is closest to the sequence that you are testing.
- 2024-03-xx sequences (the majority): use the default calibration files
- 2024-05-20 sequences (bodleian library-02): either calibration from 2024-03-19 or 2024-06-29 
- 2024-07-09 sequences (new college): use calibration from 2024-06-29 
