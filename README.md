## Overview

This project includes Python scripts designed for two major tasks in computer vision: camera calibration and image deblurring. These tasks are accomplished using OpenCV and NumPy, critical libraries for computer vision and numerical computations in Python.

## Scripts

### Camera Calibration

These scripts are used to calibrate single and stereo camera systems using images of a checkerboard pattern to determine camera parameters.

#### 1. Single Camera Calibration

##### `calibrate.py`

**Purpose:** Calibrates a single camera using frames containing a checkerboard pattern.

**Usage:**

```bash
python calibrate.py --camera_data_path <path_to_images_or_video> --output_folder_path <output_directory> --rows_num <num_rows> --columns_num <num_columns> --world_scaling <scaling_factor>
```

- **Arguments:**
  - `--camera_data_path`: Path to the directory or video file used for calibration.
  - `--output_folder_path`: Directory for storing calibration results.
  - `--rows_num`: Number of inner corners per chessboard row (default: 9).
  - `--columns_num`: Number of inner corners per chessboard column (default: 6).
  - `--world_scaling`: Scaling factor for the checkerboard squares (default: 1.0).

#### 2. Stereo Camera Calibration

##### `stereo_calibration.py`

**Purpose:** Performs stereo calibration on two cameras using frames containing a checkerboard pattern.

**Usage:**

```bash
python stereo_calibration.py --camera_1_data_path <path_to_camera1_data> --camera_2_data_path <path_to_camera2_data> --output_folder_path <output_directory> --rows_num <num_rows> --columns_num <num_columns> --world_scaling <scaling_factor>
```

- **Arguments:**
  - `--camera_1_data_path`: Path to images or video for the first camera.
  - `--camera_2_data_path`: Path to images or video for the second camera.
  - `--output_folder_path`: Directory for storing stereo calibration results.
  - `--rows_num`: Number of inner corners per chessboard row (default: 9).
  - `--columns_num`: Number of inner corners per chessboard column (default: 6).
  - `--world_scaling`: Scaling factor for the checkerboard squares (default: 1.0).

### Image Deblurring

These scripts apply different techniques to remove blur from images and videos, estimating kernels to correct motion blur.

#### 1. Apply Kernel to Image

##### `apply_kernel.py`

**Purpose:** Applies a motion blur kernel to an image, facilitating experiments with image deblurring.

**Usage:**

```bash
python apply_kernel.py --image_path <path_to_image> --output_path <output_image> --angle <motion_angle> --distance <motion_distance> --sz <size> --snr <signal_to_noise_ratio>
```

- **Arguments:**
  - `--image_path`: Path to the input image.
  - `--output_path`: Path to save the output image (default: 'result.png').
  - `--kernel_path`: Path to an existing kernel image, if not generating one.
  - `--angle`: Angle for the motion blur in degrees.
  - `--distance`: Distance of motion.
  - `--sz`: Size of the kernel (default: 65).
  - `--snr`: Signal-to-noise ratio for the applied kernel (default: 10).

#### 2. Estimate Motion Kernel

##### `estimate_kernel.py`

**Purpose:** Estimates the motion kernel between two images, essential for correcting motion blur.

**Usage:**

```bash
python estimate_kernel.py --prev_image_path <path_to_previous_image> --image_path <path_to_current_image>
```

- **Arguments:**
  - `--prev_image_path`: Path to the previous image.
  - `--image_path`: Path to the current image.

#### 3. Extract Kernel from Video

##### `extract_kernel_from_video.py`

**Purpose:** Extracts a kernel trajectory from a video, enabling deblurring through frame analysis.

**Usage:**

```bash
python extract_kernel_from_video.py --video_path <path_to_video> --image_time <timestamp_image_start> --end_video_time <timestamp_video_end> --exp_time <exposure_time>
```

- **Arguments:**
  - `--video_path`: Path to the video file.
  - `--image_time`: Timestamp in video for the starting frame extraction (HH:MM:SS.sss).
  - `--end_video_time`: Timestamp for the end of the video analysis (HH:MM:SS.sss).
  - `--exp_time`: Duration for the exposure time in analysis.
  - `--kernel_size`: Size of the computed kernel.
  - `--period`: Interval period for sampling frames (default: 4).
  - `--block_size`: Size of each block used for matching (default: 100).
  - `--win_size`: Window size for the search around the block (default: 150).
  - `--kernel_width`: Width of the trajectory representation (default: 1).

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

To install the necessary Python packages, use pip:

```bash
pip install opencv-python numpy
```

## Notes

- Calibration scripts require precision control; ensure images capture a fully visible checkerboard.
- Maintain correct file paths to prevent errors when reading images and videos.
- For stereo camera calibration, synchronization between both cameras is crucial for accurate results.
