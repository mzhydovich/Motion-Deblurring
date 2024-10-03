import argparse
import os
import pickle
import cv2
from typing import Optional, Tuple, List
from numpy.typing import NDArray

import numpy as np

from calibrate import calibrate_camera, read_frames


def stereo_calibrate(
    frames1: List[NDArray[np.uint8]],
    frames2: List[NDArray[np.uint8]],
    output_folder_path: str,
    rows_num: int = 5,
    columns_num: int = 8,
    world_scaling: float = 1.0,
    criteria: Optional[Tuple[int, int, float]] = None
) -> Tuple[float, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Calibrates two cameras stereoscopically using sets of image frames containing a checkerboard pattern.

    Args:
        frames1 (List[NDArray[np.uint8]]): List of image frames for camera 1.
        frames2 (List[NDArray[np.uint8]]): List of image frames for camera 2.
        output_folder_path (str): Directory where calibration data will be saved.
        rows_num (int): Number of inner corners per each chessboard row.
        columns_num (int): Number of inner corners per each chessboard column.
        world_scaling (float): Scaling factor for the checkerboard squares.
        criteria (Optional[Tuple[int, int, float]]): Termination criteria for the stereo calibration algorithm.

    Returns:
        Tuple: A tuple containing return value, camera matrices, distortion coefficients,
        rotation matrix, translation vector, essential matrix, and fundamental matrix.
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    os.makedirs(output_folder_path, exist_ok=True)
    output_folder_path1 = os.path.join(output_folder_path, 'camera_1')
    output_folder_path2 = os.path.join(output_folder_path, 'camera_2')

    print('\nCamera 1 calibration...')
    objpoints, imgpoints1, ret1, mtx1, dist1, rvecs1, tvecs1 = calibrate_camera(
        frames1,
        output_folder_path1,
        rows_num,
        columns_num,
        world_scaling,
        criteria
    )

    print('\nCamera 2 calibration...')
    objpoints, imgpoints2, ret2, mtx2, dist2, rvecs2, tvecs2 = calibrate_camera(
        frames2,
        output_folder_path2,
        rows_num,
        columns_num,
        world_scaling,
        criteria
    )

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

    print('\nStereo calibration is started...')
    ret, cam_matrix_1, dist_1, cam_matrix_2, dist_2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, frames1[0].shape[1::-1], criteria=criteria, flags=stereocalibration_flags
    )

    calibration_info = {
        "ret": ret,
        "camera_matrix_1": cam_matrix_1,
        "camera_matrix_2": cam_matrix_2,
        "distortion_coeff_1": dist_1,
        "distortion_coeff_2": dist_2,
        "rotation_matrix": R,
        "translation_vector": T,
        "essential_matrix": E,
        "fundamental_matrix": F
    }

    calibration_info_path = os.path.join(output_folder_path, 'calibration_info.pkl')
    with open(calibration_info_path, 'wb') as file:
        pickle.dump(calibration_info, file)

    print(f'Stereo calibration is finished! Results path: {output_folder_path}\n')

    return ret, cam_matrix_1, dist_1, cam_matrix_2, dist_2, R, T, E, F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_1_data_path', required=True)
    parser.add_argument('--camera_2_data_path', required=True)
    parser.add_argument('--output_folder_path', required=True)
    parser.add_argument('--rows_num', type=int, default=9)
    parser.add_argument('--columns_num', type=int, default=6)
    parser.add_argument('--world_scaling', type=float, default=1.0)
    args = parser.parse_args()

    print('Frames reading...')
    frames1 = read_frames(args.camera_1_data_path)
    frames2 = read_frames(args.camera_2_data_path)

    # sync frames ???
    assert len(frames1) == len(frames2), "Frame counts for the two cameras must be equal for stereo calibration."

    print(f'Camera1 frames num: {len(frames1)}')
    print(f'Camera2 frames num: {len(frames2)}\n')

    ret, cam_matrix_1, dist_1, cam_matrix_2, dist_2, R, T, E, F = stereo_calibrate(
        frames1,
        frames2,
        args.output_folder_path,
        args.rows_num,
        args.columns_num,
        args.world_scaling
    )

    print('ret:', ret)
