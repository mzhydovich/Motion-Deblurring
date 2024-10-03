import argparse
import os
import glob
import pickle
import cv2
import tqdm
from typing import Optional, List, Tuple
from numpy.typing import NDArray

import numpy as np


def calibrate_camera(
    images: List[NDArray[np.uint8]],
    output_folder_path: str,
    rows_num: int = 5,
    columns_num: int = 8,
    world_scaling: float = 1.0,
    criteria: Optional[Tuple[int, int, float]] = None
) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], float, NDArray[np.float32], NDArray[np.float32], List[NDArray[np.float32]], List[NDArray[np.float32]]]:
    """
    Calibrates a single camera using a set of image frames containing a checkerboard pattern.

    Args:
        images (List[NDArray[np.uint8]]): List of image frames containing a checkerboard pattern.
        output_folder_path (str): Directory where calibration images and data will be saved.
        rows_num (int): Number of inner corners per a chessboard row.
        columns_num (int): Number of inner corners per a chessboard column.
        world_scaling (float): Scaling factor for the checkerboard's squares.
        criteria (Optional[Tuple[int, int, float]]): Termination criteria for the cornerSubPix algorithm.

    Returns:
        Tuple: A tuple containing object points, image points, return value,
        camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((rows_num * columns_num, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows_num, 0:columns_num].T.reshape(-1, 2)
    objp *= world_scaling

    imgpoints = []
    objpoints = []
    print('Calibration is started...\n')

    output_corners_path = os.path.join(output_folder_path, f'chessboard_corners')
    os.makedirs(output_corners_path, exist_ok=True)

    for index, frame in tqdm.tqdm(enumerate(images)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (rows_num, columns_num), None)
        
        if ret:
            conv_size = (11, 11)
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows_num, columns_num), corners, ret)

            output_path = os.path.join(output_corners_path, f'corners_{index}.png')
            cv2.imwrite(output_path, frame)

            objpoints.append(objp)
            imgpoints.append(corners)

    success_frames_num = len(imgpoints)
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (gray.shape[::-1]), None, None)

    calibration_info = {
        "ret": ret,
        "camera_matrix": cam_mtx,
        "distortion_coeff": dist,
        "rotation_vectors": rvecs,
        "translation_vectors": tvecs,
        "success_frames_num": success_frames_num
    }

    calibration_info_path = os.path.join(output_folder_path, 'calibration_info.pkl')
    with open(calibration_info_path, 'wb') as file:
        pickle.dump(calibration_info, file)

    print(f'Success frames num used for calibration: {success_frames_num}\n')
    print(f'Calibration is finished! Results path: {output_folder_path}\n')

    return objpoints, imgpoints, ret, cam_mtx, dist, rvecs, tvecs


def read_frames(data_path: str) -> List[NDArray[np.uint8]]:
    """
    Reads frames from a directory of images or a video file.

    Args:
        data_path (str): Path to a folder of images or a video file.

    Returns:
        List[NDArray[np.uint8]]: A list of image frames.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    frames = []

    if os.path.isdir(data_path):
        images_names = glob.glob(os.path.join(data_path, '*'))
        for im_name in images_names:
            im = cv2.imread(im_name, 1)
            frames.append(im)

    elif os.path.isfile(data_path) and data_path.endswith(video_extensions):
        video_cap = cv2.VideoCapture(data_path)
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            frames.append(frame)
        video_cap.release()
    else:
        raise ValueError('Error! Check if the path or its contents are correct.')

    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_data_path', required=True)
    parser.add_argument('--output_folder_path', required=True)
    parser.add_argument('--rows_num', type=int, default=9)
    parser.add_argument('--columns_num', type=int, default=6)
    parser.add_argument('--world_scaling', type=float, default=1.0)
    args = parser.parse_args()

    print('Frames reading...\n')
    frames = read_frames(args.camera_data_path)
    print(f'Frames num: {len(frames)}')

    objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        frames,
        args.output_folder_path,
        args.rows_num,
        args.columns_num,
        args.world_scaling
    )
