import argparse
import cv2
import numpy as np
from datetime import datetime, timedelta
from numpy.typing import NDArray
from typing import List, Tuple


def find_similar_blocks(img1: NDArray[np.uint8], img2: NDArray[np.uint8], block_size: int = 100, w_size: int = 120) -> List[Tuple[int, int]]:
    """
    Finds similar blocks between two images.

    Args:
        img1 (NDArray[np.uint8]): The first image.
        img2 (NDArray[np.uint8]): The second image.
        block_size (int): The size of the block to compare.
        w_size (int): The width of the search window.

    Returns:
        List[Tuple[int, int]]: The offset of the best matching block in (x, y) coordinates.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if block_size * 2 > min(h1, w1, h2, w2):
        raise ValueError("Block size must not exceed image size")

    center_x = w1 // 2
    center_y = h1 // 2
    center_block = img1[center_y-block_size:center_y+block_size, center_x-block_size:center_x+block_size]

    center_x2 = w2 // 2
    center_y2 = h2 // 2
    start_x2, start_y2 = center_x2 - w_size, center_y2 - w_size
    end_x2, end_y2 = center_x2 + w_size, center_y2 + w_size
    center_region = img2[start_y2:end_y2, start_x2:end_x2]

    min_diff = -2
    best_offset = (0, 0)
    for y2 in range(block_size, center_region.shape[0] - block_size + 1):
        for x2 in range(block_size, center_region.shape[1] - block_size + 1):
            block2 = center_region[y2-block_size:y2+block_size, x2-block_size:x2+block_size]
            corr = np.corrcoef(center_block.flatten(), block2.flatten())[0, 1]
            if corr > min_diff:
                min_diff = corr
                best_offset = (x2 - center_x + start_x2, y2 - center_y + start_y2)

    return [best_offset]


def extract_frames(video_capture: cv2.VideoCapture, start_time: float, end_time: float) -> List[NDArray[np.uint8]]:
    """
    Extracts frames from a video within a specific time range.

    Args:
        video_capture (cv2.VideoCapture): The video capture object.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        List[NDArray[np.uint8]]: A list of extracted frames.
    """
    frames = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if start_time <= current_time <= end_time:
            frames.append(frame)
        if current_time > end_time:
            break
    video_capture.release()

    return frames


def kernel_points(frames: List[NDArray[np.uint8]], period: int, block_size: int, win_size: int) -> List[List[Tuple[int, int]]]:
    """
    Computes kernel points for given frames.

    Args:
        frames (List[NDArray[np.uint8]]): List of video frames.
        period (int): Period to sample frames.
        block_size (int): Block size for block matching.
        win_size (int): Size of the search window.

    Returns:
        List[List[Tuple[int, int]]]: A list of kernel points.
    """
    first = frames[0]
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    coords = []
    for i in range(1, len(frames), period):
        frame = frames[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coords.append(find_similar_blocks(first, frame, block_size, win_size))
        first = frame
    return coords


def mat_array_normalization(array_of_matrix: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    Normalizes an array of matrices by their sum.

    Args:
        array_of_matrix (NDArray[np.uint8]): An array of matrices to normalize.

    Returns:
        NDArray[np.float32]: The normalized matrices.
    """
    normalized_matrix = np.empty_like(array_of_matrix, dtype=np.float32)
    for i, matr in enumerate(array_of_matrix):
        sum_values = np.sum(matr)
        if sum_values != 0:
            normalized_matrix[i] = matr.astype(np.float32) / sum_values
        else:
            normalized_matrix[i] = matr.astype(np.float32)
    return normalized_matrix

def mat_sum(array_of_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Computes the sum of matrices in an array.

    Args:
        array_of_matrix (NDArray[np.float32]): An array of matrices.

    Returns:
        NDArray[np.float32]: The sum of the matrices.
    """
    sum_matrix = np.sum(array_of_matrix, axis=0)
    return sum_matrix


def draw_line(matr: NDArray[np.uint8], x: Tuple[int, int], y: Tuple[int, int], width: int = 1) -> NDArray[np.uint8]:
    """
    Draws a line on an image matrix.

    Args:
        matr (NDArray[np.uint8]): The image matrix.
        x (Tuple[int, int]): Starting coordinate of the line.
        y (Tuple[int, int]): Ending coordinate of the line.
        width (int): Width of the line.

    Returns:
        NDArray[np.uint8]: The image matrix with the line drawn.
    """
    cv2.line(matr, x, y, 255, width, lineType=cv2.LINE_AA)
    return matr


def draw_coordinates(image_array: NDArray[np.uint8], center_x: int, center_y: int, offsets: List[List[Tuple[int, int]]], width: int) -> NDArray[np.uint8]:
    """
    Draws lines between coordinates on each image in the array of images.

    Args:
        image_array (NDArray[np.uint8]): Array of image matrices.
        center_x (int): Center x-coordinate.
        center_y (int): Center y-coordinate.
        offsets (List[List[Tuple[int, int]]]): List of offsets.
        width (int): Width of the line.

    Returns:
        NDArray[np.uint8]: The array of images with lines drawn.
    """
    for count_iter, shift in enumerate(offsets):
        x, y = shift[0]
        end_x = center_x + (x * 2 + 1)
        end_y = center_y + (y * 2)
        x1, y1 = (center_x, center_y), (end_x, end_y)
        image_array[count_iter] = draw_line(image_array[count_iter], x1, y1, width)
        center_x, center_y = end_x, end_y
    return image_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--image_time', required=True)
    parser.add_argument('--end_video_time', required=True)
    parser.add_argument('--exp_time', required=True)
    parser.add_argument('--kernel_size', required=True)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=100)
    parser.add_argument('--win_size', type=int, default=150)
    parser.add_argument('--kernel_width', type=int, default=1)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    image_time = datetime.strptime(args.image_time, "%H:%M:%S.%f")
    video_end_time = datetime.strptime(args.end_video_time, "%H:%M:%S.%f")
    video_start_time = (video_end_time - timedelta(seconds=duration))

    start_frame_time = (image_time - video_start_time).total_seconds()
    end_frame_time = start_frame_time + float(args.exp_time)

    selected_frames = extract_frames(cap, start_frame_time, end_frame_time)
    coords = kernel_points(selected_frames, args.period, args.block_size, args.win_size)

    kernel_size = int(args.kernel_size)
    array_of_matrix = np.zeros((len(coords), kernel_size, kernel_size), dtype=np.uint8)
    k = draw_coordinates(array_of_matrix, kernel_size // 2, kernel_size // 2, coords, args.kernel_width)
    k = mat_array_normalization(k)
    kernel = mat_sum(k)

    cv2.imwrite('kernel.png', kernel * 1500.0)
