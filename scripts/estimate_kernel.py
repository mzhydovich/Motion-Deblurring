import argparse
import cv2
import numpy as np
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev_image_path', required=True)
    parser.add_argument('--image_path', required=True)
    args = parser.parse_args()

    prev_image = cv2.imread(args.prev_image_path)
    image = cv2.imread(args.image_path)
    coords = find_similar_blocks(prev_image, image)
    print(coords)
