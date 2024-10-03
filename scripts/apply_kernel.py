import argparse
import cv2
import numpy as np
from numpy.typing import NDArray


def motion_kernel(angle: float, d: int, sz: int = 65) -> NDArray[np.float32]:
    """
    Generates a motion blur kernel.
    
    Args:
        angle (float): The angle of motion in radians.
        d (int): The distance of motion.
        sz (int): Size of the kernel matrix.

    Returns:
        NDArray[np.float32]: The motion blur kernel.
    """
    d = int(d)
    sz = int(sz)
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def apply_kernel(img: NDArray[np.float32], kernel: NDArray[np.float32], noise: int = 10) -> NDArray[np.float32]:
    """
    Applies a specified kernel to an image with optional noise.
    
    Args:
        img (NDArray[np.float32]): The input image in frequency domain.
        kernel (NDArray[np.float32]): The kernel to apply.
        noise (int): Signal-to-noise ratio.

    Returns:
        NDArray[np.float32]: The processed image.
    """
    noise = 10 ** (-0.1 * noise)
    kernel /= kernel.sum()

    kernel_pad = np.zeros_like(img[:, :, 0])
    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel

    PSF = cv2.dft(kernel_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
    PSF2 = (PSF ** 2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
    RES = cv2.mulSpectrums(img, iPSF, 0)
    
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    res = np.roll(res, -kh // 2, axis=0)
    res = np.roll(res, -kw // 2, axis=1)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_path', default='result.png')
    parser.add_argument('--kernel_path')
    parser.add_argument('--angle', type=float)
    parser.add_argument('--distance', type=int)
    parser.add_argument('--sz', type=int, default=65)
    parser.add_argument('--snr', type=int, default=10)
    args = parser.parse_args()

    img = cv2.imread(args.image_path, 0)
    img = np.float32(img) / 255.0

    img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    if args.kernel_path:
        kernel = cv2.imread(args.kernel_path, 0)
        kernel = np.float32(kernel)
    else:
        angle_in_radians = np.deg2rad(args.angle)
        kernel = motion_kernel(angle_in_radians, args.distance, args.sz)

    res = apply_kernel(img, kernel, noise=args.snr)
    cv2.imwrite(args.output_path, res * 255)
