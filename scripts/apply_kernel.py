import argparse
import cv2
import numpy as np


def motion_kernel(angle, d, sz=65):
    d = int(d)
    sz = int(sz)
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def apply_kernel(img, kernel, noise=10):
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

    res = np.roll(res, -kh // 2, 0)
    res = np.roll(res, -kw // 2, 1)

    return res


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True)
parser.add_argument('--output_path', default='result_800_l10_w10.png')
parser.add_argument('--kernel_path')
parser.add_argument('--angle')
parser.add_argument('--distance')
parser.add_argument('--sz', default=65)
parser.add_argument('--snr', default=10)
if __name__ == '__main__':
    args = parser.parse_args()

    img = cv2.imread(args.image_path, 0)
    img = np.float32(img) / 255.0

    img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    if args.kernel_path:
        kernel = cv2.imread(args.kernel_path, 0)
        kernel = np.float32(kernel)
    else:
        angle_in_radians = np.deg2rad(float(args.angle))
        kernel = motion_kernel(angle_in_radians, args.distance, args.sz)

    res = apply_kernel(img, kernel, noise=int(args.snr))
    cv2.imwrite(args.output_path, res * 255)
