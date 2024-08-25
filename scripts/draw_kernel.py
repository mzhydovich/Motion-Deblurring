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


parser = argparse.ArgumentParser()
parser.add_argument('--angle', required=True)
parser.add_argument('--distance', required=True)
parser.add_argument('--sz', default=65)
parser.add_argument('--output_path', default='kernel.png')
if __name__ == '__main__':
    args = parser.parse_args()

    angle_in_radians = np.deg2rad(float(args.angle))
    kernel = motion_kernel(angle_in_radians, args.distance, args.sz)

    cv2.imwrite(args.output_path, kernel * 255)
