import argparse
import cv2
import numpy as np


def estimate_motion(prev_image, image):
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

  flow = cv2.calcOpticalFlowFarneback(prev_image_gray, image_gray, None ,0.1,3,3,31,15,1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  mean_angle = np.mean(angle)
  mean_magnitude = np.mean(magnitude)

  angle_degrees = (mean_angle * 180) / np.pi

  return angle_degrees, mean_magnitude


parser = argparse.ArgumentParser()
parser.add_argument('--prev_image_path', required=True)
parser.add_argument('--image_path', required=True)
if __name__ == '__main__':
    args = parser.parse_args()

    prev_image = cv2.imread(args.prev_image_path)
    image = cv2.imread(args.image_path)

    angle_degrees, distance = estimate_motion(prev_image, image)

    print(f'Angle: {angle_degrees}')
    print(f'Distance: {distance}')
