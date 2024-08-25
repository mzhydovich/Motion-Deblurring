import argparse
import cv2
import numpy as np
from datetime import datetime, timedelta

def estimate_motion(prev_image, image):
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

  flow = cv2.calcOpticalFlowFarneback(prev_image_gray, image_gray, None ,0.1,3,3,31,15,1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  mean_angle = np.mean(angle)
  mean_magnitude = np.mean(magnitude)

  angle_degrees = (mean_angle * 180) / np.pi

  return angle_degrees, mean_magnitude


def extract_frames(video_capture, start_time, end_time):
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


def find_kernel_by_frames(frames):
    combined_angle = 0
    combined_distance = 0

    for i in range(1, len(frames)):
        angle, distance = estimate_motion(frames[i-1], frames[i])
        combined_angle += angle
        combined_distance += distance

    avg_angle = combined_angle / (len(frames) - 1)
    avg_distance = int(combined_distance / (len(frames) - 1))
    
    avg_angle = np.deg2rad(float(avg_angle))
    kernel = motion_kernel(avg_angle, avg_distance)

    return kernel, avg_angle, avg_distance


parser = argparse.ArgumentParser()
parser.add_argument('--video_path', required=True)
parser.add_argument('--start_time', required=True)
parser.add_argument('--end_time', required=True)
parser.add_argument('--exp_time', required=True)
if __name__ == '__main__':
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    image_time = datetime.strptime(args.start_time, "%H:%M:%S")
    video_end_time = datetime.strptime(args.end_time, "%H:%M:%S")
    video_start_time = (video_end_time - timedelta(seconds=duration))

    start_frame_time = (image_time - video_start_time).total_seconds()
    end_frame_time = start_frame_time + int(args.exp_time)

    selected_frames = extract_frames(cap, start_frame_time, end_frame_time)
    blur_kernel, avg_angle, avg_distance = find_kernel_by_frames(selected_frames)

    print(f'Angle: {avg_angle}')
    print(f'Distance: {avg_distance}')
