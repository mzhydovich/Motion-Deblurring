import argparse
import cv2
import numpy as np
from scipy.signal import correlate2d
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


def find_kernel_by_frames_mean(frames):
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


def find_kernel_by_frames(frames, kernel_length=3, kernel_width=1):
    trajectory_x = []
    trajectory_y = []
    for i in range(1, len(frames)):
        image_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_image_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_image_gray, image_gray, None ,0.1,3,3,31,15,1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        avg_flow_x = np.mean(flow[:, :, 0])
        avg_flow_y = np.mean(flow[:, :, 1])

        trajectory_x.append(avg_flow_x)
        trajectory_y.append(avg_flow_y)

    trajectory_image = np.zeros((800, 800), dtype=np.float32)

    trajectory_length = kernel_length
    trajectory_x_center = int(trajectory_image.shape[1] / 2)
    trajectory_y_center = int(trajectory_image.shape[0] / 2)
    
    prev_x = trajectory_x_center
    prev_y = trajectory_y_center
    for i in range(len(trajectory_x)):
        next_x = int(prev_x + trajectory_x[i] * trajectory_length)
        next_y = int(prev_y + trajectory_y[i] * trajectory_length)

        l = ((trajectory_x[i] * trajectory_length) ** 2 + (trajectory_y[i] * trajectory_length) ** 2) ** 0.5
        clr = 1.0 / l

        cv2.line(trajectory_image, (prev_x, prev_y), (next_x, next_y), (clr), kernel_width)
        prev_x = next_x
        prev_y = next_y

    cv2.imwrite('kernel_800_l10_w100.png', trajectory_image * 1500.0)


def find_kernel_by_frames_corr(frames, kernel_length=1, kernel_width=1):
    trajectory_x = []
    trajectory_y = []
    for i in range(1, len(frames)):
        image_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_image_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        cr2d = correlate2d(image_gray[450:550, 450:550], prev_image_gray[450:550, 450:550])
        coords = np.unravel_index(np.argmax(cr2d, axis=None), cr2d.shape)

        avg_flow_x = coords[1]
        avg_flow_y = coords[0]

        trajectory_x.append(avg_flow_x)
        trajectory_y.append(avg_flow_y)
    
    trajectory_image = np.zeros((frames[0].shape[0], frames[1].shape[1]), dtype=np.float32)

    trajectory_length = kernel_length
    trajectory_x_center = int(trajectory_image.shape[1] / 4)
    trajectory_y_center = int(trajectory_image.shape[0] / 2)
    
    prev_x = trajectory_x_center
    prev_y = trajectory_y_center
    for i in range(len(trajectory_x)):
        next_x = int(prev_x + trajectory_x[i] / 8 * trajectory_length)
        next_y = int(prev_y + trajectory_y[i] * trajectory_length)

        l = ((trajectory_x[i] * trajectory_length) ** 2 + (trajectory_y[i] * trajectory_length) ** 2) ** 0.5
        clr = 1.0 / l

        cv2.line(trajectory_image, (prev_x, prev_y), (next_x, next_y), (clr), kernel_width)
        prev_x = next_x
        prev_y = next_y
    
    return trajectory_image


parser = argparse.ArgumentParser()
parser.add_argument('--video_path', required=True)
parser.add_argument('--image_time', required=True)
parser.add_argument('--end_video_time', required=True)
parser.add_argument('--exp_time', required=True)
parser.add_argument('--kernel_width', default=1)
parser.add_argument('--kernel_length', default=1)
parser.add_argument('--mode')
if __name__ == '__main__':
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    image_time = datetime.strptime(args.image_time, "%H:%M:%S.%f")
    video_end_time = datetime.strptime(args.end_video_time, "%H:%M:%S.%f")
    video_start_time = (video_end_time - timedelta(seconds=duration))

    start_frame_time = (image_time - video_start_time).total_seconds()
    end_frame_time = start_frame_time + int(args.exp_time)

    selected_frames = extract_frames(cap, start_frame_time, end_frame_time)

    if args.mode == 'mean':
        kernel, avg_angle, avg_distance = find_kernel_by_frames_mean(selected_frames)
        print(f'Angle: {avg_angle}')
        print(f'Distance: {avg_distance}')
    else:
        kernel = find_kernel_by_frames_corr(selected_frames, args.kernel_length, args.kernel_width)

    cv2.imwrite('kernel.png', kernel * 1500.0)
