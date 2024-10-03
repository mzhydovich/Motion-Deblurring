import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--calibration_result_path', required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.calibration_result_path, 'rb') as file:
        calibration_result = pickle.load(file)

    print(calibration_result)

    # # apply calibration to image by camers_mtx and dist_coeff

    # img = cv.imread(img_path)
    # h,  w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # # undistort
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y: y + h, x: x + w]
    # cv.imwrite('calibrate_result.png', dst)


    # # compute calibration error
    # mean_error = 0
    # for i in range(len(objpoints)):
    # imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    # mean_error += error

    # print( "total error: {}".format(mean_error/len(objpoints)) )
