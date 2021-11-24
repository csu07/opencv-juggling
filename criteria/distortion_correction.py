import numpy as np
import cv2

mtx = None
dist = None


def calibrate(cam):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    x = 9
    y = 6

    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    count = 0  # count 检测成功的次数

    while True:
        res, frame = cam.read()

        if cv2.waitKey(1) & 0xFF == ord(' '):

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            res, corners = cv2.findChessboardCorners(gray, (x, y), None)  # Find the corners

            if res is True:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)
                cv2.drawChessboardCorners(frame, (x, y), corners, res)
                count += 1

                if count > 20:
                    break

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    res, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print(mtx, dist)

    mean_error = 0
    for i in range(len(obj_points)):
        img_points1, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points1, cv2.NORM_L2) / len(img_points1)
        mean_error += error

    print("total error: ", mean_error / len(obj_points))
    # # When everything done, release the capture

    np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])


def distortion_correction(img, mtx, dist):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]

    return dst


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    mtx = []
    dist = []

    try:
        npzfile = np.load('calibrate.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        calibrate(cap)

    while True:

        res, frame = cap.read()

        frame = distortion_correction(frame, mtx, dist[0:4])
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
