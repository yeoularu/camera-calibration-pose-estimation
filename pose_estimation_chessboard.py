import numpy as np
import cv2 as cv

# The given video and calibration data
input_file = './data/chessboard.mov'
K = np.array([[1.43624645e+03, 0, 8.21399444e+02],
              [0, 1.43814864e+03, 5.27026483e+02],
              [0, 0, 1]])
dist_coeff = np.array([4.60598063e-02,  1.99846779e-01, -9.12081076e-04,  6.43551674e-04,
                       -8.00178409e-01])
board_pattern = (10, 6)
board_cellsize = 0.015
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + \
    cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_file

# Prepare a 3D box for simple AR
box_lower = board_cellsize * \
    np.array([[4, 2,  0], [5, 2,  0], [5, 3,  0], [4, 3,  0]])
box_upper = board_cellsize * \
    np.array([[4, 2, -1], [5, 3, -1], [5, 2, -1], [4, 3, -1]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * \
    np.array([[c, r, 0] for r in range(board_pattern[1])
             for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, img_points = cv.findChessboardCorners(
        img, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)

        # Print the camera position
        # Alternative) scipy.spatial.transform.Rotation
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25),
                   cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
