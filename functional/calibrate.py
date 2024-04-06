import cv2
import numpy as np
import math
import os
from functional.general import kernel


def contour_valid(img, contour, area_thresh=0.0025, centre_thresh=0.2):
    h, w = img.shape[:2]
    area = cv2.contourArea(contour) > math.prod(img.shape[:2])*area_thresh
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centre = w*centre_thresh < cX < w*(1-centre_thresh) and h*area_thresh < cY < h*(1-area_thresh)
    return area and centre


def get_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thw = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    _, thb = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(thw, thb)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel(5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel(5))
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [contour for contour in contours if contour_valid(img, contour)]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, cnts[:1], 0, 255, -1)
    chessboard = cv2.bitwise_and(grey, grey, mask=mask)
    _, binary_mask = cv2.threshold(chessboard, 150, 255, cv2.THRESH_BINARY)
    binary_mask -= 255
    return binary_mask


def harris_corners(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(grey, 2, 15, 0.07)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    v = np.percentile(dst_norm_scaled, 99)
    img[dst_norm_scaled > v] = (0, 0, 255)
    return img


def draw_corners(img, size=(4, 7)):
    mask = get_mask(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(mask, size, flags)
    if ret:
        cnrs = cv2.cornerSubPix(mask, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, size, cnrs, ret)
    return img


def find_points(img, size=(4, 7)):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(img, size, flags)
    if not ret:
        size = size[::-1]
        ret, corners = cv2.findChessboardCorners(img, size, flags)
    if ret:
        objpt = np.zeros((math.prod(size), 3), np.float32)
        objpt[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        objpts = [objpt]
        imgpts = [corners]
        return objpts, imgpts, size
    return None, None, None


def calibrate_undis(img, size=(4, 7)):
    mask = get_mask(img)
    objpts, imgpts, _ = find_points(mask, size)
    if imgpts is None:
        return img

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img.shape[::-1], None, None)
    # dist *= 0.1
    # undistortion
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
    # crop
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst


def calibrate_remap(img, size=(4, 7)):  # technically the same
    mask = get_mask(img)
    objpts, imgpts, _ = find_points(mask, size)
    if imgpts is None:
        return img

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img.shape[::-1], None, None)
    dist *= 0.1
    # undistortion
    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return dst


def remap(pts, size):
    h, w = size
    ret = [[0 for _ in range(h)] for _ in range(w)]
    for r in range(w):
        for c in range(h):
            ret[r][c] = pts[c][w-r-1]
    return ret


def stereo_essential_mat(frameL, frameR, size=(4, 7)):
    maskL = get_mask(frameL)
    maskR = get_mask(frameR)
    _, imgptsL, sizeL = find_points(maskL, size)
    _, imgptsR, sizeR = find_points(maskR, size)
    if imgptsL is None or imgptsR is None:
        return None, None

    if sizeL != sizeR:
        imgptsR = remap(imgptsR, sizeR)
    e, mask = cv2.findEssentialMat(imgptsL[0], imgptsR[0])
    return e, mask


def essential_matrix(img1, img2, mask1, mask2):
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    pass


def project_point(img, size=(4, 7)):
    pt = (100, 100)
    mask = get_mask(img)
    objpts, imgpts = find_points(mask)
    if imgpts is not None:
        corners = imgpts[0]
        world_origin = list(map(int, corners[len(corners)//2][0]))
        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, mask.shape[::-1], None, None)

        intrinstic = np.matrix(mtx)[:, :-1]
        cam_coord = np.linalg.inv(intrinstic) @ [*pt, 1]
        print(cam_coord)
        rvec, tvec = cv2.solvePnP(objpts, imgpts, intrinstic)
        extrinsic = np.hstack((intrinstic, np.zeros((intrinstic.shape[0], 1), dtype=intrinstic.dtype)))
        print(np.linalg.inv(intrinstic))
        world_coord = np.linalg.inv(intrinstic) @ cam_coord.T
        print(world_coord)


if __name__ == "__main__":
    # img = cv2.imread('../data/calibration/fps10/chessboard.jpg')
    # cv2.imshow('corners', draw_corners(img))
    # cv2.imshow('undistort', calibrate_undis(img))
    # project_point(img)
    img_dir = '../data/calibration/K203_K238/chessboard'
    frame_no = 2189
    frameL = cv2.imread(os.path.join(img_dir, f'l/{frame_no}.jpg'))
    frameR = cv2.imread(os.path.join(img_dir, f'r/{frame_no}.jpg'))
    e, mask = stereo_essential_mat(frameL, frameR)
    print(e)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
