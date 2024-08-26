import cv2
import numpy as np
import math
import os
from utils.general import kernel


def contour_valid(im, contour, area_thresh=0.0025, centre_thresh=0.2):
    h, w = im.shape[:2]
    area = cv2.contourArea(contour) > math.prod(im.shape[:2]) * area_thresh
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centre = w*centre_thresh < cX < w*(1-centre_thresh) and h*area_thresh < cY < h*(1-area_thresh)
    return area and centre


def get_mask(im):
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thw = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    _, thb = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(thw, thb)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel(5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel(5))
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [contour for contour in contours if contour_valid(im, contour)]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

    mask = np.zeros(im.shape[:2], np.uint8)
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


def find_points(im, size=(4, 7)):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_FAST_CHECK
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_USE_INTRINSIC_GUESS)
    ret, corners = cv2.findChessboardCorners(im, size, flags)
    if not ret:
        size = size[::-1]
        ret, corners = cv2.findChessboardCorners(im, size, flags)
    if ret:
        objpt = np.zeros((math.prod(size), 3), np.float32)
        objpt[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        objpts = [objpt]
        imgpts = [corners]
        return objpts, imgpts, size
    return None, None, None


def find_corners(im, size=(4, 7)):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_FAST_CHECK
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_USE_INTRINSIC_GUESS)
    ret, corners = cv2.findChessboardCorners(im, size, flags)
    if not ret:
        size = size[::-1]
        ret, corners = cv2.findChessboardCorners(im, size, flags)
    if ret:
        return corners, size
    return None, None


def calibrate(im, size=(4, 7)):
    mask = get_mask(im)
    objpts, imgpts, _ = find_points(mask, size)
    if imgpts is not None:
        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, im.shape[1::-1], None, None)
        if ret:
            return mtx, dist
    return None, None


def remap(pts, size):
    h, w = size
    ret = [[0 for _ in range(h)] for _ in range(w)]
    for r in range(w):
        for c in range(h):
            ret[r][c] = pts[c][w-r-1]
    return ret


def remap_(pts):
    return np.flip(pts.T, axis=1)


def stereo_essential_mat(frameL, frameR, size=(4, 7)):
    _, imgptsL, sizeL = find_points(frameL, size)
    _, imgptsR, sizeR = find_points(frameR, size)
    if imgptsL is None or imgptsR is None:
        return None, None

    if sizeL != sizeR:
        imgptsR = remap_(imgptsR, sizeR)
    e, mask = cv2.findEssentialMat(imgptsL[0], imgptsR[0])
    return e, mask


if __name__ == '__main__':
    img_dir = '../data/calibration/K203_K238/chessboard'
    frame_no = 2189
    frameL = cv2.imread(os.path.join(img_dir, f'l/{frame_no}.jpg'))
    frameR = cv2.imread(os.path.join(img_dir, f'r/{frame_no}.jpg'))
    e, mask = stereo_essential_mat(frameL, frameR)
    print(e)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
