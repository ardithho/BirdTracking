import cv2
import numpy as np
from functional.general import kernel


def getMask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thw = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    _, thb = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(thw, thb)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel(5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel(5))
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [contour for contour in contours if cv2.contourArea(contour) > 2048]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, cnts[:1], 0, 255, -1)
    chessBoard = cv2.bitwise_and(grey, grey, mask=mask)
    _, binaryMask = cv2.threshold(chessBoard, 150, 255, cv2.THRESH_BINARY)
    binaryMask -= 255
    return binaryMask


def harrisCorners(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(grey, 2, 15, 0.07)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    v = np.percentile(dst_norm_scaled, 99)
    img[dst_norm_scaled > v] = (0, 0, 255)
    return img


def drawCorners(img, mask):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    size = (4, 7)  # (r, c)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(mask, size, flags)
    if ret:
        cnrs = cv2.cornerSubPix(mask, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, size, cnrs, ret)
        # cv2.circle(img, list(map(int, corners[len(corners)//2][0])), 3, (255, 0, 0), -1)
    return img


def calibrateUndis(img, mask):
    size = (4, 7)  # (r, c)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(mask, size, flags)
    if ret:
        objp = np.zeros((7 * 4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:4, 0:7].T.reshape(-1, 2)
        objpts = [objp]
        imgpts = [corners]
        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, mask.shape[::-1], None, None)
        # dist *= 0.1
        # undistortion
        h, w = mask.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        # crop (not quite useful :)))
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        return dst
    return img


def calibrateRemap(img, mask):  # technically the same (I guess :)
    size = (4, 7)  # (r, c)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(mask, size, flags)
    if ret:
        objp = np.zeros((7 * 4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:4, 0:7].T.reshape(-1, 2)
        objpts = [objp]
        imgpts = [corners]

        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, mask.shape[::-1], None, None)
        dist *= 0.1
        # undistortion
        h, w = mask.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        return dst
    return img


def projectPoint(img, mask):
    pt = (100, 100)
    size = (4, 7)  # (r, c)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(mask, size, flags)
    if ret:
        objp = np.zeros((7 * 4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:4, 0:7].T.reshape(-1, 2)
        objpts = [objp]
        imgpts = [corners]
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
    img = cv2.imread('../vid/fps10/chessboard.jpg')
    binaryMask = getMask(img)
    cv2.imshow('corners', drawCorners(img, binaryMask))
    cv2.imshow('undistort', calibrateUndis(img, binaryMask))
    projectPoint(img, binaryMask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
