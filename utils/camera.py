import cv2
import numpy as np
import yaml
import pycolmap

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.calibrate import find_corners, get_mask, remap, obj_pts
from utils.general import kernel


class Camera:
    def __init__(self, path, skip=0, flash=-1, K=None, dist=None, ext=None, P=None, mre=-1):
        if Path(path).suffix == '.yaml':
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.path = cfg['path']
            self.h = cfg['h']
            self.w = cfg['w']
            self.flash = cfg['flash'] if 'flash' in cfg.keys() else flash
            self.K = np.array(cfg['K']).reshape(3, 3)
            self.dist = np.array(cfg['dist']) if 'dist' in cfg.keys() else None
            self.ext = np.array(cfg['ext']).reshape(3, 4)
            self.P = np.array(cfg['P']).reshape(3, 4) if 'P' in cfg.keys() else self.K @ self.ext
            self.mre = cfg['mre'] if 'mre' in cfg.keys() else mre
            self.setup_colmap()
        else:
            self.path = path
            self.cap = cv2.VideoCapture(path)
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.flash = flash
            self.K = K
            self.dist = dist
            self.ext = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1) if ext is None else ext
            if K is None:
                self.objpts = []
                self.imgpts = []
                self.mpts = []
                self.mpts_ = []
                self.colmap = None
            else:
                self.setup_colmap()
            if P is None and self.K is not None:
                self.P = self.K @ self.ext
            else:
                self.P = P
            self.mre = mre
        self.skip = skip
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.skip)

    def first_flash(self, kernel_size=5):
        print('Detecting camera flash...')
        count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                count += 1
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

                low = np.array([0, 0, 255])
                high = np.array([0, 0, 255])
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, low, high)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(kernel_size))

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) >= 1:
                    self.flash = count + self.skip
                    break
            else:
                print('not ret')
                self.cap.release()
                break

    def setup_colmap(self):
        if self.dist is not None:
            self.colmap = pycolmap.Camera(
                model='OPENCV',
                width=self.w,
                height=self.h,
                params=(self.K[0, 0], self.K[1, 1],  # fx, fy
                        self.K[0, 2], self.K[1, 2],  # cx, cy
                        *self.dist[:4]),  # dist: k1, k2, p1, p2
            )
        else:
            self.colmap = pycolmap.Camera(
                model='SIMPLE_PINHOLE',
                width=self.w,
                height=self.h,
                params=(self.K[0, 0],  # focal length
                        self.K[0, 2], self.K[1, 2]),  # cx, cy
            )

    def add_chessboard_pts(self, corners, size):
        if corners is not None and size is not None:
            self.objpts.append(obj_pts(*size))
            self.imgpts.append(corners)

    def add_matched_pts(self, corners):
        self.mpts.append(corners)
        self.mpts_.append(corners.squeeze)

    def calibrate(self, find_pts=True, shape=(4, 7), stride=30, resize=0.5, display=True):
        if find_pts:
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((shape[0] * shape[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)

            while self.cap.isOpened():
                for i in range(stride):
                    if self.cap.isOpened():
                        _ = self.cap.grab()
                    else:
                        break
                ret, frame = self.cap.retrieve()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_, corners = cv2.findChessboardCorners(gray, shape, None)
                    if ret_:
                        self.objpts.append(objp)
                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        self.imgpts.append(corners)
                        # Draw and display the corners
                        cv2.drawChessboardCorners(frame, shape, corners, ret)
                    if display:
                        cv2.imshow('Calibration',
                                   cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC))
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    break
            self.cap.release()
            cv2.destroyAllWindows()
        # Calibrate camera
        ret, self.K, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpts, self.imgpts, (self.w, self.h),
            None, None)
        self.dist = self.dist.flatten()
        self.P = self.K @ self.ext
        # Re-projection error
        if ret:
            self.mre = 0
            for i in range(len(self.objpts)):
                imgpts_, _ = cv2.projectPoints(self.objpts[i], rvecs[i], tvecs[i], self.K, self.dist)
                error = cv2.norm(self.imgpts[i], imgpts_, cv2.NORM_L2) / len(imgpts_)
                self.mre += error
            self.mre /= len(self.objpts)

    def undistort(self, im):
        return cv2.undistort(im, self.K, self.dist)

    def undistort_pts(self, pts):
        return cv2.undistortImagePoints(pts, self.K, self.dist)

    def save(self, path):
        with open(path, 'w') as f:
            data = {'path': self.path,
                    'K': self.K.flatten().tolist(),
                    'dist': self.dist.flatten().tolist(),
                    'ext': self.ext.flatten().tolist(),
                    'P': self.P.flatten().tolist(),
                    'h': self.h,
                    'w': self.w,
                    'flash': self.flash}
            f.write(yaml.dump(data, sort_keys=False))
        print('Camera config saved to {}'.format(path))


class Stereo:
    def __init__(self, path=None, vidL=None, vidR=None, skip=1500, stride=30, timeout=5400, size=(4, 7)):
        if path is not None:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.camL = Camera(cfg['pathL'], skip=skip,
                               flash=cfg['flashL'] if 'flashL' in cfg.keys() else -1,
                               K=np.array(cfg['KL']).reshape(3, 3),
                               dist=np.array(cfg['distL']) if 'distL' in cfg.keys() else None,
                               ext=np.array(cfg['extL']).reshape(3, 4),
                               P=np.array(cfg['PL']).reshape(3, 4) if 'PL' in cfg.keys() else None,
                               mre=cfg['mreL'] if 'mreL' in cfg.keys() else -1)
            self.camR = Camera(cfg['pathR'], skip=skip,
                               flash=cfg['flashR'] if 'flashR' in cfg.keys() else -1,
                               K=np.array(cfg['KR']).reshape(3, 3),
                               dist=np.array(cfg['distR']) if 'distR' in cfg.keys() else None,
                               ext=np.array(cfg['extR']).reshape(3, 4),
                               P=np.array(cfg['PR']).reshape(3, 4) if 'PR' in cfg.keys() else None,
                               mre=cfg['mreR'] if 'mreR' in cfg.keys() else -1)
            self.R = np.asarray(cfg['R']).reshape(3, 3) if 'R' in cfg.keys() else None
            self.T = np.asarray(cfg['T']).reshape(3, 1) if 'T' in cfg.keys() else None
            self.E = np.asarray(cfg['E']).reshape(3, 3) if 'E' in cfg.keys() else None
            self.F = np.asarray(cfg['F']).reshape(3, 3) if 'F' in cfg.keys() else None
            self.offsetL = cfg['offsetL'] if 'offsetL' in cfg.keys() else 0
            self.offsetR = cfg['offsetR'] if 'offsetR' in cfg.keys() else 0
        else:
            self.camL = Camera(vidL, skip)
            self.camR = Camera(vidR, skip)
            self.size = size
            self.R = None
            self.T = None
            self.E = None
            self.F = None
            self.calibrated = False
            self.offsetL = 0
            self.offsetR = 0
            self.objpts = []
            self.sync(stride=stride, timeout=timeout)

    def sync(self, stride, timeout):
        self.camL.first_flash()
        self.camR.first_flash()
        if self.camL.flash >= 0 and self.camR.flash >= 0:
            print('Calibrating cameras...')
            self.offsetL = self.camL.flash
            self.offsetR = self.camR.flash
            self.camL.cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsetL)
            self.camR.cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsetR)
            count = 0
            while count < timeout and self.camL.cap.isOpened() and self.camR.cap.isOpened():
                for i in range(stride):
                    _ = self.camL.cap.grab()
                    _ = self.camR.cap.grab()
                self.offsetL += stride
                self.offsetR += stride
                count += stride
                retL, frameL = self.camL.cap.retrieve()
                retR, frameR = self.camR.cap.retrieve()
                if retL and retR:
                    self.find_chessboard(frameL, frameR)
                else:
                    break
            self.camL.calibrate(find_pts=False)
            self.camR.calibrate(find_pts=False)
            print('Syncing cameras...')
            self.calibrate()
        else:
            print('Camera sync failed.')
        self.camL.cap.release()
        self.camR.cap.release()

    def find_chessboard(self, frameL, frameR, show=False):
        cnrL, sizeL = find_corners(get_mask(frameL), self.size)
        cnrR, sizeR = find_corners(get_mask(frameR), self.size)
        self.camL.add_chessboard_pts(cnrL, sizeL)
        self.camR.add_chessboard_pts(cnrR, sizeR)
        if show:
            cv2.drawChessboardCorners(frameL, sizeL, cnrL, cnrL is not None)
            cv2.drawChessboardCorners(frameR, sizeR, cnrR, cnrR is not None)
            cv2.imshow('chessboard',
                       cv2.resize(cv2.vconcat([frameL, frameR]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1)
        if cnrL is not None and cnrR is not None:
            if sizeL != sizeR:
                cnrR = remap(cnrR, sizeR)
            self.camL.add_matched_pts(cnrL)
            self.camR.add_matched_pts(cnrR)
            self.objpts.append(obj_pts(*sizeL))

    def calibrate(self):
        self.calibrated, self.camL.K, self.camL.dist, self.camR.K, self.camR.dist, self.R, self.T, self.E, self.F \
            = cv2.stereoCalibrate(
            self.objpts, self.camL.mpts, self.camR.mpts,
            self.camL.K, self.camL.dist,
            self.camR.K, self.camR.dist,
            (self.camL.w, self.camL.h),
            self.R, self.T, self.E, self.F)
        self.camL.P = self.camL.K @ self.camL.ext
        self.camL.setup_colmap()
        self.camR.ext = np.concatenate([self.R, self.T], axis=1)
        self.camR.P = self.camR.K @ self.camR.ext
        self.camR.setup_colmap()

    def save(self, path):
        with open(path, 'w') as f:
            data = {'pathL': self.camL.path,
                    'KL': self.camL.K.flatten().tolist(),
                    'distL': self.camL.dist.flatten().tolist(),
                    'extL': self.camL.ext.flatten().tolist(),
                    'PL': self.camL.P.flatten().tolist(),
                    'hL': self.camL.h,
                    'wL': self.camL.w,
                    'flashL': self.camL.flash,
                    'offsetL': self.offsetL,
                    'pathR': self.camR.path,
                    'KR': self.camR.K.flatten().tolist(),
                    'distR': self.camR.dist.flatten().tolist(),
                    'extR': self.camR.ext.flatten().tolist(),
                    'PR': self.camR.P.flatten().tolist(),
                    'hR': self.camR.h,
                    'wR': self.camR.w,
                    'flashR': self.camR.flash,
                    'offsetR': self.offsetR,
                    'R': self.R.flatten().tolist(),
                    'T': self.T.tolist(),
                    'E': self.E.flatten().tolist(),
                    'F': self.F.flatten().tolist()}
            f.write(yaml.dump(data, sort_keys=False))
        print('Stereo config saved to {}'.format(path))


if __name__ == '__main__':
    import yaml
    from pathlib import Path

    STRIDE = 30
    vidL = str(ROOT / 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4')
    vidR = str(ROOT / 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4')

    # sync videos and calibrate cameras
    stereo = Stereo(vidL=vidL, vidR=vidR, stride=STRIDE)
    stereo.save(ROOT / 'data/calibrate/cam.yaml')
