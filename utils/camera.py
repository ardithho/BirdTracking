import math
import cv2
import numpy as np
from utils.general import kernel
from utils.calibrate import find_corners, get_mask, remap


class Camera:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.flash = -1
        self.first_flash()
        self.objpts = []
        self.imgpts = []
        self.mtx = None
        self.dist = None

    def first_flash(self, kernel_size=5, save=False):
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
                    self.flash = count
                    if save:
                        cv2.imwrite(f'data/img/sync/frame_{count}.jpg', frame)
                        cv2.imwrite(f'data/img/sync/mask_{count}.jpg', mask)
                    break
            else:
                self.cap.release()
                break

    def add_chessboard_pts(self, corners, size):
        if corners is not None and size is not None:
            o = np.zeros((math.prod(size), 3), np.float32)
            o[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
            self.objpts.append(o)
            self.imgpts.append(corners)

    def calibrate(self):
        self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            self.objpts, self.imgpts, (self.w, self.h),
            None, None)


class Stereo:
    def __init__(self, vidL, vidR, skip=1800, stride=30, size=(4, 7)):
        self.camL = Camera(vidL)
        self.camR = Camera(vidR)
        self.size = size
        self.e = None
        self.offsetL = 0
        self.offsetR = 0
        self.sync(skip=skip, stride=stride)

    def sync(self, skip, stride):
        if self.camL.flash >= 0 and self.camR.flash >= 0:
            self.offsetL = self.camL.flash + skip
            self.offsetR = self.camR.flash + skip
            self.camL.cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsetL)
            self.camR.cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsetR)
            while self.camL.cap.isOpened() and self.camR.cap.isOpened():
                for i in range(stride):
                    _ = self.camL.cap.grab()
                    _ = self.camR.cap.grab()
                self.offsetL += stride
                self.offsetR += stride
                retL, frameL = self.camL.cap.retrieve()
                retR, frameR = self.camR.cap.retrieve()
                if retL and retR:
                    if self.e is None:
                        self.calibrate(frameL, frameR)
                        if self.e is not None:
                            self.camL.calibrate()
                            self.camR.calibrate()
                            break
                else:
                    break
                self.camL.cap.release()
                self.camR.cap.release()

    def calibrate(self, frameL, frameR):
        cnrL, sizeL = find_corners(get_mask(frameL), self.size)
        cnrR, sizeR = find_corners(get_mask(frameR), self.size)
        self.camL.add_chessboard_pts(cnrL, sizeL)
        self.camR.add_chessboard_pts(cnrR, sizeR)
        if cnrL and cnrR:
            if sizeL != sizeR:
                cnrR = remap(cnrR, sizeR)
                self.e, mask = cv2.findEssentialMat(cnrL, cnrR)

