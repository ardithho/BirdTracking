import cv2
import numpy as np
import yaml
from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker
from utils.camera import Stereo
from utils.structs import Bird, Birds
from utils.reconstruct import solvePnP
from utils.sim import *

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS


STRIDE = 30

tracker = Tracker('yolov8/weights/head.pt')
predictor_head = Predictor('yolov8/weights/head.pt')

vidL = 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

'''
# sync videos and calibrate cameras
stereo = Stereo(vidL, vidR, stride=STRIDE)
capL = cv2.VideoCapture(vidL)
capR = cv2.VideoCapture(vidR)
# skip chessboard calibration frames
capL.set(cv2.CAP_PROP_POS_FRAMES, stereo.offsetL)
capR.set(cv2.CAP_PROP_POS_FRAMES, stereo.offsetR)

birdsL = Birds()
birdsR = Birds()
prev_frames = None
while capL.isOpened() and capR.isOpened():
    for i in range(STRIDE):
        _ = capL.grab()
        _ = capR.grab()
    stereo.offsetL += STRIDE
    stereo.offsetR += STRIDE
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if retL and retR:
        headL = tracker.tracks(frameL)[0].boxes.cpu().numpy()
        headR = tracker.tracks(frameR)[0].boxes.cpu().numpy()
        featL = detect_features(frameL, headL)
        featR = detect_features(frameR, headR)
        birdsL.update([Bird(head, feat) for head, feat in zip(headL, featL)], frameL)
        birdsR.update([Bird(head, feat) for head, feat in zip(headR, featR)], frameR)

        prev_frames = {'l': frameL, 'r': frameR}
'''

mtx_path = 'data/mtx.yaml'
with open(mtx_path, 'r') as f:
    mtx = yaml.safe_load(f)
    k = np.asarray(mtx['kR']).reshape(3, 3)
    dist = np.asarray(mtx['distR'])

cap = cv2.VideoCapture('data/vid/fps120/K203_K238_1_GH040045.mp4')
birds = Birds()
prev_frame = None
T = np.eye(4)
prev_T = np.eye(4)

while cap.isOpened():
    for i in range(4):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        print('Detecting head...')
        head = list(tracker.tracks(frame))[0].boxes.cpu().numpy()
        print('Detecting features...')
        feat = detect_features(frame, head)
        print('Sorting features...')
        birds.update([Bird(head, feat) for head, feat in zip(head, feat)], frame)
        print('Reconstructing head pose...')
        bird = birds['m'] if birds['m'] is not None else birds['f']
        if bird is not None:
            pnp, r, t, _ = solvePnP(bird, k, dist)
            if pnp:
                print(r, t)
                R, _ = cv2.Rodrigues(r)
                T[:3, :3] = prev_T[:3, :3].T @ R
                # T[:3, 3] = t.T - prev_T[:3, 3]
                print(T)
                prev_T[:3, :3] = R
                # prev_T[:3, 3] = t.T
        cv2.imshow('frame', cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
        sim.update(T)
        prev_frame = frame
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
sim.close()
