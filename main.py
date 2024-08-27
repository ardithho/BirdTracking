import cv2
from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker
from utils.camera import Stereo
from utils.structs import Bird, Birds

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS


STRIDE = 30

tracker = Tracker('yolov8/weights/head.pt')
predictor_head = Predictor('yolov8/weights/head.pt')

vidL = 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

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
        stereo.offsetL += 1
        stereo.offsetR += 1
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
