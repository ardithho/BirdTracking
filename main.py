import cv2
from yolov8.detect import Detect
from yolov8.detect_feat import detect_features
from functional.sync import sync


STRIDE = 30

model_head = Detect('yolov8/weights/head.pt')
model_feat = Detect('yolov8/weights/feat.pt')


vidL = 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

offsetL, offsetR, e, _ = sync(vidL, vidR, stride=STRIDE)
capL = cv2.VideoCapture(vidL)
capR = cv2.VideoCapture(vidR)
capL.set(cv2.CAP_PROP_POS_FRAMES, offsetL)
capR.set(cv2.CAP_PROP_POS_FRAMES, offsetR)

prev_frame = None
prev_boxes = None
while capL.isOpened() and capR.isOpened():
    for i in range(STRIDE):
        _ = capL.grab()
        _ = capR.grab()
        offsetL += 1
        offsetR += 1
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if retL and retR:
        headL = model_head.predictions(frameL)[0].boxes
        headR = model_head.predictions(frameR)[0].boxes
        featL = detect_features(frameL, headL)
        featR = detect_features(frameR, headR)
