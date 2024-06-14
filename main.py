import cv2
from yolov8.predict import Predictor, detect_features
from functional.sync import sync


STRIDE = 30

model_head = Predictor('yolov8/weights/head.pt')
model_feat = Predictor('yolov8/weights/feat.pt')


vidL = 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

# sync videos and calibrate cameras
offsetL, offsetR, e, _ = sync(vidL, vidR, stride=STRIDE)
capL = cv2.VideoCapture(vidL)
capR = cv2.VideoCapture(vidR)
# skip chessboard calibration frames
capL.set(cv2.CAP_PROP_POS_FRAMES, offsetL)
capR.set(cv2.CAP_PROP_POS_FRAMES, offsetR)

prev_frames = None
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
        headL = model_head.predictions(frameL)[0].boxes.cpu().numpy()
        headR = model_head.predictions(frameR)[0].boxes.cpu().numpy()
        featL = detect_features(frameL, headL)
        featR = detect_features(frameR, headR)
        prev_frames = {'l': frameL,  'r': frameR}
