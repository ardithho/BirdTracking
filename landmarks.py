import cv2
import numpy as np

from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker
from utils.structs import Bird, Birds

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS


STRIDE = 1
SAVE = True

tracker = Tracker('yolov8/weights/head.pt')
predictor_head = Predictor('yolov8/weights/head.pt')

vid = 'data/vid/fps120/K203_K238_1_GH040045.mp4'
cap = cv2.VideoCapture(vid)

if SAVE:
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('data/out/landmarks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps//STRIDE, (w, h))

birds = Birds()
while cap.isOpened():
    for i in range(STRIDE):
        _ = cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        head = tracker.tracks(frame)[0].boxes.cpu().numpy()
        feat = detect_features(frame, head)
        birds.update([Bird(head, feat) for head, feat in zip(head, feat)], frame)

        display = birds.plot()
        cv2.imshow('display', display)
        if SAVE: writer.write(display)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)
    else:
        break

cap.release()
if SAVE: writer.release()
cv2.destroyAllWindows()
