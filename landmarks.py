import cv2

from yolov8.predict import Predictor, detect_features

from utils.box import pad_boxes
from utils.structs import Bird, Birds


RESIZE = .5
STRIDE = 1
SAVE = True
PADDING = 30

predictor = Predictor('yolov8/weights/head.pt')

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
        head = pad_boxes(predictor.predictions(frame)[0].boxes.cpu().numpy(), frame.shape, PADDING)
        feat = detect_features(frame, head)
        birds.update([Bird(head, feat) for head, feat in zip(head, feat)], frame)

        display = birds.plot()
        cv2.imshow('display',
                   cv2.resize(display, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        if SAVE:
            writer.write(display)
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
