from yolov8.detect import Detect
from functional.sync import sync


model_head = Detect('yolov8/weights/head.pt')
model_feat = Detect('yolov8/weights/feat.pt')


vidL = 'data'
vidR = 'data'

offsetL, offsetR, e, _ = sync(vidL, vidR)
