import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker

from utils.box import pad_boxes
from utils.reconstruct import solvePnP
from utils.structs import Bird, Birds
from utils.sim import *


STRIDE = 4
PADDING = 30

tracker = Tracker(ROOT / 'yolov8/weights/head.pt')
predictor = Predictor(ROOT / 'yolov8/weights/head.pt')

vidL = ROOT / 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = ROOT / 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

cfg_path = ROOT / 'data/calibration/cam.yaml'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/pnp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h * 2))

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['KR']).reshape(3, 3)
    dist = np.asarray(cfg['distR'])

sim = Sim()

cap = cv2.VideoCapture(str(ROOT / 'data/vid/fps120/K203_K238_1_GH040045.mp4'))
birds = Birds()
sim.flip()
T = np.eye(4)
prev_T = np.eye(4)
sim.update(T)
while cap.isOpened():
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        head = pad_boxes(predictor.predictions(frame)[0].boxes.cpu().numpy(), frame.shape, PADDING)
        feat = detect_features(frame, head)
        birds.update([Bird(head, feat) for head, feat in zip(head, feat)], frame)
        bird = birds['m'] if birds['m'] is not None else birds['f']
        if bird is not None:
            pnp, r, t, _ = solvePnP(bird, K, dist)
            if pnp:
                R, _ = cv2.Rodrigues(r)
                T[:3, :3] = R @ prev_T[:3, :3].T
                # T[:3, 3] = t.T - prev_T[:3, 3]
                prev_T[:3, :3] = R
                # prev_T[:3, 3] = t.T
                sim.update(T)
        cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

        out = cv2.vconcat([cv2.resize(birds.plot(), (w, h), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
        cv2.imshow('out', out)
        writer.write(out)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
