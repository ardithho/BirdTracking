import cv2
import numpy as np
import os

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.camera import Stereo
from utils.structs import Bird, Birds
from utils.reconstruct import triangulate_estimate
from utils.sim import *
from utils.odometry import draw_bird_matches


STRIDE = 1
BLENDER_ROOT = ROOT / 'data/blender'
NAME = 'marked'

renders_dir = BLENDER_ROOT / 'renders'
vidL = renders_dir / f'vid/{NAME}_l.mp4'
vidR = renders_dir / f'vid/{NAME}_r.mp4'
input_dir = renders_dir / 'renders'
cfg_path = os.path.join(input_dir, 'cam.yaml')
trans_path = os.path.join(input_dir, 'transforms.txt')

stereo = Stereo(path=cfg_path)
capL = cv2.VideoCapture(str(vidL))
capR = cv2.VideoCapture(str(vidR))

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/tri.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 1.5)))

birdsL = Birds()
birdsR = Birds()
T = np.eye(4)
prev_T = np.eye(4)
sim.update(T)
cam_w, cam_h = capL.get(cv2.CAP_PROP_FRAME_WIDTH), capL.get(cv2.CAP_PROP_FRAME_HEIGHT)
dummy_head = Box(0, conf=[1.],
                 xywh=np.array([[cam_w/2, cam_h/2, cam_w, cam_h]]),
                 xywhn=np.array([[.5, .5, 1., 1.]]),
                 xyxy=np.array([[0., 0., cam_w, cam_h]]),
                 xyxyn=np.array([[0., 0., 1., 1.]]))
while capL.isOpened() and capR.isOpened():
    for i in range(STRIDE):
        _ = capL.grab()
        _ = capR.grab()
    stereo.offsetL += STRIDE
    stereo.offsetR += STRIDE
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if retL and retR:
        birdsL.update([Bird(dummy_head, extract_features(frameL))], frameL)
        birdL = birdsL['m'] if birdsL['m'] is not None else birdsL['f']
        birdsR.update([Bird(dummy_head, extract_features(frameR))], frameR)
        birdR = birdsR['m'] if birdsR['m'] is not None else birdsR['f']
        ret, T, _ = triangulate_estimate(birdL, birdR, stereo)
        if ret:
            sim.update(T)
        matches = draw_bird_matches(frameL, birdL, frameR, birdR)
        out = cv2.vconcat([cv2.resize(matches, (w, int(h / 2)), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
        cv2.imshow('out', out)
        writer.write(out)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)
    else:
        break

capL.release()
capR.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
