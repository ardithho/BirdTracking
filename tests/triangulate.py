import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.camera import Stereo
from utils.structs import Bird, Birds
from utils.reconstruct import triangulate
from utils.sim import *
from utils.odometry import draw_bird_matches


STRIDE = 1
COLOURS = {'bill': 'r', 'left_eye': 'y', 'left_tear': 'g', 'right_eye': 'y', 'right_tear': 'g'}
MARKERS = {'bill': '^', 'left_eye': 'o', 'left_tear': 'o', 'right_eye': 'o', 'right_tear': 'o'}
BLENDER_ROOT = ROOT / 'blender'
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
writer = cv2.VideoWriter(str(ROOT / 'data/out/tri_plot.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 1.5)))

birdsL = Birds()
birdsR = Birds()
T = np.eye(4)
prev_T = np.eye(4)
sim.update(T)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
scatter_plots = [None, None, None, None, None]

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

        matches = draw_bird_matches(frameL, birdL, frameR, birdR)

        feat_pts, head_pts = triangulate(birdL, birdR, stereo)
        visible = [k for k in CLS_DICT.keys() if birdL.feats[k] is not None and birdR.feats[k] is not None]
        if len(feat_pts) > 0:
            pts = np.squeeze(cv2.convertPointsFromHomogeneous(feat_pts.T), axis=1)
            for i, ft, pt in zip(range(len(visible)), visible, pts):
                scatter_plots[i] = ax.scatter(*pt, c=COLOURS[ft], marker=MARKERS[ft])
            fig.canvas.draw()
            im_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            im_plot = im_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            for i in range(len(pts)):
                scatter_plots[i].remove()

        out = cv2.vconcat([cv2.resize(matches, (w, int(h / 2)), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(im_plot, (w, h), interpolation=cv2.INTER_CUBIC)])
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
