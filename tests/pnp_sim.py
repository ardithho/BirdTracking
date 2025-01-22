import yaml
import cv2
import numpy as np
import os

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.general import RAD2DEG
from utils.camera import Stereo
from utils.structs import Bird, Birds
from utils.sim import *
from utils.reconstruct import solvePnP


STRIDE = 1

vid_path = ROOT / 'data/blender/marked_f.mp4'

renders_dir = ROOT / 'data/blender/marked'
cfg_path = os.path.join(renders_dir, 'cam.yaml')
trans_path = os.path.join(renders_dir, 'transforms.txt')

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/pnp_sim.mp4'), cv2.VideoWriter_fourcc(*'MPEG'), 10, (w, int(h * 2)))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
sim.flip()
T = np.eye(4)
prev_T = T.copy()
sim.update(T)
gt = np.eye(4)
cam_w, cam_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
dummy_head = Box(0, conf=[1.],
                 xywh=np.array([[cam_w/2, cam_h/2, cam_w, cam_h]]),
                 xywhn=np.array([[.5, .5, 1., 1.]]),
                 xyxy=np.array([[0., 0., cam_w, cam_h]]),
                 xyxyn=np.array([[0., 0., 1., 1.]]))
while cap.isOpened():
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        birds.update([Bird(dummy_head, extract_features(frame))], frame)
        bird = birds['m'] if birds['m'] is not None else birds['f']
        pnp, r, t, _ = solvePnP(bird, K)
        gt = transforms[frame_no] @ gt
        if pnp:
            r -= cv2.Rodrigues(ext[:3, :3])[0]
            # r = r[[0, 2, 1]]
            R, _ = cv2.Rodrigues(r)
            # R = R.T
            T[:3, :3] = R @ prev_T[:3, :3].T
            # T[:3, 3] = t.T - prev_T[:3, 3]
            # r, _ = cv2.Rodrigues(R*transforms[frame_no][:3, :3])
            # error = np.linalg.norm(r)
            # print(r, error)
            print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0]*RAD2DEG))
            print('gt:', *np.rint(cv2.Rodrigues(transforms[frame_no][:3, :3])[0]*RAD2DEG))

            print('esT:', *np.rint(r*RAD2DEG))
            print('gtT:', *np.rint(cv2.Rodrigues(gt[:3, :3])[0]*RAD2DEG))
            print('')
            prev_T[:3, :3] = R
            # prev_T[:3, 3] = t.T
            sim.update(T)

        cv2.imshow('frame', cv2.resize(birds.plot(frame), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
        out = cv2.vconcat([cv2.resize(birds.plot(frame), (w, h), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])

        cv2.imshow('out', out)
        writer.write(out)

        frame_no += 1
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
