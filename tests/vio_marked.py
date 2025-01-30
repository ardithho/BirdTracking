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
from utils.odometry import bird_vio, draw_matches


STRIDE = 1
BLENDER_ROOT = ROOT / 'data/blender'
NAME = 'marked'

vid_path = BLENDER_ROOT / f'vid/{NAME}_f.mp4'
renders_dir = BLENDER_ROOT / 'renders'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/vio_marked.mp4'), cv2.VideoWriter_fourcc(*'MPEG'), 10, (w, int(h * 1.5)))

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
prev_frame = None
frame_no = 0
sim.flip()
T = np.eye(4)
sim.update(T)
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
        if prev_frame is not None:
            prev_bird = birds.caches['m'][-2] if birds.caches['m'][-2] is not None else birds.caches['f'][-2]
            # vio, _, R, t, _ = estimate_vio(prev_frame, frame, prev_bird.mask(prev_frame.shape[:2]), bird.mask(frame.shape[:2]), k)
            vio, Rs, ts = bird_vio(prev_bird, bird, K=K)
            if vio:
                T[:3, :3] = Rs[0].T
                # T[:3, 3] = -t.T
                # r, _ = cv2.Rodrigues(R*transforms[frame_no][:3, :3])
                # error = np.linalg.norm(r)
                # print(r, error)
                for R in Rs:
                    print('vo:', *np.rint(cv2.Rodrigues(R.T)[0]*RAD2DEG))
                print('gt:', *np.rint(cv2.Rodrigues(transforms[frame_no][:3, :3])[0]*RAD2DEG))
                print('')
                sim.update(T)
            matches = draw_matches(prev_frame, prev_bird, frame, bird)
        # cv2.imshow('frame', cv2.resize(birds.plot(frame), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

        # out = cv2.vconcat([cv2.resize(birds.plot(frame), (w, h), interpolation=cv2.INTER_CUBIC),
        #                    cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
            out = cv2.vconcat([cv2.resize(matches, (w, int(h/2)), interpolation=cv2.INTER_CUBIC),
                               cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
        else:
            out = cv2.vconcat([cv2.resize(cv2.hconcat([frame, frame]), (w, int(h / 2)), interpolation=cv2.INTER_CUBIC),
                               cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
        cv2.imshow('out', out)
        writer.write(out)

        prev_frame = frame
        frame_no += 1
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
