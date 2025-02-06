import yaml
import cv2
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.general import RAD2DEG
from utils.camera import Stereo
from utils.structs import Bird, Birds
from utils.sim import *
from utils.odometry import estimate_vio, find_matches, find_matching_pts, draw_kp_matches

STRIDE = 1
METHOD = 'lg'
BLENDER_ROOT = ROOT / 'data/blender'
NAME = 'vanilla'

renders_dir = BLENDER_ROOT / 'renders'
vid_path = renders_dir / f'vid/{NAME}_f.mp4'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / f'data/out/vio_{METHOD}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 1.5)))

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['K']).reshape(3, 3)
    dist = None

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
prev_frame = None
frame_no = 0
T = np.eye(4)
sim.update(T)
while cap.isOpened():
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        if prev_frame is not None:
            vio, R, t, _ = estimate_vio(prev_frame, frame, K=K, method=METHOD)
            if vio:
                R_ = cv2.Rodrigues(cv2.Rodrigues(R.T)[0][[1, 0, 2]])[0]
                T[:3, :3] = R_
                # T[:3, 3] = -t.T
                # r, _ = cv2.Rodrigues(R*transforms[frame_no][:3, :3])
                # error = np.linalg.norm(r)
                # print(r, error)
                print('vo:', *np.rint(cv2.Rodrigues(R.T)[0] * RAD2DEG))
                print('gt:', *np.rint(cv2.Rodrigues(transforms[frame_no][:3, :3])[0] * RAD2DEG))
                print('')
                sim.update(T)
            if METHOD == 'lg':
                kp1, kp2 = find_matching_pts(prev_frame, frame, method=METHOD)
                match = draw_kp_matches(prev_frame, kp1, frame, kp2)
            else:
                kp1, kp2, matches = find_matches(prev_frame, frame, thresh=.2, method=METHOD)
                match = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

        # out = cv2.vconcat([cv2.resize(birds.plot(), (w, h), interpolation=cv2.INTER_CUBIC),
        #                    cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
            out = cv2.vconcat([cv2.resize(match, (w, int(h / 2)), interpolation=cv2.INTER_CUBIC),
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
