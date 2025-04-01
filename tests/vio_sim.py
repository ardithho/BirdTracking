from scipy.spatial.transform import Rotation as R

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.structs import Birds
from utils.sim import *
from utils.odometry import estimate_vio, find_matches, find_matching_pts, draw_kp_matches


RESIZE = 0.5
STRIDE = 1
METHOD = 'orb'
BLENDER_ROOT = ROOT / 'data/blender'
NAME = 'vanilla'

renders_dir = BLENDER_ROOT / 'renders'
vid_path = renders_dir / f'vid/{NAME}.mp4'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / f'data/out/vio_{METHOD}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 1.5)))

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['KF']).reshape(3, 3)
    dist = None

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

sim = Sim()

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
T = np.eye(4)
abs_T = T.copy()
sim.update(T)
gt = np.eye(4)
prev_frame = None
while cap.isOpened():
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        gt = transforms[frame_no] @ gt
        if prev_frame is not None:
            vio, rmat, tvec, _ = estimate_vio(prev_frame, frame, K=K, method=METHOD)
            if vio:
                r = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                # colmap to o3d notation
                r[0] *= -1
                rmat = R.from_euler('xyz', r, degrees=True).as_matrix()
                rmat = rmat.T

                tvec[0] *= -1

                T[:3, :3] = rmat
                T[:3, 3] = -tvec.T
                abs_T = T @ abs_T

                esD = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
                gtD = R.from_matrix(transforms[frame_no][:3, :3]).as_euler('xyz', degrees=True) * np.array([1., 1., 1.])
                esT = R.from_matrix(abs_T[:3, :3]).as_euler('xyz', degrees=True)
                gtT = R.from_matrix(gt[:3, :3]).as_euler('xyz', degrees=True) * np.array([1., 1., 1.])

                print('esD:', *np.rint(esD))
                print('gtD:', *np.rint(gtD))
                print('esT:', *np.rint(esT))
                print('gtT:', *np.rint(gtT))
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
        cv2.imshow('out', cv2.resize(out, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        writer.write(out)

        frame_no += 1
        prev_frame = frame
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
