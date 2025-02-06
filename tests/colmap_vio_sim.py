import pycolmap
import yaml
import cv2
import os

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.general import RAD2DEG
from utils.camera import Stereo
from utils.sim import *
from utils.odometry import find_matches, find_matching_pts, draw_lg_matches


RESIZE = 1.
STRIDE = 1
METHOD = 'lg'
BLENDER_ROOT = ROOT / 'data/blender'
NAME = 'vanilla'

renders_dir = BLENDER_ROOT / 'renders'
vid_path = renders_dir / f'vid/{NAME}.mp4'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / f'data/out/colmap_vio_sim_{METHOD}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 1.5)))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cam = pycolmap.Camera(
    camera_id=0,
    model='SIMPLE_PINHOLE',
    width=int(K[0, 2]*2),
    height=int(K[1, 2]*2),
    params=(K[0, 0],  # focal length
            K[0, 2], K[1, 2]),  # cx, cy
)
options = pycolmap.TwoViewGeometryOptions(compute_relative_pose=True)

cap = cv2.VideoCapture(str(vid_path))
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
            pts1, pts2 = find_matching_pts(prev_frame, frame, method=METHOD)
            matches = np.asarray(list(zip(list(range(len(pts1))), list(range(len(pts2))))))
            vio = pycolmap.estimate_calibrated_two_view_geometry(camera1=cam,
                                                                 points1=pts1.reshape(-1, 2),
                                                                 camera2=cam,
                                                                 points2=pts2.reshape(-1, 2),
                                                                 matches=matches,
                                                                 options=options)
            if vio is not None:
                rig = vio.cam2_from_cam1  # Rigid3d
                R = rig.rotation.matrix()
                R = R @ ext[:3, :3].T  # undo camera extrinsic rotation
                r = cv2.Rodrigues(R)[0]
                # colmap to o3d notation
                r[0] = -r[0]
                R, _ = cv2.Rodrigues(r)
                R = R.T
                T[:3, :3] = R
                abs_T = T @ abs_T
                print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0]*RAD2DEG))
                print('gt:', *np.rint(
                    cv2.Rodrigues(transforms[frame_no][:3, :3])[0]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

                print('esT:', *np.rint(cv2.Rodrigues(abs_T[:3, :3])[0]*RAD2DEG))
                print('gtT:', *np.rint(
                    cv2.Rodrigues(gt[:3, :3])[0]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

                print('')
                sim.update(T)
            if METHOD == 'lg':
                kp1, kp2 = find_matching_pts(prev_frame, frame, method=METHOD)
                match = draw_lg_matches(prev_frame, kp1, frame, kp2)
            else:
                kp1, kp2, matches = find_matches(prev_frame, frame, thresh=.2, method=METHOD)
                match = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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

