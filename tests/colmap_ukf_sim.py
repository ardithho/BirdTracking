import pycolmap
import yaml
import cv2
import os
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.general import RAD2DEG
from utils.camera import Stereo
from utils.filter import ukf
from utils.structs import Bird, Birds
from utils.sim import *
from utils.reconstruct import get_head_feat_pts


RESIZE = .5
STRIDE = 1
BLENDER_ROOT = ROOT / 'data/blender'
EXTENSION = ''
NAME = f'marked{EXTENSION}'

renders_dir = BLENDER_ROOT / 'renders'
vid_path = renders_dir / f'vid/{NAME}_f.mp4'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / f'data/out/colmap{EXTENSION}_ukf_sim.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 2)))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cam = pycolmap.Camera(
    model='SIMPLE_PINHOLE',
    width=stereo.camL.w,
    height=stereo.camL.h,
    params=(K[0, 0],  # focal length
            K[0, 2], K[1, 2]),  # cx, cy
)

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
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
        gt = transforms[frame_no] @ gt
        birds.update([Bird(dummy_head, extract_features(frame))], frame)
        bird = birds['m'] if birds['m'] is not None else birds['f']
        head_pts, feat_pts = get_head_feat_pts(bird)
        if head_pts.shape[0] > 0:
            pnp = pycolmap.estimate_and_refine_absolute_pose(feat_pts, head_pts, cam)
            if pnp is not None:
                rig = pnp['cam_from_world']  # Rigid3d
                R = rig.rotation.matrix()
                R = R @ ext[:3, :3].T  # undo camera extrinsic rotation
                r = cv2.Rodrigues(R)[0]
                obs = np.array([*Rotation.from_euler('xyz', -r.flatten()).as_quat(), *-rig.translation])
                mean, cov = ukf.filter_update(
                    filtered_state_mean=ukf.initial_state_mean,
                    filtered_state_covariance=ukf.initial_state_covariance,
                    observation=np.array(obs)
                )
                ukf.initial_state_mean = mean
                ukf.initial_state_covariance = cov
                # colmap to o3d notation
                r = Rotation.from_quat(mean[:4]).as_rotvec()
                r[0] = -r[0]
                R, _ = cv2.Rodrigues(r)
                # R = R.T
                T[:3, :3] = R @ prev_T[:3, :3].T
                print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0]*RAD2DEG))
                print('gt:', *np.rint(
                    cv2.Rodrigues(transforms[frame_no][:3, :3])[0][[0, 1, 2]]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

                print('esT:', *np.rint(cv2.Rodrigues(R)[0]*RAD2DEG))
                print('gtT:', *np.rint(
                    cv2.Rodrigues(gt[:3, :3])[0][[0, 1, 2]]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

                print('')
                prev_T[:3, :3] = R
                sim.update(T)

        cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        out = cv2.vconcat([cv2.resize(birds.plot(), (w, h), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])

        cv2.imshow('out', cv2.resize(out, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
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
