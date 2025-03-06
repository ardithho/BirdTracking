import pycolmap
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker

from utils.camera import Stereo
from utils.general import RAD2DEG
from utils.filter import ukf, OBS_COV_HIGH, OBS_COV_LOW
from utils.reconstruct import get_head_feat_pts
from utils.sim import *
from utils.structs import Bird, Birds


RESIZE = .5
STRIDE = 1
FPS = 120
PADDING = 20

tracker = Tracker(ROOT / 'yolov8/weights/head.pt')
predictor_head = Predictor(ROOT / 'yolov8/weights/head.pt')

vid_path = ROOT / 'data/vid/fps120/GH140045_solo.mp4'

cfg_path = ROOT / 'data/calibration/cam.yaml'
blender_cfg = ROOT / 'data/blender/configs/cam.yaml'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/colmap_ukf.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS//STRIDE, (w, h * 2))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['KR']).reshape(3, 3)
    dist = np.asarray(cfg['distR'])

with open(blender_cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    ext = np.array(cfg['ext']).reshape(3, 4)

cam = pycolmap.Camera(
    model='OPENCV',
    width=stereo.camR.w,
    height=stereo.camR.h,
    params=(K[0, 0], K[1, 1],  # fx, fy
            K[0, 2], K[1, 2],  # cx, cy
            *dist[:4]),  # dist: k1, k2, p1, p2
    )

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
T = np.eye(4)
prev_T = np.eye(4)
sim.update(T)
state_mean = ukf.initial_state_mean
state_cov = ukf.initial_state_covariance
while cap.isOpened():
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        head = tracker.tracks(frame)[0].boxes.cpu().numpy()
        feat = detect_features(frame, head, PADDING)
        birds.update([Bird(head, feat, PADDING, *frame.shape[:2][::-1]) for head, feat in zip(head, feat)], frame)
        bird = birds['m'] if birds['m'] is not None else birds['f']
        if bird is not None:
            head_pts, feat_pts = get_head_feat_pts(bird)
            if head_pts.shape[0] > 0:
                pnp = pycolmap.estimate_and_refine_absolute_pose(feat_pts, head_pts, cam)
                if pnp is not None:
                    rig = pnp['cam_from_world']  # Rigid3d
                    R = rig.rotation.matrix()
                    R = R @ ext[:3, :3].T  # undo camera extrinsic rotation
                    r = cv2.Rodrigues(R)[0]
                    obs = np.array([*Rotation.from_euler('xyz', -r.flatten()).as_quat(), *-rig.translation])
                    state_mean, state_cov = ukf.filter_update(
                        filtered_state_mean=state_mean,
                        filtered_state_covariance=state_cov,
                        observation=np.array(obs),
                        observation_covariance=OBS_COV_HIGH if head_pts.shape[0] < 4 else OBS_COV_LOW
                    )
                    # colmap to o3d notation
                    r = Rotation.from_quat(state_mean[:4]).as_rotvec()
                    r[0] *= -1
                    R, _ = cv2.Rodrigues(r)
                    # R = R.T
                    T[:3, :3] = R @ prev_T[:3, :3].T
                    prev_T[:3, :3] = R
                    sim.update(T)
                    if head_pts.shape[0] >= 4:
                        print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0] * RAD2DEG))
                        print('esT:', *np.rint(cv2.Rodrigues(R)[0] * RAD2DEG))
                        print('')
        cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))

        out = cv2.vconcat([cv2.resize(birds.plot(), (w, h), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
        cv2.imshow('out', cv2.resize(out, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        writer.write(out)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()
