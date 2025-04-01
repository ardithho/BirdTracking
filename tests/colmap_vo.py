import pycolmap

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov8.predict import Predictor, detect_features
from yolov8.track import Tracker

from utils.box import pad_boxes
from utils.camera import Stereo
from utils.general import RAD2DEG
from utils.odometry import find_matches, find_matching_pts, draw_kp_matches
from utils.sim import *
from utils.structs import Bird, Birds


RESIZE = 1.
STRIDE = 1
PADDING = 30
METHOD = 'lg'
BLENDER_ROOT = ROOT / 'data/blender'

tracker = Tracker(ROOT / 'yolov8/weights/head.pt')
predictor = Predictor(ROOT / 'yolov8/weights/head.pt')

vid_path = ROOT / 'data/vid/fps120/K203_K238_1_GH040045.mp4'

cfg_path = ROOT / 'data/calibration/cam.yaml'
blender_cfg = ROOT / 'data/blender/configs/cam.yaml'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / f'data/out/colmap_vo_{METHOD}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, int(h * 1.5)))

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
options = pycolmap.TwoViewGeometryOptions(compute_relative_pose=True)

sim = Sim()

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
T = np.eye(4)
abs_T = T.copy()
sim.update(T)
prev_frame = None
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
        prev_bird = birds.caches['m'][-1] if birds.caches['m'][-1] is not None else birds.caches['f'][-1]
        if bird is not None and prev_frame is not None:
            prev_mask = prev_bird.mask(prev_frame)
            curr_mask = bird.mask(frame)
            pts1, pts2 = find_matching_pts(prev_frame, frame, prev_mask, curr_mask, method=METHOD)
            matches = np.asarray(list(zip(list(range(len(pts1))), list(range(len(pts2))))))
            vo = pycolmap.estimate_calibrated_two_view_geometry(camera1=cam,
                                                                points1=pts1.reshape(-1, 2),
                                                                camera2=cam,
                                                                points2=pts2.reshape(-1, 2),
                                                                matches=matches,
                                                                options=options)
            if vo is not None:
                rig = vo.cam2_from_cam1  # Rigid3d
                R = rig.rotation.matrix()
                R = R @ ext[:3, :3].T  # undo camera extrinsic rotation
                r = cv2.Rodrigues(R)[0]
                # colmap to o3d notation
                r[0] *= -1
                R, _ = cv2.Rodrigues(r)
                R = R.T
                T[:3, :3] = R
                abs_T = T @ abs_T
                print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0]*RAD2DEG))
                print('esT:', *np.rint(cv2.Rodrigues(abs_T[:3, :3])[0]*RAD2DEG))
                print('')
                sim.update(T)
            if METHOD == 'lg' or METHOD == 'of':
                kp1, kp2 = find_matching_pts(prev_frame, frame, prev_mask, curr_mask, method=METHOD)
                match = draw_kp_matches(prev_frame, kp1, frame, kp2)
            else:
                kp1, kp2, matches = find_matches(prev_frame, frame, prev_mask, curr_mask, thresh=.2, method=METHOD)
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
