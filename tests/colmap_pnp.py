import pycolmap
from scipy.spatial.transform import Rotation as R

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
from utils.reconstruct import get_head_feat_pts, reproj_error, reproj_error_
from utils.sim import *
from utils.structs import Bird, Birds


RESIZE = 0.5  # resize display window
STRIDE = 1
FPS = 120
SPEED = 0.5
PADDING = 30

tracker = Tracker(ROOT / 'yolov8/weights/head.pt')
predictor = Predictor(ROOT / 'yolov8/weights/head.pt')

vid_path = ROOT / 'data/vid/fps120/GH140045_solo.mp4'

cfg_path = ROOT / 'data/calibration/cam.yaml'
blender_cfg = ROOT / 'data/blender/configs/cam.yaml'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/colmap_pnp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS//STRIDE*SPEED, (w, h * 2))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['KR']).reshape(3, 3)
    dist = np.asarray(cfg['distR'])

with open(blender_cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    ext = np.array(cfg['ext']).reshape(3, 4)
    cam_rmat = ext[:3, :3]
    cam_rvec = cv2.Rodrigues(cam_rmat)[0]
    cam_tvec = ext[:3, 3]

cam = pycolmap.Camera(
    model='OPENCV',
    width=stereo.camR.w,
    height=stereo.camR.h,
    params=(K[0, 0], K[1, 1],  # fx, fy
            K[0, 2], K[1, 2],  # cx, cy
            *dist[:4]),  # dist: k1, k2, p1, p2
    )

sim = Sim()

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_count = 0
re_sum = 0

T = np.eye(4)
prev_T = np.eye(4)
proj_T = np.eye(4)
sim.update(T)
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
        if bird is not None:
            head_pts, feat_pts = get_head_feat_pts(bird)
            if head_pts.shape[0] >= 4:
                pnp = pycolmap.estimate_and_refine_absolute_pose(feat_pts, head_pts, cam)
                if pnp is not None:
                    rig = pnp['cam_from_world']  # Rigid3d
                    rmat = rig.rotation.matrix()
                    rmat = rmat @ cam_rmat  # camera to world
                    rmat = rmat.T
                    r = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                    # tvec = -(rig.translation - cam_tvec)
                    tvec = -(rig.translation + cam_tvec)

                    proj_T[:3, :3] = rig.rotation.matrix() @ cam_rmat.T
                    proj_T[:3, 3] = rig.translation - cam_tvec

                    # colmap to o3d notation
                    r[0] *= -1
                    rmat = R.from_euler('xyz', r, degrees=True).as_matrix()
                    tvec[0] *= -1

                    T[:3, :3] = rmat @ prev_T[:3, :3].T
                    T[:3, 3] = tvec - prev_T[:3, 3]
                    print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0] * RAD2DEG))
                    print('esT:', *np.rint(cv2.Rodrigues(rmat)[0] * RAD2DEG))

                    error = reproj_error(feat_pts, head_pts, proj_T, cam_rvec, cam_tvec, K, dist)
                    print('error:', error)
                    error_ = reproj_error_(feat_pts, head_pts,
                                           cv2.Rodrigues(rig.rotation.matrix())[0],
                                           rig.translation, K, dist)
                    print('error_:', error_)
                    print('')

                    re_sum += error_
                    frame_count += 1

                    prev_T[:3, :3] = rmat
                    prev_T[:3, 3] = tvec
                    sim.update(T)
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

print('MRE:', re_sum / frame_count)
