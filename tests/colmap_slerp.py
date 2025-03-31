import pycolmap
from scipy.spatial.transform import Rotation

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
from utils.general import RAD2DEG, cosine, slerp
from utils.reconstruct import get_head_feat_pts
from utils.sim import *
from utils.structs import Bird, Birds


RESIZE = .5
STRIDE = 1
PADDING = 30
OFFSET = 10

tracker = Tracker(ROOT / 'yolov8/weights/head.pt')
predictor = Predictor(ROOT / 'yolov8/weights/head.pt')

vid_path = ROOT / 'data/vid/fps120/K203_K238_1_GH040045.mp4'

cfg_path = ROOT / 'data/calibration/cam.yaml'
blender_cfg = ROOT / 'data/blender/configs/cam.yaml'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(ROOT / 'data/out/colmap_slerp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h * 2))

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

sim = Sim()

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
cam_w, cam_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
dummy_head = Box(0, conf=[1.],
                 xywh=np.array([[cam_w/2, cam_h/2, cam_w, cam_h]]),
                 xywhn=np.array([[.5, .5, 1., 1.]]),
                 xyxy=np.array([[0., 0., cam_w, cam_h]]),
                 xyxyn=np.array([[0., 0., 1., 1.]]))
qs = []
ptr = None
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
        updated = False
        if bird is not None:
            head_pts, feat_pts = get_head_feat_pts(bird)
            if head_pts.shape[0] >= 4:
                pnp = pycolmap.estimate_and_refine_absolute_pose(feat_pts, head_pts, cam)
                if pnp is not None:
                    rig = pnp['cam_from_world']  # Rigid3d
                    R = rig.rotation.matrix()
                    R = R @ ext[:3, :3].T  # undo camera extrinsic rotation
                    r = cv2.Rodrigues(R)[0]
                    # colmap to o3d notation
                    r[0] *= -1
                    q = Rotation.from_euler('xyz', -r.flatten()).as_quat()
                    if ptr is not None and frame_no > ptr + 1:
                        steps = frame_no - ptr
                        if steps <= OFFSET:
                            q_ = qs[ptr].copy()
                            if ptr >= 1 and qs[ptr-1] is not None:
                                if cosine(qs[ptr-1], slerp(-q_, q, 1/steps)) < cosine(qs[ptr-1], slerp(q_, q, 1/steps)):
                                    q_ *= -1
                            for i in range(ptr+1, frame_no):
                                qs[i] = slerp(q_, q, (i-ptr)/steps)
                    qs.append(q)
                    updated = True
                    ptr = frame_no
        if not updated:
            qs.append(None)
        frame_no += 1
        print(ptr, frame_no, qs[-1])
        cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

frame_no = 0
cap = cv2.VideoCapture(str(vid_path))
T = np.eye(4)
prev_T = T.copy()
sim.update(T)
gt = np.eye(4)
while cap.isOpened() and frame_no < len(qs):
    for i in range(STRIDE):
        if cap.isOpened():
            _ = cap.grab()
        else:
            break
    ret, frame = cap.retrieve()
    if ret:
        q = qs[frame_no]
        if q is not None:
            R = Rotation.from_quat(q).as_matrix()
            T[:3, :3] = R @ prev_T[:3, :3].T
            prev_T[:3, :3] = R
            sim.update(T)
        print('es:', *np.rint(cv2.Rodrigues(T[:3, :3])[0]*RAD2DEG))
        print('esT:', *np.rint(cv2.Rodrigues(R)[0]*RAD2DEG))
        print('')

        out = cv2.vconcat([cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])

        cv2.imshow('out', cv2.resize(out, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        writer.write(out)

        frame_no += 1
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
sim.close()

