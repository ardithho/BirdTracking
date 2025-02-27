import pycolmap
import yaml
import cv2
import os
from scipy.spatial.transform import Rotation, Slerp

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.camera import Stereo
from utils.general import RAD2DEG, cosine, slerp
from utils.reconstruct import get_head_feat_pts
from utils.sim import *
from utils.structs import Bird, Birds


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
writer = cv2.VideoWriter(str(ROOT / f'data/out/colmap{EXTENSION}_slerp_sim.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 2)))

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
        birds.update([Bird(dummy_head, extract_features(frame))], frame)
        bird = birds['m'] if birds['m'] is not None else birds['f']
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
                    # key_qs = [qs[ptr], q]
                    # key_times = [ptr, frame_no]
                    # mid_time = (frame_no + ptr) / 2
                    # slerp = Slerp(key_times, Rotation.from_quat(key_qs))
                    # mid_q = -slerp(mid_time).as_quat()
                    # slerp = Slerp([ptr, mid_time, frame_no], Rotation.from_quat([qs[ptr], mid_q, q]))
                    # interp_qs = slerp(list(range(ptr+1, frame_no))).as_quat()
                    # for i, interp_q in zip(range(ptr+1, frame_no), interp_qs):
                    #     qs[i] = interp_q
                    q_ = qs[ptr].copy()
                    steps = frame_no - ptr
                    if ptr >= 1 and qs[ptr-1] is not None:
                        if cosine(qs[ptr-1], slerp(-q_, q, 1/steps)) < cosine(qs[ptr-1], slerp(q_, q, 1/steps)):
                            q_ *= -1
                    for i in range(ptr+1, frame_no):
                        qs[i] = slerp(q_, q, (i-ptr)/steps)
                else:
                    qs.append(q)
                ptr = frame_no
        else:
            qs.append(None)
        frame_no += 1
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
        print('gt:', *np.rint(
            cv2.Rodrigues(transforms[frame_no][:3, :3])[0][[0, 1, 2]]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

        print('esT:', *np.rint(cv2.Rodrigues(R)[0]*RAD2DEG))
        print('gtT:', *np.rint(
            cv2.Rodrigues(gt[:3, :3])[0][[0, 1, 2]]*np.array([-1., 1., 1.]).reshape((-1, 1))*RAD2DEG))

        print('')

        out = cv2.vconcat([cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC),
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

