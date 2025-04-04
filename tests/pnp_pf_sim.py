import pycolmap
from scipy.spatial.transform import Rotation as R

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.camera import Stereo
from utils.filter import ParticleFilter
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
out_dir = ROOT / 'data/out/pnp'

h, w = (720, 1280)
writer = cv2.VideoWriter(str(out_dir / f'pnp_pf_sim{EXTENSION}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(h * 2)))

stereo = Stereo(path=cfg_path)
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)
    cam_rmat = ext[:3, :3]
    cam_tvec = ext[:3, 3]

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

sim = Sim()
pf = ParticleFilter()

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
frame_count = 0
ae_sum = np.zeros(3)
te_sum = np.zeros(3)

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
                rmat = rig.rotation.matrix()
                rmat = cam_rmat @ rmat  # camera to world
                tvec = -(rig.translation + cam_tvec)

                obs = np.array([*R.from_matrix(rmat.T).as_euler('xyz'), *tvec])
                pf.transition()
                pf.observe(obs)
                pf.resample()
                estimate = pf.estimate()

                # colmap to o3d notation
                r = estimate[:3]
                r[0] *= -1
                rmat = R.from_euler('xyz', r).as_matrix()

                tvec = estimate[3:6]
                tvec[0] *= -1

                T[:3, :3] = rmat @ prev_T[:3, :3].T
                T[:3, 3] = tvec - prev_T[:3, 3]

                esD = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
                gtD = R.from_matrix(transforms[frame_no][:3, :3]).as_euler('xyz', degrees=True) * np.array([1., 1., 1.])
                esT = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                gtT = R.from_matrix(gt[:3, :3]).as_euler('xyz', degrees=True) * np.array([1., 1., 1.])
                est = tvec
                gtt = gt[:3, 3]

                ae = np.abs(gtT - esT)
                ae_sum += ae
                te = np.abs(gtt - est)
                te_sum += te
                frame_count += 1

                print('esD:', *np.rint(esD))
                print('gtD:', *np.rint(gtD))
                print('esT:', *np.rint(esT))
                print('gtT:', *np.rint(gtT))
                print('ae:', *ae)
                print('est:', *est)
                print('gtt:', *gtt)
                print('te:', *te)
                print('')

                prev_T[:3, :3] = rmat
                prev_T[:3, 3] = tvec
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

mae = ae_sum / frame_count
print('MAE:', *mae, np.mean(mae))
mte = te_sum / frame_count
print('MTE:', *mte, np.mean(mte))
