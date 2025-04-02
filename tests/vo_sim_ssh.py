from scipy.spatial.transform import Rotation as R

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.odometry import estimate_vo
from utils.sim import *
from utils.structs import Birds


RESIZE = 0.5
STRIDE = 1
METHOD = 'lg'
BLENDER_ROOT = ROOT / 'data/blender'
EXTENSION = ''
NAME = f'marked{EXTENSION}'

renders_dir = BLENDER_ROOT / 'renders'
vid_path = renders_dir / f'vid/{NAME}_f.mp4'
input_dir = renders_dir / NAME
cfg_path = input_dir / 'cam.yaml'
trans_path = input_dir / 'transforms.txt'

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['KF']).reshape(3, 3)
    dist = None

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]

cap = cv2.VideoCapture(str(vid_path))
birds = Birds()
frame_no = 0
frame_count = 0
ae_sum = np.zeros(3)
te_sum = np.zeros(3)

T = np.eye(4)
abs_T = T.copy()
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
            vo, rmat, tvec, _ = estimate_vo(prev_frame, frame, K=K, method=METHOD)
            if vo:
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
                est = T[:3, 3]
                gtt = transforms[frame_no][:3, 3]

                ae = np.abs(esD - gtD)
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

        frame_no += 1
        prev_frame = frame
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

mae = ae_sum / frame_count
print('MAE:', *mae, np.mean(mae))
mte = te_sum / frame_count
print('MTE:', *mte, np.mean(mte))
