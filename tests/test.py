from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.general import DEG2RAD, RAD2DEG
from utils.filter import ukf, OBS_COV_HIGH, OBS_COV_LOW
from utils.sim import *


RESIZE = .5
BLENDER_ROOT = ROOT / 'data/blender'
cfg_path = BLENDER_ROOT / 'configs/cam.yaml'

rotations = [Rotation.from_euler('xyz', np.array((0, 0, 0))*DEG2RAD).as_quat(),
             Rotation.from_euler('xyz', np.array((0, 0, 5))*DEG2RAD).as_quat(),
             Rotation.from_euler('xyz', np.array((0, 0, 90))*DEG2RAD).as_quat()]

sim = Sim()

T = np.eye(4)
prev_T = T.copy()
sim.update(T)
state_mean = ukf.initial_state_mean
state_cov = ukf.initial_state_covariance
sims = []
for rotation in rotations:
    obs = np.array([*rotation, 0, 0, 0])
    state_mean, state_cov = ukf.filter_update(
        filtered_state_mean=state_mean,
        filtered_state_covariance=state_cov,
        observation=obs,
        observation_covariance=OBS_COV_HIGH
    )
    # colmap to o3d notation
    r = Rotation.from_quat(state_mean[:4]).as_rotvec()
    r[0] *= -1
    R, _ = cv2.Rodrigues(r)
    # R = R.T
    T[:3, :3] = R @ prev_T[:3, :3].T
    prev_T[:3, :3] = R
    sim.update(T)
    sims.append(sim.screen)
    print(*np.rint(r*RAD2DEG))

cv2.imwrite(str(ROOT / 'data/out/ukf_sim.jpg'), cv2.vconcat(sims))
sim.close()
