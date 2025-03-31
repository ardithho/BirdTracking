from scipy.spatial.transform import Rotation

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.sim import *
from utils.general import DEG2RAD


input_dir = ROOT / 'data/blender/renders/marked'
cfg_path = os.path.join(input_dir, 'cam.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)
    R = ext[:3, :3]
    t = ext[:3, 3]

sim = Sim()

T = np.eye(4)
q = np.array([0, 0, 0, 1])
# r = np.array([0, 0, 0], dtype=np.float64)
# T[:3, :3] = cv2.Rodrigues(r * DEG2RAD)[0]
T[:3, :3] = Rotation.from_quat(-q).as_matrix()
r_1 = np.array([0, 0, 0], dtype=np.float64)
r_2 = np.array([0, 0, 180], dtype=np.float64)
print(Rotation.from_euler('xyz', r_1, degrees=True).as_quat())
print(Rotation.from_euler('xyz', r_2, degrees=True).as_quat())
sim.update(T)
sim.run()
