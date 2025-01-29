import cv2
import yaml
import os

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.sim import *
from utils.general import DEG2RAD


renders_dir = ROOT / 'data/blender/marked'
cfg_path = os.path.join(renders_dir, 'cam.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)
    R = ext[:3, :3]
    t = ext[:3, 3]

T = np.eye(4)
r = np.array([0, 0, 0], dtype=np.float64)
T[:3, :3] = cv2.Rodrigues(r * DEG2RAD)[0]
sim.update(T)
o3d.visualization.draw_geometries([sim.mesh],
                                  front=R.T@t[[0, 2, 1]],
                                  lookat=[0, 0, 0],
                                  up=[0, 1, 0],
                                  zoom=1.
                                  )
