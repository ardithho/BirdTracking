import yaml
import cv2
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.box import Box
from utils.structs import Bird, Birds
from utils.sim import extract_features
from utils.reconstruct import solvePnP, HEAD_CFG
from utils.general import RAD2DEG


def foo(arr):
    return arr[:2] / arr[-1]


input_dir = ROOT / 'data/blender/marked'
im_path = input_dir / 'f/001.png'
cfg_path = input_dir / 'cam.yaml'
im = cv2.imread(str(im_path))
cam_h, cam_w = im.shape[:2]

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)

dummy_head = Box(0, conf=[1.],
             xywh=np.array([[cam_w/2, cam_h/2, cam_w, cam_h]]),
             xywhn=np.array([[.5, .5, 1., 1.]]),
             xyxy=np.array([[0., 0., cam_w, cam_h]]),
             xyxyn=np.array([[0., 0., 1., 1.]]))

birds = Birds()
birds.update([Bird(dummy_head, extract_features(im))], im)
bird = birds['m'] if birds['m'] is not None else birds['f']
pnp, r, t, inliers = solvePnP(bird, K)
print(inliers)
new_ext = np.zeros((3, 4))
new_ext[:3, :3] = cv2.Rodrigues(r)[0]
new_ext[:3, 3] = t.T
reproj = {k: np.rint(foo(K @ new_ext @ (v + [1]))) for k, v in HEAD_CFG.items()}
r -= cv2.Rodrigues(ext[:3, :3])[0]
print(*r*RAD2DEG)
print(*t)
print(cam_h, cam_w)
print('im', bird.feats)
print('rp', reproj)
gt = {k: np.rint(foo(K @ ext @ (v + [1]))) for k, v in HEAD_CFG.items()}
print('gt', gt)
error = np.linalg.norm(np.array(list(reproj.values())) - np.array(list(bird.feats.values()))) / len(reproj)
# error = sum([np.linalg.norm(p2-p1) for p1, p2 in zip(bird.feats.values(), reproj.values())])
print(error)
