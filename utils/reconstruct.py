import yaml
import cv2
import numpy as np


cfg_path = 'data/blender/config.yaml'
with open(cfg_path, 'r') as f:
    HEAD_CFG = yaml.safe_load(f)


def solvePnP(bird, k, dist=None):
    head_pts = np.array([HEAD_CFG[k] for k, v in bird.feats.items() if v is not None], dtype=np.float32)
    feat_pts = np.array([v for k, v in bird.feats.items() if v is not None], dtype=np.float32)
    if head_pts.shape[0] >= 4:
        return cv2.solvePnPRansac(head_pts, feat_pts, k, dist)
    return False, None, None, None
