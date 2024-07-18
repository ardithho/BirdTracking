import yaml
import cv2


HEAD_CFG = yaml.safe_load('data/blender/config.yaml')


def solvePnP(bird, k):
    head_pts = [HEAD_CFG[k] for k, v in bird.feats.items() if v is not None]
    feat_pts = [v for k, v in bird.feats.items() if v is not None]
    return cv2.solvePnPRansac(head_pts, feat_pts, k, None)
