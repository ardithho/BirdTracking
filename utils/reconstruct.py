import cv2
import numpy as np

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.configs import CLS_DICT, HEAD_CFG


def get_head_feat_pts(bird):
    keys = [k for k, v in bird.feats.items() if v is not None]
    head_pts = np.array([HEAD_CFG[k] for k in keys], dtype=np.float32)
    feat_pts = np.array([bird.feats[k] for k in keys], dtype=np.float32)
    return head_pts, feat_pts


def reproj_error(img_pts, obj_pts, transform, rvec, tvec, K, dist=None):
    """
    :param img_pts: (N, 2)
    :param obj_pts: (N, 3)
    :param transform: (4, 4) transformation matrix
    :param rvec: camera rotation extrinsic
    :param tvec: camera translation extrinsic
    :param K: camera matrix
    :param dist: distortion coefficients
    :return: reprojection error
    """
    obj_pts_t = (transform[:3, :3] @ obj_pts.T).T + transform[:3, 3]
    proj_pts, _ = cv2.projectPoints(obj_pts_t, rvec, tvec, K, dist)
    proj_pts = np.squeeze(proj_pts, axis=1)
    img_pts = img_pts.astype(proj_pts.dtype)
    error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(img_pts)
    return error


def reproj_error_(img_pts, obj_pts, rvec, tvec, K, dist=None):
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj_pts = np.squeeze(proj_pts, axis=1)
    img_pts = img_pts.astype(proj_pts.dtype)
    error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(img_pts)
    return error


def solvePnP(bird, K, dist=None):
    head_pts, feat_pts = get_head_feat_pts(bird)
    if head_pts.shape[0] >= 4:
        return cv2.solvePnPRansac(head_pts, feat_pts, K, dist, reprojectionError=4.)
        # return True, cv2.solvePnPRefineLM(head_pts, feat_pts, K, dist, None, None), None
    return False, None, None, None


def triangulate(birdL, birdR, stereo):
    visible = [k for k in CLS_DICT.keys() if birdL.feats[k] is not None and birdR.feats[k] is not None]
    if len(visible) == 0:
        return [], []
    head_pts = np.array([HEAD_CFG[k] for k in visible])
    feat_ptsL = np.array([birdL.feats[k] for k in visible]).T
    feat_ptsR = np.array([birdR.feats[k] for k in visible]).T
    feat_pts = cv2.triangulatePoints(stereo.camL.P, stereo.camR.P, feat_ptsL, feat_ptsR)
    return feat_pts, head_pts


def triangulate_estimate(birdL, birdR, stereo):
    feat_pts, head_pts = triangulate(birdL, birdR, stereo)
    if len(feat_pts) == 0:
        return 0, None, None
    return cv2.estimateAffine3D(head_pts, cv2.convertPointsFromHomogeneous(feat_pts.T))
