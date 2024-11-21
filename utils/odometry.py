import cv2
import numpy as np

from utils.general import RAD2DEG
from utils.structs import CLS_DICT


def extract_features(frame, mask=None):
    orb = cv2.ORB_create()
    return orb.detectAndCompute(frame, mask)


def find_matches(prev_frame, curr_frame, prev_mask=None, curr_mask=None, thresh=0.8):
    kp1, des1 = extract_features(prev_frame, prev_mask)
    kp2, des2 = extract_features(curr_frame, curr_mask)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(bf.match(des1, des2))
    matches.sort(key=lambda x: x.distance)
    filtered = matches[:int(len(matches) * thresh)]
    return filtered, kp1, kp2


def find_matching_pts(prev_frame, curr_frame, prev_mask=None, curr_mask=None, thresh=0.8):
    matches, kp1, kp2 = find_matches(prev_frame, curr_frame, prev_mask, curr_mask, thresh)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def estimate_homography(prev_frame, curr_frame, prev_mask=None, curr_mask=None):
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


def estimate_essential_mat(prev_frame, curr_frame, prev_mask=None, curr_mask=None, K=None, dist=None):
    # https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    if K is not None:
        return cv2.findEssentialMat(src_pts, dst_pts, K, dist, K, dist)
    return cv2.findEssentialMat(src_pts, dst_pts)


# visual odometry
def estimate_vio(prev_frame, curr_frame, prev_mask=None, curr_mask=None, K=None, dist=None, thresh=.8):
    # return: retval, R, t, mask
    src_pts, dst_pts = find_matching_pts(prev_frame, curr_frame, prev_mask, curr_mask, thresh)
    # return cv2.recoverPose(src_pts, dst_pts, K, dist, K, dist, threshold=thresh)
    return estimate_vio_pts(src_pts, dst_pts, K, dist)


def estimate_vio_pts(src_pts, dst_pts, K, dist=None):
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, dist, K, dist, threshold=.8)
    return cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)


def bird_vio(prev_bird, curr_bird, K=None, dist=None, thresh=.8):
    visible = [k for k in CLS_DICT.keys() if prev_bird.feats[k] is not None and curr_bird.feats[k] is not None]
    if len(visible) < 5:
        return False, None, None
    prev_pts = np.array([prev_bird.feats[k] for k in visible]).reshape(-1, 1, 2)
    curr_pts = np.array([curr_bird.feats[k] for k in visible]).reshape(-1, 1, 2)
    E, _ = cv2.findEssentialMat(prev_pts, curr_pts, K)
    Es = [E[i*3:(i+1)*3, :] for i in range(E.shape[0]//3)]
    Rs = []
    ts = []
    for E in Es:
        ret, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, K)
        if not np.any(np.abs(cv2.Rodrigues(R.T)[0] * RAD2DEG) > 5):
            Rs.append(R)
            ts.append(t)
    return len(Rs), Rs, ts


def draw_matches(im1, bird1, im2, bird2):
    out = cv2.hconcat([im1, im2])
    visible = [k for k in CLS_DICT.keys() if bird1.feats[k] is not None and bird2.feats[k] is not None]
    for k in visible:
        pt1 = bird1.feats[k].astype(int)
        pt2 = bird2.feats[k].astype(int) + [im1.shape[1], 0]
        cv2.line(out, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(out, pt1, 3, (0, 255, 255), -1)
        cv2.circle(out, pt2, 3, (0, 255, 255), -1)
    return out


if __name__ == '__main__':
    cap = cv2.VideoCapture('../data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4')
    if cap.isOpened():
        prev = cap.read()[1]
    while cap.isOpened():
        ret, curr = cap.read()
        if ret:
            print('Homography:', estimate_homography(prev, curr)[0])
            print('Essential:', estimate_essential_mat(prev, curr)[0])
            prev = curr
    cap.release()
