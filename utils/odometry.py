import cv2
import numpy as np

from lightglue import LightGlue, SuperPoint, match_pair, viz2d
from lightglue.utils import load_image, numpy_image_to_torch

from utils.general import RAD2DEG
from utils.structs import CLS_DICT


FLANN_INDEX_KDTREE = 1


def extract_features(frame, mask=None, method='orb'):
    if method == 'orb':
        extractor = cv2.ORB_create()
    else:
        extractor = cv2.SIFT_create(nfeatures=1000)
    return extractor.detectAndCompute(frame, mask)


def find_matches(prev_frame, curr_frame,
                 prev_mask=None, curr_mask=None, thresh=.8, method='orb'):
    kp1, des1 = extract_features(prev_frame, prev_mask, method)
    kp2, des2 = extract_features(curr_frame, curr_mask, method)

    if method == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = list(bf.match(des1, des2))
        matches.sort(key=lambda x: x.distance)
        filtered = matches[:int(len(matches) * thresh)]
    else:
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        filtered = []
        for m, n in matches:
            if m.distance < thresh * n.distance:
                filtered.append(m)
    return kp1, kp2, filtered


def find_matching_pts(prev_frame, curr_frame,
                      prev_mask=None, curr_mask=None, thresh=.8, method='orb'):
    if method == 'lg':
        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
        feats0, feats1, matches01 = match_pair(
            extractor, matcher,
            numpy_image_to_torch(prev_frame).cuda(),
            numpy_image_to_torch(curr_frame).cuda())
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        return m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy()

    kp0, kp1, matches = find_matches(prev_frame, curr_frame,
                                     prev_mask, curr_mask, thresh, method)
    src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def estimate_homography(prev_frame, curr_frame,
                        prev_mask=None, curr_mask=None, thresh=.8, method='orb'):
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame,
                                         curr_mask, prev_mask, thresh, method)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


def estimate_essential_mat(prev_frame, curr_frame,
                           prev_mask=None, curr_mask=None,
                           K=None, dist=None, thresh=.8, method='orb'):
    # https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame,
                                         curr_mask, prev_mask, thresh, method)
    if K is not None:
        return cv2.findEssentialMat(src_pts, dst_pts, K, dist, K, dist)
    return cv2.findEssentialMat(src_pts, dst_pts)


# visual odometry
def estimate_vio(prev_frame, curr_frame,
                 prev_mask=None, curr_mask=None,
                 K=None, dist=None, thresh=.8, method='orb'):
    # return: retval, R, t, mask
    src_pts, dst_pts = find_matching_pts(prev_frame, curr_frame,
                                         prev_mask, curr_mask, thresh, method)
    # return cv2.recoverPose(src_pts, dst_pts, K, dist, K, dist, threshold=thresh)
    if len(src_pts) > 0:
        return estimate_vio_pts(src_pts, dst_pts, K, dist)
    return False, None, None, None


def estimate_vio_pts(src_pts, dst_pts, K, dist=None):
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, dist, K, dist, threshold=.5)
    if E is None:
        return False, None, None, None
    if len(E) == 3:
        return cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    Es = [E[i * 3:(i + 1) * 3, :] for i in range(E.shape[0] // 3)]
    bestE = None
    min_mag = 1000
    for E in Es:
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
        rvec = cv2.Rodrigues(R)[0]
        if not np.any(np.abs(cv2.Rodrigues(R.T)[0] * RAD2DEG) > 25):
            mag = np.linalg.norm(rvec)
            if mag < min_mag:
                bestE = E
                min_mag = mag
    if bestE is None:
        return False, None, None, None
    return cv2.recoverPose(bestE, src_pts, dst_pts, K, mask=mask)


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
        if not np.any(np.abs(cv2.Rodrigues(R.T)[0] * RAD2DEG) > 20):
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
