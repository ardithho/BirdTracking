import cv2
import numpy as np


def extract_features(frame, mask=None):
    orb = cv2.ORB_create()
    return orb.detectAndCompute(frame, mask)


def find_matching_pts(curr_frame, prev_frame, curr_mask=None, prev_mask=None):
    kp1, des1 = extract_features(prev_frame, prev_mask)
    kp2, des2 = extract_features(curr_frame, curr_mask)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, corssCheck=True)
    match = bf.match(des1, des2)
    distThreshold = 0.9
    filteredMatch = [m for m, n in match if m.distance < distThreshold * n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in filteredMatch]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in filteredMatch]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def estimate_homography(curr_frame, prev_frame, curr_mask=None, prev_mask=None):
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


def estimate_essential_mat(curr_frame, prev_frame, curr_mask=None, prev_mask=None):
    # https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    return cv2.findEssentialMat(src_pts, dst_pts)
