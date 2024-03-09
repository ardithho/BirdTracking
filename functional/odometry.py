import cv2
import numpy as np

# https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf

def extractFeatures(frame):
    orb = cv2.ORB_create()
    return orb.detectAndCompute()


def estimateMotion(curr_frame, prev_frame):
    kp1, des1 = extractFeatures(prev_frame)
    kp2, des2 = extractFeatures(curr_frame)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, corssCheck=True)
    match = bf.match(des1, des2)
    distThreshold = 0.9
    filteredMatch = [m for m, n in match if m.distance < distThreshold * n.distance]

    # for m in filteredMatch:
    #     u1, v1 = kp1[m.queryIdx].pt
    #     u2, v2 = kp2[m.trainIdx].pt
    src_pts = np.float32([kp1[m.queryIdx].pt for m in filteredMatch]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in filteredMatch]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
