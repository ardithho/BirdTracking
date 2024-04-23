import cv2
import numpy as np


def extract_features(frame, mask=None):
    orb = cv2.ORB_create()
    return orb.detectAndCompute(frame, mask)


def find_matches(prev_frame, curr_frame, prev_mask=None, curr_mask=None):
    kp1, des1 = extract_features(prev_frame, prev_mask)
    kp2, des2 = extract_features(curr_frame, curr_mask)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(bf.match(des1, des2))
    matches.sort(key=lambda x: x.distance)
    thresh = 0.9
    filtered = matches[:int(len(matches) * thresh)]
    return filtered, kp1, kp2


def find_matching_pts(prev_frame, curr_frame, prev_mask=None, curr_mask=None):
    matches, kp1, kp2 = find_matches(prev_frame, curr_frame, prev_mask, curr_mask)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def estimate_homography(prev_frame, curr_frame, prev_mask=None, curr_mask=None):
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


def estimate_essential_mat(prev_frame, curr_frame, prev_mask=None, curr_mask=None):
    # https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    src_pts, dst_pts = find_matching_pts(curr_frame, prev_frame, curr_mask, prev_mask)
    return cv2.findEssentialMat(src_pts, dst_pts)


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
