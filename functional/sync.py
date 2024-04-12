import os
import cv2
import numpy as np
from calibrate import stereo_essential_mat


def calculate_offset(flash1, flash2):
    offset = int(abs(flash1 - flash2))
    offset_vid = int(((flash2 - flash1) / offset + 1) / 2)
    return offset, offset_vid


def get_offset(vid_path1, vid_path2):
    flash1 = first_flash(vid_path1)
    print('Video 1 first flash:', flash1)
    flash2 = first_flash(vid_path2)
    print('Video 2 first flash:', flash2)
    return calculate_offset(flash1, flash2)


def first_flash(vid_path, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cap = cv2.VideoCapture(vid_path)
    count = 0
    flash = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

            low = np.array([0, 0, 255])
            high = np.array([0, 0, 255])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, low, high)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >= 1:
                flash = count
                break
        else:
            break

    cap.release()
    return flash


def sync(vidL, vidR, skip=1800, stride=30):
    capL = cv2.VideoCapture(vidL)
    capR = cv2.VideoCapture(vidR)
    offsetL = first_flash(vidL) + skip
    offsetR = first_flash(vidR) + skip
    capL.set(cv2.CAP_PROP_POS_FRAMES, offsetL)
    capR.set(cv2.CAP_PROP_POS_FRAMES, offsetR)

    e = None
    while capL.isOpened() and capR.isOpened():
        for i in range(stride):
            _ = capL.grab()
            _ = capR.grab()
            offsetL += 1
            offsetR += 1
        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()
        if retL and retR:
            if e is None:
                e, mask = stereo_essential_mat(frameL, frameR)
                if e is not None:
                    break
        else:
            break
        capL.release()
        capR.release()

    return offsetL, offsetR, e, mask


def main():
    ROOT = os.path.dirname(os.getcwd())
    vid_root = os.path.join(ROOT, 'data/vid/fps120/K203_K238')
    vid_dir1 = os.path.join(vid_root, 'GOPRO2')
    vid_dir2 = os.path.join(vid_root, 'GOPRO1')
    vid_paths1 = [os.path.join(vid_dir1, path) for path in os.listdir(vid_dir1) if path[-3:] == 'MP4']
    vid_paths2 = [os.path.join(vid_dir2, path) for path in os.listdir(vid_dir2) if path[-3:] == 'MP4']
    vid_paths = [vid_paths1, vid_paths2]

    print(sync(vid_paths[0][0], vid_paths[1][0]))


if __name__ == '__main__':
    main()
