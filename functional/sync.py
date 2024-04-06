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
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
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


def main():
    ROOT = os.path.dirname(os.getcwd())
    vid_root = os.path.join(ROOT, 'data/vid/fps120/K203_K238')
    vid_dir1 = os.path.join(vid_root, 'GOPRO1')
    vid_dir2 = os.path.join(vid_root, 'GOPRO2')
    vid_paths1 = [os.path.join(vid_dir1, path) for path in os.listdir(vid_dir1) if path[-3:] == 'MP4']
    vid_paths2 = [os.path.join(vid_dir2, path) for path in os.listdir(vid_dir2) if path[-3:] == 'MP4']
    vid_paths = [vid_paths1, vid_paths2]

    # offset, offset_vid = get_offset(vid_paths1[0], vid_paths2[0])
    caps = [cv2.VideoCapture(vid_path[0]) for vid_path in vid_paths]
    skip = 1700
    offset1 = first_flash(vid_paths1[0])
    offset2 = first_flash(vid_paths2[0])
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, offset1+skip)
    caps[1].set(cv2.CAP_PROP_POS_FRAMES, offset2+skip)
    # caps[offset_vid].set(cv2.CAP_PROP_POS_FRAMES, offset)

    count = 0
    e = None
    while caps[0].isOpened() and caps[1].isOpened():
        for i in range(10):
            count += 1
            for j in range(2):
                caps[j].grab
        ret1, frame1 = caps[0].retrieve()
        ret2, frame2 = caps[1].retrieve()
        if ret1 and ret2:
            frame1 = cv2.resize(frame1, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            frame2 = cv2.resize(frame2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('frame', cv2.vconcat([frame1, frame2]))
            cv2.waitKey(1)
            if e is None:
                e, mask = stereo_essential_mat(frame1, frame2)
                print(count, e)
                if e: break
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
