import os
import cv2
import numpy as np


def get_offset(vid_path1, vid_path2, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    flash1 = first_flash(vid_path1, kernel)
    flash2 = first_flash(vid_path2, kernel)
    offset = int(abs(flash1 - flash2))
    offset_vid = int(((flash2 - flash1) / offset + 1) / 2)
    return offset, offset_vid


def first_flash(vid_path, kernel):
    cap = cv2.VideoCapture(vid_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)
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
            # cv2.imshow('frame', frame)
            # cv2.imshow('mask', mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >= 1:
                flash = count
                print(flash)
                break
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
        else:
            print(count)
            break

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    return flash


if __name__ == '__main__':
    ROOT = os.path.dirname(os.getcwd())
    vid_root = os.path.join(ROOT, 'vid/fps120/K203_K238')
    # vid_root = 'D:/orange_cheek/K203_K238'
    vid_dir1 = os.path.join(vid_root, 'GOPRO1')
    vid_dir2 = os.path.join(vid_root, 'GOPRO2')
    vid_paths1 = [os.path.join(vid_dir1, path) for path in os.listdir(vid_dir1) if path[-3:] == 'MP4']
    vid_paths2 = [os.path.join(vid_dir2, path) for path in os.listdir(vid_dir2) if path[-3:] == 'MP4']
    vid_paths = [vid_paths1, vid_paths2]

    offset, offset_vid = get_offset(vid_paths1[0], vid_paths2[0])
    caps = [cv2.VideoCapture(vid_paths[i][0]) for i in range(2)]
    caps[offset_vid].set(cv2.CAP_PROP_POS_FRAMES, offset)

    while caps[0].isOpened() and caps[1].isOpened():
        ret1, frame1 = caps[0].read()
        ret2, frame2 = caps[1].read()
        if ret1 and ret2:
            frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('GOPRO1', frame1)
            cv2.imshow('GOPRO2', frame2)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()