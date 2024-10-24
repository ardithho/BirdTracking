import cv2
import numpy as np


def plot_box(im, xyxy, colour):
    return cv2.rectangle(im, np.rint(xyxy[:2]).astype(np.uint32), np.rint(xyxy[2:]).astype(np.uint32), colour, 3)


def plot_feat(im, bill, eyes, tear_marks, start=(0, 0)):
    line_colours = [(0, 255, 0), (0, 0, 255)]  # L, R
    if bill is not None:
        bill = [round(start[i]+bill[i]) for i in range(2)]
        for i in range(2):
            eye = eyes[i]
            if eye is not None:
                eye = [round(start[i]+eye[i]) for i in range(2)]
                cv2.line(im, eye, bill, line_colours[i], 2)
                cv2.circle(im, eye, 4, (0, 255, 255), -1)
            tear_mark = tear_marks[i]
            if tear_mark is not None:
                tear_mark = [round(start[i]+tear_mark[i]) for i in range(2)]
                cv2.line(im, tear_mark, bill, line_colours[i], 2)
                cv2.circle(im, tear_mark, 4, (0, 150, 255), -1)
        cv2.circle(im, bill, 4, (255, 255, 0), -1)
    else:
        for i in range(2):
            eye = eyes[i]
            if eye is not None:
                eye = [round(start[i]+eye[i]) for i in range(2)]
                cv2.circle(im, eye, 4, (0, 255, 255), -1)
                cv2.circle(im, eye, 5, line_colours[i], 2)
            tear_mark = tear_marks[i]
            if tear_mark is not None:
                tear_mark = [round(start[i]+tear_mark[i]) for i in range(2)]
                cv2.circle(im, tear_mark, 4, (0, 150, 255), -1)
                cv2.circle(im, tear_mark, 5, line_colours[i], 2)
    return im
