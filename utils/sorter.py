import cv2
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.colour import bill_mask
from utils.configs import FEAT_DICT
from utils.general import euc_dist, angle, cosine, cnt_centroid


# filter eyes and tear marks that is within a distance to the bill tip
# distance is determined by the size of the bill
def bound_feat(im, bill, bill_conf, eyes, tear_marks):
    if bill is not None:
        mask = bill_mask(im)
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1 and cv2.contourArea(contours[0]) > 0:
            cnt = contours[0]
            centroid = cnt_centroid(cnt)

            # bounding rect for features to be included
            ratio = 3
            dist = euc_dist(bill, centroid) * 2
            length = dist * ratio
            x, y, w, h = cv2.boundingRect(cnt)
            w += length
            h += length
            dx = centroid[0] - bill[0]
            dy = centroid[1] - bill[1]
            if dx >= 0 and dy != 0 and abs(dx/dy) < 1:
                x -= w * (1 - abs(dx/dy)) / 2
            elif dx < 0:
                if dy != 0 and abs(dx/dy) <= 1:
                    x += w * (1 - abs(dx/dy)) / 2 - length
                else:
                    x -= length
            if dy >= 0 and dx != 0 and abs(dy/dx) < 1:
                y -= h * (1 - abs(dy/dx)) / 2
            elif dy < 0:
                if dx != 0 and abs(dy/dx) <= 1:
                    y += h * (1 - abs(dy/dx)) / 2 - length
                else:
                    y -= length
            x = round(x)
            y = round(y)
            w = round(w)
            h = round(h)

            if x <= bill[0] <= x + w and y <= bill[1] <= y + h:
                for eye in eyes:
                    if not (x <= eye[0] <= x + w and y <= eye[1] <= y + h):
                        eyes.remove(eye)
                for tear in tear_marks:
                    if not (x <= tear[0] <= x + w and y <= tear[1] <= y + h):
                        tear_marks.remove(tear)
            else:
                bill = None
        elif bill_conf < 0.75:
            bill = None
    return bill, eyes, tear_marks


def group_feat(feats):
    feats = np.array(feats)

    # Extract the two sets of points
    pts1 = feats[:, 0, :]  # (n, 2)
    pts2 = feats[:, 1, :]  # (n, 2)

    # Compute pairwise Euclidean distances within each set (A and B)
    distance_matrix_A = cdist(pts1, pts1, metric='euclidean')
    distance_matrix_B = cdist(pts2, pts2, metric='euclidean')

    # Sum distance matrices to get a cost matrix for assignment
    cost_matrix = distance_matrix_A + distance_matrix_B

    # Solve assignment using the Hungarian algorithm
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Assign points based on the optimal matching
    grp1 = pts1[row_idx]  # One point from each pair
    grp2 = pts2[col_idx]  # The other point from each pair

    return np.array([grp1, grp2])


# determine left right of a pair of features
def sort_lr(pivot, top, bot):
    eye_angle = angle(pivot, top)
    tear_angle = angle(pivot, bot)
    if (eye_angle-tear_angle) % (2*math.pi) < math.pi:
        return 'left'
    return 'right'


# sort left right features
def sort_feat(bill, eyes, tear_marks, bill_liners=None):
    encoding = [len(eyes), len(tear_marks)]
    if len(eyes) == 0:
        eyes = [None, None]  # L, R
    if len(tear_marks) == 0:
        tear_marks = [None, None]  # L, R
    if bill_liners is None or len(bill_liners) == 0:
        bill_liners = [None, None]  # L, R
        encoding.append(0)
    else:
        encoding.append(len(bill_liners))

    if encoding == [1, 0, 0]:
        if bill is not None:
            if angle(bill, eyes[0]) >= 0:
                eyes.append(None)
            else:
                eyes.insert(0, None)
        else:
            eyes.append(None)
    elif encoding == [2, 0, 0]:
        if ((bill is not None and sort_lr(bill, *eyes) == 'left')
                or (bill is None and eyes[0][0] < eyes[1][0])):
            eyes = eyes[::-1]
    elif encoding == [0, 1, 0]:
        if bill is not None:
            if angle(bill, tear_marks[0]) >= 0:
                tear_marks.append(None)
            else:
                tear_marks.insert(0, None)
        else:
            tear_marks.append(None)
    elif encoding == [0, 2, 0]:
        if tear_marks[0][0] < tear_marks[1][0]:
            tear_marks = tear_marks[::-1]
    elif encoding == [0, 0, 1]:
        if bill is not None:
            if angle(bill, bill_liners[0]) >= 0:
                bill_liners.append(None)
            else:
                bill_liners.insert(0, None)
        else:
            bill_liners.append(None)
    elif encoding == [0, 0, 2]:
        if bill_liners[0][0] < bill_liners[1][0]:
            bill_liners = bill_liners[::-1]
    elif encoding == [1, 1, 0]:
        if bill is not None:
            if sort_lr(bill, eyes[0], tear_marks[0]) == 'left':
                eyes.append(None)
                tear_marks.append(None)
            else:
                eyes.insert(0, None)
                tear_marks.insert(0, None)
        else:
            eyes.append(None)
            tear_marks.append(None)
    elif encoding == [1, 0, 1]:
        if bill is not None:
            if sort_lr(bill, eyes[0], bill_liners[0]) == 'left':
                eyes.append(None)
                bill_liners.append(None)
            else:
                eyes.insert(0, None)
                bill_liners.insert(0, None)
        else:
            eyes.append(None)
            bill_liners.append(None)
    elif encoding == [0, 1, 1]:
        if bill is not None:
            if sort_lr(bill, bill_liners[0], tear_marks[0]) == 'left':
                tear_marks.append(None)
                bill_liners.append(None)
            else:
                tear_marks.insert(0, None)
                bill_liners.insert(0, None)
        else:
            tear_marks.append(None)
            bill_liners.append(None)
    elif encoding == [1, 1, 1]:
        if sort_lr(bill_liners[0], eyes[0], tear_marks[0]) == 'left':
            eyes.append(None)
            tear_marks.append(None)
            bill_liners.append(None)
        else:
            eyes.insert(0, None)
            tear_marks.insert(0, None)
            bill_liners.insert(0, None)
    elif encoding == [2, 1, 0]:
        if bill is not None:
            if cosine(eyes[0], tear_marks[0], bill) >= cosine(eyes[1], tear_marks[0], bill):
                if sort_lr(bill, eyes[0], tear_marks[0]) == 'left':
                    tear_marks.append(None)
                else:
                    eyes = eyes[::-1]
                    tear_marks.insert(0, None)
            else:
                if sort_lr(bill, eyes[1], tear_marks[0]) == 'left':
                    eyes = eyes[::-1]
                    tear_marks.append(None)
                else:
                    tear_marks.insert(0, None)
        else:
            if euc_dist(eyes[0], tear_marks[0]) <= euc_dist(eyes[1], tear_marks[0]):
                if sort_lr(eyes[1], eyes[0], tear_marks[0]) == 'left':
                    tear_marks.append(None)
                else:
                    eyes = eyes[::-1]
                    tear_marks.insert(0, None)
            else:
                if sort_lr(eyes[0], eyes[1], tear_marks[0]) == 'left':
                    eyes = eyes[::-1]
                    tear_marks.append(None)
                else:
                    tear_marks.insert(0, None)
    elif encoding == [2, 0, 1]:
        if bill is not None:
            if cosine(eyes[0], bill_liners[0], bill) >= cosine(eyes[1], bill_liners[0], bill):
                if sort_lr(bill, eyes[0], bill_liners[0]) == 'left':
                    bill_liners.append(None)
                else:
                    eyes = eyes[::-1]
                    bill_liners.insert(0, None)
            else:
                if sort_lr(bill, eyes[1], bill_liners[0]) == 'left':
                    eyes = eyes[::-1]
                    bill_liners.append(None)
                else:
                    bill_liners.insert(0, None)
        else:
            if euc_dist(eyes[0], bill_liners[0]) <= euc_dist(eyes[1], bill_liners[0]):
                if sort_lr(eyes[1], eyes[0], bill_liners[0]) == 'left':
                    bill_liners.append(None)
                else:
                    eyes = eyes[::-1]
                    bill_liners.insert(0, None)
            else:
                if sort_lr(eyes[0], eyes[1], bill_liners[0]) == 'left':
                    eyes = eyes[::-1]
                    bill_liners.append(None)
                else:
                    bill_liners.insert(0, None)
    elif encoding == [1, 2, 0]:
        if bill is not None:
            if cosine(eyes[0], tear_marks[0], bill) >= cosine(eyes[0], tear_marks[1], bill):
                if sort_lr(bill, eyes[0], tear_marks[0]) == 'left':
                    eyes.append(None)
                else:
                    tear_marks = tear_marks[::-1]
                    eyes.insert(0, None)
            else:
                if sort_lr(bill, eyes[0], tear_marks[1]) == 'left':
                    tear_marks = tear_marks[::-1]
                    eyes.append(None)
                else:
                    eyes.insert(0, None)
        else:
            if euc_dist(eyes[0], tear_marks[0]) <= euc_dist(eyes[0], tear_marks[1]):
                if sort_lr(tear_marks[1], eyes[0], tear_marks[0]) == 'left':
                    eyes.append(None)
                else:
                    tear_marks = tear_marks[::-1]
                    eyes.insert(0, None)
            else:
                if sort_lr(tear_marks[0], eyes[0], tear_marks[1]) == 'left':
                    tear_marks = tear_marks[::-1]
                    eyes.append(None)
                else:
                    eyes.insert(0, None)
    elif encoding == [0, 2, 1]:
        if bill is not None:
            if cosine(bill_liners[0], tear_marks[0], bill) >= cosine(bill_liners[0], tear_marks[1], bill):
                if sort_lr(bill, bill_liners[0], tear_marks[0]) == 'left':
                    bill_liners.append(None)
                else:
                    tear_marks = tear_marks[::-1]
                    bill_liners.insert(0, None)
            else:
                if sort_lr(bill, bill_liners[0], tear_marks[1]) == 'left':
                    tear_marks = tear_marks[::-1]
                    bill_liners.append(None)
                else:
                    bill_liners.insert(0, None)
        else:
            if euc_dist(bill_liners[0], tear_marks[0]) <= euc_dist(bill_liners[0], tear_marks[1]):
                if sort_lr(tear_marks[1], bill_liners[0], tear_marks[0]) == 'left':
                    bill_liners.append(None)
                else:
                    tear_marks = tear_marks[::-1]
                    bill_liners.insert(0, None)
            else:
                if sort_lr(tear_marks[0], bill_liners[0], tear_marks[1]) == 'left':
                    tear_marks = tear_marks[::-1]
                    bill_liners.append(None)
                else:
                    bill_liners.insert(0, None)
    elif encoding == [1, 0, 2]:
        if bill is not None:
            if cosine(eyes[0], bill_liners[0], bill) >= cosine(eyes[0], bill_liners[1], bill):
                if sort_lr(bill, eyes[0], bill_liners[0]) == 'left':
                    eyes.append(None)
                else:
                    bill_liners = bill_liners[::-1]
                    eyes.insert(0, None)
            else:
                if sort_lr(bill, eyes[0], bill_liners[1]) == 'left':
                    bill_liners = bill_liners[::-1]
                    eyes.append(None)
                else:
                    eyes.insert(0, None)
        else:
            if euc_dist(eyes[0], bill_liners[0]) <= euc_dist(eyes[0], bill_liners[1]):
                if sort_lr(bill_liners[1], eyes[0], bill_liners[0]) == 'left':
                    eyes.append(None)
                else:
                    bill_liners = bill_liners[::-1]
                    eyes.insert(0, None)
            else:
                if sort_lr(bill_liners[0], eyes[0], bill_liners[1]) == 'left':
                    bill_liners = bill_liners[::-1]
                    eyes.append(None)
                else:
                    eyes.insert(0, None)
    elif encoding == [0, 1, 2]:
        if bill is not None:
            if cosine(bill_liners[0], tear_marks[0], bill) >= cosine(bill_liners[1], tear_marks[0], bill):
                if sort_lr(bill, bill_liners[0], tear_marks[0]) == 'left':
                    tear_marks.append(None)
                else:
                    bill_liners = bill_liners[::-1]
                    tear_marks.insert(0, None)
            else:
                if sort_lr(bill, bill_liners[1], tear_marks[0]) == 'left':
                    bill_liners = bill_liners[::-1]
                    tear_marks.append(None)
                else:
                    tear_marks.insert(0, None)
        else:
            if euc_dist(bill_liners[0], tear_marks[0]) <= euc_dist(bill_liners[1], tear_marks[0]):
                if sort_lr(bill_liners[1], bill_liners[0], tear_marks[0]) == 'left':
                    tear_marks.append(None)
                else:
                    bill_liners = bill_liners[::-1]
                    tear_marks.insert(0, None)
            else:
                if sort_lr(bill_liners[0], bill_liners[1], tear_marks[0]) == 'left':
                    bill_liners = bill_liners[::-1]
                    tear_marks.append(None)
                else:
                    tear_marks.insert(0, None)
    elif encoding == [2, 1, 1]:
        if (euc_dist(eyes[1], tear_marks[0]) + euc_dist(eyes[1], bill_liners[0]) <
        euc_dist(eyes[0], tear_marks[0]) + euc_dist(eyes[0], bill_liners[0])):
            eyes = eyes[::-1]
        if sort_lr(bill_liners[0], eyes[0], tear_marks[0]) == 'left':
            tear_marks.append(None)
            bill_liners.append(None)
        else:
            eyes = eyes[::-1]
            tear_marks.insert(0, None)
            bill_liners.insert(0, None)
    elif encoding == [1, 2, 1]:
        if (euc_dist(tear_marks[1], eyes[0]) + euc_dist(tear_marks[1], bill_liners[0]) <
                euc_dist(tear_marks[0], eyes[0]) + euc_dist(tear_marks[0], bill_liners[0])):
            tear_marks = tear_marks[::-1]
        if sort_lr(bill_liners[0], eyes[0], tear_marks[0]) == 'left':
            eyes.append(None)
            bill_liners.append(None)
        else:
            tear_marks = tear_marks[::-1]
            eyes.insert(0, None)
            bill_liners.insert(0, None)
    elif encoding == [1, 1, 2]:
        if (euc_dist(bill_liners[1], eyes[0]) + euc_dist(bill_liners[1], tear_marks[0]) <
                euc_dist(bill_liners[0], eyes[0]) + euc_dist(bill_liners[0], tear_marks[0])):
            bill_liners = bill_liners[::-1]
        if sort_lr(bill_liners[0], eyes[0], tear_marks[0]) == 'left':
            eyes.append(None)
            tear_marks.append(None)
        else:
            bill_liners = bill_liners[::-1]
            eyes.insert(0, None)
            tear_marks.insert(0, None)
    elif encoding == [2, 2, 0] or encoding == [2, 2, 1]:
        if all([euc_dist(eyes[0], tear_marks[i]) < euc_dist(eyes[1], tear_marks[i]) for i in range(2)]):
            if euc_dist(eyes[1], tear_marks[0]) < euc_dist(eyes[1], tear_marks[1]):
                tear_marks = tear_marks[::-1]
        elif all([euc_dist(eyes[1], tear_marks[i]) < euc_dist(eyes[0], tear_marks[i]) for i in range(2)]):
            if euc_dist(eyes[0], tear_marks[1]) < euc_dist(eyes[0], tear_marks[0]):
                tear_marks = tear_marks[::-1]
        elif euc_dist(eyes[0], tear_marks[1]) < euc_dist(eyes[0], tear_marks[0]):
            tear_marks = tear_marks[::-1]
        pivotL = bill if bill is not None else eyes[1]
        pivotR = bill if bill is not None else eyes[0]
        if sort_lr(pivotL, eyes[0], tear_marks[0]) == 'right' and sort_lr(pivotR, eyes[1], tear_marks[1]) == 'left':
            eyes = eyes[::-1]
            tear_marks = tear_marks[::-1]
        if encoding[2] == 1:
            if (euc_dist(bill_liners[0], eyes[1]) + euc_dist(bill_liners[0], tear_marks[1]) <
                    euc_dist(bill_liners[0], eyes[0]) + euc_dist(bill_liners[0], tear_marks[0])):
                bill_liners.insert(0, None)
            else:
                bill_liners.append(None)
    elif encoding == [2, 0, 2] or encoding == [2, 1, 2]:
        if all([euc_dist(eyes[0], bill_liners[i]) < euc_dist(eyes[1], bill_liners[i]) for i in range(2)]):
            if euc_dist(eyes[1], bill_liners[0]) < euc_dist(eyes[1], bill_liners[1]):
                bill_liners = bill_liners[::-1]
        elif all([euc_dist(eyes[1], bill_liners[i]) < euc_dist(eyes[0], bill_liners[i]) for i in range(2)]):
            if euc_dist(eyes[0], bill_liners[1]) < euc_dist(eyes[0], bill_liners[0]):
                bill_liners = bill_liners[::-1]
        elif euc_dist(eyes[0], bill_liners[1]) < euc_dist(eyes[0], bill_liners[0]):
            bill_liners = bill_liners[::-1]
        pivotL = bill if bill is not None else eyes[1]
        pivotR = bill if bill is not None else eyes[0]
        if sort_lr(pivotL, eyes[0], bill_liners[0]) == 'right' and sort_lr(pivotR, eyes[1], bill_liners[1]) == 'left':
            eyes = eyes[::-1]
            bill_liners = bill_liners[::-1]
        if encoding[1] == 1:
            if (euc_dist(tear_marks[0], eyes[1]) + euc_dist(tear_marks[0], bill_liners[1]) <
                    euc_dist(tear_marks[0], eyes[0]) + euc_dist(tear_marks[0], bill_liners[0])):
                tear_marks.insert(0, None)
            else:
                tear_marks.append(None)
    elif encoding == [0, 2, 2] or encoding == [1, 2, 2]:
        if all([euc_dist(bill_liners[0], tear_marks[i]) < euc_dist(bill_liners[1], tear_marks[i]) for i in range(2)]):
            if euc_dist(bill_liners[1], tear_marks[0]) < euc_dist(bill_liners[1], tear_marks[1]):
                tear_marks = tear_marks[::-1]
        elif all([euc_dist(bill_liners[1], tear_marks[i]) < euc_dist(bill_liners[0], tear_marks[i]) for i in range(2)]):
            if euc_dist(bill_liners[0], tear_marks[1]) < euc_dist(bill_liners[0], tear_marks[0]):
                tear_marks = tear_marks[::-1]
        elif euc_dist(bill_liners[0], tear_marks[1]) < euc_dist(bill_liners[0], tear_marks[0]):
            tear_marks = tear_marks[::-1]
        pivotL = bill if bill is not None else bill_liners[1]
        pivotR = bill if bill is not None else bill_liners[0]
        if sort_lr(pivotL, bill_liners[0], tear_marks[0]) == 'right' and sort_lr(pivotR, bill_liners[1], tear_marks[1]) == 'left':
            bill_liners = bill_liners[::-1]
            tear_marks = tear_marks[::-1]
        if encoding[0] == 1:
            if (euc_dist(eyes[0], tear_marks[1]) + euc_dist(eyes[0], bill_liners[1]) <
            euc_dist(eyes[0], tear_marks[0]) + euc_dist(eyes[0], bill_liners[0])):
                eyes.insert(0, None)
            else:
                eyes.append(None)
    elif encoding == [2, 2, 2]:
        grouped = group_feat([eyes, tear_marks, bill_liners])
        eyes = grouped[:, 0]
        tear_marks = grouped[:, 1]
        bill_liners = grouped[:, 2]
        if sort_lr(bill_liners[0], eyes[0], tear_marks[0]) == 'right' and sort_lr(bill_liners[1], eyes[1], tear_marks[1]) == 'left':
            eyes = eyes[::-1]
            bill_liners = bill_liners[::-1]
            tear_marks = tear_marks[::-1]
    return bill, eyes, tear_marks, bill_liners


# check for overlapping labels
def match_labels(target, labels, dist):
    match = False
    i = 0
    while not match and i < len(labels):
        label = labels[i]
        if all([label[i]-dist <= target[i] <= label[i]+dist for i in range(2)]):
            match = True
        i += 1
    return match


# remove overlapping labels and return best n labels
def process_labels(labels, n, dist=3, im_shape=None):
    i = 0
    count = 0
    out_labels = []
    while count < n and i < len(labels):
        label = labels[i]
        if im_shape is not None:
            label = [round(label[1+i] * im_shape[1-i]) for i in range(2)]
        if not match_labels(label, out_labels, dist):
            out_labels.append(label)
            count += 1
        i += 1
    return out_labels


def filter_feat(im, det):
    bill_labels = sorted([label for label in det if label[0] == FEAT_DICT['bill'] and label[-1] >= 0.1], key=lambda x: x[-1],
                          reverse=True)
    if bill_labels:
        # choose bill with the highest confidence
        bill_label = bill_labels[0]
        bill = [round(bill_label[1 + i] * im.shape[1 - i]) for i in range(2)]
        bill_conf = bill_label[-1]
    else:
        bill = None
        bill_conf = 0

    # sort by confidence
    eyes_labels = sorted([label for label in det if label[0] in FEAT_DICT['eyes'] and label[-1] >= 0.1], key=lambda x: x[-1])
    eyes = process_labels(eyes_labels, 2, im.shape[:2])

    tear_labels = sorted([label for label in det if label[0] in FEAT_DICT['tear_marks'] and label[-1] >= 0.1], key=lambda x: x[-1])
    tear_marks = process_labels(tear_labels, 2, im.shape[:2])
    return sort_feat(*bound_feat(im, bill, bill_conf, eyes, tear_marks))


def to_txt(im, bill, eyes, tear_marks, bill_liners=None, start=(0, 0)):
    shape = im.shape[:2]
    if bill is not None:
        bill = [(start[i] + bill[i])/shape[1-i] for i in range(2)]
    for i in range(2):
        if eyes[i]:
            eyes[i] = [(start[j]+eyes[i][j])/shape[1-j] for j in range(2)]
        if tear_marks[i]:
            tear_marks[i] = [(start[j]+tear_marks[i][j])/shape[1-j] for j in range(2)]
        if bill_liners is not None and bill_liners[i]:
            bill_liners[i] = [(start[j]+bill_liners[i][j])/shape[1-j] for j in range(2)]
    if bill_liners is None:
        return bill, eyes, tear_marks
    return bill, eyes, tear_marks, bill_liners


def to_dict(bill, eyes, tear_marks, bill_liners=None):
    dict = {'bill': bill,
            'left_eye': eyes[0],
            'right_eye': eyes[1],
            'left_tear': tear_marks[0],
            'right_tear': tear_marks[1]}
    if bill_liners is not None:
        dict['left_liner'] = bill_liners[0]
        dict['right_liner'] = bill_liners[1]
    return dict
