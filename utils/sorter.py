import cv2
import math
from general import bill_mask, euc_dist, angle


def bound_feat(img, bill, bill_conf, eyes, tear_marks):
    if bill:
        mask = bill_mask(img)
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1 and cv2.contourArea(contours[0]) > 0:
            cnt = contours[0]
            M = cv2.moments(cnt)
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            # bounding rect
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


def sort_lr(pivot, eye, tear_mark):  # determine left right of a pair of features
    eye_angle = angle(pivot, eye)
    tear_angle = angle(pivot, tear_mark)
    if abs(tear_angle - eye_angle) < math.pi:
        if tear_angle >= eye_angle:
            return 'left'
        return 'right'
    if tear_angle < eye_angle:
        return 'left'
    return 'right'


def sort_feat(bill, eyes, tear_marks):  # sort left right features
    if len(eyes) == 0 and len(tear_marks) == 0:
        eyes = [None, None]  # L, R
        tear_marks = [None, None]  # L, R
    elif len(eyes) == 0:
        eyes = [None, None]
        if len(tear_marks) == 1:
            if bill:
                if tear_marks[0][0] >= bill[0]:
                    tear_marks.append(None)
                else:
                    tear_marks.insert(0, None)
            else:
                tear_marks.append(None)
        else:
            if tear_marks[0][0] < tear_marks[1][0]:
                tear_marks = [tear_marks[1], tear_marks[0]]
    elif len(tear_marks) == 0:
        tear_marks = [None, None]
        if len(eyes) == 1:
            if bill:
                if eyes[0][0] >= bill[0]:
                    eyes.append(None)
                else:
                    eyes.insert(0, None)
            else:
                eyes.append(None)
        else:
            if eyes[0][0] < eyes[1][0]:
                eyes = [eyes[1], eyes[0]]
    elif len(eyes) == 1 and len(tear_marks) == 1:
        if bill:
            if sort_lr(bill, eyes[0], tear_marks[0]) == 'left':
                eyes.append(None)
                tear_marks.append(None)
            else:
                eyes.insert(0, None)
                tear_marks.insert(0, None)
        else:
            eyes.append(None)
            tear_marks.append(None)
    elif len(eyes) > len(tear_marks):
        if euc_dist(eyes[0], tear_marks[0]) <= euc_dist(eyes[1], tear_marks[0]):
            pivot = bill if bill else eyes[1]
            if sort_lr(pivot, eyes[0], tear_marks[0]) == 'left':
                tear_marks.append(None)
            else:
                eyes = [eyes[1], eyes[0]]
                tear_marks.insert(0, None)
        else:
            pivot = bill if bill else eyes[0]
            if sort_lr(pivot, eyes[1], tear_marks[0]) == 'left':
                eyes = [eyes[1], eyes[0]]
                tear_marks.append(None)
            else:
                tear_marks.insert(0, None)
    elif len(eyes) < len(tear_marks):
        if euc_dist(eyes[0], tear_marks[0]) <= euc_dist(eyes[0], tear_marks[1]):
            pivot = bill if bill else tear_marks[1]
            if sort_lr(pivot, eyes[0], tear_marks[0]) == 'left':
                eyes.append(None)
            else:
                tear_marks = [tear_marks[1], tear_marks[0]]
                eyes.insert(0, None)
        else:
            pivot = bill if bill else tear_marks[0]
            if sort_lr(pivot, eyes[0], tear_marks[1]) == 'left':
                tear_marks = [tear_marks[1], tear_marks[0]]
                eyes.append(None)
            else:
                eyes.insert(0, None)
    else:  # full set
        if all([euc_dist(eyes[0], tear_marks[i]) < euc_dist(eyes[1], tear_marks[i]) for i in range(2)]):
            if euc_dist(eyes[1], tear_marks[0]) < euc_dist(eyes[1], tear_marks[1]):
                tear_marks = [tear_marks[1], tear_marks[0]]
        elif all([euc_dist(eyes[1], tear_marks[i]) < euc_dist(eyes[0], tear_marks[i]) for i in range(2)]):
            if euc_dist(eyes[0], tear_marks[1]) < euc_dist(eyes[0], tear_marks[0]):
                tear_marks = [tear_marks[1], tear_marks[0]]
        elif euc_dist(eyes[0], tear_marks[1]) < euc_dist(eyes[0], tear_marks[0]):
            tear_marks = [tear_marks[1], tear_marks[0]]
        pivotL = bill if bill else eyes[1]
        pivotR = bill if bill else eyes[0]
        if sort_lr(pivotL, eyes[0], tear_marks[0]) == 'right' and sort_lr(pivotR, eyes[1], tear_marks[1]) == 'left':
            eyes = [eyes[1], eyes[0]]
            tear_marks = [tear_marks[1], tear_marks[0]]
    return bill, eyes, tear_marks


def match_labels(target, labels, dist):  # check for overlapping labels
    match = False
    i = 0
    while not match and i < len(labels):
        label = labels[i]
        if all([label[i]-dist <= target[i] <= label[i]+dist for i in range(2)]):
            match = True
        i += 1
    return match


def process_labels(labels, n, img_shape):  # remove overlapping labels and return best n
    i = 0
    count = 0
    out_labels = []
    while count < n and i < len(labels):
        label = labels[i]
        label = [round(label[1+i] * img_shape[1-i]) for i in range(2)]
        if not match_labels(label, out_labels, 3):
            out_labels.append(label)
            count += 1
        i += 1
    return out_labels


def filter_feat(img, det, classes):
    bill_labels = sorted([label for label in det if label[0] == classes[0] and label[-1] >= 0.1], key=lambda x: x[-1],
                          reverse=True)
    if bill_labels:
        # choose bill with the highest confidence
        bill_label = bill_labels[0]
        bill = [round(bill_label[1 + i] * img.shape[1 - i]) for i in range(2)]
        bill_conf = bill_label[-1]
    else:
        bill = None
        bill_conf = 0

    # sort by confidence
    eyes_labels = sorted([label for label in det if label[0] in classes[1] and label[-1] >= 0.1], key=lambda x: x[-1])
    eyes = process_labels(eyes_labels, 2, img.shape[:2])

    tear_labels = sorted([label for label in det if label[0] in classes[2] and label[-1] >= 0.1], key=lambda x: x[-1])
    tear_marks = process_labels(tear_labels, 2, img.shape[:2])
    return sort_feat(*bound_feat(img, bill, bill_conf, eyes, tear_marks))


def plot_feat(img, bill, eyes, tear_marks, start=(0, 0)):
    line_colours = [(0, 255, 0), (0, 0, 255)]  # L, R
    if bill:
        bill = [round(start[i]+bill[i]) for i in range(2)]
        for i in range(2):
            eye = eyes[i]
            if eye:
                eye = [round(start[i]+eye[i]) for i in range(2)]
                cv2.line(img, eye, bill, line_colours[i], 2)
                cv2.circle(img, eye, 4, (0, 255, 255), -1)
            tear_mark = tear_marks[i]
            if tear_mark:
                tear_mark = [round(start[i]+tear_mark[i]) for i in range(2)]
                cv2.line(img, tear_mark, bill, line_colours[i], 2)
                cv2.circle(img, tear_mark, 4, (0, 150, 255), -1)
        cv2.circle(img, bill, 4, (255, 255, 0), -1)
    else:
        for i in range(2):
            eye = eyes[i]
            if eye:
                eye = [round(start[i]+eye[i]) for i in range(2)]
                cv2.circle(img, eye, 4, (0, 255, 255), -1)
                cv2.circle(img, eye, 5, line_colours[i], 2)
            tear_mark = tear_marks[i]
            if tear_mark:
                tear_mark = [round(start[i]+tear_mark[i]) for i in range(2)]
                cv2.circle(img, tear_mark, 4, (0, 150, 255), -1)
                cv2.circle(img, tear_mark, 5, line_colours[i], 2)
    return img


def to_txt(img, bill, eyes, tear_marks, start=(0, 0)):
    shape = img.shape[:2]
    if bill:
        bill = [(start[i] + bill[i])/shape[1-i] for i in range(2)]
    for i in range(2):
        if eyes[i]:
            eyes[i] = [(start[j]+eyes[i][j])/shape[1-j] for j in range(2)]
        if tear_marks[i]:
            tear_marks[i] = [(start[j]+tear_marks[i][j])/shape[1-j] for j in range(2)]
    return bill, eyes, tear_marks


def to_dict(bill, eyes, tear_marks):
    return {'bill': bill,
            'left_eye': eyes[0],
            'left_tear': tear_marks[0],
            'right_eye': eyes[1],
            'right_tear': tear_marks[1]}
