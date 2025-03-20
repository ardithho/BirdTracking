import numpy as np


def iou(box1, box2):
    xyxy1 = box1.xyxy
    xyxy2 = box2.xyxy
    iw = max(min(xyxy1[2], xyxy2[2]) - max(xyxy1[0], xyxy2[0]), 0)
    ih = max(min(xyxy1[3], xyxy2[3]) - max(xyxy1[1], xyxy2[1]), 0)
    intersect = iw * ih
    union = box1.area + box2.area - intersect
    return intersect / union


def extract_boxes(boxes):
    ret = []
    for cls, conf, id, xywh, xywhn, xyxy, xyxyn \
            in boxes.cls, boxes.conf, boxes.id, boxes.xywh, boxes.xywhn, boxes.xyxy, boxes.xyxyn:
        ret.append(Box(cls, conf, id, xywh, xywhn, xyxy, xyxyn))
    return ret


def pad_boxes(boxes, im_shape, padding=0):
    h, w = im_shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = list(box.xyxy[0])
        x0, x1 = max(x0-padding, 0), min(x1+padding, w)
        y0, y1 = max(y0-padding, 0), min(y1+padding, h)
        box.xyxy[0] = np.array([x0, y0, x1, y1])
        box.xyxyn[0] = box.xyxy[0] / np.array([w, h, w, h])
        box.xywh[0][0] = (x0 + x1) / 2
        box.xywh[0][1] = (y0 + y1) / 2
        box.xywh[0][2] = x1 - x0
        box.xywh[0][3] = y1 - y0
        box.xywhn[0] = box.xywh[0] / np.array([w, h, w, h])
    return boxes


class Box:
    def __init__(self, cls, conf=0., id=None, xywh=None, xywhn=None, xyxy=None, xyxyn=None):
        self.cls = cls
        self.conf = conf
        self.id = id
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn
        # self.area = xywh[2] * xywh[3]
        # self.arean = xywhn[2] * xywhn[3]
