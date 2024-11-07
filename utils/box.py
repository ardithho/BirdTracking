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
