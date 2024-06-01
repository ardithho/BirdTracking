def iou(box1, box2):
    xyxy1 = box1.xyxy
    xyxy2 = box2.xyxy
    intersect = (min(xyxy1[2], xyxy2[2]) - max(xyxy1[0], xyxy2[0])) * (min(xyxy1[3], xyxy2[3]) - max(xyxy1[1], xyxy2[1]))
    union = box1.area() + box2.area() - intersect
    return intersect / union


class Box:
    def __init__(self, cls, conf, id, xywh, xywhn, xyxy, xyxyn):
        self.cls = cls
        self.conf = conf
        self.id = id
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn

    def area(self, normalised=False):
        if normalised:
            return self.xywhn[2] * self.xywhn[3]
        return self.xywh[2] * self.xywh[3]
