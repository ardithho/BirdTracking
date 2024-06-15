class Feature:
    def __init__(self, feat, xy, xyn):
        self.cls = feat.cls
        self.conf = feat.conf
        self.xy = xy
        self.xyn = xyn


class Bird:
    def __init__(self, head, feats):
        self.conf = head.conf
        self.id = head.id
        self.xywh = head.xywh
        self.xywhn = head.xywhn
        self.xyxy = head.xyxy
        self.xyxyn = head.xyxyn
        self.feats_unsorted = self.globalise(feats)
        self.feats = {}

    def globalise(self, feats):
        globalised = []
        for feat in feats:
            xy = self.xyxy[:2] + self.xywh[2:] * feat.xywh[:2]
            xyn = self.xyxyn[:2] + self.xywhn[2:] * feat.xywhn[:2]
            globalised.append(Feature(feat, xy, xyn))
        return globalised


class Birds:
    def __init__(self):
        self.m = None
        self.f = None


class Cache:
    def __init__(self, size=5):
        self.size = size
        self.cache = [None] * self.size
        self.ptr = 0

    def update(self, bird):
        self.cache[self.ptr] = bird
        self.ptr = (self.ptr + 1) % self.size

    def interpolate(self):
        x = self[0]

    def __getitem__(self, idx):
        return self.cache[(self.ptr + idx) % self.size]

    def __setitem__(self, idx, obj):
        self.cache[(self.ptr + idx) % self.size] = obj
