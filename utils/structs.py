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
        self.featsUnsorted = self.globalise(feats)
        self.feats = {}

    def globalise(self, feats):
        globalised = []
        for feat in feats:
            xy = self.xyxy[:2] + self.xywh[2:] * feat.xywh[:2]
            xyn = self.xyxyn[:2] + self.xywhn[2:] * feat.xywhn[:2]
            globalised.append(Feature(feat, xy, xyn))
        return globalised

    def sort(self):
        pass


class Birds:
    def __init__(self):
        self.current = {}
        self.caches = {'m': Cache(), 'f': Cache()}
        self.ids = None

    def update(self, birds, frame):
        if self.ids is None:
            self.sort(birds, frame)
        unseen = ['m', 'f']
        for bird in birds:
            self.current[self.ids[bird.id]] = bird
            self.caches[self.ids[bird.id]].update(bird)
            unseen.pop(self.ids[bird.id])
        if len(unseen) > 0:
            for sex in unseen:
                self.current[sex] = None
                self.caches[sex].update(None)

    def sort(self, birds, frame):
        self.ids = {}

    def __getitem__(self, sex):
        return self.current[sex]


class Cache:
    def __init__(self, size=5):
        self.size = size
        self.cache = [None] * self.size
        self.ptr = 0

    def update(self, obj):
        self.cache[self.ptr] = obj
        self.ptr = (self.ptr + 1) % self.size

    def interpolate(self):
        x = self[0]

    def __getitem__(self, idx):
        return self.cache[(self.ptr + idx) % self.size]

    def __setitem__(self, idx, obj):
        self.cache[(self.ptr + idx) % self.size] = obj
