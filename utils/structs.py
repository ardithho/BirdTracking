import numpy as np
from utils.sorter import sort_feat, to_dict
from utils.colour import cheek_mask, mask_ratio


CLS_DICT = {'bill': 0,
            'left_eye': 1,
            'left_tear': 2,
            'right_eye': 3,
            'right_tear': 4}

FEAT_DICT = {'bill': CLS_DICT['bill'],
             'eyes': [CLS_DICT['left_eye'], CLS_DICT['right_eye']],
             'tear_marks': [CLS_DICT['left_tear'], CLS_DICT['right_tear']]}


class Feature:
    def __init__(self, feat, xy, xyn):
        self.cls = feat.cls
        self.conf = feat.conf
        self.xy = xy
        self.xyn = xyn


class Bird:
    def __init__(self, head, feats):
        self.conf = head.conf[0]
        self.id = int(head.id[0])
        self.xywh = head.xywh[0]
        self.xywhn = head.xywhn[0]
        self.xyxy = head.xyxy[0]
        self.xyxyn = head.xyxyn[0]
        self.featsUnsorted = self.globalise(feats)
        self.feats = self.sort()

    def globalise(self, feats):
        '''
        Re-localise feature points to image coordinates from box coordinates
        Args:
            feats: YOLO Result
        Returns:
            globalised features
        '''
        globalised = []
        for feat in feats:
            xy = self.xyxy[:2] + self.xywh[2:] * feat.xywh[0, :2]
            xyn = self.xyxyn[:2] + self.xywhn[2:] * feat.xywhn[0, :2]
            globalised.append(Feature(feat, xy, xyn))
        return globalised

    def sort(self):
        bill = [feat.xy for feat in self.featsUnsorted if feat.cls == FEAT_DICT['bill']][0]
        eyes = [feat.xy for feat in self.featsUnsorted if feat.cls in FEAT_DICT['eyes']]
        tear_marks = [feat.xy for feat in self.featsUnsorted if feat.cls in FEAT_DICT['tear_marks']]
        return to_dict(*sort_feat(bill, eyes, tear_marks))


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
            unseen.remove(self.ids[bird.id])
        if len(unseen) > 0:
            for sex in unseen:
                self.current[sex] = None
                self.caches[sex].update(None)

    def sort(self, birds, frame, thres=0.1):
        if len(birds) > 0:
            self.ids = {}
            if len(birds) == 2:
                x0, y0, x1, y1 = np.rint(birds[0].xyxy).astype(np.uint32)
                ratio0 = mask_ratio(cheek_mask(frame[y0:y1, x0:x1]))
                x0, y0, x1, y1 = np.rint(birds[1].xyxy).astype(np.uint32)
                ratio1 = mask_ratio(cheek_mask(frame[y0:y1, x0:x1]))
                if ratio0 > ratio1:
                    self.ids[birds[0].id] = 'm'
                    self.ids[birds[1].id] = 'f'
                else:
                    self.ids[birds[1].id] = 'm'
                    self.ids[birds[0].id] = 'f'
            else:
                x0, y0, x1, y1 = np.rint(birds[0].xyxy).astype(np.uint32)
                ratio = mask_ratio(cheek_mask(frame[y0:y1, x0:x1]))
                if ratio > thres:
                    self.ids[birds[0].id] = 'm'
                    self.ids[birds[0].id+1] = 'f'
                else:
                    self.ids[birds[0].id+1] = 'm'
                    self.ids[birds[0].id] = 'f'

    def __getitem__(self, sex):
        return self.current[sex]


class Cache:
    def __init__(self, size=5):
        self.size = size
        self.cache = [None] * self.size
        self.ptr = 0

    def update(self, obj):
        # Add detection from new frame to (circular) queue
        self.cache[self.ptr] = obj
        self.ptr = (self.ptr + 1) % self.size

    def interpolate(self):
        x = self[0]

    def __getitem__(self, idx):
        return self.cache[(self.ptr + idx) % self.size]

    def __setitem__(self, idx, obj):
        self.cache[(self.ptr + idx) % self.size] = obj
