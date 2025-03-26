import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.box import iou
from utils.colour import cheek_mask, head_mask, mask_ratio
from utils.configs import FEAT_DICT
from utils.plot import plot_box, plot_feat
from utils.sorter import sort_feat, to_dict, process_labels


class Feature:
    def __init__(self, feat, xy, xyn):
        self.cls = feat.cls
        self.conf = feat.conf
        self.xy = xy
        self.xyn = xyn


class Bird:
    def __init__(self, head, feats):
        self.conf = head.conf[0]
        self.id = int(head.id[0]) if head.id is not None else -1
        self.xywh = head.xywh[0]
        self.xywhn = head.xywhn[0]
        self.xyxy = head.xyxy[0]
        self.xyxyn = head.xyxyn[0]
        self.area = self.xywh[2] * self.xywh[3]
        self.arean = self.xywhn[2] * self.xywhn[3]
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
            xy = self.xyxy[:2] + feat.xywh[0, :2]
            xyn = self.xyxyn[:2] + self.xywhn[2:] * feat.xywhn[0, :2]
            globalised.append(Feature(feat, xy, xyn))
        return globalised

    def sort(self):
        bill = [feat.xy for feat in self.featsUnsorted if feat.cls == FEAT_DICT['bill']]
        bill = bill[0] if len(bill) > 0 else None
        eyes = process_labels([feat.xy for feat in self.featsUnsorted if feat.cls in FEAT_DICT['eyes']], 2)
        tear_marks = process_labels([feat.xy for feat in self.featsUnsorted if feat.cls in FEAT_DICT['tear_marks']], 2)
        bill_liners = process_labels([feat.xy for feat in self.featsUnsorted if feat.cls in FEAT_DICT['bill_liners']], 2)
        return to_dict(*sort_feat(bill, eyes, tear_marks, bill_liners))

    def mask(self, im):
        im_shape = im.shape[:2]
        mask = np.zeros(im_shape, dtype=np.uint8)
        x1, y1, x2, y2 = np.rint(self.xyxyn.reshape((2, 2)) * im_shape[::-1]).astype(np.uint32).flatten()
        mask[y1:y2, x1:x2] = 255
        mask &= head_mask(im)
        return mask


class Birds:
    def __init__(self, tracked=False, iou=0.7, cache=5):
        self.current = {}
        self.caches = {'m': Cache(cache), 'f': Cache(cache)}
        self.frames = Cache(cache)
        self.ids = None
        self.tracked = tracked
        self.iou = iou

    def update(self, birds, frame):
        birds = birds[:2]
        if not self.tracked:
            birds = self.track(birds)
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
        self.frames.update(frame)

    def track(self, birds):
        if self.ids is None:
            for i in range(len(birds)):
                birds[i].id = i
            return birds
        if len(birds) >= 1:
            ious = [[0, 0] for _ in range(len(birds))]
            max_ids = [0 for _ in range(len(birds))]
            for i in range(len(birds)):
                for id in self.ids.keys():
                    bird = self.current[self.ids[id]]
                    if bird is not None:
                        ious[i][id] = iou(birds[i], bird)
                        if ious[i][id] > max(ious[i]):
                            max_ids[i] = id
            if len(birds) == 1:
                birds[0].id = max_ids[0]
                return birds
            if max_ids[0] != max_ids[1]:
                for i in range(len(birds)):
                    birds[i].id = max_ids[i]
                return birds
            if ious[0][max_ids[0]] > ious[1][max_ids[0]]:
                birds[0].id = max_ids[0]
                birds[1].id = 1 - max_ids[0]
            else:
                birds[0].id = 1 - max_ids[0]
                birds[1].id = max_ids[0]
            return birds
        return birds

    def sort(self, birds, frame, thres=0.01):
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
    #
    # def interpolate(self):
    #     for id in self.ids.values():
    #         if self.current[id] is not None:
    #             cache = self.caches[id]
    #             curr_bird = self.current[id]
    #             for i in range(cache.size-2):
    #                 for feat in curr_bird.feats:
    #                     if feat.cls in cache[i].feats_cls:
    #                         for j in range(i, cache.size-1):
    #                             if feat.cls not in cache[j].feats_cls:
    #                                 new_feat = Feature()

    def plot(self):
        frame = self.frames[-1]
        if self['m'] is not None:
            frame = plot_box(frame, self['m'].xyxy, (255, 0, 0))
            frame = plot_feat(frame, self['m'].feats['bill'],
                              [self['m'].feats['left_eye'], self['m'].feats['right_eye']],
                              [self['m'].feats['left_tear'], self['m'].feats['right_tear']],
                              [self['m'].feats['left_liner'], self['m'].feats['right_liner']])
        if self['f'] is not None:
            frame = plot_box(frame, self['f'].xyxy, (255, 0, 255))
            frame = plot_feat(frame, self['f'].feats['bill'],
                              [self['f'].feats['left_eye'], self['f'].feats['right_eye']],
                              [self['f'].feats['left_tear'], self['f'].feats['right_tear']],
                              [self['f'].feats['left_liner'], self['f'].feats['right_liner']])
        return frame

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

    def __getitem__(self, idx):
        # 0 is the old item
        return self.cache[(self.ptr + idx) % self.size]

    def __setitem__(self, idx, obj):
        self.cache[(self.ptr + idx) % self.size] = obj
