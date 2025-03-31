import cv2
import time

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import euc_dist


class DetectionsDataloader:
    def __init__(self, directory, no_of_features=6, offset=3, resize=0.5, pt_size=3, thickness=2):
        self.dir = directory
        self.n = no_of_features
        self.offset = offset
        self.resize = resize
        self.ptSize = pt_size
        self.thickness = thickness
        self.filenames = sorted(os.listdir(self.dir), key=lambda x: get_frame_no(x))
        self.noOfFrames = get_frame_no(self.filenames[-1])
        maxFrameNo = max([get_frame_no(filename) for filename in self.filenames])
        if self.noOfFrames != maxFrameNo:
            print(f'{self.noOfFrames} {maxFrameNo}; max frame numbers do not match')
        self.detections = [[[None] * self.n for i in range(2)] for i in range(self.noOfFrames)]
        self.unsorted = None
        self.frameSkips = [0] * self.noOfFrames
        self.headCounts = [0] * self.noOfFrames  # head count for each frame
        self.pairFrameSkips = [[0] * 2 for i in range(self.noOfFrames)]

    def load(self):
        print('Loading detections...', end='')
        timeStart = time.time()

        counter = 0
        frameSkip = 0
        headDetected = False
        for filename in self.filenames:
            path = os.path.join(self.dir, filename)
            frameNo = get_frame_no(filename)
            while frameNo > (counter + 1) and self.noOfFrames > (counter + 1):
                frameSkip += 1
                counter += 1
            with open(path) as f:
                heads = int(f.readline())
                if heads == 0:
                    frameSkip += 1
                else:
                    self.headCounts[counter] = heads
                    if headDetected and frameSkip > 0:
                        self.frameSkips[counter] = frameSkip
                    frameSkip = 0
                    if not headDetected:
                        headDetected = True
                        self.firstHead = frameNo - 1
                    for headNo in range(heads):  # write feature detections
                        for featNo in range(self.n):
                            pos = [float(x) for x in f.readline().split()[1:3]]
                            if pos[0] >= 0 and pos[1] >= 0:
                                self.detections[counter][headNo][featNo] = pos
            counter += 1

        timeEnd = time.time()
        print(f'finished in %.3fms' % ((timeEnd - timeStart)*1000))

    def head_dist(self, head1, head2):
        pairs = [(head1[i], head2[i]) for i in range(self.n) if head1[i] is not None and head2[i] is not None]
        return sum([abs(euc_dist(*pairs[i])) for i in range(len(pairs))]) / len(pairs)

    def sort_detections(self):
        print('Sorting detections...', end='')
        timeStart = time.time()

        self.unsorted = self.detections.copy()
        if self.headCounts[0] == 1:
            self.detections[0][1] = None
        elif self.headCounts[0] == 0:
            self.detections[0] = [None] * 2
        for currNo in range(1, self.noOfFrames):
            prevNo = currNo - 1 - self.frameSkips[currNo]
            previous = self.detections[prevNo]
            current = self.detections[currNo]
            if self.headCounts[prevNo] > 0 and self.headCounts[currNo] > 0:
                if self.frameSkips[currNo] <= self.offset:
                    if self.headCounts[prevNo] == 1 and self.headCounts[currNo] == 1:
                        if previous[0] is None:
                            self.detections[currNo] = [None, current[0]]
                        else:
                            self.detections[currNo][1] = None
                    elif self.headCounts[prevNo] < self.headCounts[currNo]:
                        index = 1 if previous[0] is None else 0
                        if self.head_dist(previous[index], current[1 - index]) < self.head_dist(previous[index], current[index]):
                            self.detections[currNo] = [current[1], current[0]]
                    elif self.headCounts[currNo] < self.headCounts[prevNo]:
                        if self.head_dist(previous[1], current[0]) < self.head_dist(previous[0], current[0]):
                            self.detections[currNo] = [None, current[0]]
                    else:
                        original = sum([self.head_dist(previous[i], current[i]) for i in range(2)])
                        flipped = sum([self.head_dist(previous[i], current[1-i]) for i in range(2)])
                        if flipped < original:
                            self.detections[currNo] = [current[1], current[0]]
                else:
                    if self.headCounts[currNo] == 1:
                        self.detections[currNo][1] = None

        frameSkip = [0, 0]
        for frameNo in range(self.firstHead, self.noOfFrames):
            for headNo in range(2):
                if self.detections[frameNo][headNo] is None:
                    frameSkip[headNo] += 1
                else:
                    if frameSkip[headNo] > 0:
                        self.pairFrameSkips[frameNo][headNo] = frameSkip[headNo]
                        frameSkip[headNo] = 0

        timeEnd = time.time()
        print(f'finished in %.3fms' % ((timeEnd - timeStart)*1000))

    def interpolate(self):
        self.sort_detections()

        print('Interpolating detections...', end='')
        timeStart = time.time()
        # interpolate missing frames
        for headNo in range(2):
            for frameNo in range(self.firstHead, self.noOfFrames):
                skip = self.pairFrameSkips[frameNo][headNo]
                if 0 < skip <= self.offset:
                    curr = self.detections[frameNo][headNo]
                    startNo = frameNo - skip - 1
                    start = self.detections[startNo][headNo]
                    for i in range(self.n):
                        self.detections[startNo+i][headNo] = [None] * self.n
                    for featNo in range(self.n):
                        if curr[featNo] is not None and start[featNo] is not None:
                            diff = [curr[featNo][i] - start[featNo][i] for i in range(2)]
                            y = start[featNo]
                            for i in range(1, skip + 1):
                                self.detections[startNo+i][headNo][featNo] = [y[j]+diff[j]*(i/skip) for j in range(2)]
        # write feature skips
        featSkip = [[0] * self.n for _ in range(2)]
        featSkips = [[[0] * self.n for _ in range(2)] for i in range(self.noOfFrames)]
        featDetected = [[False] * self.n for _ in range(2)]
        for frameNo in range(self.firstHead, self.noOfFrames):
            curr = self.detections[frameNo]
            for i in range(2):
                if curr[i] is not None:
                    for j in range(self.n):
                        if curr[i][j] is not None:
                            if featDetected[i][j]:
                                featSkips[frameNo][i][j] = featSkip[i][j]
                            else:
                                featDetected[i][j] = True
                            featSkip[i][j] = 0
                        else:
                            featSkip[i][j] += 1
                else:
                    for j in range(self.n):
                        featSkip[i][j] += 1
        # interpolate missing features
        for headNo in range(2):
            for featNo in range(self.n):
                for frameNo in range(self.firstHead, self.noOfFrames):
                    skip = featSkips[frameNo][headNo][featNo]
                    if self.offset >= skip > 0:
                        curr = self.detections[frameNo][headNo][featNo]
                        startNo = frameNo - skip - 1
                        start = self.detections[startNo][headNo][featNo]
                        diff = [curr[i] - start[i] for i in range(2)]
                        for i in range(1, skip + 1):
                            self.detections[startNo+i][headNo][featNo] = [start[j]+diff[j]*(i/skip) for j in range(2)]

        timeEnd = time.time()
        print(f'finished in %.3fms' % ((timeEnd - timeStart)*1000))

    def compare(self, filepath):
        colours = [(255, 255, 0), (0, 255, 255), (0, 255, 255), (0, 150, 255), (0, 150, 255)]
        line_colours = [(0, 255, 0), (0, 0, 255)]
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameNo = 0

        print('Press \'q\' to stop.')

        while cap.isOpened() and frameNo < self.noOfFrames:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_CUBIC)
                shape = frame.shape[:2]
                iFrame = frame.copy()
                for headNo in range(2):
                    currHead = self.unsorted[frameNo][headNo]
                    if currHead is not None:
                        head_pt = None
                        if currHead[1] is not None:
                            head_pt = [round(currHead[1][i]*shape[1-i]) for i in range(2)]
                        for featNo in range(2, self.n):
                            if currHead[featNo] is not None:
                                pt = [round(currHead[featNo][i]*shape[1-i]) for i in range(2)]
                                if head_pt is not None:
                                    cv2.line(frame, head_pt, pt, line_colours[featNo%2], self.thickness)
                                cv2.circle(frame, pt, self.ptSize, colours[featNo-1], -1)
                        if head_pt is not None:
                            cv2.circle(frame, head_pt, self.ptSize, colours[0], -1)
                    currHead = self.detections[frameNo][headNo]
                    if currHead is not None:
                        head_pt = None
                        if currHead[1] is not None:
                            head_pt = [round(currHead[1][i] * shape[1-i]) for i in range(2)]
                        for featNo in range(2, self.n):
                            if currHead[featNo] is not None:
                                pt = [round(currHead[featNo][i]*shape[1-i]) for i in range(2)]
                                if head_pt is not None:
                                    cv2.line(iFrame, head_pt, pt, line_colours[featNo%2], self.thickness)
                                cv2.circle(iFrame, pt, self.ptSize, colours[featNo-1], -1)
                        if head_pt is not None:
                            cv2.circle(iFrame, head_pt, self.ptSize, colours[0], -1)

                output = cv2.vconcat([frame, iFrame])
                cv2.imshow('compare', output)
                frameNo += 1

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def get_frame_no(filename):
    return int(filename.split('_')[-1].split('.')[0])


if __name__ == '__main__':
    det_dir = os.path.join(ROOT, 'runs/detect/exp7/labels')
    vid_path = os.path.join(ROOT, 'vid/fps120/K203_K238_1_GH020045_cut.mp4')
    detections = DetectionsDataloader(det_dir, resize=0.3)
    detections.load()
    detections.interpolate()
    detections.compare(vid_path)
