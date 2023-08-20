import os
from general import eucDist


def load_txt(dir, n):
    filenames = os.listdir(dir)
    counter = 0
    frameSkip = 0
    noOfFrames = int(filenames[-1].split('_')[-1])
    detections = [[[None] * n] * 2] * noOfFrames
    frameSkips = [None] * noOfFrames
    headCounts = [0] * noOfFrames  # head count for each frame
    headDetected = False
    for filename in filenames:
        path = os.path.join(dir, filename)
        frameNo = int(filename.split('_')[-1])
        if frameNo == (counter + 1):
            with open(path) as f:
                heads = int(f.readline())
                if heads == 0:
                    frameSkip += 1
                else:
                    headCounts[counter] = heads
                    if headDetected and frameSkip > 0:
                        frameSkips[counter] = frameSkip
                    frameSkip = 0
                    if not headDetected:
                        headDetected = True
                        firstHead = frameNo - 1
                    for headNo in range(heads):  # write feature detections
                        for featNo in range(n):
                            pos = [float(x) for x in f.readline().split()[1:3]]
                            if pos[0] >= 0 and pos[1] >= 0:
                                detections[counter][headNo][featNo] = pos
        else:  # should not happen
            frameSkip += 1
        counter += 1
    return detections, headCounts, frameSkips, firstHead


def head_dist(head1, head2, n):
    pairs = [(head1[i], head2[i]) for i in range(n) if head1[i] >= 0 and head2[i] >= 0]
    return sum([abs(eucDist(*pairs[i])) for i in range(len(pairs))]) / len(pairs)


def sort_detections(detections, headCounts, frameSkips, n, offset, firstHead):
    if headCounts[0] == 1:
        detections[0][1] = None
    elif headCounts[0] == 0:
        detections[0] = [None] * 2
    for currNo in range(1, len(detections)):
        prevNo = currNo - 1 - frameSkips[currNo]
        previous = detections[prevNo]
        current = detections[currNo]
        if frameSkips[currNo] <= offset:
            if headCounts[prevNo] == 1 and headCounts[currNo] == 1:
                if previous[0] is None:
                    detections[currNo] = [None, current[0]]
                else:
                    detections[currNo][1] = None
            elif headCounts[prevNo] < headCounts[currNo]:
                index = 1 if previous[0] is None else 0
                if head_dist(previous[index], current[1-index], n) < head_dist(previous[index], current[index], n):
                    detections[currNo] = [current[1], current[0]]
            elif headCounts[currNo] < headCounts[prevNo]:
                if head_dist(previous[1], current[0]) < head_dist(previous[0], current[0]):
                    detections[currNo] = [None, current[0]]
            else:
                original = sum([head_dist(previous[i], current[i], n)] for i in range(2))
                flipped = sum([head_dist(previous[i], current[1-i], n)] for i in range(2))
                if flipped < original:
                    detections[currNo] = [current[1], current[0]]
        else:
            if headCounts[currNo] == 1:
                detections[currNo][1] = None

    frameSkip = [0, 0]
    for frameNo in range(firstHead, len(detections)):
        for headNo in range(2):
            if detections[frameNo][headNo] is None:
                frameSkip[headNo] += 1
            else:
                if frameSkip[headNo] > 0:
                    frameSkips[frameNo][headNo] = frameSkip[headNo]
                    frameSkip[headNo] = 0
    return detections, frameSkips


def interpolate(detections, frameSkips, n, offset, firstHead):
    noOfFrames = len(detections)
    featSkip = [[0] * n] * 2
    featSkips = [[[0] * n] * 2] * noOfFrames
    featDetected = [[False] * n] * 2
    # interpolate missing frames
    for headNo in range(2):
        for frameNo in range(firstHead, noOfFrames):
            skip = frameSkips[frameNo][headNo]
            if offset >= skip > 0:
                curr = detections[frameNo][headNo]
                startNo = frameNo - skip - 1
                start = detections[startNo][headNo]
                for featNo in range(n):
                    if curr[featNo] is not None and start[featNo] is not None:
                        diff = [curr[featNo][i]-start[featNo][i] for i in range(2)]
                        y = start[featNo]
                        for i in range(1, skip+1):
                            detections[startNo+i][headNo][featNo] = [round(y[j]+diff[j]*(i/skip)) for j in range(2)]
    # write feature skips
    for frameNo in range(firstHead, noOfFrames):
        curr = detections[frameNo]
        for i in range(2):
            if curr[i] is not None:
                for j in range(n):
                    if curr[i][j] is not None:
                        if featDetected[i][j]:
                            featSkips[frameNo][i][j] = featSkip[i][j]
                        else:
                            featDetected[i][j] = True
                        featSkip[i][j] = 0
                    else:
                        featSkip[i][j] += 1
    # interpolate missing features
    for headNo in range(2):
        for featNo in range(n):
            for frameNo in range(firstHead, noOfFrames):
                skip = featSkips[frameNo][headNo][featNo]
                if offset >= skip > 0:
                    curr = detections[frameNo][headNo][featNo]
                    startNo = frameNo - skip - 1
                    start = detections[startNo][headNo][featNo]
                    diff = [curr[i]-start[i] for i in range(2)]
                    for i in range(1, skip+1):
                        detections[startNo+i][headNo][featNo] = [round(start[j]+diff[j]*(i/skip)) for j in range(2)]
    return detections


def load_detections(dir):
    n = 6  # number of features including head
    offset = 5
    detections, headCounts, frameSkips, firstHead = load_txt(dir, n)
    detections, frameSkips = sort_detections(detections, headCounts, frameSkips, n, offset)
    detections = interpolate(detections, frameSkips, n, offset, firstHead)
    return detections
