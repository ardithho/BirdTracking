import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from functional.colour import *
from itertools import combinations
from sklearn.cluster import KMeans, k_means
from collections import Counter
from skimage import feature, exposure


def imgDiff(img, sub):
    diff = cv2.absdiff(cv2.blur(img, (5, 5)), cv2.blur(sub, (5, 5)))
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel(5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel(15))
    return thresh


def combine(masks):
    andMasks = []
    combs = combinations(range(len(masks)), 2)
    for comb in combs:
        andMasks.append(cv2.bitwise_and(masks[comb[0]], masks[comb[1]]))
    if len(andMasks) > 1:
        combined = cv2.bitwise_xor(andMasks[0], andMasks[1])
        if len(andMasks) > 2:
            for i in range(2, len(andMasks)):
                combined = cv2.bitwise_xor(combined, andMasks[i])
    else:
        combined = andMasks[0]
    return combined


def kernel(length):
    return np.ones((length, length), np.uint8)


def orbKP(img):
    orb = cv2.ORB_create(nfeatures=50)
    kp, des = orb.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, kp, None, color=(255, 255, 0), flags=0)
    return img


def orbSingleKP(img):
    img_copy = img.copy()
    orb = cv2.ORB_create(nfeatures=10)
    kp, des = orb.detectAndCompute(img, None)
    pt = significantKP(kp)
    cv2.circle(img_copy, pt, 4, (255, 255, 0), -1)
    return img_copy


def orbKM(img):
    colours = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0),
               (255, 150, 0), (255, 0, 0), (255, 0, 150), (255, 0, 255), (150, 0, 255)]
    clrs = [colours[i] for i in range(len(colours)) if i % 2 == 0]
    orb = cv2.ORB_create(100)
    kp, des = orb.detectAndCompute(img, None)
    km = KMeans(n_clusters=6)
    labels = km.fit_predict(des)
    for i in range(len(kp)):
        img = cv2.drawKeypoints(img, [kp[i]], None, color=clrs[labels[i]], flags=0)
    return img


def fastKP(img):  # corner detection
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 255))
    return img


def siftKP(img):
    sift = cv2.SIFT_create(nfeatures=1000)
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return img


def siftKM(img):
    colours = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0),
               (255, 150, 0), (255, 0, 0), (255, 0, 150), (255, 0, 255), (150, 0, 255)]
    clrs = [colours[i] for i in range(len(colours)) if i % 2 == 0]
    sift = cv2.SIFT_create(nfeatures=100)
    kp, des = sift.detectAndCompute(img, None)
    km = KMeans(n_clusters=6)
    labels = km.fit_predict(des)
    for i in range(len(kp)):
        img = cv2.drawKeypoints(img, [kp[i]], None, color=clrs[labels[i]], flags=0)
    return img


def surfKP(img):
    surf = cv2.SURF()
    kp, des = surf.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 150))
    return img


def orbSiftKM(img):
    colours = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0),
               (255, 150, 0), (255, 0, 0), (255, 0, 150), (255, 0, 255), (150, 0, 255)]
    clrs = [colours[i] for i in range(len(colours)) if i % 2 == 0]
    co = colours[:6]
    cs = colours[6:]
    n = 100
    orb = cv2.ORB_create(n)
    okp, odes = orb.detectAndCompute(img, None)
    sift = cv2.SIFT_create(n)
    skp, sdes = sift.detectAndCompute(img, None)
    oexkps = []
    oexdes = []
    for i in range(len(okp)):
        overlap = False
        j = 0
        while not overlap and j < len(skp):
            if cv2.KeyPoint().overlap(okp[i], skp[j]):
                oexkps.append(okp[i])
                oexdes.append(odes[i])
                overlap = True
            j += 1
    sexkps = []
    sexdes = []
    for i in range(len(skp)):
        overlap = False
        j = 0
        while not overlap and j < len(oexkps):
            if cv2.KeyPoint().overlap(skp[i], oexkps[j]):
                sexkps.append(skp[i])
                sexdes.append(sdes[i])
                overlap = True
            j += 1

    km = KMeans(n_clusters=6)
    olabels = km.fit_predict(oexdes)
    for i in range(len(oexkps)):
        img = cv2.drawKeypoints(img, [oexkps[i]], None, color=co[olabels[i]], flags=None)
    slabels = km.fit_predict(sexdes)
    for i in range(len(sexkps)):
        img = cv2.drawKeypoints(img, [sexkps[i]], None, color=cs[slabels[i]], flags=None)
    return img


def orbMatch(imga, imgb):
    orb = cv2.ORB_create()
    imga = cv2.resize(imga, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    imgb = cv2.resize(imgb, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    kpa, desa = orb.detectAndCompute(imga, None)
    kpb, desb = orb.detectAndCompute(imgb, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if desa is not None and desb is not None:
        matches = list(bf.match(desa, desb))
        matches.sort(key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 20)]
        img = cv2.drawMatches(imga, kpa, imgb, kpb, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    else:
        img = cv2.hconcat([imga, imgb])
    return img


def orbSamePos(imga, imgb):
    orb = cv2.ORB_create()
    kpa, desa = orb.detectAndCompute(imga, None)
    kpb, desb = orb.detectAndCompute(imgb, None)
    ptsa = [kp.pt for kp in kpa]
    kps = [kp for kp in kpb if kp.pt in ptsa]
    img = cv2.drawKeypoints(imga, kps, None, color=(0, 255, 255), flags=0)
    return img


def orbOverlapping(imga, imgb):
    orb = cv2.ORB_create()
    kpa, desa = orb.detectAndCompute(imga, None)
    kpb, desb = orb.detectAndCompute(imgb, None)
    kps = []
    for ptb in kpb:
        i = 0
        overlap = False
        while i < len(kpa) and not overlap:
            if cv2.KeyPoint().overlap(kpa[i], ptb) > 0.75:
                kps.append(ptb)
                overlap = True
            i += 1
    img = cv2.drawKeypoints(imga, kps, None, color=(255, 255, 0), flags=0)
    return img


def distinctKP(kps):  # group clusters of kps
    grps = []
    for kp in kps:
        if len(grps) > 0:
            added = False
            i = 0
            while not added and i < len(grps):
                j = 0
                while not added and j < len(grps[i]):
                    if cv2.KeyPoint().overlap(grps[i][j], kp) > 0.5:
                        grps[i].append(kp)
                        added = True
                    j += 1
                i += 1
            if not added:
                grps.append([kp])
        else:
            grps = [[kp]]
    return grps


def significantKP(kps):
    grps = distinctKP(kps)
    grps = [[kp.pt for kp in grp] for grp in grps]
    largest = []
    maxLen = 0
    for grp in grps:
        if len(grp) > maxLen:
            largest = grp
            maxLen = len(grp)
        elif len(grp) == maxLen and maxLen >= 2:
            if circleRadius(grp) < circleRadius(largest):
                largest = grp
    return ptsCentroid(largest)


def manDist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def eucDist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)


def ptsCentroid(arr):
    l = len(arr)
    sum_x = np.sum([pt[0] for pt in arr])
    sum_y = np.sum([pt[1] for pt in arr])
    return int(sum_x/l), int(sum_y/l)


def welzl(P, R):  # Welzl's algorithm (emo guy)
    if len(P) == 0 or len(R) == 3:
        return list(R)
    p = set(random.sample(P, 1))
    D = welzl(P.difference(p), R)
    if ptInCircle(list(p)[0], D):  # prolly doesn't work this way
        return D
    return welzl(P.difference(p), R.union(p))


def circleRadius(pts):
    cpts = welzl(set(pts), set())
    if len(cpts) == 3:
        return circleProperties(cpts)[-1]
    elif len(cpts) == 2:
        return eucDist(*cpts)
    return 0  # technically shouldn't happen


def circleProperties(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    if y31 * x12 - y21 * x13 == 0 or x31 * y12 - x21 * y13 == 0:
        h = (x1 + x2 + x3) / 3
        k = (y1 + y2 + y3) /3
        r = 0
    else:
        f = ((sx13 * x12 + sy13 * x12 +
              sx21 * x13 + sy21 * x13) //
             (2 * (y31 * x12 - y21 * x13)))  # division by zero problem

        g = ((sx13 * y12 + sy13 * y12 +
              sx21 * y13 + sy21 * y13) //
             (2 * (x31 * y12 - x21 * y13)))

        c = (-pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1)

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        h = -g
        k = -f
        r = math.sqrt(h * h + k * k - c)
    return h, k, r


def ptInCircle(p, D):
    if len(D) == 3:
        cx, cy, r = circleProperties(D)
        if eucDist(p, (cx, cy)) <= r:
            return True
    return False


def edgeCanny(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(grey, 100, 200)
    return edges


def hogFeatures(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, hogImg = feature.hog(grey, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    hogImg = exposure.rescale_intensity(hogImg, out_range=(0, 255))
    hogImg = hogImg.astype('uint8')
    return hogImg


def detectCorners(img):
    img_copy = img.copy()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, 5)
    dst = cv2.cornerHarris(grey, 5, 5, 0.2)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    v = np.percentile(dst_norm_scaled, 99.5)
    # img[dst_norm_scaled > v] = (255, 0, 255)

    points = np.argwhere(dst_norm_scaled > v)

    if len(points) > 0:
        n = int(math.sqrt(len(np.unique(points))))
        km = k_means(points, n_clusters=n)
        corners = km[0].astype('uint64')

        for cnr in corners:
            cv2.circle(img_copy, cnr[::-1], 4, (255, 200, 0), -1)
    return img_copy


def getContours(img):
    img_copy = img.copy()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours, -1, (0, 255, 255), 3)
    return img_copy


def getColours(img, n):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mod = hsv.reshape(hsv.shape[0] * hsv.shape[1], 3)
    clf = KMeans(n_clusters=n)
    labels = clf.fit_predict(mod)
    counts = Counter(labels)
    del counts[list(counts.keys())[0]]  # delete largest count
    while len(counts) > 10:
        del counts[list(counts.keys())[0]]
    centre_colours = clf.cluster_centers_
    hsv_colours = [centre_colours[i] for i in counts.keys()]
    return hsv_colours, counts


def pie(hsv_colours, counts):
    norm_hsv = normalise_hsv(hsv_colours)
    rgb_colours = denormalise_rgb([hsv_to_rgb(norm_hsv[i]) for i in range(len(norm_hsv))])
    hex_colours = [rgb2hex(colour) for colour in rgb_colours]
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colours, colors=hex_colours)
    plt.show()


def colourMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # orange cheeks (+feet
    lowo = np.array([0, 140, 100])
    higho = np.array([40, 255, 255])
    masko = cv2.inRange(hsv, lowo, higho)
    # red bill
    lowr = np.array([0, 150, 100])
    highr = np.array([5, 255, 255])
    maskr = cv2.inRange(hsv, lowr, highr)
    # black tear marks
    lowb = np.array([0, 0, 0])
    highb = np.array([180, 255, 50])
    maskb = cv2.inRange(hsv, lowb, highb)
    mask = cv2.bitwise_or(masko, maskr)
    mask = maskr
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(5))
    colMask = cv2.bitwise_and(img, img, mask=mask)
    return colMask


def hueMask(img, hue):
    low = np.array([hue - 3, 200, 200])
    high = np.array([hue + 3, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    return mask


def billMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low1 = np.array([0, 80, 80])
    high1 = np.array([8, 255, 255])
    mask1 = cv2.inRange(hsv, low1, high1)
    low2 = np.array([172, 80, 80])
    high2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(3))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, contours, 0, 255, -1)

    colMask = cv2.bitwise_and(img, img, mask=mask)
    return colMask


def trackHead(img):
    img_copy = img.copy()
    mask = billMask(img)
    orb = cv2.ORB_create(nfeatures=10)
    kp, des = orb.detectAndCompute(mask, None)
    if len(kp) > 0:
        pt = significantKP(kp)

        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        M = cv2.moments(cnt)
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # bounding rect
        ratio = 3
        dist = eucDist(pt, centroid) * 2
        length = dist * ratio
        x, y, w, h = cv2.boundingRect(cnt)
        w += length
        h += length
        dx = centroid[0] - pt[0]
        dy = centroid[1] - pt[1]
        if dx >= 0 and dy != 0 and abs(dx/dy) < 1:
            x -= w * (1 - abs(dx/dy)) / 2
        elif dx < 0:
            if dy != 0 and abs(dx/dy) < 1:
                x += w * (1 - abs(dx/dy)) / 2 - w
            else:
                x -= length
        if dy >= 0 and dx != 0 and abs(dy/dx) < 1:
            y -= h * (1 - abs(dy/dx)) / 2
        elif dy < 0:
            if dx != 0 and abs(dy/dx) < 1:
                y += h * (1 - abs(dy/dx)) / 2 - h
            else:
                y -= length
        x = round(x)
        y = round(y)
        w = round(w)
        h = round(h)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(img_copy, pt, centroid, (0, 255, 255), 2)
        cv2.circle(img_copy, pt, 4, (255, 0, 255), -1)
    return img_copy


def grpFeat(img, bill_label1, bill_label5, eyes_labels1, eyes_labels5):
    if bill_label1 is not None and bill_label5 is not None:
        pos = [(bill_label1[i]+bill_label5[i])/2 for i in range(1, 3)]
        bill = [round(pos[i] * img.shape[1 - i]) for i in range(2)]
        bill_conf = (bill_label1[-1] + bill_label5[-1]) / 2
    elif bill_label1 is not None:
        bill = [round(bill_label1[1 + i] * img.shape[1 - i]) for i in range(2)]
        bill_conf = bill_label1[-1]
    elif bill_label5 is not None:
        bill = [round(bill_label5[1 + i] * img.shape[1 - i]) for i in range(2)]
        bill_conf = bill_label5[-1]
    else:
        bill = None
        bill_conf = 0
    eyes = []
    for label in eyes_labels1:
        i = 0
        match = False
        pos1 = label[1:3]
        while not match and i < len(eyes_labels5):
            pos5 = eyes_labels5[i][1:3]
            if eucDist(pos1, pos5) < 3 / eucDist((0, 0), img.shape[:2][::-1]):
                pos1 = [(pos1[i]+pos5[i])/2 for i in range(2)]
                eyes_labels5.pop(i)
                match = True
            i += 1
        if match or label[-1] > 0.4:
            pt = [round(pos1[i] * img.shape[1-i]) for i in range(2)]
            eyes.append(pt)
    for label in eyes_labels5:
        if label[-1] > 0.4:
            pt = [round(label[1+i] * img.shape[1-i]) for i in range(2)]
            eyes.append(pt)
    return bill, eyes, bill_conf


def boundFeat(img, bill, bill_conf, eyes):
    if bill:
        mask = billMask(img)
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1 and cv2.contourArea(contours[0]) > 0:
            cnt = contours[0]
            M = cv2.moments(cnt)
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            # bounding rect
            ratio = 3
            dist = eucDist(bill, centroid) * 2
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
            else:
                bill = None
        elif bill_conf < 0.75:
            bill = None
    return bill, eyes


def filterFeat(img, det):
    bill_labels = sorted([label for label in det if label[0] == 0 and label[-1] >= 0.1], key=lambda x: x[-1],
                          reverse=True)
    if bill_labels:
        bill_label = bill_labels[0]
        bill = [round(bill_label[1 + i] * img.shape[1 - i]) for i in range(2)]
        bill_conf = bill_label[-1]
    else:
        bill = None
        bill_conf = 0
    eyes_labels = sorted([label for label in det if label[0] != 0 and label[-1] >= 0.1], key=lambda x: x[-1])
    eyes = [[round(label[1+i] * img.shape[1-i]) for i in range(2)] for label in eyes_labels]
    return boundFeat(img, bill, bill_conf, eyes)


def filterFeat2(img, det1, det5):
    bill_labels1 = sorted([label for label in det1 if label[0] == 0 and label[-1] >= 0.1], key=lambda x: x[-1],
                          reverse=True)
    eyes_labels1 = sorted([label for label in det1 if label[0] != 0 and label[-1] >= 0.1], key=lambda x: x[-1])
    bill_labels5 = sorted([label for label in det5 if label[0] == 0 and label[-1] >= 0.1], key=lambda x: x[-1],
                          reverse=True)
    eyes_labels5 = sorted([label for label in det5 if label[0] != 0 and label[-1] >= 0.1], key=lambda x: x[-1])
    bill_label1 = bill_labels1[0] if bill_labels1 else None
    bill_label5 = bill_labels5[0] if bill_labels5 else None
    bill, eyes, bill_conf = grpFeat(img, bill_label1, bill_label5, eyes_labels1, eyes_labels5)
    return boundFeat(img, bill, bill_conf, eyes)


def plotFeat(img, bill, eyes, start=(0, 0)):
    if bill:
        bill = [round(start[i]+bill[i]) for i in range(2)]
        for eye in eyes:
            eye = [round(start[i]+eye[i]) for i in range(2)]
            cv2.line(img, eye, bill, (255, 0, 255), 2)
            cv2.circle(img, eye, 4, (0, 255, 255), -1)
        cv2.circle(img, bill, 4, (255, 255, 0), -1)
    return img


def visualise(img, bill_label1, bill_label5, eyes_labels1, eyes_labels5):
    bill, eyes = grpFeat(img, bill_label1, bill_label5, eyes_labels1, eyes_labels5)
    if bill:
        mask = billMask(img)
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1:
            cnt = contours[0]
            M = cv2.moments(cnt)
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            # bounding rect
            ratio = 3
            dist = eucDist(bill, centroid) * 2
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
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.line(img, bill, centroid, (0, 255, 255), 2)

            for eye in eyes:
                if x <= eye[0] <= x + w and y <= eye[1] <= y + h:
                    cv2.line(img, eye, bill, (255, 0, 255), 2)
                cv2.circle(img, eye, 4, (0, 255, 255), -1)
            if x <= bill[0] <= x + w and y <= bill[1] <= y + h:
                cv2.circle(img, bill, 4, (255, 255, 0), -1)
            else:
                cv2.circle(img, bill, 4, (255, 0, 0), -1)
        else:
            for eye in eyes:
                cv2.line(img, eye, bill, (255, 0, 255), 2)
                cv2.circle(img, eye, 4, (0, 255, 255), -1)
            cv2.circle(img, bill, 4, (255, 0, 0), -1)
    else:
        for eye in eyes:
            cv2.circle(img, eye, 4, (0, 255, 255), -1)
    return img

