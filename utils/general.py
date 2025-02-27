import math
import random
import cv2
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans


DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


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


def orbKP(img, mask=None):
    orb = cv2.ORB_create(nfeatures=50)
    kp, des = orb.detectAndCompute(img, mask)
    img = cv2.drawKeypoints(img, kp, None, color=(255, 255, 0), flags=0)
    return img


def orbSingleKP(img, mask=None):
    img_copy = img.copy()
    orb = cv2.ORB_create(nfeatures=10)
    kp, des = orb.detectAndCompute(img, mask)
    pt = significantKP(kp)
    cv2.circle(img_copy, pt, 4, (255, 255, 0), -1)
    return img_copy


def orbKM(img, mask=None):
    colours = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0),
               (255, 150, 0), (255, 0, 0), (255, 0, 150), (255, 0, 255), (150, 0, 255)]
    clrs = [colours[i] for i in range(len(colours)) if i % 2 == 0]
    orb = cv2.ORB_create(100)
    kp, des = orb.detectAndCompute(img, mask)
    km = KMeans(n_clusters=6)
    labels = km.fit_predict(des)
    for i in range(len(kp)):
        img = cv2.drawKeypoints(img, [kp[i]], None, color=clrs[labels[i]], flags=0)
    return img


def fastKP(img, mask=None):  # corner detection
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, mask)
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 255))
    return img


def siftKP(img, mask=None):
    sift = cv2.SIFT_create(nfeatures=1000)
    kp = sift.detect(img, mask)
    img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return img


def siftKM(img, mask=None):
    colours = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0),
               (255, 150, 0), (255, 0, 0), (255, 0, 150), (255, 0, 255), (150, 0, 255)]
    clrs = [colours[i] for i in range(len(colours)) if i % 2 == 0]
    sift = cv2.SIFT_create(nfeatures=100)
    kp, des = sift.detectAndCompute(img, mask)
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


def man_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euc_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)


def ptsCentroid(arr):
    l = len(arr)
    sum_x = np.sum([pt[0] for pt in arr])
    sum_y = np.sum([pt[1] for pt in arr])
    return int(sum_x/l), int(sum_y/l)


def welzl(P, R):  # Welzl's algorithm
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
        return euc_dist(*cpts)
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
        if euc_dist(p, (cx, cy)) <= r:
            return True
    return False


def cnt_centroid(cnt):
    M = cv2.moments(cnt)
    return np.array((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))  # x, y


def angle(pivot, point):
    return np.arctan2(*(point - pivot))


# cosine similarity
def cosine(v1, v2, pivot=None):
    if pivot is not None:
        v1 -= pivot
        v2 -= pivot
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def slerp(q0, q1, t):
    omega = np.arccos(cosine(q0, q1))
    if omega != 0:
        return (q1*np.sin(t*omega) + q0*np.sin((1-t)*omega)) / np.sin(omega)
    return q0
