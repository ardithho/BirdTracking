import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from collections import Counter
from sklearn.cluster import KMeans
from general import kernel


def bgr2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[2]), int(colour[1]), int(colour[0]))


def rgb2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


def bgr2rgb(colour):
    return [colour[2], colour[1], colour[0]]


def normalise_hsv(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        norm[i][0] *= (1/180)
        for j in range(2):
            norm[i][j+1] *= (1/255)
    return norm


def normalise_rgb(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        for j in range(3):
            norm[i][j] *= (1/255)
    return norm


def denormalise_hsv(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        arr[i][0] *= 180
        for j in range(2):
            arr[i][j+1] *= 255
    return arr


def denormalise_rgb(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        for j in range(3):
            arr[i][j] *= 255
    return arr


def get_colours(im, n):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
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


def mask_ratio(mask):
    return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])


def colour_mask(im, kernel_size=5):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(kernel_size))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(5))
    return cv2.bitwise_and(im, im, mask=mask)


def hue_mask(im, hue):
    low = np.array([hue - 3, 200, 200])
    high = np.array([hue + 3, 255, 255])
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    return mask


def bill_mask(im, kernel_size=5):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    low1 = np.array([0, 60, 60])
    high1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, low1, high1)
    low2 = np.array([170, 60, 60])
    high2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(kernel_size))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(3))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.drawContours(mask, contours, 0, 255, -1)

    return cv2.bitwise_and(im, im, mask=mask)


def cheek_mask(im, kernel_size=5):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # orange cheeks
    low = np.array([0, 140, 100])
    high = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(kernel_size))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.drawContours(mask, contours, 0, 255, -1)
    return mask
