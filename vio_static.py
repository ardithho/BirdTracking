import yaml
import os
import cv2
import numpy as np
from utils.general import RAD2DEG
from utils.odometry import estimate_vio, find_matches


src_dir = 'data/blender/renders'
cfg_path = os.path.join(src_dir, 'cam.yaml')
trans_path = os.path.join(src_dir, 'transforms.txt')


with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.asarray(cfg['K']).reshape(3, 3)
    dist = None

with open(trans_path, 'r') as f:
    lines = f.readlines()
    transforms = [np.array(list(map(float, line.strip().split()[1:]))).reshape((4, 4)) for line in lines]


im1 = cv2.imread(os.path.join(src_dir, '001.png'))
im2 = cv2.imread(os.path.join(src_dir, '002.png'))

vio, R, t, _ = estimate_vio(im1, im2, K, thresh=.2)
if vio:
    print('vo:', *np.rint(cv2.Rodrigues(R.T)[0]*RAD2DEG))
    print('gt:', *np.rint(cv2.Rodrigues(transforms[0][:3, :3])[0]*RAD2DEG))

im1 = im1[100:-100, 600:-600]
im2 = im2[100:-100, 600:-600]
matches, kp1, kp2 = find_matches(im1, im2, thresh=.2)
orb = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('data/out/matches.jpg', orb)
cv2.imshow('orb', orb)
cv2.waitKey(0)
cv2.destroyAllWindows()
