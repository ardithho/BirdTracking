import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.odometry import estimate_vio_pts
from utils.general import RAD2DEG, DEG2RAD


save_dir = 'data/out/vio'
cfg_path = 'data/blender/marked/cam.yaml'
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    K = np.array(cfg['KF']).reshape(3, 3)
    ext = np.array(cfg['extF']).reshape(3, 4)
    rvec = cv2.Rodrigues(ext[:3, :3])[0]
    tvec = ext[:3, 3]
    # rvec = np.zeros((3, 1))
    # tvec = np.zeros((3, 1))
dist = np.array([-0.06665328200820067, 0.6597426404260505,
                 0.030050421934352787, -0.021281447560983234, -4.553479330755196])

obj_pts1 = np.array([[-0.7, -0.7,  1. ],
                     [-0.1, -0.1,  0.9],
                     [ 0.6,  0.4,  0.1],
                     [-0.5, -1. ,  0.1],
                     [-0.9, -0.9,  0.9],
                     [ 0.7,  1. , -0.3],
                     [-0.5, -0.7, -0.4],
                     [-0.9, -0.7,  0.9],
                     [ 0.9, -0.1,  0.9],
                     [ 0.4, -0.8,  0.3]])

# x = np.linspace(-.1, .1, 3)
# y = np.linspace(-.1, .1, 3)
# z = np.linspace(-.1, .1, 3)
# x, y, z = np.meshgrid(x, y, z)
# obj_pts1 = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
rot = np.array([3., 0., 0.])
obj_pts2 = (cv2.Rodrigues(rot.T * DEG2RAD)[0] @ obj_pts1.T).T

img_pts1, _ = cv2.projectPoints(obj_pts1, rvec, tvec, K, dist)
img_pts2, _ = cv2.projectPoints(obj_pts2, rvec, tvec, K, dist)

ret, R, t, _ = estimate_vio_pts(img_pts1, img_pts2, K, dist)
print(*cv2.Rodrigues(R.T)[0] * RAD2DEG)
vio_pts = (R.T @ obj_pts1.T).T
print(np.linalg.norm(vio_pts - obj_pts2))

fig = plt.figure(figsize=(15, 10))

# Plot Point Cloud 1
ax1 = fig.add_subplot(231, projection='3d')  # 1st subplot
ax1.scatter(obj_pts1[:, 0], obj_pts1[:, 1], obj_pts1[:, 2], c='r', marker='o')
ax1.set_title('obj pts 1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot Point Cloud 2
ax2 = fig.add_subplot(232, projection='3d')  # 2nd subplot
ax2.scatter(obj_pts2[:, 0], obj_pts2[:, 1], obj_pts2[:, 2], c='g', marker='^')
ax2.set_title('obj pts 2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Plot Point Cloud 3
ax3 = fig.add_subplot(233, projection='3d')  # 3rd subplot
ax3.scatter(vio_pts[:, 0], vio_pts[:, 1], vio_pts[:, 2], c='b', marker='s')
ax3.set_title('vio pts')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# 2D Plot 1
ax4 = fig.add_subplot(234)  # 4th plot (2D)
ax4.scatter(img_pts1[:, 0, 0], img_pts1[:, 0, 1], color='orange')
ax4.set_title("img pts 1")
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.legend()

# 2D Plot 2
ax5 = fig.add_subplot(235)  # 5th plot (2D)
ax5.scatter(img_pts2[:, 0, 0], img_pts2[:, 0, 1], c='purple')
ax5.set_title("img pts 2")
ax5.set_xlabel("X")
ax5.set_ylabel("Cos(X)")
ax5.legend()

ax6 = fig.add_subplot(236, projection='3d')  # 1st subplot
ax6.scatter(obj_pts1[:, 0], obj_pts1[:, 1], obj_pts1[:, 2], c='r', marker='o')
ax6.scatter(obj_pts2[:, 0], obj_pts2[:, 1], obj_pts2[:, 2], c='g', marker='^')
ax6.scatter(vio_pts[:, 0], vio_pts[:, 1], vio_pts[:, 2], c='b', marker='s')
ax6.set_title('all pts')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
def conv(x):
    return str(abs(int(x)))
fig.savefig(os.path.join(save_dir, f'{"_".join(list(map(conv, list(rot.reshape(3)))))}.png'))
