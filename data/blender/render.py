import bpy
import os

import cv2
import yaml
import numpy as np
from pathlib import Path
from mathutils import Matrix
from scipy.spatial.transform import Rotation as R


parent_dir = Path(__file__).parent.parent
output_dir = os.path.join(parent_dir, 'renders')
os.makedirs(output_dir, exist_ok=True)

cam = bpy.data.objects["Camera"].data
scene = bpy.context.scene
assert scene.render.resolution_percentage == 100
assert cam.sensor_fit != 'VERTICAL'
f_in_mm = cam.lens
sensor_width_in_mm = cam.sensor_width
w = scene.render.resolution_x
h = scene.render.resolution_y
pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
fx = f_in_mm / sensor_width_in_mm * w
fy = fx * pixel_aspect
# yes, shift_x is inverted
cx = w * (0.5 - cam.shift_x)
# and shift_y is still a percentage of width
cy = h * 0.5 + w * cam.shift_y
k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
with open(os.path.join(output_dir, 'cam.yaml'), 'w') as f:
    data = {'K': k.flatten().tolist()}
    f.write(yaml.dump(data, sort_keys=False))


f = open(os.path.join(output_dir, 'transforms.txt'), 'w')

bpy.ops.object.select_all(action='DESELECT')
mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
for obj in mesh:
    obj.select_set(state=True)
    bpy.context.view_layer.objects.active = obj
bpy.ops.object.join()

head = bpy.context.active_object
T = np.eye(4)
for i in range(100):
    f.write(' '.join([str(i+1), *map(str, T.flatten())]) + '\n')
    head.matrix_world @= Matrix(T)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    scene.render.filepath = os.path.join(output_dir, '%03d.jpg' % (i+1))
    bpy.ops.render.render(write_still=True, use_viewport=True)

    T[:3, 3] = np.random.rand(3) * 0.005
    # T[:3, :3] = R.from_euler('zyx', np.random.randint(0, 5, 3), degrees=True).as_matrix()
    T[:3, :3] = cv2.Rodrigues(np.random.randint(0, 5, 3)*np.pi/180)[0]

f.close()
