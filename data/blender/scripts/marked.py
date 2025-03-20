import bpy
import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from mathutils import Matrix
from scipy.spatial.transform import Rotation as R


BLENDER_ROOT = Path(__file__).parent.parent.parent
renders_dir = BLENDER_ROOT / 'renders'
output_dir = os.path.join(renders_dir, 'marked')
l_dir = os.path.join(output_dir, 'l')
r_dir = os.path.join(output_dir, 'r')
f_dir = os.path.join(output_dir, 'f')
os.makedirs(l_dir, exist_ok=True)
os.makedirs(r_dir, exist_ok=True)
os.makedirs(f_dir, exist_ok=True)

scene = bpy.context.scene


def extrinsic_mat(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, 1, 0),
         (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    # location = cam.location
    # rotation = cam.rotation_euler
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = np.array([[*R_world2cv[0][:], T_world2cv[0]],
                   [*R_world2cv[1][:], T_world2cv[1]],
                   [*R_world2cv[2][:], T_world2cv[2]]])
    return RT


def camera_data(cam):
    data = cam.data
    f_in_mm = data.lens
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = data.sensor_width
    sensor_height_in_mm = data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (data.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    fx = f_in_mm * s_u
    fy = f_in_mm * s_v
    cx = resolution_x_in_px * scale / 2
    cy = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    k = np.array([[fx, skew, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    ext = extrinsic_mat(cam)
    return k, ext


camL = bpy.data.objects['cam_l']
camR = bpy.data.objects['cam_r']
camF = bpy.data.objects['cam_f']
kL, extL = camera_data(camL)
kR, extR = camera_data(camR)
kF, extF = camera_data(camF)
with open(os.path.join(output_dir, 'cam.yaml'), 'w') as f:
    data = {'pathL': l_dir,
            'KL': kL.flatten().tolist(),
            'extL': extL.flatten().tolist(),
            'pathR': r_dir,
            'KR': kR.flatten().tolist(),
            'extR': extR.flatten().tolist(),
            'pathF': f_dir,
            'KF': kF.flatten().tolist(),
            'extF': extF.flatten().tolist()}
    f.write(yaml.dump(data, sort_keys=False))


f = open(os.path.join(output_dir, 'transforms.txt'), 'w')

bpy.ops.object.select_all(action='DESELECT')
mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
for obj in mesh:
    obj.select_set(state=True)
    bpy.context.view_layer.objects.active = obj
bpy.ops.object.join()

head = bpy.context.active_object
head.hide_render = False
T = np.eye(4)
for i in range(100):
    f.write(' '.join([str(i+1), *map(str, T.flatten())]) + '\n')
    head.matrix_world = Matrix(T) @ head.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

    scene.camera = camL
    scene.render.filepath = os.path.join(l_dir, '%03d.jpg' % (i+1))
    bpy.ops.render.render(write_still=True, use_viewport=True)
    scene.camera = camR
    scene.render.filepath = os.path.join(r_dir, '%03d.jpg' % (i+1))
    bpy.ops.render.render(write_still=True, use_viewport=True)
    scene.camera = camF
    scene.render.filepath = os.path.join(f_dir, '%03d.jpg' % (i+1))
    bpy.ops.render.render(write_still=True, use_viewport=True)

    # T[:3, 3] = np.random.rand(3) * 0.005
    # T[:3, :3] = R.from_euler('zyx', np.random.randint(0, 5, 3), degrees=True).as_matrix()
    T[:3, :3] = cv2.Rodrigues(np.random.randint(0, 5, 3)*np.pi/180)[0]

f.close()
