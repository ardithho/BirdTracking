import bpy
import os
import numpy as np
from mathutils import Matrix
from scipy.spatial.transform import Rotation as R


output_dir = '/data/blender/renders'
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
    T[:3, 3] = np.random.rand(3) * 0.005
    T[:3, :3] = R.from_euler('xyz', np.random.randint(0, 10, 3), degrees=True).as_matrix()
    print(T)
    f.write(' '.join([str(i+1), *map(str, T.flatten())]) + '\n')
    head.matrix_world @= Matrix(T)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    bpy.context.scene.render.filepath = os.path.join(output_dir, '%03d.jpg' % (i+1))
    bpy.ops.render.render(write_still=True, use_viewport=True)

f.close()
