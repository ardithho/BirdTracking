import open3d as o3d
from pathlib import Path


ROOT = Path(__file__).parent.parent
mesh = o3d.io.read_triangle_mesh(str(ROOT / 'data/blender/full_model.obj'), enable_post_processing=True)
print(mesh)
mesh.compute_vertex_normals()


if __name__ == '__main__':
    o3d.visualization.draw_geometries([mesh])
