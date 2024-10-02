import open3d as o3d
from pathlib import Path


ROOT = Path(__file__).parent.parent


class Sim:
    def __init__(self, path=ROOT / 'data/blender/full_model.obj'):
        self.mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        self.mesh.compute_vertex_normals()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.mesh)

    def update(self, T):
        self.mesh.transform(T)
        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


sim = Sim()


if __name__ == '__main__':
    o3d.visualization.draw_geometries([sim.mesh])

