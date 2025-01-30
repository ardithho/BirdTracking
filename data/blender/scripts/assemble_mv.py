import cv2
import os
from pathlib import Path


NAME = 'marked'
BLENDER_ROOT = Path(__file__).parent.parent
renders_dir = BLENDER_ROOT / 'renders'
input_dir = renders_dir / NAME
output_dir = renders_dir / 'vid'

l_dir = os.path.join(input_dir, 'l')
r_dir = os.path.join(input_dir, 'r')
f_dir = os.path.join(input_dir, 'f')

out_path = os.path.join(output_dir, f'{NAME}.mp4')
l_out_path = os.path.join(output_dir, f'{NAME}_l.mp4')
r_out_path = os.path.join(output_dir, f'{NAME}_r.mp4')
f_out_path = os.path.join(output_dir, f'{NAME}_f.mp4')

w, h = (1920, 1080)
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w*2, h))
writer_l = cv2.VideoWriter(l_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
writer_r = cv2.VideoWriter(r_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
writer_f = cv2.VideoWriter(f_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

l_renders = [x for x in os.listdir(l_dir) if x.endswith('.png') or x.endswith('.jpg')]
l_renders.sort()
r_renders = [x for x in os.listdir(r_dir) if x.endswith('.png') or x.endswith('.jpg')]
r_renders.sort()
f_renders = [x for x in os.listdir(f_dir) if x.endswith('.png') or x.endswith('.jpg')]
f_renders.sort()

for l_render, r_render, f_render in zip(l_renders, r_renders, f_renders):
    l_render = cv2.imread(os.path.join(l_dir, l_render))
    r_render = cv2.imread(os.path.join(r_dir, r_render))
    f_render = cv2.imread(os.path.join(f_dir, f_render))
    render = cv2.hconcat([l_render, r_render])
    writer.write(render)
    writer_l.write(l_render)
    writer_r.write(r_render)
    writer_f.write(f_render)

writer.release()
cv2.destroyAllWindows()
print('Rendered video saved at {}'.format(out_path))
