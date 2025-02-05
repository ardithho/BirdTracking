import cv2
import os
from pathlib import Path


NAME = 'vanilla'
VIEW = 'f'
BLENDER_ROOT = Path(__file__).parent.parent
renders_dir = BLENDER_ROOT / 'renders'
input_dir = renders_dir / NAME / VIEW
output_dir = renders_dir / 'vid'

out_path = os.path.join(output_dir, f'{NAME}.mp4')

w, h = (1920, 1080)
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

renders = [x for x in os.listdir(input_dir) if x.endswith('.png') or x.endswith('.jpg')]
renders.sort()

for render in renders:
    writer.write(cv2.imread(os.path.join(input_dir, render)))

writer.release()
cv2.destroyAllWindows()
print('Rendered video saved at {}'.format(out_path))
