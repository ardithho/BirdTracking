import cv2
import os
import numpy as np
from pathlib import Path


parent_dir = output_dir = Path(__file__).parent
input_dir = os.path.join(parent_dir, 'renders')
out_path = os.path.join(output_dir, 'render.mp4')
w, h = (1920, 1080)
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MPEG'), 10, (w, h))
renders = os.listdir(input_dir)
renders.sort()

for render in renders:
    if render.endswith('.png') or render.endswith('.jpg'):
        writer.write(cv2.imread(os.path.join(input_dir, render)))

writer.release()
cv2.destroyAllWindows()
print('Rendered video saved at {}'.format(out_path))
