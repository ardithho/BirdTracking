import cv2

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 1

data_dir = ROOT / 'data'
test_dir = data_dir / 'test'
out_dir = data_dir / 'out/pnp'
os.makedirs(out_dir, exist_ok=True)

vid_path = test_dir / f'bird/test_{TEST}.mp4'
calib_path = test_dir / f'calib/test_{TEST}.mp4'

cap = cv2.VideoCapture(str(calib_path))
ret, frame = cap.read()
cv2.imwrite(f'out.jpg', frame)
cap.release()
