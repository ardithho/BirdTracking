from ultralytics import YOLO

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
YOLO_ROOT = ROOT / 'yolov8'
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))


if __name__ == '__main__':
    print(ROOT)
    name = 'head'
    model = YOLO_ROOT / f'weights/{name}.pt'
    model = YOLO(model)
    metrics = model.val()
