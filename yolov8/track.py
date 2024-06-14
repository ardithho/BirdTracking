import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = os.path.join(Path.cwd(), 'yolov8')  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

PROJECT_ROOT = Path.cwd()  # this project root directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))  # add PROJECT_ROOT to PATH
PROJECT_ROOT = Path(os.path.relpath(PROJECT_ROOT, Path.cwd()))  # relative


class Tracker:
    def __init__(self, model_path=ROOT / 'weights/pose.pt'):
        self.model = YOLO(model_path)

    def track(self, **kwargs):
        self.model.track(**kwargs)

    def tracks(self, **kwargs):
        return self.model.tracks(**kwargs)
