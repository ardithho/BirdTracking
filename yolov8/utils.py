import argparse

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
YOLO_ROOT = ROOT / 'yolov8'
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))


# initialise parser
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default=YOLO_ROOT / 'weights/pose.pt', help='model path or triton URL')
parser.add_argument('--source', type=str, default=ROOT / 'data/img', help='file/dir/URL/glob/screen/0(webcam)')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
parser.add_argument('--bird-stride', type=int, default=1, help='video frame-rate stride')
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--show', action='store_true', help='show results')
parser.add_argument('--no-save', action='store_true', help='do not save results to file')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--show-labels', default=True, action='store_true', help='show labels')
parser.add_argument('--show-conf', default=True, action='store_true', help='show confidences')
parser.add_argument('--show-boxes', default=True, action='store_true', help='show bounding boxes')
parser.add_argument('--line-width', default=3, type=int, help='bounding box thickness (pixels)')


def parse_opt(parser):
    opt = parser.parse_args()
    return opt
