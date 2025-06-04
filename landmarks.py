import cv2
import argparse

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov8.predict import Predictor, detect_features

from utils.box import pad_boxes
from utils.structs import Bird, Birds


predictor = Predictor('yolov8/weights/head.pt')
out_dir = ROOT / 'out'

# initialise parser
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='file/dir/URL/glob/screen/0(webcam)')
parser.add_argument('--output', type=str, default=out_dir, help='output directory')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU threshold')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
parser.add_argument('--stride', type=int, default=1, help='inference video frame-rate stride')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--speed', type=int, default=1, help='output video speed')
parser.add_argument('--padding', type=int, default=30, help='head bounding box padding')
parser.add_argument('--resize', type=float, default=0.5, help='resize display window')
parser.add_argument('--no-save', action='store_true', help='don\'t save output video')


def parse_opt(parser):
    opt = parser.parse_args()
    return opt


def run(
        source,
        output=out_dir,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        half=False,
        device='',
        max_det=300,
        stride=1,
        agnostic_nms=False,
        classes=None,
        speed=1,
        padding=30,
        resize=0.5,
        no_save=False
):
    os.makedirs(output, exist_ok=True)
    cap = cv2.VideoCapture(source)

    if not no_save:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(os.path.join(output, 'landmarks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps//stride*speed, (w, h))

    birds = Birds()
    while cap.isOpened():
        for i in range(stride):
            _ = cap.grab()
        ret, frame = cap.retrieve()
        if ret:
            head = pad_boxes(
                predictor.predictions(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    half=half,
                    device=device,
                    max_det=max_det,
                    agnostic_nms=agnostic_nms,
                    classes=classes,
                )[0].boxes.cpu().numpy(), frame.shape, padding)
            feat = detect_features(frame, head)
            birds.update([Bird(head, feat) for head, feat in zip(head, feat)], frame)

            display = birds.plot()
            cv2.imshow('display',
                       cv2.resize(display, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC))
            if not no_save:
                writer.write(display)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.waitKey(0)
        else:
            break

    cap.release()
    if not no_save:
        writer.release()
    cv2.destroyAllWindows()


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt(parser)
    main(opt)

