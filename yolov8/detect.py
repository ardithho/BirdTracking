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


class Detect:
    """
    Detect bird head features in source using our pre-trained model.
    """

    def __init__(self, model_path=ROOT / 'weights/full.pt'):
        self.model = YOLO(model_path)

    def predict(self, **kwargs):
        self.model.predict(**kwargs)

    def predictions(self, **kwargs):
        return self.model.predict(**kwargs)


feat_model = Detect(ROOT / 'weights/feat.pt')
def detect_features(img, boxes):
    feat = []
    for xyxy in boxes.xyxy:
        x0, y0, x1, y1 = list(map(int, list(xyxy.cpu().numpy())))
        feat.append(feat_model.predictions(img[y0:y1, x0:x1])[0].boxes)
    return feat


def run(
        weights=ROOT / 'weights/full.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        conf=0.25,  # confidence threshold
        iou=0.7,  # NMS IOU threshold
        imgsz=640,  # inference size
        half=False,  # use FP16 half-precision inference
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        max_det=300,  # maximum detections per image
        vid_stride=1,  # video frame-rate stride
        visualize=False,  # visualize features
        augment=False,  # augmented inference
        agnostic_nms=False,  # class-agnostic NMS
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        show=False,  # show results
        no_save=False,  # do not save results to file
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        show_labels=True,  # show labels
        show_conf=True,  # show confidences
        boxes=True,  # show boxes
        line_width=3  # bounding box thickness (pixels)
):
    model = Detect(weights)
    model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        half=half,
        device=device,
        max_det=max_det,
        vid_stride=vid_stride,
        visualize=visualize,
        augment=augment,
        agnostic_nms=agnostic_nms,
        classes=classes,
        show=show,
        save=not no_save,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show_labels=show_labels,
        show_conf=show_conf,
        boxes=boxes,
        line_width=line_width
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/full.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
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
    parser.add_argument('--boxes', default=True, action='store_true', help='show bounding boxes')
    parser.add_argument('--line-width', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
