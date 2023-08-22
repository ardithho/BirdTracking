import argparse
import sys
from pathlib import Path
from functional.dataloaders import DetectionsDataloader


ROOT = Path.cwd()  # this project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(
        labels=ROOT / 'data/detect/labels',
        source=ROOT / 'data/detect/K213_K268_1_GH010020_cut.mp4',
        offset=3,
        resize=0.5,
        pt_size=3
):
    detections = DetectionsDataloader(str(labels), offset=offset, resize=resize, pt_size=pt_size)
    detections.load()
    detections.interpolate()
    detections.compare(str(source))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default=ROOT / 'data/detect/labels')
    parser.add_argument('--source', type=str, default=ROOT / 'data/detect/K213_K268_1_GH010020_cut.mp4')
    parser.add_argument('--offset', type=int, default=3, help='number of frames to interpolate over')
    parser.add_argument('--resize', type=float, default=0.5, help='resize output frame size')
    parser.add_argument('--pt-size', type=int, default=3, help='size of annotation points')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
