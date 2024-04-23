import argparse
import sys
from pathlib import Path
from functional.dataloaders import DetectionsDataloader


ROOT = Path.cwd()  # this project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(
        labels=ROOT / 'data/detect/labels',
        source=ROOT / 'data/detect/K213_K268.mp4',
        offset=3,
        resize=0.5,
        pt_size=3,
        thickness=2
):
    detections = DetectionsDataloader(str(labels), offset=offset, resize=resize, pt_size=pt_size, thickness=thickness)
    detections.load()
    detections.interpolate()
    detections.compare(str(source))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default=ROOT / 'data/detect/labels', help='directory of labels')
    parser.add_argument('--source', type=str, default=ROOT / 'data/detect/K213_K268.mp4', help='video filepath')
    parser.add_argument('--offset', type=int, default=3, help='number of frames to interpolate over (int)')
    parser.add_argument('--resize', type=float, default=0.5, help='resize output frame size (float)')
    parser.add_argument('--pt-size', type=int, default=3, help='size of annotation points (int)')
    parser.add_argument('--thickness', type=int, default=2, help='thickness of annotation lines (int)')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
