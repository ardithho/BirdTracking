from yolov8.predict import main
from yolov8.utils import parser, parse_opt


if __name__ == "__main__":
    opt = parse_opt(parser)
    main(opt)
