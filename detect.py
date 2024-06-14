from yolov8.predict import main, parse_opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
