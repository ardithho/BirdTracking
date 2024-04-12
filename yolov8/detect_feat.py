from detect import Detect


model = Detect('yolov8/weights/feat.pt')


def detect_features(img, boxes):
    feat = []
    for xyxy in boxes.xyxy:
        x0, y0, x1, y1 = list(xyxy.cpu().numpy())
        feat.append(model.predictions(img[y0:y1, x0:x1])[0].boxes)
    return feat
