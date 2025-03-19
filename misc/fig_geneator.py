import cv2

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from utils.camera import Stereo


RESIZE = .5  # resize display window
STRIDE = 1

out_path = ROOT / 'data/out/stereo_frame.jpg'

vidL = ROOT / 'data/vid/fps120/K203_K238/GOPRO2/GH010039.MP4'
vidR = ROOT / 'data/vid/fps120/K203_K238/GOPRO1/GH010045.MP4'

cfg_path = ROOT / 'data/calibration/cam.yaml'
stereo = Stereo(path=cfg_path)
capL = cv2.VideoCapture(str(vidL))
capR = cv2.VideoCapture(str(vidR))
# skip chessboard calibration frames
capL.set(cv2.CAP_PROP_POS_FRAMES, stereo.offsetL+1800)
capR.set(cv2.CAP_PROP_POS_FRAMES, stereo.offsetR+1800)

while capL.isOpened() and capR.isOpened():
    for i in range(STRIDE):
        _ = capL.grab()
        _ = capR.grab()
    stereo.offsetL += STRIDE
    stereo.offsetR += STRIDE
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if retL and retR:
        frame = cv2.hconcat([frameL, frameR])
        cv2.imshow('display', cv2.resize(frame, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(str(out_path), frame)
            break
capL.release()
capR.release()
cv2.destroyAllWindows()
