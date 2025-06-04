# BirdTracking

## Install
Clone repo and install [requirements.txt](https://github.com/teethoe/BirdTracking/blob/master/requirements.txt) in a 
[**Python>=3.7.0**](https://www.python.org/) environment, including 
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). 
(For a more detailed installation guide please refer to 
[INSTALL.md](https://github.com/teethoe/BirdTracking/blob/master/INSTALL.md).)

```bash
git clone https://github.com/ardithho/BirdTracking  # clone
cd BirdTracking
pip install -r requirements.txt  # install
```

## Demo
Run `main.py` for a demo of the 3D bird head pose estimation.
```bash
python main.py --source {path/to/vid} --calib {path/to/calibration/vid/or/yaml}
```
The calibration video is required to calibrate the camera at the start. 
It should include chessboard calibration frames of the camera same as the source. 
Alternatively, a `.yaml` file containing the camera info of a calibrated camera can be used. 
If the `--calib` input is a video, a `.yaml` file of the camera info will be saved after calibration, which can be used for other videos from the same camera.

For more options, run
```bash
python main.py --help
```

## Detect
`landmarks.py` is the standard head and facial landmark detection pipeline used in this project.
```bash
python landmarks.py --source {path/to/vid}
```
For more options, run
```bash
python landmarks.py --help
```
`detect.py` runs inference on a variety of sources and saving results to `runs/detect`.
```bash
python detect.py --source 0                               # webcam
                           img.jpg                         # image
                           vid.mp4                         # video
                           screen                          # screenshot
                           path/                           # directory
                           list.txt                        # list of images
                           list.streams                    # list of streams
                           'path/*.jpg'                    # glob
                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
