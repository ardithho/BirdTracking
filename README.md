# BirdTracking

## Installation
Clone repo and install [requirements.txt](https://github.com/teethoe/BirdTracking/blob/master/requirements.txt) in a 
[**Python>=3.7.0**](https://www.python.org/) environment, including 
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). 
(For a more detailed installation guide please refer to 
[INSTALL.md](https://github.com/teethoe/BirdTracking/blob/master/INSTALL.md).)

```bash
git clone https://github.com/teethoe/BirdTracking  # clone
cd BirdTracking
pip install -r requirements.txt  # install
```

## Detect
`detect.py` runs inference on a variety of sources and saving results to `runs/detect`.
```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
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
Use the full path or relative path of files instead of just the file names, 
the full path of images, videos or directories can be found by 
selecting the file in file explorer and then pressing `Ctrl+Shift+C` for Windows, 
or right-clicking the file and select `Get Info` for macOS.
