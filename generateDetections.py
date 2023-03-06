import os


ROOT = os.getcwd()

yolo_path = os.path.join(ROOT, 'yolov5')
detect_full_path = os.path.join(yolo_path, 'detect_full.py')
detect_head_path = os.path.join(yolo_path, 'detect_head.py')

vid_root = os.path.join(ROOT, 'vid/fps120')
vid_names = os.listdir(vid_root)

CONF = 0.5
CONF_HEAD = 0.1

for vid_name in vid_names:
    vid_path = os.path.join(vid_root, vid_name)

    # detect features in videos
    # os.system(f'python {detect_full_path} --source {vid_path} --conf {CONF} --save-txt --save-conf --nosave')
    os.system(f'python {detect_head_path} --source {vid_path} --conf {CONF_HEAD} --save-crop --nosave')
