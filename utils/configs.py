import yaml

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


cfg_dir = ROOT / 'data/configs'
yolo_cfg_path = cfg_dir / 'yolo.yaml'
blender_cfg_path = cfg_dir / 'blender.yaml'
colour_cfg_path = cfg_dir / 'colour.yaml'

with open(yolo_cfg_path, 'r') as f:
    data = yaml.safe_load(f)
    CLS_DICT = data['cls']
    FEAT_DICT = data['feat']


with open(blender_cfg_path, 'r') as f:
    HEAD_CFG = yaml.safe_load(f)


with open(colour_cfg_path, 'r') as f:
    COLOUR_DICT = yaml.safe_load(f)
