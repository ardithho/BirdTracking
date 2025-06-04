import argparse
import pycolmap
from scipy.spatial.transform import Rotation as R

import os
import sys
from pathlib import Path
ROOT = Path(os.path.abspath(__file__)).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov8.predict import Predictor, detect_features

from utils.box import pad_boxes
from utils.camera import Camera, Stereo
from utils.reconstruct import get_head_feat_pts, reproj_error
from utils.sim import *
from utils.structs import Bird, Birds


predictor = Predictor(ROOT / 'yolov8/weights/head.pt')
out_dir = ROOT / 'out'
os.makedirs(out_dir, exist_ok=True)

blender_cfg = ROOT / 'data/blender/configs/cam.yaml'
with open(blender_cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    ext = np.array(cfg['ext']).reshape(3, 4)
    cam_rmat = ext[:3, :3]
    cam_rvec = cv2.Rodrigues(cam_rmat)[0]
    cam_tvec = ext[:3, 3]

# initialise parser
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='file/dir/URL/glob/screen/0(webcam)')
parser.add_argument('--calib', type=str, default=ROOT / 'data/configs/cam.yaml', help='calibration video or yaml file')
parser.add_argument('--output', type=str, default=out_dir, help='output directory')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU threshold')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
parser.add_argument('--stride', type=int, default=1, help='inference video frame-rate stride')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--speed', type=int, default=1, help='output video speed')
parser.add_argument('--padding', type=int, default=30, help='head bounding box padding')
parser.add_argument('--resize', type=float, default=0.5, help='resize display window')


def parse_opt(parser):
    opt = parser.parse_args()
    return opt


def run(
        source,
        calib=ROOT / 'data/configs/cam.yaml',
        output=out_dir,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        half=False,
        device='',
        max_det=300,
        stride=1,
        agnostic_nms=False,
        classes=None,
        speed=1,
        padding=30,
        resize=0.5
):
    # Setup camera
    if Path(calib).suffix == '.yaml':
        # Setuop camera from yaml config file
        cam = Camera(calib).colmap
    else:
        # Calibrate camera
        cam_path = out_dir / 'cam.yaml'
        cam = Camera(calib)
        print('Calibrating camera...')
        cam.calibrate()
        print(f'Calibration MRE: {round(cam.mre, 3)}')
        cam.save(cam_path)
        cam = cam.colmap

    cap = cv2.VideoCapture(str(source))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    h, w = (720, 1280)
    out_path = output / f'out.mp4'
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps // stride * speed, (w, h * 2))

    birds = Birds()
    frame_count = 0
    re_sum = 0

    sim = Sim()
    T = np.eye(4)
    prev_T = np.eye(4)
    proj_T = np.eye(4)
    sim.update(T)
    while cap.isOpened():
        for i in range(stride):
            if cap.isOpened():
                _ = cap.grab()
            else:
                break
        ret, frame = cap.retrieve()
        if ret:
            head = pad_boxes(
                predictor.predictions(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    half=half,
                    device=device,
                    max_det=max_det,
                    agnostic_nms=agnostic_nms,
                    classes=classes,
                )[0].boxes.cpu().numpy(), frame.shape, padding)
            feat = detect_features(frame, head)
            birds.update([Bird(head, feat) for head, feat in zip(head, feat)][:1], frame)
            bird = birds['m'] if birds['m'] is not None else birds['f']
            if bird is not None:
                head_pts, feat_pts = get_head_feat_pts(bird)
                if head_pts.shape[0] >= 4:
                    pnp = pycolmap.estimate_and_refine_absolute_pose(feat_pts, head_pts, cam)
                    if pnp is not None:
                        rig = pnp['cam_from_world']  # Rigid3d
                        rmat = rig.rotation.matrix()
                        rmat = cam_rmat @ rmat  # camera to world
                        r = R.from_matrix(rmat).as_euler('xyz', degrees=True)
                        tvec = rig.translation + cam_tvec

                        # error projection
                        proj_T[:3, :3] = rmat
                        proj_T[:3, 3] = tvec

                        # colmap to o3d notation
                        r[0] *= -1
                        rmat = R.from_euler('xyz', r, degrees=True).as_matrix()
                        tvec[0] *= -1

                        # camera pose to head pose
                        rmat = rmat.T
                        tvec = -tvec

                        T[:3, :3] = rmat @ prev_T[:3, :3].T
                        # T[:3, 3] = tvec - prev_T[:3, 3]
                        print('es:', *np.rint(R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)))
                        print('esT:', *np.rint(-r))

                        frame_count += 1

                        prev_T[:3, :3] = rmat
                        # prev_T[:3, 3] = tvec
                        sim.update(T)
            cv2.imshow('frame', cv2.resize(birds.plot(), None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC))

            out = cv2.vconcat([cv2.resize(birds.plot(), (w, h), interpolation=cv2.INTER_CUBIC),
                               cv2.resize(sim.screen, (w, h), interpolation=cv2.INTER_CUBIC)])
            cv2.imshow('out', cv2.resize(out, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC))

            writer.write(out)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    sim.close()

    print('Video saved to:', str(out_path))


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt(parser)
    main(opt)
