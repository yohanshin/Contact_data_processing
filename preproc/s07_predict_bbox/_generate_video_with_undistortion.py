import os
import sys
sys.path.append('./')
sys.path.append(os.getcwd())
import glob
import json
import argparse

import cv2
import numpy as np
import imageio
from tqdm import trange
from preproc import config as _C

from utils.pose_utils import _xywh2cs
from utils.subprocess_utils import image_to_video
from preproc.s07_predict_bbox.util import create_camera_grid

from vis import get_affine_transform


def main(args, target_frames=None):
    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    bboxes = []
    for camera in _C.CAMERA_NAMES:
        bbox_pth_list = sorted(glob.glob(os.path.join(bbox_dir, camera, 'bbox/*.npy')))
        if camera not in proc_info["sync_frame"]:
            bboxes.append(np.ones_like(bboxes[-1]) * (-1))    
            continue
        bboxes.append(np.stack([np.load(bbox_pth) for bbox_pth in bbox_pth_list]))

    max_len = max([_bbox.shape[0] for _bbox in bboxes])
    for bbox_i, _bbox in enumerate(bboxes):
        if _bbox.shape[0] < max_len:
            _bbox = np.concatenate((_bbox, np.ones((max_len - _bbox.shape[0], 4)) * (-1) ))
            bboxes[bbox_i] = _bbox.copy()
    
    bboxes = np.stack(bboxes, axis=1)
    n_frames = bboxes.shape[0]
    results_video_pth = os.path.join(bbox_dir, 'grid_vis_w_undist.avi')

    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")
    calib = dict(np.load(calib_pth))
    Ks, effective_Ks, maps_list = [], [], []
    for camera_idx in range(6):
        dist = calib["dists"][camera_idx]
        K = calib["Ks"][camera_idx]

        height, width = 2160, 3840
        effective_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, (width, height), np.eye(3), balance=0.0
        )
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), effective_K, (width, height), cv2.CV_16SC2
        )

        Ks.append(K)
        effective_Ks.append(effective_K)
        maps_list.append((mapx, mapy))

    writer = imageio.get_writer(results_video_pth, fps=60, mode="I", format="FFMPEG", macro_block_size=None)
    pad = np.zeros((256, 192, 3)).astype(np.uint8)
    for frame in trange(n_frames, leave=False, dynamic_ncols=True):
        if target_frames is not None:
            if not frame in target_frames: continue
        
        frame_image = []
        for i, camera in enumerate(_C.CAMERA_NAMES):
            if i != 1 or frame != 6840:
                continue
            
            bbox = bboxes[frame, i].copy()
            image_pth = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera, f"{frame + 1:05d}.jpg")
            
            if np.all(bbox <= 0) or not os.path.exists(image_pth):
                frame_image.append(pad)
                continue
            
            image = cv2.imread(image_pth)
            undist_image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)
            image = undist_image.copy()

            x, y, w, h = bbox
            corner = np.array([[x, y], [x+w, y+h]]).astype(float)
            undist_corner = cv2.fisheye.undistortPoints(
                corner.reshape(-1, 1, 2), K, dist, R=np.eye(3), P=effective_K
            ).reshape(4)
            
            x, y = undist_corner[:2].astype(int)
            w, h = (undist_corner[2:] - undist_corner[:2]).astype(int)

            if x + w / 2.0 < 0 or x + w / 2.0 > width:
                frame_image.append(pad)
                continue

            if y + h / 2.0 < 0 or y + h / 2.0 > height:
                frame_image.append(pad)
                continue

            bbox = np.array([x, y, w, h])

            center, trans = _xywh2cs(bbox)
            trans = get_affine_transform(center, trans, 0.0, (192, 256))
            cropped_image = cv2.warpAffine(
                image.copy(),
                trans,
                (192, 256),
                flags=cv2.INTER_LINEAR
            )
            import pdb; pdb.set_trace()
            
            frame_image.append(cropped_image)
        
        grid_image = create_camera_grid(frame_image, frame, cam_names=_C.CAMERA_NAMES)
        writer.append_data(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))

    writer.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    main(args)