import os
import sys
sys.path.append('./')

import cv2
import imageio
import argparse
import numpy as np
from tqdm import tqdm

from preproc import config as _C
from utils.pose_utils import _xywh2cs
from utils.transform import get_affine_transform, affine_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME, )
    grid_video_pth = os.path.join(results_dir, "grid_video.mp4")
    grid_video = imageio.get_writer(grid_video_pth, fps=60, mode="I", format="FFMPEG", macro_block_size=None)

    colors = np.load('/home/soyongs/Data/body_models/downsample_colors.npy')[:, :3] * 255
    npy_pth_list = sorted([f for f in os.listdir(results_dir) if f.endswith(".npy")])
    results = [np.load(os.path.join(results_dir, f)) for f in npy_pth_list]
    n_frames = min([len(result) for result in results])
    results = np.stack([result[:n_frames] for result in results], axis=1)
    results = results[..., [1, 0, 2]]
    
    bbox_results_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    camera_list = sorted([f for f in os.listdir(bbox_results_dir) if f.startswith("cam")])
    bboxes = []
    for camera in camera_list:
        bbox_pth_list = sorted(os.listdir(os.path.join(bbox_results_dir, camera, "bbox")))
        _bboxes = []
        for bbox_pth in bbox_pth_list:
            bbox = np.load(os.path.join(bbox_results_dir, camera, "bbox", bbox_pth))
            if bbox.shape[-1] == 1:
                bbox = np.ones(4) * -1
            _bboxes.append(bbox)
        # bbox = np.stack([np.load(os.path.join(bbox_results_dir, camera, "bbox", bbox_pth)) for bbox_pth in bbox_pth_list], axis=0)
        bbox = np.stack(_bboxes, axis=0)
        bboxes.append(bbox[:n_frames])
    bboxes = np.stack(bboxes, axis=1)

    grid_image_names = sorted(os.listdir(os.path.join(bbox_results_dir, "grid_vis")))
    grid_image_pth_list = [os.path.join(bbox_results_dir, "grid_vis", grid_image_name) for grid_image_name in grid_image_names]

    rows, cols = (2, 3)
    grid_video_pth = os.path.join(results_dir, "grid_video_colored.mp4")
    grid_video = imageio.get_writer(grid_video_pth, fps=60, mode="I", format="FFMPEG", macro_block_size=None)
    for frame_i, grid_image_pth in tqdm(enumerate(grid_image_pth_list), total=len(grid_image_pth_list), dynamic_ncols=True, leave=False):
        grid_image = cv2.imread(grid_image_pth)
        if frame_i >= bboxes.shape[0]: break
        
        bbs = bboxes[frame_i].copy()
        rslts = results[frame_i].copy()

        for camera_i, (bbox, result) in enumerate(zip(bbs, rslts)):
            row = camera_i // cols
            col = camera_i % cols

            center, scale = _xywh2cs(bbox)
            trans = get_affine_transform(center, scale, 0.0, (192, 256))
            _result = np.stack([affine_transform(result[i, :2], trans) for i in range(result.shape[0])], axis=0)
            
            _result[:, 0] = _result[:, 0] + col * 192
            _result[:, 1] = _result[:, 1] + 40 + row * 256

            for joint_i, (xy, color) in enumerate(zip(_result, colors)):
                if results[frame_i, camera_i, joint_i, 2] < 0.5: 
                    continue
                cv2.circle(grid_image, (int(xy[0]), int(xy[1])), color=color, radius=1, thickness=-1)
        
        grid_video.append_data(grid_image[..., ::-1])
    grid_video.close()