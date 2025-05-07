import os
import sys
sys.path.append('./')

import cv2
import imageio
import argparse
import numpy as np
from tqdm import tqdm

from preproc import config as _C
from preproc.s07_predict_bbox.util import create_camera_grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-m', '--model', default='vitpose', choices=['vitpose', 'dense_vitpose'])
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    results_dir = os.path.join(_C.VITPOSE_RESULTS_DIR if args.model == 'vitpose' else _C.LANDMARK_RESULTS_DIR, 
                               _C.SEQUENCE_NAME, )
    
    grid_video_pth = os.path.join(results_dir, "grid_video.mp4")
    grid_video = imageio.get_writer(grid_video_pth, fps=60, mode="I", format="FFMPEG", macro_block_size=None)

    vid_pth_list = sorted([f for f in os.listdir(results_dir) if f.startswith("cam") and f.endswith(".mp4")])
    vidcaps = [cv2.VideoCapture(os.path.join(results_dir, vid_pth)) for vid_pth in vid_pth_list]
    n_frames = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in vidcaps])
    pbar = tqdm(range(n_frames), dynamic_ncols=True, leave=False)

    frame_i = 0
    h, w = (256, 192) if args.model == "vitpose" else (384, 288)
    BG = np.zeros((h, w, 3)).astype(np.uint8) * 55

    while True:
        imgs = []
        rets = []
        for cap in vidcaps:
            ret, img = cap.read()
            rets.append(ret)
            if not ret:
                imgs.append(BG.copy())
            else:
                imgs.append(img)

        if not any(rets):
            break

        grid_image = create_camera_grid(imgs, frame_i)
        grid_video.append_data(grid_image[..., ::-1])
        frame_i += 1
        pbar.update(1)