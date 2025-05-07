import os
import sys
sys.path.append("./")
import glob

import cv2
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from preproc import config as _C
from utils.fitting_utils.loader import read_data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-c', '--camera-idx', default=0, type=int)
    parser.add_argument('--image-overlay', action='store_true')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    vitpose_results_dir = os.path.join(_C.VITPOSE_RESULTS_DIR, _C.SEQUENCE_NAME)
    dense_vitpose_results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)
    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")

    kpts, bboxes, calibs = read_data(vitpose_results_dir, dense_vitpose_results_dir, bbox_dir, calib_pth)

    kpts2d = kpts[:, args.camera_idx]
    BG = np.ones((2160, 3840, 3)).astype(np.uint8) * 120
    writer = imageio.get_writer(f"outputs/vis/kpts2d/{args.sequence}_{args.camera_idx}.mp4", 
                                mode="I", 
                                format="FFMPEG", 
                                fps=60)
    for kpt in tqdm(kpts2d):
        # Draw keypoints
        img = BG.copy()
        for i, xyc in enumerate(kpt):
            color = (255, 0, 0) if i < 17 else (0, 255, 0)
            x = int(xyc[0])
            y = int(xyc[1])
            img = cv2.circle(img, (x, y), radius=3, color=color, thickness=-1)

        writer.append_data(cv2.resize(img[..., ::-1], None, fx=0.25, fy=0.25))

    writer.close()