import os
import sys
sys.path.append('./')
import glob
import commentjson as json
import argparse
import subprocess

import cv2
import torch
import numpy as np

from preproc import config as _C
from vis.animation.renderables import addKeypointsSequence
from vis.animation import render_scene_list

from utils.fitting_utils.kp_utils import (filter_keypoints_2d, 
                                          filter_keypoints_3d, 
                                          smooth_keypoints)
from utils.fitting_utils.triangulation import (iterative_triangulation, 
                                               simple_triangulation)
from utils.fitting_utils.distortion import do_undistortion, do_undistort_fisheye
from utils.fitting_utils.loader import read_data



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('--vitpose_only', action='store_true')
    parser.add_argument('-i', '--ignore_camera_idx', default=-1, type=str)
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    vitpose_results_dir = os.path.join(_C.VITPOSE_RESULTS_DIR, _C.SEQUENCE_NAME)
    dense_vitpose_results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)
    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")

    kpts, bboxes, calibs = read_data(vitpose_results_dir, dense_vitpose_results_dir, bbox_dir, calib_pth)
    if args.vitpose_only: 
        kpts = kpts[..., :17, :]
        min_valid_points = 6
    else:
        min_valid_points = 50
    
    json_pth = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, "bbox_removal.json")
    if os.path.exists(json_pth):
        with open(json_pth, "r", encoding="utf-8") as fopen:
            ignore_dict = json.load(fopen)

        for camera_i, (camera, frames) in enumerate(ignore_dict.items()):
            for frame in frames:
                if len(frame.split("-")) == 2:
                    start, end = [int(f) for f in frame.split("-")]
                    frames_to_update = list(range(start, end+1))

                else:
                    frames_to_update = [int(frame)]

                kpts[frames_to_update, camera_i, :, -1] = 0.0
    
    # Undistort keypoints
    undist_kpts, new_K = do_undistort_fisheye(kpts[..., :2].astype(float), 
                                              calibs['Ks'].astype(float), 
                                              calibs['dists'].astype(float), 
                                              (3840, 2160))
    undist_kpts = np.concatenate((undist_kpts, kpts[..., -1:]), axis=-1)
    calibs["Ks"] = new_K.copy()
    # undist_kpts[:, 1, ..., -1] *= 0.0
    # undist_kpts = do_undistortion(kpts, calibs)

    # Filter keypoints
    smoothed_kpts = undist_kpts.copy()
    for c in range(kpts.shape[1]):
        smoothed_kpts[:, c] = smooth_keypoints(undist_kpts[:, c], dim=2, kernel_size=5, kernel_type='gaussian')

    # kpts3d = iterative_triangulation(smoothed_kpts, calibs, bboxes, conf_thr=0.3, reproj_thr=0.03, n_repeat=1, min_valid_points=min_valid_points)
    kpts3d = simple_triangulation(smoothed_kpts, calibs, apply_conf=True, conf_thr=0.3, min_valid_points=min_valid_points)
    kpts3d[kpts3d[..., -1] < 0.1] = 0.0
    kpts3d = filter_keypoints_3d(kpts3d, )
    kpts3d = smooth_keypoints(kpts3d, dim=3, kernel_size=5, kernel_type='gaussian')
    kpts3d[np.all(kpts3d[..., :3] == 0, axis=-1)] = 0.0

    scene_list = []
    scene_list.append(addKeypointsSequence(kpts3d[..., :3], ))
    render_scene_list(scene_list)