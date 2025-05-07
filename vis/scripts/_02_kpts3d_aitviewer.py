import os
import sys
sys.path.append('./')
import glob
import json
import argparse
import subprocess

import cv2
import torch
import pycolmap
import numpy as np

from preproc import config as _C
from vis.animation.renderables import addKeypointsSequence
from vis.animation import render_scene_list

from utils.fitting_utils.kp_utils import (filter_keypoints_2d, 
                                          filter_keypoints_3d, 
                                          smooth_keypoints)
from utils.fitting_utils.triangulation import (iterative_triangulation, 
                                               simple_triangulation)
from utils.fitting_utils.distortion import do_undistortion
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
    colmap_dir = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "workspace")

    reconstruction = pycolmap.Reconstruction(colmap_dir)
    import pdb; pdb.set_trace()

    kpts, bboxes, calibs = read_data(vitpose_results_dir, dense_vitpose_results_dir, bbox_dir, calib_pth)
    if args.vitpose_only: 
        kpts = kpts[..., :17, :]
        min_valid_points = 6
    else:
        min_valid_points = 50
    
    