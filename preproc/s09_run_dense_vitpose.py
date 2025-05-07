import os
import sys
sys.path.append('./')
import glob
import argparse
import subprocess

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from preproc import config as _C


import os
import sys
sys.path.append('./')
import glob
import json
import subprocess

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from preproc import config as _C


def run_dense_landmark(working_dir, conda_env, cmd):
    """
    Run a command in a specific working directory using a specific conda environment.
    Output goes directly to the terminal (so tqdm bars still render).
    """
    current_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        # Build a proper argument list instead of a shell string
        full_cmd = ["conda", "run", "-n", conda_env,
                    "--no-capture-output",     # don’t buffer/capture output internally
                    ] + cmd
        
        print(f"Running: {' '.join(full_cmd)}")
        # Don't redirect anything—inherit the parent’s fds
        process = subprocess.Popen(full_cmd)
        ret = process.wait()
        if ret != 0:
            print(f"❌ Command failed with exit code {ret}")
            return False
        print("✅ Command succeeded")
        return True

    except Exception as e:
        print(f"Exception while running samurai: {e}")
        return False
    finally:
        os.chdir(current_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    conda_env = "dgl"
    landmark_results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)
    os.makedirs(landmark_results_dir, exist_ok=True)
    for camera in tqdm(_C.CAMERA_NAMES, dynamic_ncols=True):
        landmark_results_pth = os.path.join(landmark_results_dir, f"{camera}.npy")
        # if os.path.exists(landmark_results_pth):
        #     continue

        working_dir = _C.LANDMARK_WORKING_DIR
        image_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera)
        bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, camera, "bbox")
        # bbox_dir = os.path.join(_C.YOLO_RESULTS_DIR, _C.SEQUENCE_NAME, camera, "bbox")
        
        run_cmd = [
            "python", "-u", 
            "-m", "scripts.fbcontact.run_detection", 
            "--img_fldr", image_dir, 
            "--bbox_fldr", bbox_dir,
            "--output_npy_pth", landmark_results_pth,
            "--visualize", "--fps", f"{_C.TARGET_FPS}"
        ]

        run_dense_landmark(working_dir, conda_env, run_cmd)