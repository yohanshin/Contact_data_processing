import os
import sys
sys.path.append('./')
import glob
import json
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from preproc import config as _C
from utils.subprocess_utils import (
    run_command_with_conda, 
    image_to_video
)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-c', '--camera', default='all')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    conda_env = "samurai"
    samurai_results_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    os.makedirs(samurai_results_dir, exist_ok=True)
    yolo_results_dir = os.path.join(_C.YOLO_RESULTS_DIR, _C.SEQUENCE_NAME)

    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    camera_list = _C.CAMERA_NAMES if args.camera == "all" else [args.camera]
    for camera in tqdm(camera_list, dynamic_ncols=True):
        if camera not in proc_info["sync_frame"]:
            continue

        if "valid_init_frames" in proc_info:
            if camera in proc_info["valid_init_frames"]:
                valid_frame = f"{int(proc_info['valid_init_frames'][camera]):05d}.jpg"
            else:
                valid_frame = '00001.jpg'
        else:
            valid_frame = '00001.jpg'
        
        image_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera)
        txt_pth = os.path.join(samurai_results_dir, 'prompt', f"{camera}.txt")
        results_vis_dir = os.path.join(samurai_results_dir, f"{camera}/vis")
        output_video_pth = os.path.join(samurai_results_dir, f"{camera}/vis.avi")
        results_bbox_dir = os.path.join(samurai_results_dir, f"{camera}/bbox")

        # if os.path.exists(output_video_pth):
        #     continue
        
        os.makedirs(results_vis_dir, exist_ok=True)
        os.makedirs(results_bbox_dir, exist_ok=True)
        working_dir = _C.SAMURAI_WORKING_DIR

        cmd = ["python", "-u",            # -u = unbuffered stdout/stderr
               "-m", "scripts.demo_long_video", 
               "--video_path", image_dir, 
               "--txt_path", txt_pth, 
               "--model_path", _C.SAMURAI_MODEL_CKPT, 
               "--image_output_dir", results_vis_dir, 
               "--bbox_output_dir", results_bbox_dir, 
               "--valid_from", valid_frame,
            #    "--save_to_image"
               ]
        run_command_with_conda(working_dir, conda_env, cmd)

        image_to_video(
            image_folder=results_vis_dir,
            output_video_path=output_video_pth,
            framerate=_C.TARGET_FPS
        )