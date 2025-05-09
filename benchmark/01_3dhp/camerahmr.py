import os
import sys
sys.path.append('./')
import argparse

from preproc import config as _C
from utils.subprocess_utils import run_command_with_conda

conda_env = "dgl"
working_dir = "/home/soyongs/Codes/srcs/CameraHMR"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    results_dir = os.path.join(_C.BENCHMARK_3DHP_RESULT_DIR, "camerahmr", _C.SEQUENCE_NAME)
    os.makedirs(results_dir, exist_ok=True)
    
    image_base_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME)
    bbox_base_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    cameras = sorted([f for f in os.listdir(image_base_dir) if f.startswith("cam")])
    for camera in cameras:
        output_file_path = os.path.join(results_dir, f"{camera}.pkl")
        image_folder = os.path.join(image_base_dir, camera)
        bbox_folder = os.path.join(bbox_base_dir, camera, "bbox")

        run_cmd = [
        "python", "-u", 
            "evaluate.py", 
            "--image_folder", image_folder, 
            "--bbox_folder", bbox_folder, 
            "--output_file_path", output_file_path,
            "--fps", 6,
        ]

        run_command_with_conda(working_dir, conda_env, run_cmd)