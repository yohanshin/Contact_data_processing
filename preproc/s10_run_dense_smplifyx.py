import os
import sys
import json
sys.path.append('./')
import argparse

from preproc import config as _C
from utils.subprocess_utils import run_command_with_conda


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    conda_env = "dgl"
    gender = proc_info["gender"]
    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")
    vitpose_results_dir = os.path.join(_C.VITPOSE_RESULTS_DIR, _C.SEQUENCE_NAME)
    dense_vitpose_results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)
    smplifyx_results_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, _C.SEQUENCE_NAME, "results.pkl")
    bbox_removal_pth = os.path.join(bbox_dir, "bbox_removal.json")
    large_regularization_json_pth = os.path.join(os.path.dirname(smplifyx_results_pth), "high_regularization_frames.json")
    os.makedirs(os.path.dirname(smplifyx_results_pth), exist_ok=True)
    
    working_dir = _C.SMPLIFYX_WORKING_DIR
    run_cmd = [
        "python", "-u", 
        "-m", "scripts.fbcontact.run_fitting", 
        "--vitpose_results_dir", vitpose_results_dir, 
        "--dense_vitpose_results_dir", dense_vitpose_results_dir, 
        "--calib_path", calib_pth,
        "--bbox_dir", bbox_dir,
        "--output_results_pth", smplifyx_results_pth,
        "--gender", gender,
        "--vitpose_ignore_json_pth", bbox_removal_pth,
        "--large_regularization_json_pth", large_regularization_json_pth,
    ]

    run_command_with_conda(working_dir, conda_env, run_cmd)