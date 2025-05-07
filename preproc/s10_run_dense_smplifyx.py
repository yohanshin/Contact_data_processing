import os
import sys
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
    
    conda_env = "dgl"
    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")
    vitpose_results_dir = os.path.join(_C.VITPOSE_RESULTS_DIR, _C.SEQUENCE_NAME)
    dense_vitpose_results_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)
    smplifyx_results_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, _C.SEQUENCE_NAME, "results.pkl")
    os.makedirs(os.path.dirname(smplifyx_results_pth), exist_ok=True)
    
    working_dir = _C.SMPLIFYX_WORKING_DIR
    run_cmd = [
        "python", "-u", 
        "-m", "scripts.fbcontact.run_fitting", 
        "--vitpose_results_dir", vitpose_results_dir, 
        "--dense_vitpose_results_dir", dense_vitpose_results_dir, 
        "--calib_path", calib_pth,
        "--bbox_dir", bbox_dir,
        "--output_results_pth", smplifyx_results_pth
    ]

    run_command_with_conda(working_dir, conda_env, run_cmd)