import os
import sys
sys.path.append('./')
import glob
import argparse
import subprocess

from preproc import config as _C
from preproc.vrs.s01_process_vrs_calibration import main as run_vrs_s01
from preproc.vrs.s02_process_data import main as run_vrs_s02


def run_bash_script(conda_env, cmd):
    """
    Run a command using a specific conda environment.
    
    Args:
        conda_env (str): Name of the conda environment to use
        cmd (list or str): Command to run
    
    Returns:
        bool: True if the command was successful, False otherwise
    """
    current_dir = os.getcwd()

    try:
        # Convert list to string if needed
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        
        # Use conda run
        full_cmd = f"conda run -n {conda_env} {cmd_str}"
        
        print(f"Executing command: {full_cmd}")
        
        # Run the command and wait for it to complete
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
        else:
            print(f"Command completed successfully")
            return True
            
    except Exception as e:
        print(f"An error occurred while executing the command: {e}")
        return False
    finally:
        # Always restore the original working directory
        os.chdir(current_dir)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    # Step 0. Copy slam results
    aria_fldr = os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, _C.ARIA_CAM_NAME)
    mps_fldr = os.path.join(
        aria_fldr, 
        [dirname for dirname in os.listdir(aria_fldr) if os.path.isdir(os.path.join(aria_fldr, dirname)) and dirname.startswith("mps")][0])
    slam_fldr = os.path.join(mps_fldr, "slam")
    trajectory_dir = os.path.join(aria_fldr, "trajectory")
    os.system(f"cp -r {slam_fldr} {trajectory_dir}")
    
    # Step 1. Extract vrs files
    extracted_images_dir = os.path.join(aria_fldr, "vrs_images")
    vrs_file = glob.glob(os.path.join(aria_fldr, "*.vrs"))[0]
    extract_cmd = ["vrs", "extract-all", vrs_file, "--to", extracted_images_dir]
    extract_conda_env = "aria"
    run_bash_script(extract_conda_env, extract_cmd)

    # Step 2. Process calibration
    vrs_calib_dir = os.path.join(aria_fldr, "vrs_calib")
    run_vrs_s01(trajectory_dir, extracted_images_dir, vrs_calib_dir)

    # Step 3. 
    target_images_dir = os.path.join(aria_fldr, "images")
    target_calib_dir = os.path.join(aria_fldr, "calib")
    run_vrs_s02(extracted_images_dir, vrs_calib_dir, target_images_dir, target_calib_dir)
    
    # Step 4. Rotate images
    os.system(f"sh ./preproc/vrs/s03_create_rotate_data.sh {os.path.abspath(target_images_dir)} {os.path.abspath('./preproc/vrs')}")