import os
import sys
sys.path.append('./')
import subprocess
from preproc import config as _C

def run_command(command):
    """Run a command and return its output."""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error message: {stderr.decode('utf-8')}")
        return False
    
    return True

def run_colmap_pipeline():
    """Run the complete COLMAP pipeline based on configuration."""
    # Define directories
    COLMAP_IMAGES_DIR = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "images")
    COLMAP_WORKSPACE_DIR = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "workplace")
    os.makedirs(COLMAP_WORKSPACE_DIR, exist_ok=True)
    
    # Define paths for different outputs
    COLMAP_DATABASE_PATH = os.path.join(COLMAP_WORKSPACE_DIR, "temp.db")
    COLMAP_SPARSE_DIR = os.path.join(COLMAP_WORKSPACE_DIR, "sparse")
    os.makedirs(COLMAP_SPARSE_DIR, exist_ok=True)
    
    # Step 1: Create database
    print("Step 1: Creating COLMAP database...")
    cmd = f"colmap database_creator --database_path {COLMAP_DATABASE_PATH}"
    if not run_command(cmd):
        return False
    
    # Step 2: Feature extraction with OPENCV_FISHEYE and shared per sub-folder
    print("Step 2: Extracting features...")
    cmd = f"colmap feature_extractor \
        --database_path {COLMAP_DATABASE_PATH} \
        --image_path {COLMAP_IMAGES_DIR} \
        --ImageReader.camera_model OPENCV_FISHEYE \
        --ImageReader.single_camera_per_folder 1"
    if not run_command(cmd):
        return False
    
    # Step 3: Feature matching
    print("Step 3: Matching features...")
    cmd = f"colmap exhaustive_matcher --database_path {COLMAP_DATABASE_PATH}"
    if not run_command(cmd):
        return False
    
    # Step 4: Sparse reconstruction
    print("Step 4: Running sparse reconstruction...")
    cmd = f"colmap mapper \
        --database_path {COLMAP_DATABASE_PATH} \
        --image_path {COLMAP_IMAGES_DIR} \
        --output_path {COLMAP_SPARSE_DIR} \
        --Mapper.num_threads {os.cpu_count()}"
    if not run_command(cmd):
        return False
    
    print("COLMAP pipeline completed successfully!")
    return True
    

if __name__ == '__main__':
    COLMAP_CONDA_ENV = "colmap"
    # Check if COLMAP is available
    try:
        subprocess.check_output(["colmap", "-h"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("COLMAP not found. Make sure COLMAP is installed and in your PATH.")
        print(f"Consider activating the conda environment: {COLMAP_CONDA_ENV}")
        sys.exit(1)
        
    # Run the pipeline
    success = run_colmap_pipeline()
    
    if success:
        print(f"COLMAP processing complete. Results stored in: {os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, 'workplace')}")
    else:
        print("COLMAP processing failed. Check the error messages above.")