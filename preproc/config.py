import os
username = os.getenv("USER")
ssd_name = "T9" if "soyong" in username else "T9"

DATA_ROOT_DIR = f"/media/{username}/{ssd_name}/projects/FBContact"
RAW_DATA_DIR = f"{DATA_ROOT_DIR}/raw_data"
EXTRACT_IMAGE_DIR = f"{DATA_ROOT_DIR}/extracted"
PROC_CALIB_DIR = f"{DATA_ROOT_DIR}/calib"
PROC_IMAGE_DIR = f"{DATA_ROOT_DIR}/images"
GRID_IMAGE_DIR = f"{DATA_ROOT_DIR}/inference/samurai/sequence/grid_vis"
PROC_JSON_PTH = f"{DATA_ROOT_DIR}/raw_data/sequence_name/proc.json"
COLMAP_WORKSPACE_DIR = f"{DATA_ROOT_DIR}/calib/sequence_name/workspace"
RESULTS_DIR = f"{DATA_ROOT_DIR}/inference"
SAMURAI_WORKING_DIR = f"/home/{username}/Codes/srcs/samurai"
SAMURAI_RESULTS_DIR = f"{RESULTS_DIR}/samurai"
SAMURAI_MODEL_CKPT = f"/home/{username}/Data/checkpoints/sam2.1_hiera_base_plus.pt"
YOLO_RESULTS_DIR = f"{RESULTS_DIR}/yolo"
YOLO_MODEL_CKPT = f"/home/{username}/Data/checkpoints/yolov8x.pt"
LANDMARK_WORKING_DIR = f"/home/{username}/Codes/projects/DeepGaitLab_beta"
VITPOSE_WORKING_DIR = f"/home/{username}/Codes/srcs/ViTPose"
SMPLIFYX_WORKING_DIR = f"/home/{username}/Codes/projects/DenseSMPLify"
LANDMARK_RESULTS_DIR = f"{RESULTS_DIR}/dense_vitpose"
VITPOSE_RESULTS_DIR = f"{RESULTS_DIR}/vitpose"
SMPLIFYX_RESULTS_DIR = f"{RESULTS_DIR}/smplifyx"
PROC_CONTACT_DIR = f"{DATA_ROOT_DIR}/contact"
BODY_MODEL_DIR = f"/home/{username}/Data/body_models"

ARIA_CAM_NAME = "aria01"
SLAM_CAM_NAME = "mobile"

# Check for every sequence
ARIA_FPS = 30
GOPRO_FPS = 60
TARGET_FPS = 30

# Frame-dependent parts
SEQUENCE_NAME = "11_titus_tepper_01"
# SEQUENCE_NAME = "09_soyongs_highland-park_02"
CAMERA_NAMES = ["cam01", "cam02", "cam03", "cam04", "cam05", "cam06"]

SENSOR_NAME_LIST = ["LHip", "RHip", "LThigh", "RThigh", "LKnee", "RKnee", 
                    "LShank", "RShank", "LHeel", "RHeel", "LToe", "RToe",
                    "Back", "Spine", "LShoulder", "RShoulder", "LBicep", "RBicep", 
                    "LElbow", "RElbow", "LForearm", "RForearm", "LHand", "RHand"]
SENSOR_NAME_MAPPER = {
    "LHip": "leftHip", 
    "RHip": "rightHip", 
    "LThigh": "leftThigh",
    "RThigh": "rightThigh",
    "LKnee": "leftKnee", 
    "RKnee": "rightKnee", 
    "LShank": "leftShank",
    "RShank": "rightShank",
    "LHeel": "leftFoot",
    "RHeel": "rightFoot",
    "LToe": "leftToeBase",
    "RToe": "rightToeBase",
    "Back": "lspine",
    "Spine": "uspine",
    "LShoulder": "leftShoulder",
    "RShoulder": "rightShoulder",
    "LBicep": "leftBicep",
    "RBicep": "rightBicep",
    "LElbow": "leftElbow",
    "RElbow": "rightElbow",
    "LForearm": "leftForeArm",
    "RForearm": "rightForeArm",
    "LHand": "leftHand",
    "RHand": "rightHand",
}