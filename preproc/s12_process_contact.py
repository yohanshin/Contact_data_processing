import os
import sys
sys.path.append('./')

import cv2
import json
import glob
import joblib
import subprocess
import numpy as np
from tqdm import tqdm

from preproc import config as _C
from utils.contact_utils import parse_arduino_data
from utils.subprocess_utils import image_to_video

def overlay_sensor_readings(image, sensor_names, sensor_readings, output_path=None):
    """
    Overlay sensor readings on an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image.
    sensor_names : list
        List of sensor names.
    sensor_readings : numpy.ndarray
        Array of sensor readings (integers).
    output_path : str, optional
        Path to save the output image. If None, the image will be displayed.
    
    Returns:
    --------
    numpy.ndarray
        Image with sensor readings overlaid.
    """
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define text properties with increased font size (50% larger)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Increased from 1.0 to 1.5 (50% larger)
    font_thickness = 3  # Increased from 2 to 3 for better visibility
    font_color = (255, 255, 255)  # White
    background_color = (0, 0, 0)   # Black
    
    # Calculate padding and spacing (adjusted for larger font)
    padding = 30  # Increased from 20
    line_spacing = 60  # Increased from 40
    
    # Calculate the starting position for the text (top-left corner)
    start_x = padding
    start_y = padding + 45  # Adjusted for larger font (was 30)
    
    # Create a copy of the image to draw on
    result = image.copy()
    
    # Overlay each sensor reading
    for i, (name, reading) in enumerate(zip(sensor_names, sensor_readings)):
        # Format the text
        text = f"{name}: {reading}"
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate text position
        text_x = start_x
        text_y = start_y + i * line_spacing
        
        # Draw a black rectangle as background for better visibility
        cv2.rectangle(
            result, 
            (text_x - 10, text_y - text_height - 10), 
            (text_x + text_width + 10, text_y + 10), 
            background_color, 
            -1
        )
        
        # Draw the text
        cv2.putText(
            result, 
            text, 
            (text_x, text_y), 
            font, 
            font_scale, 
            font_color, 
            font_thickness
        )
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result


if __name__ == '__main__':
    fps = 29.97
    contact_txt_path = os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, "contact", "DATALOG.TXT")
    sensor_timestamps, sensor_readings = parse_arduino_data(contact_txt_path)

    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)
    sensor_sync_idx = proc_info["contact_sync"]["sensor"]
    camera_sync_idx = proc_info["contact_sync"]["camera"]
    repr_camera = proc_info["repr_camera"]
    sensor_names = proc_info["sensor_names"]

    camera_mspf = 1000.0 / fps
    num_camera_frames = len(
        glob.glob(os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, _C.CAMERA_NAMES[0], "*.jpg"))
    )
    camera_timestamps = np.arange(num_camera_frames) * camera_mspf
    offset = sensor_timestamps[sensor_sync_idx] - camera_timestamps[camera_sync_idx]
    camera_timestamps = camera_timestamps + offset

    proc_sensor_readings = np.zeros((num_camera_frames, sensor_readings.shape[1]))
    for sensor_i in range(sensor_readings.shape[1]):
        proc_sensor_readings[:, sensor_i] = np.interp(
            camera_timestamps, sensor_timestamps, sensor_readings[:, sensor_i]
        )

    proc_sensor_readings = proc_sensor_readings.astype(np.int64)
    os.makedirs(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME), exist_ok=True)
    np.save(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "contact.npy"), proc_sensor_readings)

    if "sensor_threshold" in proc_info.keys():
        thresholds = proc_info["sensor_threshold"]
        sensor_thresholds = {k: int(v) for k, v in zip(sensor_names, thresholds)}
        joblib.dump(sensor_thresholds, os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "threshold.pkl"))

    # # Create video
    # image_path_list = sorted(glob.glob(
    #     os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, repr_camera, "*.jpg")
    # ))
    # image_path_list = sorted(image_path_list)
    # proc_sensor_vis_dir = os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "vis")
    # os.makedirs(proc_sensor_vis_dir, exist_ok=True)

    # max_frames = 9999
    # for frame_i, image_path in tqdm(enumerate(image_path_list), total=len(image_path_list), leave=False, dynamic_ncols=True):
    #     output_image_path = os.path.join(proc_sensor_vis_dir, os.path.basename(image_path))
    #     image = cv2.imread(image_path)
    #     overlay_sensor_readings(image, sensor_names, proc_sensor_readings[frame_i], output_image_path)
    #     if frame_i > max_frames: break

    # image_to_video(proc_sensor_vis_dir, os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "video.avi"))