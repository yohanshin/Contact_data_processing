import os
import sys
sys.path.append('./')

import json
import glob
import argparse
import numpy as np

from preproc import config as _C
from utils.contact_utils import parse_arduino_data
from preproc.s12_process_contact.generate_video import main as get_video


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

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
    contact_binary = np.zeros((proc_sensor_readings.shape[0], len(_C.SENSOR_NAME_MAPPER))).astype(bool)
    target_segment_names = [k for k in _C.SENSOR_NAME_MAPPER.keys()]
    
    for segm_idx, segm_name in enumerate(proc_info["sensor_names"]):
        thr = int(proc_info["sensor_threshold"][segm_idx])
        binary = proc_sensor_readings[:, segm_idx] > thr
        contact_binary[:, target_segment_names.index(segm_name)] = np.logical_or(contact_binary[:, target_segment_names.index(segm_name)], binary)

    os.makedirs(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME), exist_ok=True)
    np.save(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "init_contact.npy"), contact_binary)

    get_video(contact_binary)