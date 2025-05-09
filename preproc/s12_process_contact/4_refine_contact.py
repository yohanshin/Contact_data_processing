import os
import sys
sys.path.append('./')

import commentjson as json
import glob
import argparse
import numpy as np

from preproc import config as _C
from preproc.s12_process_contact.generate_video import main as get_video


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    contact_dir = os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME)
    history_dir = os.path.join(contact_dir, "refinement_history")
    os.makedirs(history_dir, exist_ok=True)

    updated_contact_pth = os.path.join(contact_dir, "refined_contact.npy")
    if os.path.exists(updated_contact_pth):
        n_history = len([f for f in os.listdir(history_dir) if f.endswith(".npy")])
        # npy_with_version = f"version_{n_history:03d}.npy"
        # cmd = f"cp {updated_contact_pth} {os.path.join(history_dir, npy_with_version)}"
        # os.system(cmd)

        contact_values = np.load(updated_contact_pth)

    else:
        n_history = 0
        # First time in the refinement
        contact_values = np.load(os.path.join(contact_dir, "init_contact.npy"))

    # Read updates
    json_pth = os.path.join(contact_dir, "refinement.json")
    with open(json_pth, "r", encoding="utf-8") as fopen:
        correction_dict = json.load(fopen)
    
    # Add first
    segm_names = _C.SENSOR_NAME_LIST
    for segm_name, frames in correction_dict["add"].items():
        segm_idx = segm_names.index(segm_name)
        for frame in frames:
            if len(frame.split("-")) == 2:
                start, end = [int(f) for f in frame.split("-")]
                frames_to_update = list(range(start, end+1))

            else:
                frames_to_update = [int(frame)]

            contact_values[frames_to_update, segm_idx] = True

    # Remove later
    segm_names = _C.SENSOR_NAME_LIST
    for segm_name, frames in correction_dict["remove"].items():
        segm_idx = segm_names.index(segm_name)
        for frame in frames:
            if len(frame.split("-")) == 2:
                start, end = [int(f) for f in frame.split("-")]
                frames_to_update = list(range(start, end+1))

            else:
                frames_to_update = [int(frame)]

            contact_values[frames_to_update, segm_idx] = False

    # Save refined contact labels in history folder
    history_contact_pth = os.path.join(history_dir, f"version_{n_history:03d}.npy")
    history_json_pth = os.path.join(history_dir, f"version_{n_history:03d}.json")
    np.save(history_contact_pth, contact_values)
    json.dump(correction_dict, open(history_json_pth, "w"))
    
    # Save refined contact labels
    np.save(updated_contact_pth, contact_values)

    # Save video
    start = correction_dict.get("start", 0)
    end = correction_dict.get("end", -1)
    get_video(contact_values, start, end)