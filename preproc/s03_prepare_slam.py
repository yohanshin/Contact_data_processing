import os
import sys
sys.path.append('./')
import glob
import json
import argparse
from preproc import config as _C

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    slam_frames = proc_info["slam_frames"]
    colmap_images_dir = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "images")
    os.makedirs(colmap_images_dir, exist_ok=True)
    
    for camera, info in slam_frames.items():
        camera_images_dir = os.path.join(_C.EXTRACT_IMAGE_DIR, _C.SEQUENCE_NAME, camera)
        camera_images = sorted(glob.glob(os.path.join(_C.EXTRACT_IMAGE_DIR, _C.SEQUENCE_NAME, camera, '*.jpg')))
        camera_colmap_images_dir = os.path.join(colmap_images_dir, camera)
        os.makedirs(camera_colmap_images_dir, exist_ok=True)
        start, end, num_frames = info
        
        skip = (end - start) // num_frames
        source_images = camera_images[start:end][::skip]
        for frame, source_image in enumerate(source_images):
            target_image = os.path.join(camera_colmap_images_dir, os.path.basename(source_image))
            cmd = f"cp {source_image} {target_image}"
            os.system(cmd)
        