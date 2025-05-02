import os
import sys
sys.path.append('./')
import glob
import json
import argparse
from tqdm import tqdm
from preproc import config as _C

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
        
    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    sync_frames = proc_info["sync_frame"]
    start_frame = proc_info["start_frame"]
    end_frame = proc_info["end_frame"]
    
    aria_fps = _C.ARIA_FPS
    camera_fps = _C.GOPRO_FPS
    target_fps = _C.TARGET_FPS
    
    for camera, frame in sync_frames.items():
        print(f"Processing {camera} ...")
        
        if "aria" in camera:
            skip = int(aria_fps // target_fps)
        else:
            skip = int(camera_fps // target_fps)

        sync_pth = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, 'sync', f'{camera}.jpg')
        target_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera)
        os.makedirs(os.path.dirname(sync_pth), exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        image_path_list = sorted(glob.glob(os.path.join(_C.EXTRACT_IMAGE_DIR, _C.SEQUENCE_NAME, camera, '*.jpg')))
        start, end = start_frame // (camera_fps // target_fps), end_frame // (camera_fps // target_fps)
        if start < 0:
            frame = frame + start * skip
            end = end - start
            sync_img_idx = -start * skip
            start = 0
        else:
            sync_img_idx = 0
        image_path_list_from_sync = image_path_list[frame:]
        
        sync_img_path = image_path_list_from_sync[sync_img_idx]
        image_path_list_after_resample = image_path_list_from_sync[::skip]

        image_path_list_after_start = image_path_list_after_resample[start:end]
        
        for frame_i, image_path in tqdm(enumerate(image_path_list_after_start), total=len(image_path_list_after_start), dynamic_ncols=True, leave=False):
            cmd = f"cp {image_path} {os.path.join(target_dir, f'{frame_i+1:05d}.jpg')}"
            os.system(cmd)

        cmd = f"cp {sync_img_path} {sync_pth}"
        os.system(cmd)