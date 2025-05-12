import os
import sys
sys.path.append("./")
import glob

import cv2
import argparse
import imageio
import numpy as np
from tqdm import tqdm

from preproc import config as _C
sys.path.append(f"/home/{_C.username}/Codes/projects/DeepGaitLab_beta")

def _xywh2cs(bbox, pixel_std=200, aspect_ratio=192/256, scale_factor=1.1):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_factor

    return center, scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-a', '--aitviewer_video_pth', default='')
    parser.add_argument('-o', '--output_pth', default='outputs/vis/tmp.mp4')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    grid_video_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, "grid_vis")
    image_pth_list = sorted(glob.glob(os.path.join(grid_video_dir, "*.jpg")))
    
    # image_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, args.camera)
    # image_pth_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    writer = imageio.get_writer(args.output_pth, fps=60, mode='I', format='FFMPEG', macro_block_size=None)
    cap = cv2.VideoCapture(args.aitviewer_video_pth)
    for image_pth in tqdm(image_pth_list, dynamic_ncols=True):
        ret, frame = cap.read()
        if not ret: break

        image = cv2.imread(image_pth)
        h, w = image.shape[:2]
        
        cropped_frame = cv2.resize(frame, (int(frame.shape[1] * w / frame.shape[0]), h))
        image = np.concatenate((image, cropped_frame), axis=1)
        if image.shape[0] % 2 == 1:
            image = image[1:]
        elif image.shape[1] % 2 == 1:
            image = image[:, 1:]
        
        writer.append_data(image[..., ::-1])

    writer.close()