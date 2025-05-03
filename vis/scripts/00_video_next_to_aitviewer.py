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
from lib.models.detector.utils.transform import get_affine_transform

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
    parser.add_argument('-c', '--camera', default='cam01')
    parser.add_argument('-a', '--aitviewer_video_pth', default='')
    parser.add_argument('-o', '--output_pth', default='outputs/vis/tmp.mp4')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    bbox_results_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, args.camera)
    bbox_pth_list = sorted(glob.glob(os.path.join(bbox_results_dir, "bbox", "*.npy")))
    
    image_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, args.camera)
    image_pth_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    writer = imageio.get_writer(args.output_pth, fps=30, mode='I', format='FFMPEG', macro_block_size=None)
    cap = cv2.VideoCapture(args.aitviewer_video_pth)
    for bbox_pth, image_pth in tqdm(zip(bbox_pth_list, image_pth_list), total=len(bbox_pth_list), dynamic_ncols=True):
        ret, frame = cap.read()
        if not ret: break

        bbox = np.load(bbox_pth)
        x1, y1, w, h = bbox.astype(int)
        x2, y2 = x1 + w, y1 + h
        if np.all(bbox <= 0): continue
        
        image = cv2.imread(image_pth)
        center, scale = _xywh2cs(bbox)
        trans = get_affine_transform(center, scale, 0.0, (288, 384))
        cropped_image = cv2.warpAffine(
            image.copy(),
            trans,
            (288, 384),
            flags=cv2.INTER_LINEAR
        )
        
        cropped_frame = cv2.resize(frame, (int(frame.shape[1] * 384 / frame.shape[0]), 384))
        image = np.concatenate((cropped_image, cropped_frame), axis=1)
        writer.append_data(image[..., ::-1])

    writer.close()