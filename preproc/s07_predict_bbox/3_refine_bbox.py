import os
import sys
sys.path.append('./')
import torch
import argparse
import commentjson as json

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from preproc import config as _C
from utils.subprocess_utils import run_command_with_conda
from vis.utils.functions import visualize_bbox
from preproc.s07_predict_bbox.generate_video import main as make_video

def run_samurai(model, frames, camera, image_dir, bbox_dir):
    # Get init bbox from YOLO
    start = frames[0]
    end = frames[-1]
    
    prompt = None
    while prompt is None:
        prompt = run_yolo(model, start, camera, image_dir, bbox_dir, save=False)
        start += 1
    start -= 1
    
    with open(f"/home/{_C.username}/.cache/bbox.txt", "w") as fopen:
        fopen.write(','.join([str(int(x)) for x in prompt]))
    
    working_dir = _C.SAMURAI_WORKING_DIR
    results_bbox_dir = os.path.join(bbox_dir, camera, "bbox")
    results_vis_dir = os.path.join(bbox_dir, camera, "vis")
    os.makedirs(results_bbox_dir, exist_ok=True)
    os.makedirs(results_vis_dir, exist_ok=True)
    
    _image_dir = os.path.join(image_dir, camera)
    cmd = ["python", "-u",
            "-m", "scripts.fix_wrong_frames", 
            "--video_path", _image_dir, 
            "--model_path", _C.SAMURAI_MODEL_CKPT, 
            "--image_output_dir", results_vis_dir, 
            "--bbox_output_dir", results_bbox_dir, 
            "--txt_path", f"/home/{_C.username}/.cache/bbox.txt", 
            "--start", f"{start}", 
            "--end", f"{end}", 
            "--save_to_image"
            ]
    
    run_command_with_conda(working_dir, "samurai", cmd)


def run_yolo(model, frame, camera, image_dir, bbox_dir, save=True):
    image_pth = os.path.join(image_dir, camera, f"{frame+1:05d}.jpg")
    image = cv2.imread(image_pth)
    
    bboxes = model.predict(
        image, device=device, classes=0, conf=0.5, save=False, verbose=False
    )[0].boxes

    _bboxes = []
    vis_image = image.copy()
    for i, bbox in enumerate(bboxes):
        xyxy = bbox[0].xyxy.detach().cpu().numpy()
        cxywh = bbox[0].xywh.detach().cpu().numpy()
        _bbox = np.zeros_like(xyxy)
        _bbox[..., :2] = xyxy[..., :2]
        _bbox[..., 2:] = cxywh[..., 2:]
        _bboxes.append(_bbox)
        if i == 0: 
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        vis_image = visualize_bbox(_bbox, vis_image.copy(), color, bbox_id=i + 1)

    cv2.imshow('Bbox ID', cv2.resize(vis_image, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Close the window

    user_input = input("Target bbox ID: ")
    if len(user_input) == 1:
        bbox_id = int(user_input)
        if bbox_id <= 0:
            print(f"User input is wrong! Maybe YOLO fails. Discard this frame!")
            bbox_pth = os.path.join(bbox_dir, camera, "bbox", f"{frame + 1:05d}.npy")
            np.save(bbox_pth, np.ones(4) * (-1))
            return None

        target_bbox = _bboxes[bbox_id - 1][0]

    else:
        bboxes = [int(i) for i in user_input.split(" ")]
        target_bbox = np.array(bboxes)
    
    if save:
        bbox_pth = os.path.join(bbox_dir, camera, "bbox", f"{frame + 1:05d}.npy")
        np.save(bbox_pth, target_bbox)
    
    return target_bbox

def remove_frames(_dict, bbox_dir):
    _updated_frames = []
    for camera, frames in _dict.items():
        if len(frames) == 0:
            continue
            
        for frame in frames:
            if len(frame.split("-")) == 2:
                start, end = [int(f) for f in frame.split("-")]
                frames_to_remove = list(range(start, end+1))
            else:
                frames_to_remove = [int(frame)]
            
            _updated_frames += frames_to_remove
            for frame_to_remove in frames_to_remove:
                bbox_pth = os.path.join(bbox_dir, camera, "bbox", f"{frame_to_remove + 1:05d}.npy")
                new_bbox = np.array([-1., -1., -1., -1.])
                np.save(bbox_pth, new_bbox)
    return _updated_frames

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    # Load bbox refinement json
    json_pth = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, "bbox_correction.json")
    with open(json_pth, "r", encoding="utf-8") as fopen:
        correction_dict = json.load(fopen)

    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    image_dir = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME)
    # Remove frames
    updated_frames = remove_frames(correction_dict["remove"], bbox_dir)
    
    # Refine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(_C.YOLO_MODEL_CKPT)
    for camera, frames in correction_dict["update"].items():
        if len(frames) == 0:
            continue

        for frame in frames:
            if len(frame.split("-")) == 2:
                start, end = [int(f) for f in frame.split("-")]
                frames_to_update = list(range(start, end+1))
                
                if end - start > 10:
                    # Run SAMURAI
                    run_samurai(yolo, frames_to_update, camera, image_dir, bbox_dir)
                    updated_frames += frames_to_update
                    continue

            else:
                frames_to_update = [int(frame)]

            updated_frames += frames_to_update

            for frame_to_update in frames_to_update:
                run_yolo(yolo, frame_to_update, camera, image_dir, bbox_dir)

    make_video(args, updated_frames)