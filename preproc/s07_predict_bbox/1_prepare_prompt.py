import os
import sys
sys.path.append('./')
import json
import argparse
from collections import defaultdict

import cv2
import torch
import numpy as np
from preproc import config as _C
from ultralytics import YOLO

from vis.utils.functions import visualize_bbox


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(_C.YOLO_MODEL_CKPT)
    
    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    results_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME, "prompt")
    os.makedirs(results_dir, exist_ok=True)
    
    for camera in _C.CAMERA_NAMES:
        if camera not in proc_info["sync_frame"]:
            continue
        
        if "valid_init_frames" in proc_info:
            if camera in proc_info["valid_init_frames"]:
                frame = f"{int(proc_info['valid_init_frames'][camera]):05d}.jpg"
            else:
                frame = '00001.jpg'
        else:
            frame = '00001.jpg'
        
        image_pth = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera, frame)
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
        bbox_id = int(user_input)

        target_bbox = _bboxes[bbox_id - 1]
        
        try:
            # Save txt
            with open(os.path.join(results_dir, f'{camera}.txt'), "w") as fopen:
                fopen.write(','.join([str(int(x)) for x in target_bbox[0]]))
        except: 
            pass