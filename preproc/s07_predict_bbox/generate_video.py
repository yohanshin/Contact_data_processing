import os
import sys
sys.path.append('./')
sys.path.append(os.getcwd())
import glob
import json
import argparse

import cv2
import numpy as np
from tqdm import trange, tqdm
from preproc import config as _C

from utils.pose_utils import _xywh2cs
from utils.subprocess_utils import image_to_video
from preproc.s07_predict_bbox.util import create_camera_grid

from vis import get_affine_transform

def collect_common_frame_ids(vis_dirs):
    frame_sets = []
    for vis_dir in vis_dirs:
        frames = {os.path.splitext(f)[0] for f in os.listdir(vis_dir) if f.endswith(".jpg")}
        frame_sets.append(frames)
    return sorted(set.intersection(*frame_sets))

def build_grid_image_from_vis(frame_id, vis_dirs):
    images = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)
    thickness = 2

    # Text position within each panel (camera name)
    cam_label_pos = (10, 25)

    for cam_idx, vis_dir in enumerate(vis_dirs):
        cam_name = _C.CAMERA_NAMES[cam_idx]
        img_path = os.path.join(vis_dir, f"{frame_id}.jpg")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = np.zeros((256, 192, 3), dtype=np.uint8)

        # Draw camera name in each tile
        cv2.putText(img, f"{cam_name}", cam_label_pos, font, font_scale, font_color, thickness, cv2.LINE_AA)
        images.append(img)

    # # Create grid (2x3)
    # top = cv2.hconcat(images[:3])
    # bottom = cv2.hconcat(images[3:])
    # grid = cv2.vconcat([top, bottom])
    
    # Create grid (3x2)
    row1 = cv2.hconcat(images[0:2])
    row2 = cv2.hconcat(images[2:4])
    row3 = cv2.hconcat(images[4:6])
    grid = cv2.vconcat([row1, row2, row3])
    
    # rows, cols = (2, 3)
    # height, width = (256, 192)
    
    # top_padding = 40  # Space for frame number at top
    
    # # Create blank canvas for the grid
    # grid_height = top_padding + rows * height
    # grid_width = cols * width
    # grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Draw frame number once on the full grid (top-left corner)
    frame_label_pos = (10, 30)
    cv2.putText(grid, f"Frame {frame_id}", frame_label_pos, font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return grid


def main(args, target_frames=None):
    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    USE_ANNOTATED_VIS = False  # 🔁 Toggle this flag to use annotated images

    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)

    if USE_ANNOTATED_VIS:
        # ========== Annotated (vis) path ========== #
        camera_vis_dirs = [os.path.join(bbox_dir, cam, "vis_all") for cam in _C.CAMERA_NAMES]
        frame_ids = collect_common_frame_ids(camera_vis_dirs)
        results_image_dir = os.path.join(bbox_dir, 'ann_vis_grid')
        results_video_pth = os.path.join(bbox_dir, 'ann_vis_grid.avi')
        os.makedirs(results_image_dir, exist_ok=True)

        for frame_id in tqdm(frame_ids, desc="Generating vis grid"):
            grid_img = build_grid_image_from_vis(frame_id, camera_vis_dirs)
            cv2.imwrite(os.path.join(results_image_dir, f"{frame_id}.jpg"), grid_img)

        image_to_video(results_image_dir, results_video_pth, framerate=60)
        return

    # ========== Original bbox + crop pipeline ========== #
    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    bboxes = []
    for camera in _C.CAMERA_NAMES:
        bbox_pth_list = sorted(glob.glob(os.path.join(bbox_dir, camera, 'bbox/*.npy')))
        if camera not in proc_info["sync_frame"]:
            bboxes.append(np.ones_like(bboxes[-1]) * (-1))    
            continue
        bboxes.append(np.stack([np.load(bbox_pth) for bbox_pth in bbox_pth_list]))

    max_len = max([_bbox.shape[0] for _bbox in bboxes])
    for bbox_i, _bbox in enumerate(bboxes):
        if _bbox.shape[0] < max_len:
            _bbox = np.concatenate((_bbox, np.ones((max_len - _bbox.shape[0], 4)) * (-1)))
            bboxes[bbox_i] = _bbox.copy()

    bboxes = np.stack(bboxes, axis=1)
    n_frames = bboxes.shape[0]
    results_image_dir = os.path.join(bbox_dir, 'grid_vis')
    results_video_pth = os.path.join(bbox_dir, 'grid_vis.avi')
    os.makedirs(results_image_dir, exist_ok=True)

    pad = np.zeros((256, 192, 3), dtype=np.uint8)
    for frame in trange(n_frames, leave=False, dynamic_ncols=True):
        if target_frames is not None and frame not in target_frames:
            continue

        frame_image = []
        for i, camera in enumerate(_C.CAMERA_NAMES):
            bbox = bboxes[frame, i].copy()
            image_pth = os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, camera, f"{frame + 1:05d}.jpg")

            if np.all(bbox <= 0) or not os.path.exists(image_pth):
                frame_image.append(pad)
                continue

            image = cv2.imread(image_pth)
            center, trans = _xywh2cs(bbox)
            trans = get_affine_transform(center, trans, 0.0, (192, 256))
            cropped_image = cv2.warpAffine(
                image.copy(),
                trans,
                (192, 256),
                flags=cv2.INTER_LINEAR
            )
            frame_image.append(cropped_image)

        grid_image = create_camera_grid(frame_image, frame, cam_names=_C.CAMERA_NAMES)
        cv2.imwrite(os.path.join(results_image_dir, f'{frame + 1:05d}.jpg'), grid_image)

    image_to_video(results_image_dir, results_video_pth, framerate=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    main(args)
