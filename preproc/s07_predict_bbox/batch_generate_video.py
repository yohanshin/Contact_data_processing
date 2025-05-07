import os
import sys
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('./')
sys.path.append(os.getcwd())

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
    cam_label_pos = (10, 25)

    for cam_idx, vis_dir in enumerate(vis_dirs):
        cam_name = _C.CAMERA_NAMES[cam_idx]
        img_path = os.path.join(vis_dir, f"{frame_id}.jpg")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = np.zeros((256, 192, 3), dtype=np.uint8)

        cv2.putText(img, f"{cam_name}", cam_label_pos, font, font_scale, font_color, thickness, cv2.LINE_AA)
        images.append(img)

    # Build grid (3x2 layout)
    row1 = cv2.hconcat(images[0:2])
    row2 = cv2.hconcat(images[2:4])
    row3 = cv2.hconcat(images[4:6])
    grid = cv2.vconcat([row1, row2, row3])

    frame_label_pos = (10, 30)
    cv2.putText(grid, f"Frame {frame_id}", frame_label_pos, font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return grid

def all_cameras_have_equal_images(cam_dirs):
    image_counts = []
    for cam_dir in cam_dirs:
        vis_dir = os.path.join(cam_dir, "vis_all")
        if not os.path.isdir(vis_dir):
            return False
        count = len([f for f in os.listdir(vis_dir) if f.endswith(".jpg")])
        image_counts.append(count)
    print("image_counts", image_counts)

    return len(set(image_counts)) == 1 and image_counts[0] > 0

def run_single_sequence():
    bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, _C.SEQUENCE_NAME)
    camera_vis_dirs = [os.path.join(bbox_dir, cam, "vis_all") for cam in _C.CAMERA_NAMES]

    frame_ids = collect_common_frame_ids(camera_vis_dirs)
    results_image_dir = os.path.join(bbox_dir, 'ann_vis_grid')
    results_video_pth = os.path.join(bbox_dir, 'ann_vis_grid.avi')
    os.makedirs(results_image_dir, exist_ok=True)

    for frame_id in tqdm(frame_ids, desc=f"Generating vis grid for {os.path.basename(bbox_dir)}"):
        grid_img = build_grid_image_from_vis(frame_id, camera_vis_dirs)
        cv2.imwrite(os.path.join(results_image_dir, f"{frame_id}.jpg"), grid_img)

    image_to_video(results_image_dir, results_video_pth, framerate=60)

def main(args):
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted([
            seq.name for seq in Path(_C.SAMURAI_RESULTS_DIR).iterdir()
            if seq.is_dir()
        ])

    print(f"Checking {len(sequences)} sequences...")

    for sequence in sequences:
        _C.SEQUENCE_NAME = sequence
        bbox_dir = os.path.join(_C.SAMURAI_RESULTS_DIR, sequence)
        cam_dirs = [os.path.join(bbox_dir, cam) for cam in _C.CAMERA_NAMES]

        if not all_cameras_have_equal_images(cam_dirs):
            print(f"⏭ Skipping {sequence}: cameras do not have equal number of images")
            continue

        print(f"\n▶ Processing sequence: {sequence}")
        try:
            run_single_sequence()
        except Exception as e:
            print(f"❌ Failed on {sequence}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--sequences',
        nargs='+',
        default=[],
        help='List of sequences to process (e.g. -s seq1 seq2)'
    )
    args = parser.parse_args()
    main(args)
