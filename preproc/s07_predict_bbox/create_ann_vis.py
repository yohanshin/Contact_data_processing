#!/usr/bin/env python3
"""
Draw bboxes from <cam>/bbox/*.npy onto the corresponding *.jpg frames
and save them to <cam>/vis_all/*.jpg  (leaves existing vis/ untouched).
"""

import os
import sys
sys.path.append('./')
from pathlib import Path
from preproc import config as _C
import numpy as np
import cv2
from tqdm import tqdm       
import re       # optional, gives you a progress bar
# pull these from your config object
SAMURAI_RESULTS_DIR = Path(_C.SAMURAI_RESULTS_DIR)
PROC_IMAGE_DIR      = Path(_C.PROC_IMAGE_DIR)

# ── helper: grab the leading integer in “25_whatever”, “03_xyz”, etc. ──
num_re = re.compile(r"^(\d+)")
def trial_key(trial_name: str) -> int:
    m = num_re.match(trial_name)
    return int(m.group(1)) if m else -1        # trials w/o a number go last

# ── discover & sort trials (25 → 24 → 23 → …) ─────────────────────────
# trials = sorted(
#     (p.name for p in SAMURAI_RESULTS_DIR.iterdir() if p.is_dir()),
#     key=trial_key,
#     reverse=True
# )

# print("Processing order:", trials)
# # ─── CONFIG ───────────────────────────────────────────────────────────
# if not trials:
#     raise RuntimeError(f"No trial directories found in {SAMURAI_RESULTS_DIR}")

# print(f"Found {len(trials)} trials: {', '.join(trials)}")
# exclude = ["22_soyong_outdoor_02", "13_titus_tepper_03"]
# trials = [t for t in trials if t not in exclude]
trials = ["14_vu_indoor_01", "15_vu_outdoor_01"]
print("Processing trials: ", trials)
# ─── MAIN PROCESSING LOOP ---------------------------------------------
for trial in trials:
    cams  = ["cam01", "cam02", "cam03", "cam04", "cam05", "cam06"]

    # drawing parameters
    COLOR      = (0, 255, 0)     # green bbox
    THICKNESS  = 3               # px
    DOWNSCALE  = 0.20             # set to 0.25 to mimic fix_wrong_frames

    # ─── MAIN LOOP ────────────────────────────────────────────────────────
    for cam in cams:
        cam_dir  = Path(SAMURAI_RESULTS_DIR) / trial / cam
        img_dir  = Path(PROC_IMAGE_DIR)      / trial / cam
        bbox_dir = cam_dir / "bbox"
        vis_dir  = cam_dir / "vis_all"
        vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ {cam}: overlaying bboxes …")
        for jpg in tqdm(sorted(img_dir.glob("*.jpg")), unit="frame"):
            bbox_file = bbox_dir / jpg.with_suffix(".npy").name

            if bbox_file.exists():
                try:
                    x, y, w, h = np.load(bbox_file)
                except Exception as e:
                    print(f"  ⚠️  failed to load {bbox_file.name}: {e}")
                    continue
            else:
                x = y = w = h = -1  # Treat missing bbox as invalid

            if (w < 0) or (h < 0):
                # Generate a blank (black) image with same size as input image
                orig_img = cv2.imread(str(jpg))
                if orig_img is None:
                    print(f"  ⚠️  could not read {jpg}")
                    continue
                h_img, w_img = orig_img.shape[:2]
                img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            else:
                img = cv2.imread(str(jpg))
                if img is None:
                    print(f"  ⚠️  could not read {jpg}")
                    continue
                # draw bbox
                cv2.rectangle(
                    img,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    COLOR,
                    THICKNESS
                )

            # optional down-scaling
            if DOWNSCALE != 1.0:
                img = cv2.resize(img, None, fx=DOWNSCALE, fy=DOWNSCALE)
            
            # cv2.imwrite(str(vis_dir / jpg.name), img)
        print(f"{trial}/{cam}:")
        print(f"  JPGs in img_dir:  {len(list(img_dir.glob('*.jpg')))}")
        print(f"  NPYS in bbox_dir: {len(list(bbox_dir.glob('*.npy')))}")
        print(f"  Written to vis:   {len(list(vis_dir.glob('*.jpg')))}")

    print("\n✅ Finished generating vis_all overlays for every camera.")
