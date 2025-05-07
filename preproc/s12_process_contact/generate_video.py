import os
import sys
sys.path.append('./')

import cv2
import glob
import imageio
import argparse
import numpy as np
from tqdm import tqdm

from preproc import config as _C

# Image dimensions
img_height = 552
img_width = 400

# Create a white background image
padded = np.ones((img_height, img_width, 3), dtype=np.uint8) * 85
segment_names = [v for v in _C.SENSOR_NAME_MAPPER]
# Choose layout
cols = 2
rows = (len(segment_names) + cols - 1) // cols

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 1
padding = 10

# Adjust vertical spacing to fill image height
total_padding = padding * (rows + 1)
available_height = img_height - total_padding
text_height = available_height // rows

def get_contact_annotation_image(contact):

    image = padded.copy()
    
    for i, (name, highlight) in enumerate(zip(segment_names, contact)):
        col = i // rows
        row = i % rows

        x = padding + col * (img_width // cols)
        y = padding + row * (text_height + padding) + text_height

        color = (144, 238, 144) if highlight else (255, 255, 255)

        # Background rectangle
        (text_w, text_h), _ = cv2.getTextSize(name, font, font_scale, thickness)
        rect_y_start = y - text_h - 5
        rect_y_end = y + 5
        cv2.rectangle(image, (x - 5, rect_y_start), (x + text_w + 5, rect_y_end), color, -1)
        cv2.rectangle(image, (x - 5, rect_y_start), (x + text_w + 5, rect_y_end), (0, 0, 0), 1)

        # Text rendering
        cv2.putText(image, name, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image

def main(contact_values, start=0, end=-1):
    # Create video
    image_path_list = sorted(glob.glob(
        os.path.join(_C.GRID_IMAGE_DIR.replace("sequence", _C.SEQUENCE_NAME), "*.jpg")
    ))
    if end == -1:
        end = len(image_path_list)
    image_path_list = image_path_list[start:end]
    contact_values = contact_values[start:end]

    output_vis_pth = os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "vis_contact.mp4")
    writer = imageio.get_writer(output_vis_pth, fps=60, mode="I", format="FFMPEG", macro_block_size=None)

    for image_path, contact_value in tqdm(zip(image_path_list, contact_values), total=len(image_path_list), dynamic_ncols=True, leave=False):
        image = cv2.imread(image_path)
        
        contact_image = get_contact_annotation_image(contact_value)
        image = np.concatenate((image, contact_image), axis=1)
        writer.append_data(image[..., ::-1])

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    if os.path.exists(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "refined_contact.npy")):
        contact_values = np.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "refined_contact.npy"))
    else:
        contact_values = np.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "init_contact.npy"))
    
    main(contact_values, start=0, end=-1)
    
    # # image_path_list = sorted(image_path_list)
    # # proc_sensor_vis_dir = os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "vis")
    # # os.makedirs(proc_sensor_vis_dir, exist_ok=True)

    # # max_frames = 9999
    # # for frame_i, image_path in tqdm(enumerate(image_path_list), total=len(image_path_list), leave=False, dynamic_ncols=True):
    # #     output_image_path = os.path.join(proc_sensor_vis_dir, os.path.basename(image_path))
    # #     image = cv2.imread(image_path)
    # #     overlay_sensor_readings(image, sensor_names, proc_sensor_readings[frame_i], output_image_path)
    # #     if frame_i > max_frames: break

    # # image_to_video(proc_sensor_vis_dir, os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "video.avi"))