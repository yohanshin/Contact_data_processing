import os
import sys
sys.path.append('./')

import cv2
import json
import glob
import torch
import joblib
import imageio
import numpy as np
from tqdm import tqdm
from smplx import SMPLX

from preproc import config as _C
from utils import rotation as r
from vis.renderer import Renderer

SMPLX_MODEL_DIR = '/home/soyongs/Data/body_models/smplx'
SMPLX_PART_SEGM_PTH = '/home/soyongs/Data/body_models/smplx_vert_segmentation.json'
SMPLX_PART_SEGM_PTH = '/home/soyongs/Data/body_models/smplx_vert_segmentation_v6.json'

if __name__ == '__main__':

    smplx_part_segm = json.load(open(SMPLX_PART_SEGM_PTH, "rb"))
    smplx = SMPLX(SMPLX_MODEL_DIR).eval()

    global_orient = torch.tensor([[np.pi, 0, 0]]).float()
    verts = smplx(global_orient=global_orient).vertices.squeeze(0)
    verts[..., 2] += 3.0
    verts[..., 1] += 0.2
    default_colors = torch.ones_like(verts).unsqueeze(0) * 0.8

    BG = np.ones((640, 640, 3)).astype(np.uint8) * 35
    renderer = Renderer(640, 640, 825, device="cuda", faces=smplx.faces)
    video = imageio.get_writer("outputs/vis/contact_segm.avi", 
                               fps=10, 
                               format="FFMPEG", 
                               mode="I")

    full_idxs = []
    for segm_name, segm_idxs in smplx_part_segm.items():
        full_idxs += segm_idxs
        colors = default_colors.clone()
        colors[:, segm_idxs] = torch.tensor(([0.5, 0.5, 1.0])).float()
        for i in range(10):
            out_image = renderer.render_mesh(verts.cuda(), BG.copy(), colors)
            cv2.putText(out_image, f"{segm_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video.append_data(out_image[..., ::-1])

    full_colors = default_colors.clone()
    full_colors[:, full_idxs] = torch.tensor(([0.5, 0.5, 1.0])).float()
    for i in range(30):
        out_image = renderer.render_mesh(verts.cuda(), BG.copy(), full_colors)
        video.append_data(out_image[..., ::-1])

    video.close()