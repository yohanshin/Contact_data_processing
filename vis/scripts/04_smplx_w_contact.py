import os
import sys
sys.path.append('./')
import glob
import json
import argparse
import subprocess

import cv2
import torch
import joblib
import numpy as np
from smplx import SMPLX

from preproc import config as _C
from vis.animation import render_scene_list, renderables

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-g', '--gender', default='neutral')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    smplifyx_results_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, _C.SEQUENCE_NAME, "results.pkl")
    results = joblib.load(smplifyx_results_pth)
    kpts3d = results['joints3d']

    # aitviewer params
    to_tensor = lambda x: torch.from_numpy(x).float()
    kwargs = {
        'poses_root': to_tensor(results['global_orient']),
        'poses_body': to_tensor(results['body_pose']),
        'betas': to_tensor(results['betas']),
        'trans': to_tensor(results['transl']),
    }

    smplx = SMPLX(os.path.join(_C.BODY_MODEL_DIR, "smplx"), 
                  num_betas=len(results['betas'][0]), 
                  gender=args.gender,
                #   gender='neutral'/
    )

    # Read contact labels
    contact_labels = np.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "refined_contact.npy"))
    contact_labels = contact_labels[:results["global_orient"].shape[0]]
    colors = np.ones((results["global_orient"].shape[0], smplx.faces.max() + 1, 4))
    colors[..., :4] = colors * 0.75

    # Load segment colors
    segm_verts_dict = json.load(open("/home/soyongs/Data/body_models/smplx_vert_segmentation_v6.json", "r"))
    
    for key, val in _C.SENSOR_NAME_MAPPER.items():
        contact_idx = _C.SENSOR_NAME_LIST.index(key)
        verts_idxs = segm_verts_dict[val]

        indices = np.ix_(contact_labels[:, contact_idx], verts_idxs)
        colors[indices] = np.array([1.0, 0.5, 0.5, 0.75])

    scene_list = []
    scene_list.append(renderables.addSMPLSequence('smplx', z_up=True, color=[0.65, 0.65, 0.65, 0.8], bm=smplx, vertex_colors=colors, **kwargs))
    # scene_list.append(renderables.addKeypointsSequence(kpts3d[..., :3], z_up=True, color=(0.0, 1.0, 0.5, 0.7)))
    render_scene_list(scene_list)