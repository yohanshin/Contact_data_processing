import os
import sys
sys.path.append("./")
import glob

import cv2
import torch
import joblib
import numpy as np
from smplx import SMPL, SMPLX
from tqdm import tqdm

from preproc import config as _C
from utils import rotation as r
from vis.renderer import Renderer

SMPL_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smpl"
SMPLX_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smplx"

if __name__ == "__main__":

    # Example sequence
    skip = 1
    # sequence = "08_soyongs_highland-park_01"
    # sequence = "24_holly_tepper_01"
    protocol = "baseline"
    sequence = "25_holly_outdoor_01"
    camera = "cam01"
    # model = "camerahmr"
    # model = "hmr2"
    model = "tokenhmr"
    model = "gvhmr"
    
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, sequence, "calib.npz")
    image_pth_list = sorted(
        glob.glob(os.path.join(_C.PROC_IMAGE_DIR, sequence, camera, "*.jpg"))
    )
    results_pth = os.path.join(_C.BENCHMARK_3DHP_RESULT_DIR, protocol, model, sequence, f'{camera}.pkl')
    
    data = joblib.load(results_pth)
    if "transl_full" in data.keys():
        transl = data["transl_full"].copy()
    else:
        transl = data["transl"].copy()
    
    if model in ["bedlam-cliff", "gvhmr"]:
        body_model = SMPLX(SMPLX_MODEL_DIR, 
                           num_betas=data["betas"].shape[-1], 
                           batch_size=data["betas"].shape[0]).eval()
    else:
        body_model = SMPL(SMPL_MODEL_DIR, ).eval()
    pred_params = dict(
        global_orient=torch.from_numpy(data["global_orient"]).float(),
        body_pose=torch.from_numpy(data["body_pose"]).float(),
        betas=torch.from_numpy(data["betas"]).float(),
        transl=torch.from_numpy(transl).float(),
    )

    with torch.no_grad():
        pred_body = body_model(
            **pred_params
        )
    
    import imageio
    # writer = imageio.get_writer("camerahmr.mp4", fps=30)
    writer = imageio.get_writer(f"{model}.mp4", fps=60)
    image_pth_list = image_pth_list[data["frame_ids"]]
    
    for trg_frame in tqdm(range(len(image_pth_list))):
        # trg_frame = 650
        if "pred_focal_length" in data:
            pred_focal_length = data['pred_focal_length'][trg_frame]
        else:
            pred_focal_length = 5000.0 / 256 * 3840
        
        image = cv2.imread(image_pth_list[trg_frame])
        height, width = image.shape[:2]
        renderer = Renderer(width, height, focal_length=pred_focal_length, device="cuda", faces=body_model.faces)

        out_image = renderer.render_mesh(pred_body.vertices[trg_frame].cuda(), image)
        writer.append_data(cv2.resize(out_image, None, fx=0.25, fy=0.25)[..., ::-1])
    
    writer.close()
    import pdb; pdb.set_trace()
    # # Example sequence for our GT
    # sequence = "08_soyongs_highland-park_01"
    # camera = "cam01"
    # gender="male"
    # skip = 5
    # calib_pth = os.path.join(_C.PROC_CALIB_DIR, sequence, "calib.npz")
    # image_pth_list = sorted(
    #     glob.glob(os.path.join(_C.PROC_IMAGE_DIR, sequence, camera, "*.jpg"))
    # )[::skip]
    # results_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, sequence, 'results.pkl')
    # groundtruth = joblib.load(results_pth)

    # body_model = SMPLX(SMPLX_MODEL_DIR, gender=gender, 
    #                    num_betas=11, 
    #                    batch_size=groundtruth["betas"][::skip].shape[0]).eval()
    # calibs = dict(np.load(calib_pth))
    
    # # Convert world-coordinate GT to camera coordinate
    # R = torch.from_numpy(calibs["Rs"][int(camera[-2:]) - 1]).unsqueeze(0).float()
    # R = R / R.norm(dim=-1)
    # T = torch.from_numpy(calibs["Ts"][int(camera[-2:]) - 1]).unsqueeze(0).float()
    
    # gt_global_orient_world = torch.from_numpy(groundtruth["global_orient"]).float().clone()
    # gt_rotmat_world = r.axis_angle_to_matrix(gt_global_orient_world)
    # gt_rotmat_cam = R @ gt_rotmat_world
    # gt_global_orient_cam = r.matrix_to_axis_angle(gt_rotmat_cam)
    
    # gt_params_world = dict(
    #     global_orient=gt_global_orient_world[::skip],
    #     body_pose=torch.from_numpy(groundtruth["body_pose"]).float()[::skip],
    #     betas=torch.from_numpy(groundtruth["betas"]).float()[::skip],
    #     transl=torch.from_numpy(groundtruth["transl"]).float()[::skip],
    # )

    # gt_params_cam = dict(
    #     global_orient=gt_global_orient_cam[::skip],
    #     body_pose=torch.from_numpy(groundtruth["body_pose"]).float()[::skip],
    #     betas=torch.from_numpy(groundtruth["betas"]).float()[::skip],
    # )
    # with torch.no_grad():
    #     gt_body_world = body_model(
    #         **gt_params_world
    #     )

    #     gt_body_cam = body_model(
    #         **gt_params_cam
    #     )

    #     gt_verts_world = gt_body_world.vertices
    #     gt_verts_cam_wo_trans = gt_body_cam.vertices
    
    # gt_verts_cam_w_trans = (R @ gt_verts_world.mT + T.unsqueeze(-1)).mT
    # trans_cam = gt_verts_cam_w_trans - gt_verts_cam_wo_trans