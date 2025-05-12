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
from benchmark.B1_3dhp.eval_utils import batch_compute_similarity_transform_torch

SMPL_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smpl"
SMPLX_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smplx"
SMPLX2SMPL_PTH = f"/home/{os.getenv('USER')}/Data/body_models/smplx2smpl.pkl"

def undistort_fisheye_image(image, K, dist, balance=0.0):
    h, w = image.shape[:2]
    
    # Compute new intrinsics for rectified (undistorted) image
    # This matches exactly what you do in do_undistort_fisheye
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, dist, (w, h), np.eye(3), balance=balance
    )
    
    # Initialize the undistortion maps
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, dist, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )
    
    # Apply undistortion
    undistorted_image = cv2.remap(
        image, mapx, mapy, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )
    
    return undistorted_image, new_K

if __name__ == "__main__":

    # Example sequence
    # sequence = "08_soyongs_highland-park_01"
    # sequence = "24_holly_tepper_01"
    protocol = "baseline"
    sequence = "25_holly_outdoor_01"
    gender = "female"
    camera = "cam01"
    model = "camerahmr"
    # model = "hmr2"
    model = "gvhmr"

    smplx2smpl = torch.from_numpy(
        joblib.load(SMPLX2SMPL_PTH)["matrix"]
    ).float()
    
    # Read GT
    results_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, sequence, 'results.pkl')
    groundtruth = joblib.load(results_pth)
    max_frame_id = len(groundtruth["betas"])
    
    image_pth_list = sorted(
        glob.glob(os.path.join(_C.PROC_IMAGE_DIR, sequence, camera, "*.jpg"))
    )
    calib_pth = os.path.join(_C.PROC_CALIB_DIR, sequence, "calib.npz")
    results_pth = os.path.join(_C.BENCHMARK_3DHP_RESULT_DIR, protocol, model, sequence, f'{camera}.pkl')
    data = joblib.load(results_pth)
    frames = data["frame_ids"]
    valid_frames = np.where(frames < max_frame_id)[0]
    valid_frames.sort()
    frames = frames[valid_frames]
    
    if "transl_full" in data.keys():
        transl = data["transl_full"][valid_frames].copy()
    else:
        transl = data["transl"][valid_frames].copy()
    
    if model in ["bedlam-cliff", "gvhmr"]:
        smpl_model = SMPLX(SMPLX_MODEL_DIR, 
                           num_betas=data["betas"].shape[-1], 
                           batch_size=len(valid_frames)).eval()
    else:
        smpl_model = SMPL(SMPL_MODEL_DIR, ).eval()
    
    pred_params = dict(
        global_orient=torch.from_numpy(data["global_orient"][valid_frames]).float(),
        body_pose=torch.from_numpy(data["body_pose"][valid_frames]).float(),
        betas=torch.from_numpy(data["betas"][valid_frames]).float(),
        transl=torch.from_numpy(transl).float(),
    )

    with torch.no_grad():
        pred_body = smpl_model(
            **pred_params
        )


    smplx_model = SMPLX(SMPLX_MODEL_DIR, gender=gender, 
                        num_betas=11, 
                        batch_size=len(frames)).eval()
    
    gt_params_world = {
        k: torch.from_numpy(groundtruth[k]).float()[frames] for k in ["global_orient", "body_pose", "transl", "betas"]
    }
    with torch.no_grad():
        gt_body_world = smplx_model(
            **gt_params_world
        )

    calibs = np.load(calib_pth, allow_pickle=True)
    gt_global_orient_world = gt_params_world["global_orient"].clone()
    R = torch.from_numpy(calibs["Rs"][int(camera[-2:]) - 1]).unsqueeze(0).float()
    R = R / R.norm(dim=-1)
    gt_rotmat_world = r.axis_angle_to_matrix(gt_global_orient_world)
    gt_rotmat_cam = R @ gt_rotmat_world
    gt_global_orient_cam = r.matrix_to_axis_angle(gt_rotmat_cam)
    gt_params_cam = {
        k: torch.from_numpy(groundtruth[k]).float()[frames] for k in ["body_pose", "betas"]
    }
    gt_params_cam.update({"global_orient": gt_global_orient_cam})

    with torch.no_grad():
        gt_body_cam = smplx_model(
            **gt_params_cam
        )

    # Compute PVE
    pred_verts, pred_pelv = pred_body.vertices, pred_body.joints[:, :1]
    if model in ["gvhmr", "bedlam-cliff"]:
        gt_verts = gt_body_cam.vertices
        gt_pelv = gt_body_cam.joints[:, :1]
    else:
        gt_verts_smplx = gt_body_cam.vertices
        gt_verts = torch.matmul(smplx2smpl, gt_verts_smplx)
        gt_pelv = torch.matmul(smpl_model.J_regressor, gt_verts)[:, :1]
    pred_verts = pred_verts - pred_pelv
    gt_verts = gt_verts - gt_pelv
    pve = (gt_verts - pred_verts).norm(dim=-1).mean(1).numpy() * 1e3

    S1_hat = batch_compute_similarity_transform_torch(pred_verts, gt_verts)
    pa_pve = (gt_verts - S1_hat).norm(dim=-1).mean(1).numpy() * 1e3

    # S1_hat = batch_compute_similarity_transform_torch(pred_verts[[540]], gt_verts[[540]])
    # (((gt_verts[[540]] - pred_verts[[540]]) ** 2).sum(-1) ** 0.5).mean(1) * 1e3
    # (((gt_verts[[540]] - S1_hat) ** 2).sum(-1) ** 0.5).mean(1) * 1e3
    # import pdb; pdb.set_trace()
    
    import imageio
    camera_i = int(camera[-1]) - 1
    video_pth = f"outputs/vis/benchmark/{model}/{sequence}_cam0{camera_i}.mp4"
    os.makedirs(os.path.dirname(video_pth))
    writer = imageio.get_writer(video_pth, fps=60)
    
    K = calibs["Ks"][camera_i].copy()
    dist = calibs["dists"][camera_i].copy()
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, dist, (3840, 2160), np.eye(3), balance=0.0
    )
    smplx_renderer = Renderer(3840, 2160, 
                              focal_length=None, 
                            #   K=torch.from_numpy(calibs["Ks"][camera_i]).float(), 
                              K=torch.from_numpy(new_K).float(), 
                              device="cuda", 
                              faces=smplx_model.faces)
    
    cam_R = torch.from_numpy(calibs["Rs"][camera_i]).float()
    cam_T = torch.from_numpy(calibs["Ts"][camera_i]).float()
    smplx_renderer.cameras = smplx_renderer.create_camera(R=cam_R, T=cam_T)
    
    for trg_frame in tqdm(range(len(frames))):
        if "pred_focal_length" in data:
            pred_focal_length = data['pred_focal_length'][trg_frame]
        else:
            pred_focal_length = 5000.0 / 256 * 3840
        
        image = cv2.imread(image_pth_list[frames[trg_frame]])
        height, width = image.shape[:2]
        pred_renderer = Renderer(width, height, focal_length=pred_focal_length, device="cuda", faces=smpl_model.faces)

        out_image = pred_renderer.render_mesh(
            pred_body.vertices[trg_frame].cuda(), 
            image, 
            colors=(0.5, 0.5, 1.0),
            alpha=0.5)
        
        undist_image, _ = undistort_fisheye_image(out_image, K, dist, balance=0.0)
        
        out_image = smplx_renderer.render_mesh(
            gt_body_world.vertices[trg_frame].cuda(), 
            undist_image, 
            colors=(0.6, 0.6, 0.6),
            alpha=0.5, 
        )
        
        cv2.rectangle(out_image, (0, 0), (1200, 400), (0, 0, 0, 128), -1)
        cv2.putText(out_image, f"PVE: {pve[trg_frame]:>8.1f}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out_image, f"PA-PVE: {pa_pve[trg_frame]:>5.1f}", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 4, cv2.LINE_AA)
        writer.append_data(cv2.resize(out_image, None, fx=0.25, fy=0.25)[..., ::-1])

    writer.close() 