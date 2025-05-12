import os
import sys
sys.path.append('./')
import glob
import argparse

import json
import joblib
import torch
import numpy as np
from smplx import SMPL, SMPLX
from tqdm import tqdm

from preproc import config as _C
from utils import rotation as r
from benchmark.B1_3dhp.eval_utils import (batch_align_by_pelvis, 
                                          batch_compute_similarity_transform_torch,
                                          compute_error_accel)

SMPL_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smpl"
SMPLX_MODEL_DIR = f"/home/{os.getenv('USER')}/Data/body_models/smplx"
SMPLX2SMPL_PTH = f"/home/{os.getenv('USER')}/Data/body_models/smplx2smpl.pkl"
test_sequences = ["08_soyongs_highland-park_01", 
                  "13_titus_tepper_03", 
                  "24_holly_tepper_01", 
                  "25_holly_outdoor_01", 
                #   "27_evy_scaife_01"
                  ]

skip_mapper = {
    "camerahmr": 5,
    "hmr2": 5,
    "tokenhmr": 5,
    "nlf": 5,
}

fps = 30
metric = {
    "mpjpe": 1e3, # mm
    "pa_mpjpe": 1e3, # mm
    "pve": 1e3, # mm
    "pa_pve": 1e3, # mm
    "accel": fps ** 2,  # m/s^2
    "jitter": fps ** 3 / 1e2,  # 10^2m/s^3
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='camerahmr')
    parser.add_argument('-p', '--protocol', default='baseline')
    args = parser.parse_args()
    smpl_model = SMPL(SMPL_MODEL_DIR, gender="neutral", num_betas=10).eval()

    smplx2smpl = torch.from_numpy(
        joblib.load(SMPLX2SMPL_PTH)["matrix"]
    ).float()

    eval_results = dict(
        pve=[],
        pa_pve=[],
        mpjpe=[],
        pa_mpjpe=[],
        accel=[],
        jitter=[], 
        jitter_gt=[], 
        vid_names=[]
    )

    # for sequence in tqdm(test_sequences):
    for sequence in ["25_holly_outdoor_01"]:
    # for sequence in ["24_holly_tepper_01"]:
        _C.SEQUENCE_NAME = sequence

        with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
            proc_info = json.load(f)

        calib_pth = os.path.join(_C.PROC_CALIB_DIR, sequence, "calib.npz")
        calibs = np.load(calib_pth, allow_pickle=True)
        
        groundtruth_pth = os.path.join(_C.SMPLIFYX_RESULTS_DIR, sequence, "results.pkl")
        groundtruth = joblib.load(groundtruth_pth)
        max_frame_id = len(groundtruth["betas"])

        results_pth_list = sorted(glob.glob(os.path.join(_C.BENCHMARK_3DHP_RESULT_DIR, args.protocol, args.model, sequence, "*.pkl")))
        for camera_i, results_pth in tqdm(enumerate(results_pth_list), total=len(results_pth_list), leave=False):
            data = joblib.load(results_pth)
            frames = data["frame_ids"]
            if len(frames) < 10: continue
            valid_frames = np.where(frames < max_frame_id)[0]
            frames = frames[valid_frames]
            
            smplx_model = SMPLX(SMPLX_MODEL_DIR, gender=proc_info["gender"], 
                                num_betas=11, 
                                batch_size=frames.shape[0]).eval()
            
            # Convert world-coordinate GT to camera coordinate
            R = torch.from_numpy(calibs["Rs"][camera_i]).unsqueeze(0).float()
            R = R / R.norm(dim=-1)
            gt_rotmat_world = r.axis_angle_to_matrix(torch.from_numpy(groundtruth["global_orient"]).float())
            gt_rotmat_cam = R @ gt_rotmat_world
            gt_global_orient = r.matrix_to_axis_angle(gt_rotmat_cam)
            
            gt_params = dict(
                global_orient=gt_global_orient[frames],
                body_pose=torch.from_numpy(groundtruth["body_pose"]).float()[frames],
                betas=torch.from_numpy(groundtruth["betas"]).float()[frames],
            )
            with torch.no_grad():
                gt_body = smplx_model(
                    **gt_params
                )

            gt_verts = torch.matmul(smplx2smpl, gt_body.vertices)
            gt_joints = torch.matmul(smpl_model.J_regressor, gt_verts)[:, :24]

            # Build prediction
            pred_params = dict(
                global_orient=torch.from_numpy(data["global_orient"][valid_frames]).float(),
                body_pose=torch.from_numpy(data["body_pose"][valid_frames]).float(),
                betas=torch.from_numpy(data["betas"][valid_frames]).float()
            )

            if args.model in ["gvhmr", "bedlam-cliff"]:
                num_betas = 10 if args.model == "gvhmr" else "bedlam-cliff"
                body_model = SMPLX(SMPLX_MODEL_DIR, gender="neutral", 
                                num_betas=num_betas, 
                                batch_size=frames.shape[0]).eval()
                
                with torch.no_grad():
                    pred_body = body_model(
                        **pred_params
                    )

                pred_verts = torch.matmul(smplx2smpl, pred_body.vertices)
                pred_joints = torch.matmul(smpl_model.J_regressor, pred_verts)[:, :24]
                
            else:
                with torch.no_grad():
                    pred_body = smpl_model(
                        **pred_params
                    )

                pred_verts = pred_body.vertices
                pred_joints = pred_body.joints[:, :24]

            gt_joints, pred_joints, gt_verts, pred_verts = batch_align_by_pelvis([gt_joints, pred_joints, gt_verts, pred_verts], [0])
            
            # MPJPE
            mpjpe = (gt_joints - pred_joints).norm(dim=-1).mean(1) * metric["mpjpe"]
            
            # PA-MPJPE
            S1_hat = batch_compute_similarity_transform_torch(pred_joints, gt_joints)
            pa_mpjpe = (gt_joints - S1_hat).norm(dim=-1).mean(1) * metric["pa_mpjpe"]

            # PVE
            pve = (gt_verts - pred_verts).norm(dim=-1).mean(1) * metric["pve"]
            
            # PA-PVE
            S1_hat = batch_compute_similarity_transform_torch(pred_verts, gt_verts)
            pa_pve = (gt_verts - S1_hat).norm(dim=-1).mean(1) * metric["pa_pve"]

            # Accel
            gt_accel = (gt_joints[2:] - 2 * gt_joints[1:-1] + gt_joints[:-2])
            pred_accel = (pred_joints[2:] - 2 * pred_joints[1:-1] + pred_joints[:-2])
            accel = (gt_accel - pred_accel).norm(dim=-1).mean(1) * metric["accel"]

            # Jitter
            pred_jitter = (pred_joints[3:] - 3 * pred_joints[2:-1] + 3 * pred_joints[1:-2] - pred_joints[:-3])
            jitter = pred_jitter.norm(dim=-1).mean(1) * metric["jitter"]

            # Jitter (GT)
            gt_jitter = (gt_joints[3:] - 3 * gt_joints[2:-1] + 3 * gt_joints[1:-2] - gt_joints[:-3])
            gt_jitter = gt_jitter.norm(dim=-1).mean(1) * metric["jitter"]

            eval_results["pve"].append(pve)
            eval_results["pa_pve"].append(pa_pve)
            eval_results["mpjpe"].append(mpjpe)
            eval_results["pa_mpjpe"].append(pa_mpjpe)
            eval_results["accel"].append(accel)
            eval_results["jitter"].append(jitter)
            eval_results["jitter_gt"].append(gt_jitter)
            
            vid_names = [f"{sequence}_cam{camera_i+1}_{frame}" for frame in frames]
            eval_results["vid_names"].append(vid_names)

    msg = f"Model {args.model} | Protocol {args.protocol}"
    for k, v in eval_results.items():
        eval_results[k] = np.concatenate(v, axis=0)
        if k != "vid_names":
            msg += f"  | {k}: {eval_results[k].mean():1f} mm"
    print(msg)
    print()
    
    # import pdb; pdb.set_trace()
    # worst_first_idxs = np.argsort(eval_results["pve"])[::-1]
    # eval_results["vid_names"][worst_first_idxs]