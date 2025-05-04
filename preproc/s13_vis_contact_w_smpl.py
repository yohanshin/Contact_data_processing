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
# SMPLX_PART_SEGM_PTH = '/home/soyongs/Data/body_models/smplx_vert_segmentation_v3.json'
SMPLX_PART_SEGM_PTH = '/home/soyongs/Data/body_models/smplx_vert_segmentation_v5.json'

name_mapper = {
    "LHip": "leftHip", 
    "RHip": "rightHip", 
    "LKnee": "leftKnee", 
    "RKnee": "rightKnee", 
    "LHand": "leftHand",
    "RHand": "rightHand",
    "LToe": "leftToeBase",
    "RToe": "rightToeBase",
    "LHeel": "leftFoot",
    "RHeel": "rightFoot",
    "LElbow": "leftElbow",
    "RElbow": "rightElbow",
    "LForearm": "leftForeArm",
    "RForearm": "rightForeArm",
    "LBicep": "leftBicep",
    "RBicep": "rightBicep",
    "LShoulder": "leftShoulder",
    "RShoulder": "rightShoulder",
    "Back": "lspine",
    "Spine": "uspine",
    "LShank": "leftShank",
    "RShank": "rightShank",
    "LThigh": "leftThigh",
    "RThigh": "rightThigh",
}

def segment_verts():
    verts = smplx().vertices.squeeze()
    joints = smplx().joints.squeeze()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    with open(_C.PROC_JSON_PTH.replace("sequence_name", _C.SEQUENCE_NAME), "rb") as f:
        proc_info = json.load(f)

    try: 
        contact = np.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "refined_contact.npy"))
    except:
        contact = np.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "contact.npy"))
    
    threshold = joblib.load(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "threshold.pkl"))
    threshold = {k: int(thr) for k, thr in zip(threshold.keys(), proc_info['sensor_threshold'])}

    smplx_part_segm = json.load(open(SMPLX_PART_SEGM_PTH, "rb"))
    smplx = SMPLX(SMPLX_MODEL_DIR).eval()
    # segment_verts()

    verts = smplx().vertices.squeeze()
    joints = smplx().joints.squeeze()
    
    # # V4 to V5
    # left = smplx_part_segm["leftShank"] + smplx_part_segm["leftKnee"]
    # right = smplx_part_segm["rightShank"] + smplx_part_segm["rightKnee"]
    # lshank = np.array(left)[torch.where(verts[left][:, 1] < -0.9329)[0]].tolist()
    # rshank = np.array(right)[torch.where(verts[right][:, 1] < -0.9277)[0]].tolist()
    # lknee = np.array(left)[torch.where(verts[left][:, 1] >= -0.9329)[0]].tolist()
    # rknee = np.array(right)[torch.where(verts[right][:, 1] >= -0.9277)[0]].tolist()
    # smplx_part_segm["leftShank"] = lshank
    # smplx_part_segm["rightShank"] = rshank
    # smplx_part_segm["leftKnee"] = lknee
    # smplx_part_segm["rightKnee"] = rknee
    
    # left = smplx_part_segm["leftHip"] + smplx_part_segm["leftThigh"]
    # right = smplx_part_segm["rightHip"] + smplx_part_segm["rightThigh"]
    # lthigh = np.array(left)[torch.where(verts[left][:, 1] < -0.5842)[0]].tolist()
    # rthigh = np.array(right)[torch.where(verts[right][:, 1] < -0.5953)[0]].tolist()
    # lhip = np.array(left)[torch.where(verts[left][:, 1] >= -0.5842)[0]].tolist()
    # rhip = np.array(right)[torch.where(verts[right][:, 1] >= -0.5953)[0]].tolist()
    # smplx_part_segm["leftThigh"] = lthigh
    # smplx_part_segm["rightThigh"] = rthigh
    # smplx_part_segm["leftHip"] = lhip
    # smplx_part_segm["rightHip"] = rhip

    # with open(SMPLX_PART_SEGM_PTH.replace("_v5.json", "_v6.json"), "w") as fopen:
    #     json.dump(smplx_part_segm, fopen)

    # # # V4 to V5
    # left = smplx_part_segm["leftShank"] + smplx_part_segm["leftKnee"]
    # right = smplx_part_segm["rightShank"] + smplx_part_segm["rightKnee"]
    # lshank = np.array(left)[torch.where(verts[left][:, 1] < -0.9029)[0]].tolist()
    # rshank = np.array(right)[torch.where(verts[right][:, 1] < -0.8977)[0]].tolist()
    # lknee = np.array(left)[torch.where(verts[left][:, 1] >= -0.9029)[0]].tolist()
    # rknee = np.array(right)[torch.where(verts[right][:, 1] >= -0.8977)[0]].tolist()
    # smplx_part_segm["leftShank"] = lshank
    # smplx_part_segm["rightShank"] = rshank

    # left = smplx_part_segm["leftThigh"] + lknee
    # right = smplx_part_segm["rightThigh"] + rknee
    # lknee = np.array(left)[torch.where(verts[left][:, 1] < -0.8029)[0]].tolist()
    # rknee = np.array(right)[torch.where(verts[right][:, 1] < -0.7977)[0]].tolist()
    # lthigh = np.array(left)[torch.where(verts[left][:, 1] >= -0.8029)[0]].tolist()
    # rthigh = np.array(right)[torch.where(verts[right][:, 1] >= -0.7977)[0]].tolist()
    # smplx_part_segm["leftKnee"] = lknee
    # smplx_part_segm["rightKnee"] = rknee
    
    # left = smplx_part_segm["leftHip"] + lthigh
    # right = smplx_part_segm["rightHip"] + rthigh
    # lthigh = np.array(left)[torch.where(verts[left][:, 1] < -0.5442)[0]].tolist()
    # rthigh = np.array(right)[torch.where(verts[right][:, 1] < -0.5553)[0]].tolist()
    # lhip = np.array(left)[torch.where(verts[left][:, 1] >= -0.5442)[0]].tolist()
    # rhip = np.array(right)[torch.where(verts[right][:, 1] >= -0.5553)[0]].tolist()
    # smplx_part_segm["leftThigh"] = lthigh
    # smplx_part_segm["rightThigh"] = rthigh
    # smplx_part_segm["leftHip"] = lhip
    # smplx_part_segm["rightHip"] = rhip

    # with open(SMPLX_PART_SEGM_PTH.replace("_v4.json", "_v5.json"), "w") as fopen:
    #     json.dump(smplx_part_segm, fopen)
    
    # # V3 to V4
    # lleg = smplx_part_segm["leftLeg"] + smplx_part_segm["leftUpLeg"] + smplx_part_segm["leftKnee"]
    # rleg = smplx_part_segm["rightLeg"] + smplx_part_segm["rightUpLeg"] + smplx_part_segm["rightKnee"]
    
    # lshank = np.array(lleg)[torch.where(verts[lleg][:, 1] < -0.8729)[0]].tolist()
    # rshank = np.array(rleg)[torch.where(verts[rleg][:, 1] < -0.8677)[0]].tolist()

    # luleg = np.array(lleg)[torch.where(verts[lleg][:, 1] > -0.7729)[0]].tolist()
    # ruleg = np.array(rleg)[torch.where(verts[rleg][:, 1] > -0.7677)[0]].tolist()
    # lknee = [i for i in lleg if not i in lshank + luleg]
    # rknee = [i for i in rleg if not i in rshank + ruleg]
    
    # lhips = smplx_part_segm["leftHip"] + luleg
    # rhips = smplx_part_segm["rightHip"] + ruleg
    # lhip = np.array(lhips)[torch.where(verts[lhips][:, 1] > -0.5142)[0]].tolist()
    # rhip = np.array(rhips)[torch.where(verts[rhips][:, 1] > -0.5253)[0]].tolist()
    # lthigh = np.array(lhips)[torch.where(verts[lhips][:, 1] <= -0.5142)[0]].tolist()
    # rthigh = np.array(rhips)[torch.where(verts[rhips][:, 1] <= -0.5253)[0]].tolist()
    # smplx_part_segm["leftShank"] = lshank
    # smplx_part_segm["leftKnee"] = lknee
    # smplx_part_segm["leftThigh"] = lthigh
    # smplx_part_segm["leftHip"] = lhip
    # smplx_part_segm["rightShank"] = rshank
    # smplx_part_segm["rightKnee"] = rknee
    # smplx_part_segm["rightThigh"] = rthigh
    # smplx_part_segm["rightHip"] = rhip
    # del smplx_part_segm["leftLeg"]
    # del smplx_part_segm["leftUpLeg"]
    # del smplx_part_segm["rightLeg"]
    # del smplx_part_segm["rightUpLeg"]
    # del smplx_part_segm["head"]
    # with open(SMPLX_PART_SEGM_PTH.replace("_v3.json", "_v4.json"), "w") as fopen:
    #     json.dump(smplx_part_segm, fopen)
    
    # # V2 to V3
    # larm = smplx_part_segm["leftForeArm"] + smplx_part_segm["leftArm"]
    # rarm = smplx_part_segm["rightForeArm"] + smplx_part_segm["rightArm"]
    # lforearm = np.array(larm)[torch.where(verts[larm][:, 0] > 0.4682)[0]].tolist()
    # rforearm = np.array(rarm)[torch.where(verts[rarm][:, 0] < -0.4729)[0]].tolist()
    # lbicep = np.array(larm)[torch.where(verts[larm][:, 0] < 0.3682)[0]].tolist()
    # rbicep = np.array(rarm)[torch.where(verts[rarm][:, 0] > -0.3729)[0]].tolist()
    # lelbow = [i for i in larm if not i in lforearm + lbicep]
    # relbow = [i for i in rarm if not i in rforearm + rbicep]
    # smplx_part_segm["leftForeArm"] = lforearm
    # smplx_part_segm["leftBicep"] = lbicep
    # smplx_part_segm["leftElbow"] = lelbow
    # smplx_part_segm["rightForeArm"] = rforearm
    # smplx_part_segm["rightBicep"] = rbicep
    # smplx_part_segm["rightElbow"] = relbow
    
    # del smplx_part_segm["leftArm"]
    # del smplx_part_segm["rightArm"]
    # del smplx_part_segm["leftEye"]
    # del smplx_part_segm["rightEye"]
    # del smplx_part_segm["leftHandIndex1"]
    # del smplx_part_segm["rightHandIndex1"]
    # del smplx_part_segm["neck"]
    # del smplx_part_segm["eyeballs"]

    # with open(SMPLX_PART_SEGM_PTH.replace(".json", "_v3.json"), "w") as fopen:
    #     json.dump(smplx_part_segm, fopen)

    # hips = smplx_part_segm['hips']
    # spine = smplx_part_segm['spine']
    # spine1 = smplx_part_segm['spine1']
    # spine2 = smplx_part_segm['spine2']

    # uspine = spine1 + spine2

    # uhips = np.array(hips)[torch.where(verts[hips][:, 1] > -0.3)[0].tolist()].tolist()
    # hips = np.array(hips)[torch.where(verts[hips][:, 1] <= -0.3)[0].tolist()].tolist()
    # lspine = uhips + spine
    
    # lhips = np.array(hips)[torch.where(verts[hips][:, 0] > 0)[0].tolist()].tolist()
    # rhips = np.array(hips)[torch.where(verts[hips][:, 0] <= 0)[0].tolist()].tolist()

    # lleg = smplx_part_segm["leftUpLeg"] + smplx_part_segm["leftLeg"]
    # rleg = smplx_part_segm["rightUpLeg"] + smplx_part_segm["rightLeg"]
    
    # luleg = np.array(lleg)[torch.where(verts[lleg][:, 1] > -0.7)[0]].tolist()
    # llleg = np.array(lleg)[torch.where(verts[lleg][:, 1] < -0.93)[0]].tolist()
    # ruleg = np.array(rleg)[torch.where(verts[rleg][:, 1] > -0.7)[0]].tolist()
    # rlleg = np.array(rleg)[torch.where(verts[rleg][:, 1] < -0.93)[0]].tolist()
    # lknee = [i for i in lleg if not i in luleg + llleg]
    # rknee = [i for i in rleg if not i in ruleg + rlleg]
    
    # del smplx_part_segm["hips"]
    # del smplx_part_segm["spine"]
    # del smplx_part_segm["spine1"]
    # del smplx_part_segm["spine2"]
    
    # smplx_part_segm["lspine"] = lspine
    # smplx_part_segm["uspine"] = uspine
    # smplx_part_segm["leftHip"] = lhips
    # smplx_part_segm["leftUpLeg"] = luleg
    # smplx_part_segm["leftLeg"] = llleg
    # smplx_part_segm["rightHip"] = rhips
    # smplx_part_segm["rightUpLeg"] = ruleg
    # smplx_part_segm["rightLeg"] = rlleg
    # smplx_part_segm["leftKnee"] = lknee
    # smplx_part_segm["rightKnee"] = rknee
    
    # with open(SMPLX_PART_SEGM_PTH.replace(".json", "_v2.json"), "w") as fopen:
    #     json.dump(smplx_part_segm, fopen)
    
    # Create sample SMPL model
    global_orient = torch.tensor([[np.pi, 0, 0]]).float()
    verts = smplx(global_orient=global_orient).vertices.squeeze(0)
    verts[..., 2] += 3.0
    verts[..., 1] += 0.2
    default_colors = torch.ones_like(verts).unsqueeze(0) * 0.8

    BG = np.ones((640, 640, 3)).astype(np.uint8) * 35
    renderer = Renderer(640, 640, 825, device="cuda", faces=smplx.faces)
    # image = renderer.render_mesh(verts.cuda(), BG.copy(), default_colors)

    video = imageio.get_writer(os.path.join(_C.PROC_CONTACT_DIR, _C.SEQUENCE_NAME, "video_w_smpl.avi"), fps=30, format="FFMPEG", mode="I")
    image_path_list = glob.glob(
        os.path.join(_C.PROC_IMAGE_DIR, _C.SEQUENCE_NAME, "cam06", "*.jpg")
    )
    image_path_list = sorted(image_path_list)
    from tqdm import tqdm
    for curr_contact, image_path in tqdm(zip(contact, image_path_list), total=len(contact)):
        colors = default_colors.clone()
        for (name, thr), value in zip(threshold.items(), curr_contact):
            if value > thr:
                name2 = name_mapper[name]
                colors[:, smplx_part_segm[name2]] = torch.tensor(([0.5, 0.5, 1.0])).float()
            
        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        w = int(640 / H * W)
        out_image = renderer.render_mesh(verts.cuda(), BG.copy(), colors)
        image = cv2.resize(image, (w, 640))
        image = np.concatenate((image, out_image), axis=1)
        
        video.append_data(image[..., ::-1])
    video.close()