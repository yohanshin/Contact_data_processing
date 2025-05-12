import os
import sys
sys.path.append('./')
from collections import defaultdict
import pickle

import torch
import pycolmap
import numpy as np

from preproc import config as _C
from utils import rotation as r


def read_intrinsics(intrinsic_fname, trg_idxs=[2, 3, 4, 5, 6, 7]):
    """
    Read camera intrinsics from a calibration file.
    
    Args:
        intrinsic_fname (str): Path to the calibration file
        
    Returns:
        numpy.ndarray: Array of shape (num_cameras, 3, 3) containing intrinsic matrices
    """
    # Read the file
    with open(intrinsic_fname, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    
    # Initialize list to store camera intrinsics
    cameras = []
    
    # Process each camera line
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 7:  # Need at least camera_id, model, width, height, fx, fy, cx, cy
            continue
            
        # Extract parameters
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        fx = float(parts[4])
        fy = float(parts[5])
        cx = float(parts[6])
        cy = float(parts[7])
        dist = np.array([float(part) for part in parts[8:]])
        
        # Create 3x3 intrinsic matrix K
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        cameras.append((camera_id, K, dist))
    
    # Sort by camera ID and extract only the K matrices
    cameras.sort(key=lambda x: x[0])
    Ks = np.array([camera[1] for camera in cameras])
    dists = np.array([camera[2] for camera in cameras])
    
    return Ks[trg_idxs], dists[trg_idxs]

def get_camera_names(extrinsic_fname):
    with open(extrinsic_fname) as f:
        extrinsics = f.readlines()
        extrinsics = extrinsics[4:] ## drop the first 4 lines
        extrinsics = extrinsics[::2] ## only alternate lines
    
    camera_mapper = dict()
    for line in extrinsics:
        line = line.strip().split()
        camera_id = int(line[-2])
        image_path = line[-1]
        camera_name = image_path.split('/')[0]

        camera_mapper[camera_name] = camera_id

    return camera_mapper


def read_extrinsics(reconstruction, failure_fname, trg_cameras):

    if os.path.exists(failure_fname):
        with open(failure_fname, 'r') as file:
            bad_images = file.read().splitlines()
    else:
        bad_images = []

    Es = defaultdict(list)
    image_ids = defaultdict(list)
    camera_names = []
    for image_id, image in reconstruction.images.items():
        camera = image.name.split('/')[0]
        if not camera in trg_cameras: continue

        if image.name in bad_images:
            continue
        
        E = image.projection_matrix()
        E = np.concatenate((E, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
        Es[camera].append(E)
        image_ids[camera].append(image_id)
        camera_names.append(camera)

    import pdb; pdb.set_trace()
    _Es = np.zeros((len(trg_cameras), 4, 4))
    for camera, E in Es.items():
        E = np.stack(E, axis=0)
        rot6d = r.matrix_to_rotation_6d(torch.from_numpy(E[:, :3, :3]))
        rot6d = rot6d.mean(0)
        rotmat = r.rotation_6d_to_matrix(rot6d).numpy()
        translation = E[:, :3, -1].mean(0)

        mean_E = np.eye(4)
        mean_E[:3, :3] = rotmat.copy()
        mean_E[:3, -1] = translation.copy()
        _Es[trg_cameras.index(camera)] = mean_E

    return np.stack(_Es)


def read_calibration(intrinsic_fname, extrinsic_fname, transform_fname, reconstruction, failure_fname, exo_cameras):
    camera_mapper = get_camera_names(extrinsic_fname)
    min_val = min(camera_mapper.values())
    trg_camera_idxs = [camera_mapper[camera] - min_val for camera in exo_cameras]
    
    Ks, dists = read_intrinsics(intrinsic_fname, trg_camera_idxs)
    # Ks, dists = read_intrinsics(intrinsic_fname)
    Es = read_extrinsics(reconstruction, failure_fname, exo_cameras)
    
    T = pickle.load(open(transform_fname, "rb"))["aria01"]
    Es = Es @ T
    Rs = Es[:, :3, :3]
    Ts = Es[:, :3, -1]
    return Ks, dists, Rs, Ts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    exo_cameras = sorted([f for f in os.listdir(os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "images")) if f.startswith("cam")])
    workspace_fldr = "workspace"
    colmap_dir = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, workspace_fldr)
    colmap_reconstruction = pycolmap.Reconstruction(colmap_dir)

    intrinsic_fname = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, workspace_fldr, "cameras.txt")
    extrinsic_fname = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, workspace_fldr, "images.txt")
    transform_fname = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, workspace_fldr, "colmap_from_aria_transforms.pkl")
    refined_calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")
    failure_fname = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, workspace_fldr, "bad_images.txt")
    
    Ks, dists, Rs, Ts = read_calibration(intrinsic_fname, extrinsic_fname, transform_fname, colmap_reconstruction, failure_fname, exo_cameras)
    np.savez(refined_calib_pth, **dict(Ks=Ks, dists=dists, Rs=Rs, Ts=Ts))
    print("Saved!")