import torch
import numpy as np

import aitviewer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.models.smpl import SMPLLayer
from aitviewer.viewer import Viewer
from aitviewer.utils import path
from aitviewer.renderables.lines import LinesTrail

from vis.utils.rotation import rotation_6d_to_matrix

def addSMPLSequence(model_type, color=[0.65, 0.65, 0.65, 1.0], bm=None, **params):
    
    body_model_layer = SMPLLayer(model_type=model_type, gender='neutral', device=C.device, num_betas=16)
    if bm is not None:
        body_model_layer.bm = bm

    scene = SMPLSequence(
        smpl_layer=body_model_layer,
        device=C.device,
        color=color,
        z_up=True,
        is_rigged=False,
        show_joint_angles=False,
        **params
    )

    return scene


def addKeypointsSequence(points, T=np.array([[[1, 0, 0], [0, 0, 1], [0, -1, 0]]]), color=(0.0, 1.0, 0.5, 1.0), **params):
    n_points = points.shape[1]
    n_frames = points.shape[0]
    points = (T @ points.transpose(0, 2, 1)).transpose(0, 2, 1)
    orientations = np.eye(3)[None, None].repeat(n_frames, axis=0).repeat(n_points, axis=1)

    scene = RigidBodies(points, orientations, radius=0.03, length=0.0, radius_cylinder=0.00, color=color)

    return scene


def addSignalSequence(signal, T=np.array([[[1, 0, 0], [0, 0, 1], [0, -1, 0]]])):
    
    n_frames = signal.shape[0]
    signal = signal.reshape(n_frames, 3, -1)

    head_t = signal[:, 0, :3]
    head_r = rotation_6d_to_matrix(torch.from_numpy(signal[:, 0, 3:9])).numpy()
    lh_t = signal[:, 1, :3]
    lh_r = rotation_6d_to_matrix(torch.from_numpy(signal[:, 1, 3:9])).numpy()
    rh_t = signal[:, 2, :3]
    rh_r = rotation_6d_to_matrix(torch.from_numpy(signal[:, 2, 3:9])).numpy()
    
    # Viz signal
    positions = np.zeros((n_frames, 3, 3))
    orientations = np.zeros((n_frames, 3, 3, 3))
    for i, (ori, pos) in enumerate([(head_r, head_t), (lh_r, lh_t), (rh_r, rh_t)]):
        positions[:, i] = (T[0] @ pos.T).T
        orientations[:, i] = (T @ ori)
    
    orientations = orientations#.transpose(0,1,3,2)
    idxs = [0, 1, 2]
    scene = RigidBodies(positions[:, idxs], orientations[:, idxs], radius=0.01, length=0.15, radius_cylinder=0.005)

    return scene


def addTrajectory(position, color=(1, 0.5, 0.5, 1), T=np.array([[[1, 0, 0], [0, 0, 1], [0, -1, 0]]])):
    n_frames = position.shape[0]
    n_points = position.shape[1]
    orientations = np.eye(3)[None, None].repeat(n_frames, axis=0).repeat(n_points, axis=1)
    positions = position.copy()

    for i, (ori, pos) in enumerate(zip(orientations.transpose(1, 0, 2, 3), positions.transpose(1, 0, 2))):
        positions[:, i] = (T[0] @ pos.T).T

    scene = LinesTrail(
        positions[:, 0],
        r_base=0.007,
        color=color,
        cast_shadow=False,
        name="Trajectory",
    )
    return scene