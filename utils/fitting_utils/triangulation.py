import cv2
import torch
import numpy as np
from tqdm import trange

def do_triangulation(pts_2d: np.ndarray, cameras: dict) -> np.ndarray:
    """
    Triangulate a 3D point from multiple 2D points across different cameras.

    Args:
        pts_2d (np.ndarray): Shape (N_views, 2), 2D points from multiple cameras.
        cameras (dict): Camera parameters dictionary containing:
                        - 'Ks': Intrinsic matrices, shape (N_views, 3, 3)
                        - 'Rs': Rotation matrices, shape (N_views, 3, 3)
                        - 'Ts': Translation vectors, shape (N_views, 3)

    Returns:
        np.ndarray: Triangulated 3D point in Cartesian coordinates, shape (3,).
    """
    N_views = pts_2d.shape[0]
    A = []
    pts_3d = np.zeros_like(pts_2d[0])
    # valid = pts_2d[..., -1].sum(0) / pts_2d.shape[0] > 0.3
    valid = (pts_2d[..., -1] > 0).sum(0) > 1
    _pts_2d = pts_2d[:, valid].copy()

    for i in range(N_views):
        u_i = _pts_2d[i, ..., 0]
        v_i = _pts_2d[i, ..., 1]
        conf = _pts_2d[i, ..., 2:]
        
        K = cameras['Ks'][i]
        R = cameras['Rs'][i]
        t = cameras['Ts'][i].reshape(3, 1)

        # Compute the projection matrix P_i = K * [R | t]
        P_i = K @ np.hstack((R, t))  # Shape: (3, 4)

        # Formulate the equations:
        A.append(conf * (u_i[..., None] * P_i[2, :] - P_i[0, :]))
        A.append(conf * (v_i[..., None] * P_i[2, :] - P_i[1, :]))

    if len(A[0].shape) == 1:
        A = np.stack(A, axis=0)
    else:
        A = np.stack(A, axis=1)

    # Solve the homogeneous system A * X = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[..., -1, :4]  # Solution is the last row of V^T
    
    X_cartesian = X_homogeneous[..., :3] / (X_homogeneous[..., 3:] + 1e-6)
    pts_3d[valid] = X_cartesian.copy()
    return pts_3d


def do_projection(pt_3d, cameras):
    """
    Project a 3D point into 2D using camera parameters.

    Args:
        pt_3d (np.ndarray): Shape (3,), 3D point.
        cameras (dict): Camera parameters dictionary.
        idx (int): Index of the camera to use.

    Returns:
        np.ndarray: Projected 2D point, shape (2,).
    """
    K = cameras['Ks']
    R = cameras['Rs']
    T = cameras['Ts']
    P = K @ np.concatenate((R, T[..., None]), axis=-1)
    
    pt_3d_hom = np.concatenate((pt_3d, np.ones_like(pt_3d[..., :1])), axis=-1)
    reprojected_hom = np.einsum('bij,nj->bni', P, pt_3d_hom)
    pt_2d = reprojected_hom[..., :2] / reprojected_hom[..., -1:]
    return pt_2d



def simple_triangulation(pt2d, cameras, apply_conf=False, conf_thr=0.3, min_valid_points=50, *args, **kwargs):
    N_f, N_c, N_j, _ = pt2d.shape
    pt3d = np.zeros_like(pt2d[:, 0])
    if apply_conf:
        confs = np.zeros_like(pt2d[:, 0, :, -1])
        conf_mask = pt2d[..., -1] < conf_thr
        pt2d[..., -1][conf_mask] = 0.0
    
    for f in trange(N_f, desc='Running triangulation', leave=False):  # Iterate over frames
        # Run by entire 3D points in a batch
        if apply_conf:
            conf = pt2d[f, ..., -1].mean(0)
            n_valid = (pt2d[f, ..., -1] > conf_thr).sum(0)
            conf[n_valid < 2] = 0.0
            confs[f] = conf.copy()
        
        n_valid_points = (pt2d[f, :][..., -1] > conf_thr).sum(-1)
        n_valid_views = (n_valid_points > min_valid_points).sum() 
        if n_valid_views < 2: continue
        
        valid_view_idxs = np.where(n_valid_points > min_valid_points)[0]

        cam_dict = {k: v[valid_view_idxs] for k, v in cameras.items() if isinstance(v, np.ndarray)}
        triang_pt3d = do_triangulation(pt2d[f, valid_view_idxs, :], cam_dict)
        pt3d[f] = triang_pt3d.copy()

    if apply_conf:
        confs[np.any(np.isinf(pt3d), axis=-1)] = 0.0
        pt3d[np.isinf(pt3d)] = 0.0
        pt3d = np.concatenate((pt3d, confs[..., None]), axis=-1)
    
    if not apply_conf:
        mask = (pt3d != 0).any(-1)
        mask = mask.astype(float)[:, :, None]
        pt3d = np.concatenate((pt3d, mask), axis=-1)

    return pt3d


def iterative_triangulation(kpts, calibs, bboxes, conf_thr=0.3, reproj_thr=0.05, n_repeat=3, min_valid_points=50):
    
    drop_view = np.zeros((kpts.shape[:3])).astype(bool)
    for i in range(n_repeat):
        kpts3d = simple_triangulation(kpts, calibs, apply_conf=True, conf_thr=conf_thr, min_valid_points=min_valid_points)
        
        # Compute reprojection loss
        proj_kpts = do_projection(kpts3d[..., :3].reshape(-1, 3), calibs).reshape(-1, *kpts3d.shape[:2], 2).transpose(1, 0, 2, 3)
        diff = proj_kpts - kpts[..., :2]
        diff = np.linalg.norm(diff, axis=-1)
        scale = (bboxes[..., 2:] - bboxes[..., :2]).max(-1)
        diff_normed = diff / scale[..., None]
        diff_normed[drop_view] = 0.0
        curr_drop_view = diff_normed == diff_normed.max(1, keepdims=True)
        curr_drop_view = np.logical_and(curr_drop_view, diff_normed > reproj_thr)
        drop_view = np.logical_or(drop_view, curr_drop_view)
        
        kpts[drop_view, ..., -1] = 0.0
    
    kpts3d = simple_triangulation(kpts, calibs, apply_conf=True, conf_thr=conf_thr, min_valid_points=min_valid_points)
    return kpts3d


def perspective_projection_torch(pt3ds: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Perspective camera projection
    
    Args:
        pt3ds: 3D points in camera coordinate system (n_f, n_j, 3)
        K: Camera intrinsic matrix  (n_f, n_c, 3, 3) or (n_f, 3, 3)
        R: Rotation matrix          (n_f, n_c, 3, 3) or (n_f, 3, 3)
        T: Translation vector       (n_f, n_c, 3) or (n_f, 3)
    """

    if K.dim() == 4:
        pt3ds = pt3ds.unsqueeze(1)
    
    pt3ds = torch.matmul(R, pt3ds.mT) + T.unsqueeze(-1)
    pt2ds = torch.matmul(K, pt3ds).mT
    pt2ds = torch.div(pt2ds[..., :2], pt2ds[..., 2:])

    return pt2ds

def undistort_and_triangulate():
    import pdb; pdb.set_trace()