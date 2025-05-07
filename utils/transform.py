import math
import cv2
import munkres
import numpy as np
import torch


# solution proposed in https://github.com/pytorch/pytorch/issues/229#issuecomment-299424875 
def flip_tensor(tensor, dim=0):
    """
    flip the tensor on the dimension dim
    """
    inv_idx = torch.arange(tensor.shape[dim] - 1, -1, -1).to(tensor.device)
    return tensor.index_select(dim, inv_idx)


#
# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'

    output_flipped = flip_tensor(output_flipped, dim=-1)

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def get_affine_transform(center, scale, pixel_std, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 1.0 * pixel_std  # It was scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def do_triangulation_pytorch(pts_2d: torch.Tensor,
                             Ks: torch.Tensor,
                             Es: torch.Tensor,
                             valid) -> torch.Tensor:
    """
    Triangulate 3D points from multiple 2D keypoints across different cameras,
    supporting a batch of examples and multiple joints.

    Args:
        pts_2d (torch.Tensor): Tensor of shape (B, N_views, N_joints, 2)
                               containing 2D keypoints (u, v).
        Ks (torch.Tensor): Tensor of shape (B, N_views, 3, 3) containing intrinsic matrices.
        Es (torch.Tensor): Tensor of shape (B, N_views, 3, 4) containing extrinsic matrices [R|t].

    Returns:
        torch.Tensor: Triangulated 3D points in Cartesian coordinates, shape (B, N_joints, 3).
    """
    B, N_views, N_joints, _ = pts_2d.shape

    # Compute the projection matrices for each view:
    P = torch.matmul(Ks, Es[..., :3, :])

    # Separate the u and v coordinates from pts_2d.
    u = pts_2d[..., 0:1]
    v = pts_2d[..., 1:2]

    # For each camera view, extract rows of the projection matrix.
    P0 = P[:, :, 0, :].unsqueeze(2)  # (B, N_views, 1, 4)
    P1 = P[:, :, 1, :].unsqueeze(2)  # (B, N_views, 1, 4)
    P2 = P[:, :, 2, :].unsqueeze(2)  # (B, N_views, 1, 4)

    # For each view and joint, form the two equations:
    row1 = u * P2 - P0
    row2 = v * P2 - P1

    # Concatenate the equations from all views.
    A = torch.cat([row1, row2], dim=1)

    # Create a valid mask for the equations by duplicating the valid mask for each view.
    valid_eq = torch.cat([valid, valid], dim=1)  # (B, 2*N_views, N_joints)
    weight = torch.sqrt(valid_eq.float())

    # Multiply each equation row by its corresponding weight.
    A = A * weight.unsqueeze(-1).unsqueeze(-1)  # (B, 2*N_views, N_joints, 4)

    # Rearrange the dimensions so that for each batch and joint we have a system of equations:
    A = A.permute(0, 2, 1, 3)

    # To perform SVD in batch over all joints, reshape the tensor by merging the batch and joint dimensions:
    A_reshaped = A.reshape(B * N_joints, -1, 4)

    # Solve the homogeneous system A X = 0 using SVD.
    U, S, Vh = torch.linalg.svd(A_reshaped)
    X_homogeneous = Vh[:, -1, :]  # shape (B*N_joints, 4)

    # Convert homogeneous coordinates to Cartesian by dividing by the last coordinate.
    X_cartesian = X_homogeneous[:, :3] / (X_homogeneous[:, 3:4] + 1e-6)

    # Reshape back to (B, N_joints, 3)
    X_cartesian = X_cartesian.reshape(B, N_joints, 3)

    return X_cartesian



def do_triangulation(pts_2d, Ks, Es) -> np.ndarray:
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

    for i in range(N_views):
        u_i = pts_2d[i, ..., 0]
        v_i = pts_2d[i, ..., 1]
        
        K = Ks[i]
        R = Es[i, :3, :3]
        t = Es[i, :3, -1:]

        # Compute the projection matrix P_i = K * [R | t]
        P_i = K @ np.hstack((R, t))  # Shape: (3, 4)

        # Formulate the equations:
        A.append((u_i[..., None] * P_i[2, :] - P_i[0, :]))
        A.append((v_i[..., None] * P_i[2, :] - P_i[1, :]))

    if len(A[0].shape) == 1:
        A = np.stack(A, axis=0)
    else:
        A = np.stack(A, axis=1)

    # Solve the homogeneous system A * X = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[..., -1, :4]  # Solution is the last row of V^T
    
    X_cartesian = X_homogeneous[..., :3] / (X_homogeneous[..., 3:] + 1e-6)

    return X_cartesian

def perspective_project_points(kpt3d, Ks, Es):
    if kpt3d.dim() == 2:
        kpt3d = kpt3d.unsqueeze(0)
        Ks = Ks.unsqueeze(0)
        Es = Es.unsqueeze(0)
    
    B, num_points, _ = kpt3d.shape
    
    num_views = Ks.shape[1]
    
    ones = torch.ones((B, num_points, 1), device=kpt3d.device, dtype=kpt3d.dtype)
    
    kpt3d_homog = torch.cat([kpt3d, ones], dim=-1)  # (B, num_points, 4)
    kpt3d_homog = kpt3d_homog.unsqueeze(1).expand(B, num_views, num_points, 4)
    
    X_cam = torch.matmul(Es, kpt3d_homog.unsqueeze(-1)).squeeze(-1)  # (B, num_views, num_points, 4)
    X_cam_3 = X_cam[..., :3]
    
    x_proj = torch.matmul(Ks, X_cam_3.unsqueeze(-1)).squeeze(-1)  # (B, num_views, num_points, 3)
    u = x_proj[..., 0] / x_proj[..., 2]
    v = x_proj[..., 1] / x_proj[..., 2]
    kpt2d = torch.stack((u, v), dim=-1)
    
    if kpt2d.shape[0] == 1:
        kpt2d = kpt2d[0]
    
    return kpt2d

def transform_keypoints(bboxes, keypoints):
    # Update after crop
    w, h = 192, 256
    pixel_std = 200
    aspect_ratio = 192/256
    num_joints = keypoints.shape[1]

    x1, y1, x2, y2 = np.split(bboxes, 4, axis=1)
    _w = x2 - x1
    _h = y2 - y1
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    
    mask = (_w > aspect_ratio * _h).copy()
    _h[mask] = _w[mask] * 1.0 / aspect_ratio
    _w[mask] = _h[mask] * aspect_ratio
    scale = np.array(
        [_w * 1.0 / pixel_std, _h * 1.0 / pixel_std],
        dtype=np.float32)
    center = center.transpose(1, 0, 2)
    scale = scale.transpose(1, 0, 2)

    x1, y1 = np.split(center - scale * 0.5 * pixel_std, 2, axis=1)
    x2, y2 = np.split(center + scale * 0.5 * pixel_std, 2, axis=1)
    x1, y1, x2, y2 = x1[:, 0, 0], y1[:, 0, 0], x2[:, 0, 0], y2[:, 0, 0]
    out_bbox = np.array([x1, y1, x2, y2]).transpose(1, 0)

    for idx, (c, s, keypoint) in enumerate(zip(center, scale, keypoints)):
        trans = get_affine_transform(c[:, 0], s[:, 0], 0, (w, h))
        joints = keypoint.copy()
        
        for i in range(num_joints):
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        keypoints[idx] = joints
        
    return keypoints, out_bbox

def update_intrinsics(bboxes, intrinsics, w=192, h=256):
    # Update after crop
    new_intrinsics = intrinsics.copy()

    # Crop the image
    x1, y1, x2, y2 = np.split(bboxes, 4, axis=1)
    x1, y1, x2, y2 = x1[:, 0], y1[:, 0], x2[:, 0], y2[:, 0]
    W, H = x2 - x1, y2 - y1
    
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    new_cx = cx - x1
    new_cy = cy - y1
    new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2] = new_cx, new_cy
    
    # Resize the image
    fx, fy, cx, cy = new_intrinsics[:, 0, 0], new_intrinsics[:, 1, 1], new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2]
    new_fx = fx * w / W
    new_fy = fy * h / H
    new_cx = (cx * w / W)
    new_cy = (cy * h / H)
    new_intrinsics[:, 0, 0], new_intrinsics[:, 1, 1], new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2] = new_fx, new_fy, new_cx, new_cy

    return new_intrinsics