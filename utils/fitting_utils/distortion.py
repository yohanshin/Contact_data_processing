import cv2
import torch
import numpy as np


def normalize_points(pt2d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Normalize points
    x_normalized = (pt2d[..., 0] - cx) / fx
    y_normalized = (pt2d[..., 1] - cy) / fy

    return np.stack([x_normalized, y_normalized], axis=-1)
    
def denormalize_points(pt2d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_pixel = pt2d[..., 0] * fx + cx
    y_pixel = pt2d[..., 1] * fy + cy

    return np.stack([x_pixel, y_pixel], axis=-1)

def do_undistortion(pts_2d, cameras):
    undistorted_pts_2d = pts_2d.copy()
    for cam_i in range(pts_2d.shape[1]):
        pt_2d = pts_2d[:, cam_i].copy()
        dist = cameras['dists'][cam_i].copy()
        K = cameras['Ks'][cam_i].copy()

        if dist.shape[-1] == 5:
            undistorted_pt_2d = undistort_points(pt_2d, dist, K)
        else:
            undistorted_pt_2d = undistort_points_fisheye(pt_2d, dist, K)
        undistorted_pt_2d = np.concatenate((undistorted_pt_2d, pt_2d[..., -1:]), axis=-1)
        undistorted_pts_2d[:, cam_i] = undistorted_pt_2d.copy()
    
    return undistorted_pts_2d

def undistort_points(pt2d, dist, intrinsics, max_iterations=10) -> torch.Tensor:
    """
    Undistort 2D keypoints using distortion parameters.

    Args:
        pt2d: (num_frames, num_joints, 2) 2D keypoints in image coordinates.
        dist: (5,) Distortion coefficients [k1, k2, p1, p2, k3].
        intrinsics: (3, 3) Camear Intrinsics
        max_iterations: Maximum number of iterations for iterative refinement.

    Returns:
        torch.Tensor: (num_frames, num_joints, 2) Undistorted 2D keypoints.
    """
    
    
    k1, k2, p1, p2, k3 = dist  # Radial and tangential distortion coefficients

    # Start with the original points as initial guess
    pt2d = normalize_points(pt2d.copy(), intrinsics)
    undistorted = pt2d.copy()  # (num_frames, num_joints, 2)

    for _ in range(max_iterations):
        x = undistorted[..., 0]
        y = undistorted[..., 1]
        r2 = x**2 + y**2  # Radial distance squared

        # Compute radial distortion
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        # Compute tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        # Apply distortion correction
        x_undistorted = (pt2d[..., 0] - x_tangential) / radial
        y_undistorted = (pt2d[..., 1] - y_tangential) / radial

        # Update undistorted points
        undistorted[..., 0] = x_undistorted
        undistorted[..., 1] = y_undistorted

    return denormalize_points(undistorted, intrinsics)


def undistort_points_fisheye(pt2d, dist, intrinsics, max_iterations=10):
    """
    Undistort 2D keypoints using fisheye distortion parameters.

    Args:
        pt2d: (num_frames, num_joints, 2) 2D keypoints in image coordinates.
        dist: (4,) Fisheye distortion coefficients [k1, k2, k3, k4].
        intrinsics: (3, 3) Camera Intrinsics
        max_iterations: Maximum number of iterations for iterative refinement.

    Returns:
        torch.Tensor: (num_frames, num_joints, 2) Undistorted 2D keypoints.
    """
    k1, k2, k3, k4 = dist  # Fisheye distortion coefficients
    
    # Start with the original points as initial guess
    pt2d_normalized = normalize_points(pt2d.copy(), intrinsics)
    undistorted = pt2d_normalized.copy()  # (num_frames, num_joints, 2)
    
    for _ in range(max_iterations):
        x = undistorted[..., 0]
        y = undistorted[..., 1]
        r = np.sqrt(x**2 + y**2)  # Radial distance
        
        # Avoid division by zero
        small_r_mask = r < 1e-8
        theta = np.arctan(r)
        theta[small_r_mask] = 0
        
        # Calculate distortion factor - fisheye model
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        
        # Scale factor
        scale = np.ones_like(r)
        non_zero_mask = ~small_r_mask
        scale[non_zero_mask] = theta_d[non_zero_mask] / r[non_zero_mask]
        
        # Apply distortion correction (inverse)
        x_distorted = pt2d_normalized[..., 0] / scale
        y_distorted = pt2d_normalized[..., 1] / scale
        
        # Update undistorted points
        undistorted[..., 0] = x_distorted
        undistorted[..., 1] = y_distorted
    
    return denormalize_points(undistorted, intrinsics)


def undistort_points_fisheye_torch(pt2d, dist, intrinsics, max_iterations=10):
    """
    Undistort 2D keypoints using fisheye distortion parameters (PyTorch version).

    Args:
        pt2d: (num_frames, num_joints, 2) 2D keypoints in image coordinates (torch.Tensor).
        dist: (4,) Fisheye distortion coefficients [k1, k2, k3, k4] (torch.Tensor).
        intrinsics: (3, 3) Camera Intrinsics (torch.Tensor)
        max_iterations: Maximum number of iterations for iterative refinement.

    Returns:
        torch.Tensor: (num_frames, num_joints, 2) Undistorted 2D keypoints.
    """
    k1, k2, k3, k4 = dist  # Fisheye distortion coefficients
    
    # Extract intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Normalize points
    pt2d_normalized = torch.zeros_like(pt2d)
    pt2d_normalized[..., 0] = (pt2d[..., 0] - cx) / fx
    pt2d_normalized[..., 1] = (pt2d[..., 1] - cy) / fy
    
    # Start with the original points as initial guess
    undistorted = pt2d_normalized.clone()
    
    for _ in range(max_iterations):
        x = undistorted[..., 0]
        y = undistorted[..., 1]
        r = torch.sqrt(x**2 + y**2)  # Radial distance
        
        # Avoid division by zero
        small_r_mask = r < 1e-8
        theta = torch.atan(r)
        theta = torch.where(small_r_mask, torch.zeros_like(theta), theta)
        
        # Calculate distortion factor - fisheye model
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        
        # Scale factor
        scale = torch.ones_like(r)
        scale = torch.where(~small_r_mask, theta_d / r, scale)
        
        # Apply distortion correction (inverse)
        x_distorted = pt2d_normalized[..., 0] / scale
        y_distorted = pt2d_normalized[..., 1] / scale
        
        # Update undistorted points
        undistorted[..., 0] = x_distorted
        undistorted[..., 1] = y_distorted
    
    # Denormalize points
    undistorted_img = torch.zeros_like(undistorted)
    undistorted_img[..., 0] = undistorted[..., 0] * fx + cx
    undistorted_img[..., 1] = undistorted[..., 1] * fy + cy
    
    return undistorted_img


def do_undistort_fisheye(kpts2d, Ks, dists, image_dim, balance=0.0):
    """
    Undistort 2D fisheye keypoints and compute new intrinsic matrix.

    Args:
        pt2d (np.ndarray): Array of shape (N, 2) or (frames, joints, 2), distorted points.
        K (np.ndarray): Camera intrinsics (3x3).
        dist_coeffs (np.ndarray): Distortion coefficients [k1, k2, k3, k4].
        image_dim (tuple): (width, height) of the image.
        balance (float): Trade-off between FoV and pixel quality (0=best pixel quality).

    Returns:
        undistorted_pts (np.ndarray): Undistorted points, same shape as input.
        new_K (np.ndarray): Updated intrinsics after undistortion.
    """
    
    undist_kpts = kpts2d.copy()
    new_Ks = Ks.copy()
    for c in range(kpts2d.shape[1]):
        pt2d = kpts2d[:, c].copy()
        K = Ks[c].copy()
        dist = dists[c].copy()

        # Ensure points have shape (N, 1, 2) as required by OpenCV
        original_shape = pt2d.shape
        pts = pt2d.reshape(-1, 1, 2).astype(np.float32)

        # Compute new intrinsics for rectified (undistorted) image
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, image_dim, np.eye(3), balance=balance
        )

        # Undistort points using OpenCV
        undistorted_pts = cv2.fisheye.undistortPoints(
            pts, K, dist, R=np.eye(3), P=new_K
        )

        # Reshape undistorted points back to original shape
        undistorted_pts = undistorted_pts.reshape(original_shape)
        undist_kpts[:, c] = undistorted_pts.copy()
        new_Ks[c] = new_K.copy()

    return undist_kpts, new_Ks