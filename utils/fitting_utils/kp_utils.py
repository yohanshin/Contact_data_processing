import os
import torch
import numpy as np

def filter_keypoints_2d(keypoints_2d, bboxes, threshold=0.1):
    sx = bboxes[..., 2] - bboxes[..., 0]
    sy = bboxes[..., 3] - bboxes[..., 1]
    scale = np.stack((sx, sy)).max(0)[..., None]
    
    masks = np.zeros_like(keypoints_2d[..., 0])
    masks[1:-1] = np.logical_or(
        np.abs(keypoints_2d[1:-1] - keypoints_2d[:-2])[..., :2].max(-1) > scale[1:-1] * threshold, 
        np.abs(keypoints_2d[2:] - keypoints_2d[1:-1])[..., :2].max(-1) > scale[1:-1] * threshold, )
    
    masks = masks.astype(bool)
    keypoints_2d[masks, -1] = 0.0
    return keypoints_2d

def filter_keypoints_3d(keypoints_3d, threshold=0.1, med_threshold=2.0):
    
    vel1 = np.linalg.norm(keypoints_3d[1:-1, ..., :3] - keypoints_3d[:-2, ..., :3], axis=-1)
    vel2 = np.linalg.norm(keypoints_3d[2:, ..., :3] - keypoints_3d[1:-1, ..., :3], axis=-1)

    mask1 = np.logical_and(vel1 > threshold, np.logical_and(keypoints_3d[1:-1, ..., -1] > 0.0, keypoints_3d[:-2, ..., -1] > 0.0))
    mask2 = np.logical_and(vel2 > threshold, np.logical_and(keypoints_3d[2:, ..., -1] > 0.0, keypoints_3d[1:-1, ..., -1] > 0.0))

    masks = np.zeros_like(keypoints_3d[..., 0])
    masks[1:-1] = np.logical_or(mask1, mask2)
    masks = masks.astype(bool)
    
    keypoints_3d[masks, -1] = 0.0
    
    median_masks = []
    for keypoint in keypoints_3d:
        median = np.median(keypoint[keypoint[:, 3] > 0.0], axis=0)[:3]
        dist_to_median = np.linalg.norm(keypoint[:, :3] - median, axis=-1)
        median_masks.append(dist_to_median > med_threshold)
    median_masks = np.stack(median_masks)
    keypoints_3d[median_masks, -1] = 0.0

    return keypoints_3d


def smooth_keypoints(keypoints: np.ndarray, dim: int = 3, kernel_size: int = 5, kernel_type: str = "uniform") -> np.ndarray:
    """
    Smooth noisy keypoints using a convolutional filter.

    Args:
        keypoints: (N_frames, N_joints, 3) Noisy keypoints.
        kernel_size: Size of the smoothing kernel (must be odd).
        kernel_type: Type of kernel ("uniform" or "gaussian").

    Returns:
        np.ndarray: (N_frames, N_joints, 3) Smoothed keypoints.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd for convolution.")
    else:
        pad = kernel_size // 2

    # Define the smoothing kernel
    if kernel_type == "uniform":
        kernel = np.ones(kernel_size) / kernel_size
    elif kernel_type == "gaussian":
        sigma = kernel_size / 6.0  # Approximate rule for Gaussian sigma
        x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize
    else:
        raise ValueError("Unsupported kernel type. Use 'uniform' or 'gaussian'.")

    # Initialize the smoothed keypoints
    smoothed_keypoints = np.zeros_like(keypoints)
    values = keypoints[..., :dim].copy()

    if keypoints.shape[-1] == dim + 1:
        conf = keypoints[..., -1].copy()
    else:
        conf = None

    # Apply convolution along the temporal dimension (axis=0)
    for j in range(keypoints.shape[1]):  # For each joint
        for c in range(dim):  # For x, y, z coordinates
            if conf is not None:
                weighted_coord = values[:, j, c] * conf[:, j]
            
                # 2. Convolve weighted coord
                num = np.convolve(weighted_coord, kernel, mode='same')

                # 3. Convolve confidence
                den = np.convolve(conf[:, j], kernel, mode='same') + 1e-6

                smoothed_keypoints[:, j, c] = num / den
            
            else:
                smoothed_keypoints[:, j, c] = np.convolve(
                    keypoints[:, j, c], kernel, mode="same"
                )
        
        if conf is not None:
            smoothed_keypoints[:, j, -1] = conf[:, j]  
    
    smoothed_keypoints[:pad] = keypoints[:pad].copy()
    smoothed_keypoints[-pad:] = keypoints[-pad:].copy()

    return smoothed_keypoints


def median_filter_keypoints(keypoints, kernel_size=11):
    """
    Apply temporal median filtering to torso keypoints.
    
    Args:
        keypoints: Tensor of shape (N_frames, 4, 4) where
                  4 is the number of torso keypoints and 
                  last dimension is (x, y, z, confidence)
        kernel_size: Size of the median filter kernel (odd number)
        padding: Padding to apply (typically (kernel_size-1)//2)
    
    Returns:
        Filtered keypoints with the same shape as input
    """
    # Check input shape
    assert len(keypoints.shape) == 3, "Expected input shape (N_frames, 4, 4)"
    N_frames, num_keypoints, dims = keypoints.shape
    
    # Separate coordinates from confidence
    coords = keypoints[:, :, :3]  # (N_frames, 4, 3)
    conf = keypoints[:, :, 3:4]  # (N_frames, 4, 1)
    
    # Make sure kernel_size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Calculate padding
    pad_size = kernel_size // 2
    
    # Create padded array by repeating edge values (not zeros)
    padded_coords = torch.zeros((N_frames + 2 * pad_size, num_keypoints, 3), device=keypoints.device)
    padded_coords[pad_size:pad_size + N_frames] = coords
    
    # Fill padding with edge values (replicate padding)
    for i in range(pad_size):
        # Beginning padding (replicate first frame)
        padded_coords[i] = coords[0]
        # End padding (replicate last frame)
        padded_coords[N_frames + pad_size + i] = coords[-1]
    
    # Create output tensor
    filtered_coords = torch.zeros_like(coords)
    
    # Apply median filter manually
    for i in range(N_frames):
        window = padded_coords[i:i + kernel_size]
        filtered_coords[i] = torch.median(window, dim=0)[0]
    
    # Combine filtered coordinates with original confidence
    filtered_keypoints = torch.cat([filtered_coords, conf], dim=2)
    return filtered_keypoints


def batch_compute_similarity_transform_torch(S1, S2, return_transform=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    if return_transform:
        return R, scale, t

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat