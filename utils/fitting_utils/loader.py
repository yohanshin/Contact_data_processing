import os
import glob
import os.path as osp
import numpy as np

def compute_points_in_bbox_mask_vectorized(points, bboxes):
    """
    Compute a mask indicating whether points are inside bounding boxes (vectorized).
    
    Args:
        points: numpy array of shape (N_frames, N_views, N_points, 2)
               where each point is (x, y)
        bboxes: numpy array of shape (N_frames, N_views, 4)
               where each bbox is (x, y, w, h) - (x, y) is the top-left corner
    
    Returns:
        mask: boolean numpy array of shape (N_frames, N_views, N_points)
              True if point is inside bbox, False otherwise
    """
    # Extract shapes
    N_frames, N_views, N_points, _ = points.shape
    
    # Reshape bboxes to broadcast properly with points
    # From (N_frames, N_views, 4) to (N_frames, N_views, 1, 4)
    bboxes_expanded = bboxes[:, :, np.newaxis, :]
    
    # Extract bbox coordinates
    bbox_x = bboxes_expanded[:, :, :, 0]  # (N_frames, N_views, 1)
    bbox_y = bboxes_expanded[:, :, :, 1]
    bbox_w = bboxes_expanded[:, :, :, 2]
    bbox_h = bboxes_expanded[:, :, :, 3]
    
    # Calculate the bottom-right corner coordinates
    bbox_x2 = bbox_x + bbox_w  # (N_frames, N_views, 1)
    bbox_y2 = bbox_y + bbox_h
    
    # Extract point coordinates
    point_x = points[:, :, :, 0]  # (N_frames, N_views, N_points)
    point_y = points[:, :, :, 1]
    
    # Check if points are inside bboxes
    x_inside = (bbox_x <= point_x) & (point_x <= bbox_x2)  # (N_frames, N_views, N_points)
    y_inside = (bbox_y <= point_y) & (point_y <= bbox_y2)
    
    # Both x and y must be inside
    mask = x_inside & y_inside  # (N_frames, N_views, N_points)
    
    return mask

def read_data(vitpose_dir, dense_vitpose_dir, bbox_dir, calib_path):
    vitpose_path_list = sorted(glob.glob(osp.join(vitpose_dir, '*.npy')))
    dense_vitpose_path_list = sorted(glob.glob(osp.join(dense_vitpose_dir, '*.npy')))
    
    # Read detection
    vitpose_results = [np.load(detection_path).reshape(-1, 17, 3) for detection_path in vitpose_path_list]
    dense_vitpose_results = [np.load(detection_path) for detection_path in dense_vitpose_path_list]
    n_frames = min([len(vitpose_result) for vitpose_result in vitpose_results])
    vitpose_results = np.stack([vitpose_result[:n_frames] for vitpose_result in vitpose_results], axis=1)
    dense_vitpose_results = np.stack([dense_vitpose_result[:n_frames] for dense_vitpose_result in dense_vitpose_results], axis=1)
    dense_vitpose_results = dense_vitpose_results[..., [1, 0, 2]]
    kpts = np.concatenate((vitpose_results, dense_vitpose_results), axis=-2)
    
    # Read bboxes
    bboxes = []
    cameras = [osp.basename(detection_path).split('.npy')[0] for detection_path in vitpose_path_list]
    for camera in cameras:
        bbox_path_list = sorted(glob.glob(osp.join(bbox_dir, camera, 'bbox', '*.npy')))
        boxes = np.stack([np.load(bbox_path) for bbox_path in bbox_path_list], axis=0)
        bboxes.append(boxes[:n_frames])
    bboxes = np.stack(bboxes, axis=1)
    
    # Remove failed kpts
    inbound_mask = compute_points_in_bbox_mask_vectorized(kpts, bboxes)
    kpts[~inbound_mask] = 0.0

    # Read calib
    calibs = dict(np.load(calib_path, allow_pickle=True))

    return kpts, bboxes, calibs