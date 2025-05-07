import sys
import numpy as np


def _xywh2cs(bbox, pixel_std=200, aspect_ratio=192/256, scale_factor=1.1):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_factor

    return center, scale
