import os

import cv2
import numpy as np


def create_camera_grid(frame_images, frame_number, grid_shape=(2, 3), 
                      cam_names=None, image_size=(256, 192)):
    """
    Create a 2x3 grid of camera images with labels
    
    Parameters:
    -----------
    frame_images : list
        List of 6 camera images
    frame_number : int
        Frame number to display
    grid_shape : tuple
        Grid layout (rows, cols)
    cam_names : list
        List of camera names (default: cam01-cam06)
    image_size : tuple
        Size of each image (height, width)
    
    Returns:
    --------
    grid_image : numpy.ndarray
        Combined grid image with labels
    """
    if cam_names is None:
        cam_names = [f"cam{i:02d}" for i in range(1, 7)]
    
    # Calculate the size of the combined image with padding for frame number
    rows, cols = grid_shape
    height, width = image_size
    top_padding = 40  # Space for frame number at top
    
    # Create blank canvas for the grid
    grid_height = top_padding + rows * height
    grid_width = cols * width
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add frame number at the top
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid_image, f"Frame: {frame_number}", (10, 30), 
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Place each camera image in the grid
    for i, img in enumerate(frame_images):
        if i >= rows * cols:
            break
            
        # Calculate position
        row = i // cols
        col = i % cols
        
        y_start = top_padding + row * height
        x_start = col * width
        
        # Make sure the image has the correct size
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
            
        # Make sure image is 3 channels (BGR)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # Place image in the grid
        grid_image[y_start:y_start+height, x_start:x_start+width] = img
        
        # Add camera name label on the top left corner of each image
        cv2.putText(grid_image, cam_names[i], (x_start + 10, y_start + 25), 
                    font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    return grid_image