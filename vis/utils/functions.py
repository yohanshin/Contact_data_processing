import cv2
import numpy as np

def visualize_bbox(bboxes, img, color=(0, 255, 0), thickness=2, font_scale=1.0, bbox_id=0):
    img_copy = img.copy()
    
    # Convert the data type to integer if it's not already
    if bboxes.dtype != np.int32:
        bboxes = bboxes.astype(np.int32)

    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        x1, y1, w, h = bbox
        
        # Convert YOLO's xywh (center_x, center_y, width, height) to opencv's x1y1x2y2
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        
        # Draw the rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare the text to display
        display_text = f"Bbox ID: {bbox_id} | {x1} {y1} {w} {h}"
        
        # Get the size of the text to better position it
        (text_width, text_height), baseline = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Calculate text position to ensure it's visible
        text_x = x1
        text_y = y1 - 10  # Default position above the box
        
        # Check if text would be outside the top of the image
        if text_y - text_height < 0:
            # Place text inside the top of the box instead
            text_y = y1 + text_height + 5
        
        # Check if text would be outside the right edge of the image
        if text_x + text_width > img_copy.shape[1]:
            # Align text to end at the right edge of the image with a small margin
            text_x = img_copy.shape[1] - text_width - 5
        
        # Check if text would be outside the left edge of the image
        if text_x < 0:
            text_x = 5  # Add a small margin from the left edge
            
        # Draw a semi-transparent background for the text to improve readability
        text_bg_x1 = text_x - 2
        text_bg_y1 = text_y - text_height - 2
        text_bg_x2 = text_x + text_width + 2
        text_bg_y2 = text_y + 2
        
        # Create a rectangle with semi-transparent background
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
        # Apply the overlay with transparency
        alpha = 0.6
        img_copy = cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0)
        
        # Put the text at the calculated position
        cv2.putText(img_copy, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), thickness)
    
    return img_copy
