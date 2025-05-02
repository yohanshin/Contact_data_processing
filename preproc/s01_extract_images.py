import os
import sys
sys.path.append('./')
import glob
import subprocess
import argparse

from preproc import config as _C

def extract_frames_ffmpeg(video_path, output_dir, start_number=1):
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        start_number (int): Frame number to start with (for consecutive numbering)
    """
    output_pattern = os.path.join(output_dir, "%05d.jpg")
    
    # Build ffmpeg command with start_number option
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "11",        # JPEG quality
        "-start_number", str(start_number),  # Start frame numbering from this value
        "-f", "image2",      # Force image2 format
        output_pattern
    ]
    
    # Print the command for debugging
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    # Run the command
    try:
        # Use subprocess.run with timeout for safety
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Error extracting frames from {os.path.basename(video_path)}:")
            print(result.stderr)
            return 0
        
        # Try to get the number of frames extracted
        frame_count = len(glob.glob(os.path.join(output_dir, "*.jpg")))
        if start_number > 1:
            # If we're using start_number, we need to count only new frames
            frame_count = len(glob.glob(os.path.join(output_dir, "*.jpg"))) - (start_number - 1)
        
        print(f"Successfully extracted {frame_count} frames from {os.path.basename(video_path)}")
        return frame_count
        
    except subprocess.TimeoutExpired:
        print(f"Process timed out after 1 hour for {os.path.basename(video_path)}")
        return 0
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence
    
    # Create the main output directory
    os.makedirs(_C.EXTRACT_IMAGE_DIR, exist_ok=True)
    print(f"Output directory: {_C.EXTRACT_IMAGE_DIR}")
    
    # Process each camera sequentially for reliability
    all_cameras = [_C.SLAM_CAM_NAME] + _C.CAMERA_NAMES
    print(f"Processing {len(all_cameras)} cameras: {all_cameras}")
    
    for camera in all_cameras:
        print(f"\n--- Processing camera: {camera} ---")
        
        # Find the video files (case insensitive pattern)
        video_files = glob.glob(os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, camera, "*.MP4"), recursive=False)
        video_files += glob.glob(os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, camera, "*.mp4"), recursive=False)
        
        if not video_files:
            print(f"No MP4 files found for camera: {camera}")
            print(f"Searched in: {os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, camera)}")
            continue
        
        # Sort video files to ensure proper processing order
        video_files.sort()
        print(f"Found {len(video_files)} video files for camera {camera}")
        
        # Create output directory for this camera
        out_image_path = os.path.join(_C.EXTRACT_IMAGE_DIR, _C.SEQUENCE_NAME, camera)
        os.makedirs(out_image_path, exist_ok=True)
        print(f"Saving frames to: {out_image_path}")
        
        # Initialize frame counter
        next_frame_number = 1
        
        # Process each video file in sequence
        for i, video_path in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
            
            # Extract frames with consecutive numbering
            frames_extracted = extract_frames_ffmpeg(
                video_path=video_path,
                output_dir=out_image_path,
                start_number=next_frame_number
            )
            
            # Update next_frame_number for consecutive numbering
            next_frame_number += frames_extracted
            
        print(f"Completed extraction for camera: {camera}, total frames: {next_frame_number-1}")
    
    # Copy Aria images
    aria_image_fldr = os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, _C.ARIA_CAM_NAME, 'images', 'rgb')
    aria_image_fldr_new = os.path.join(_C.EXTRACT_IMAGE_DIR, _C.SEQUENCE_NAME, _C.ARIA_CAM_NAME)
    os.system(f'cp -r {aria_image_fldr} {aria_image_fldr_new}')

    print("\n--- All processing complete ---")