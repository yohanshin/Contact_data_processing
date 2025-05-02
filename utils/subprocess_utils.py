import os
import glob
import subprocess

def run_command_with_conda(working_dir, conda_env, cmd):
    """
    Run a command in a specific working directory using a specific conda environment.
    Output goes directly to the terminal (so tqdm bars still render).
    """
    current_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        # Build a proper argument list instead of a shell string
        full_cmd = ["conda", "run", "-n", conda_env,
                    "--no-capture-output",     # don’t buffer/capture output internally
                    ] + cmd
        
        print(f"Running: {' '.join(full_cmd)}")
        # Don't redirect anything—inherit the parent’s fds
        process = subprocess.Popen(full_cmd)
        ret = process.wait()
        if ret != 0:
            print(f"❌ Command failed with exit code {ret}")
            return False
        print("✅ Command succeeded")
        return True

    except Exception as e:
        print(f"Exception while running samurai: {e}")
        return False
    finally:
        os.chdir(current_dir)




def image_to_video(
    image_folder,
    output_video_path,
    framerate=30,
    output_format="avi"  # Try avi format which has wider codec support
):
    """
    Create a video from images using the most basic FFmpeg command possible.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check image files
    image_pattern = os.path.join(image_folder, "[0-9]*.jpg")
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"No image files found with pattern {image_pattern}")
        print(f"Directory contents: {os.listdir(image_folder)[:10] if os.path.exists(image_folder) else 'Directory not found'}")
        return False
    
    # Force output file extension to match format
    base_output = os.path.splitext(output_video_path)[0]
    output_path = f"{base_output}.{output_format}"
    
    # Build a very basic FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),
        '-i', os.path.join(image_folder, '%05d.jpg'),
        output_path
    ]
    
    # Print command
    print(f"\nExecuting simple command: {' '.join(cmd)}")
    
    # Run it
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Capture output
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error: FFmpeg returned code {process.returncode}")
            print("Error details:")
            print(stderr)
            return False
        
        print(f"Video successfully created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Exception while running FFmpeg: {str(e)}")
        return False