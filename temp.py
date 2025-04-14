import cv2
import os

def save_video_from_frames(frame_directory: str, output_video_path: str, fps: int = 30, resolution: tuple = (640, 480)):
    """
    Create a video from a sequence of images in the specified directory.

    Parameters:
    - frame_directory (str): Path to the directory containing image frames.
    - output_video_path (str): Path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    - resolution (tuple): Resolution for the output video (width, height).
    """
    # Get all the frame image files in the directory, sorted by name (ensure they are in the correct order)
    frame_files = sorted(os.listdir(frame_directory))

    # Check if there are frames in the directory
    if not frame_files:
        raise ValueError("No frames found in the provided directory.")
    
    # Initialize the VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format, adjust for other formats
    out = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
    
    for frame_file in frame_files:
        # Full path to the frame image
        frame_path = os.path.join(frame_directory, frame_file)

        # Check if it's an image file (optional but ensures only image files are processed)
        if not frame_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        # Read the image frame
        frame = cv2.imread(frame_path)
        
        # Resize the frame to the specified resolution (if necessary)
        if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
            frame = cv2.resize(frame, resolution)

        # Write the frame to the video
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_video_path}")
